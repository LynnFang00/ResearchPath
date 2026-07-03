import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from train_v6_1_cross_encoder_reranker import (  # noqa: E402
    BASELINE_METHODS,
    baseline_score,
    ranking_metrics,
    write_jsonl,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_VAL = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_val.jsonl"
DEFAULT_TEST = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_test.jsonl"
DEFAULT_VAL_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_predictions_val.jsonl"
DEFAULT_TEST_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_predictions_test.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_1_neural_vs_baselines_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_1_neural_vs_baselines_report.md"
DEFAULT_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_1_1_neural_validation_failure_examples.jsonl"
DEFAULT_VAL_OUT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_1_neural_vs_baselines_predictions_val.jsonl"
DEFAULT_TEST_OUT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_1_neural_vs_baselines_predictions_test.jsonl"

NEURAL_METHOD = "v6_1_cross_encoder"
METHODS = [NEURAL_METHOD, *BASELINE_METHODS]
MISSING_SCORE = -1.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["query_id"]), int(row["paper_id"])


def join_rows(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    predictions_by_key = {row_key(row): row for row in predictions}
    output = []
    for row in sorted(rows, key=lambda item: (item["query_id"], int(item["paper_id"]))):
        prediction = predictions_by_key[row_key(row)]
        baseline_scores = {}
        baseline_coverage = {}
        for method in BASELINE_METHODS:
            score = baseline_score(row, method)
            baseline_scores[method] = score
            baseline_coverage[method] = score is not None
        output.append(
            {
                **row,
                "split": split,
                "v6_1_neural_score": float(prediction["neural_score"]),
                "baseline_scores": baseline_scores,
                "baseline_score_available": baseline_coverage,
                "bm25_eval_score": baseline_scores["bm25"] if baseline_scores["bm25"] is not None else MISSING_SCORE,
                "v3_3_ltr_eval_score": baseline_scores["v3_3_ltr"] if baseline_scores["v3_3_ltr"] is not None else MISSING_SCORE,
                "v4_1_blend_eval_score": baseline_scores["v4_1_blend"] if baseline_scores["v4_1_blend"] is not None else MISSING_SCORE,
                "v4_9_guarded_text_blend_eval_score": baseline_scores["v4_9_guarded_text_blend"] if baseline_scores["v4_9_guarded_text_blend"] is not None else MISSING_SCORE,
            }
        )
    return output


def prediction_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "schema_version": "v6.1.1_neural_vs_baselines_prediction",
                "split": row["split"],
                "query_id": row["query_id"],
                "query": row["query"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "reading_value_score": float(row["reading_value_score"]),
                "topic_match_score": float(row["topic_match_score"]),
                "relevance_bucket": int(row["relevance_bucket"]),
                "hard_negative": bool(row["hard_negative"]),
                "section_label": row.get("section_label"),
                "generic_survey": bool(row.get("generic_survey")),
                "abstract_word_count": int(row.get("abstract_word_count") or 0),
                "scores": {
                    NEURAL_METHOD: float(row["v6_1_neural_score"]),
                    "bm25": row["baseline_scores"]["bm25"],
                    "v3_3_ltr": row["baseline_scores"]["v3_3_ltr"],
                    "v4_1_blend": row["baseline_scores"]["v4_1_blend"],
                    "v4_9_guarded_text_blend": row["baseline_scores"]["v4_9_guarded_text_blend"],
                },
                "score_available": {
                    NEURAL_METHOD: True,
                    **row["baseline_score_available"],
                },
            }
        )
    return output


def score_field_for_method(method: str) -> str:
    if method == NEURAL_METHOD:
        return "v6_1_neural_score"
    return f"{method}_eval_score"


def method_coverage(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    if method == NEURAL_METHOD:
        return {
            "available_rows": len(rows),
            "total_rows": len(rows),
            "coverage": 1.0 if rows else 0.0,
            "fully_comparable": True,
            "missing_score_policy": None,
            "unavailable_reason": None,
        }
    available = sum(1 for row in rows if row["baseline_score_available"][method])
    reason = None
    if available == 0:
        reason = "No exact V6.0.2 rank/score metadata is present for this method; runtime scoring was not invoked in this offline diagnosis."
    elif available < len(rows):
        reason = "Partial rank/score metadata only; missing rows receive -1.0 for ranking metrics and coverage is reported."
    return {
        "available_rows": available,
        "total_rows": len(rows),
        "coverage": available / len(rows) if rows else 0.0,
        "fully_comparable": available == len(rows),
        "missing_score_policy": "missing_scores_ranked_last_as_-1.0" if available < len(rows) else None,
        "unavailable_reason": reason,
    }


def method_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    coverage = method_coverage(rows, method)
    if coverage["available_rows"] == 0 and method != NEURAL_METHOD:
        return {"coverage": coverage, "metrics": None, "per_topic": None}
    metrics = ranking_metrics(rows, score_field_for_method(method))
    return {
        "coverage": coverage,
        "metrics": metrics["aggregate"],
        "per_topic": metrics["per_topic"],
    }


def evaluate_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {method: method_metrics(rows, method) for method in METHODS}


def relevance_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["relevance_bucket"]) for row in rows).items()))


def abstract_length_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    lengths = sorted(int(row.get("abstract_word_count") or 0) for row in rows)
    if not lengths:
        return {"count": 0, "average": 0.0, "min": 0, "p50": 0, "p90": 0, "max": 0, "short_count": 0}
    return {
        "count": len(lengths),
        "average": sum(lengths) / len(lengths),
        "min": lengths[0],
        "p50": lengths[len(lengths) // 2],
        "p90": lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))],
        "max": lengths[-1],
        "short_count": sum(1 for value in lengths if value < 50),
    }


def topic_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for topic in sorted({row["query_id"] for row in rows}):
        topic_rows = [row for row in rows if row["query_id"] == topic]
        output[topic] = {
            "row_count": len(topic_rows),
            "relevance_distribution": relevance_distribution(topic_rows),
            "hard_negative_count": sum(1 for row in topic_rows if row["hard_negative"]),
            "hard_negative_rate": sum(1 for row in topic_rows if row["hard_negative"]) / len(topic_rows),
            "section_distribution": dict(sorted(Counter(str(row.get("section_label")) for row in topic_rows).items())),
            "difficulty_distribution": dict(sorted(Counter(str(row.get("difficulty_label")) for row in topic_rows).items())),
            "generic_survey_count": sum(1 for row in topic_rows if row.get("generic_survey")),
            "abstract_length": abstract_length_summary(topic_rows),
        }
    return output


def ranked_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (float(row[score_field_for_method(method)]), -int(row["paper_id"])), reverse=True)


def validation_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for topic in sorted({row["query_id"] for row in rows}):
        topic_rows = [row for row in rows if row["query_id"] == topic]
        ranked = ranked_rows(topic_rows, NEURAL_METHOD)
        for rank, row in enumerate(ranked[:10], start=1):
            if row["hard_negative"] or int(row["relevance_bucket"]) == 0:
                failures.append(failure_row(row, rank, "hard_or_negative_promoted"))
            if row.get("section_label") == "application":
                failures.append(failure_row(row, rank, "application_only_promoted"))
        for rank, row in enumerate(ranked, start=1):
            if int(row["relevance_bucket"]) == 2 and rank > 10:
                failures.append(failure_row(row, rank, "good_paper_demoted_below_top10"))
                break
    return failures


def failure_row(row: dict[str, Any], rank: int, failure_type: str) -> dict[str, Any]:
    return {
        "schema_version": "v6.1.1_neural_validation_failure_example",
        "failure_type": failure_type,
        "query_id": row["query_id"],
        "paper_id": int(row["paper_id"]),
        "title": row["title"],
        "rank": rank,
        "neural_score": float(row["v6_1_neural_score"]),
        "reading_value_score": float(row["reading_value_score"]),
        "topic_match_score": float(row["topic_match_score"]),
        "relevance_bucket": int(row["relevance_bucket"]),
        "hard_negative": bool(row["hard_negative"]),
        "section_label": row.get("section_label"),
        "generic_survey": bool(row.get("generic_survey")),
    }


def failure_summary(failures: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["failure_type"] for row in failures).items()))


def win_loss_tie_vs_v49(evaluations: dict[str, Any]) -> dict[str, Any]:
    v49 = evaluations.get("v4_9_guarded_text_blend") or {}
    neural = evaluations.get(NEURAL_METHOD) or {}
    if not v49.get("per_topic"):
        return {
            "available": False,
            "reason": "V4.9 guarded text blend exact scores are unavailable for V6.0.2 val/test rows.",
            "wins": None,
            "losses": None,
            "ties": None,
            "severe_regression_count_reading_delta_lte_minus_0_05": None,
        }
    wins = losses = ties = severe = 0
    for topic, neural_metrics in neural["per_topic"].items():
        delta = neural_metrics["reading_ndcg_at_10"] - v49["per_topic"][topic]["reading_ndcg_at_10"]
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
        if delta <= -0.05:
            severe += 1
    return {
        "available": True,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "severe_regression_count_reading_delta_lte_minus_0_05": severe,
    }


def validation_harder_assessment(val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], val_eval: dict[str, Any], test_eval: dict[str, Any]) -> dict[str, Any]:
    val_profile = topic_profile(val_rows)
    test_profile = topic_profile(test_rows)
    val_hard_rate = sum(row["hard_negative"] for row in val_rows) / len(val_rows)
    test_hard_rate = sum(row["hard_negative"] for row in test_rows) / len(test_rows)
    val_neural = val_eval[NEURAL_METHOD]["metrics"]
    test_neural = test_eval[NEURAL_METHOD]["metrics"]
    return {
        "val_topics": sorted(val_profile),
        "test_topics": sorted(test_profile),
        "val_hard_negative_rate": val_hard_rate,
        "test_hard_negative_rate": test_hard_rate,
        "val_reading_ndcg": val_neural["reading_ndcg_at_10"],
        "test_reading_ndcg": test_neural["reading_ndcg_at_10"],
        "validation_appears_harder_than_test": val_neural["reading_ndcg_at_10"] + 0.05 < test_neural["reading_ndcg_at_10"],
        "likely_high_variance": True,
        "high_variance_reason": "Only two validation topics and two test topics are used; topic composition dominates aggregate metrics.",
    }


def model_selection_recommendation(val_eval: dict[str, Any], test_eval: dict[str, Any]) -> dict[str, Any]:
    neural_val = val_eval[NEURAL_METHOD]["metrics"]
    neural_test = test_eval[NEURAL_METHOD]["metrics"]
    if neural_val["reading_ndcg_at_10"] < 0.70 and neural_test["reading_ndcg_at_10"] >= 0.80:
        rec = "improve_cross_validation_first"
        reason = "Validation is weak while test is strong, which points to high split variance; stabilize model selection before pairwise training."
    elif neural_val["reading_ndcg_at_10"] >= 0.70:
        rec = "proceed_to_v6_2_pairwise_training"
        reason = "Validation ranking is adequate and pairwise data is available."
    else:
        rec = "add_more_targeted_labels_first"
        reason = "Validation ranking remains weak."
    return {"recommendation": rec, "reason": reason}


def leave_one_topic_out_note(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "available": False,
        "reason": "Frozen V6.1 predictions exist only for val/test topics, not all 16 topics. A true leave-one-topic-out estimate needs frozen predictions for every judged topic or a saved checkpoint to score all topics.",
        "topics_with_predictions": sorted({row["query_id"] for row in rows}),
    }


def build_report(
    *,
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    val_eval: dict[str, Any],
    test_eval: dict[str, Any],
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    validation_diagnosis = {
        "val_topics": sorted({row["query_id"] for row in val_rows}),
        "test_topics": sorted({row["query_id"] for row in test_rows}),
        "val_topic_profile": topic_profile(val_rows),
        "test_topic_profile": topic_profile(test_rows),
        "per_topic_neural_metrics_val": val_eval[NEURAL_METHOD]["per_topic"],
        "per_topic_neural_metrics_test": test_eval[NEURAL_METHOD]["per_topic"],
        "per_topic_baseline_metrics_val": {method: val_eval[method]["per_topic"] for method in BASELINE_METHODS if val_eval[method]["per_topic"]},
        "per_topic_baseline_metrics_test": {method: test_eval[method]["per_topic"] for method in BASELINE_METHODS if test_eval[method]["per_topic"]},
        "failure_summary": failure_summary(failures),
        "top_validation_failure_examples": failures[:25],
        "validation_harder_assessment": validation_harder_assessment(val_rows, test_rows, val_eval, test_eval),
    }
    return {
        "schema_version": "v6.1.1_neural_vs_baselines_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "model_retrained_by_this_script": False,
        "dataset": {
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "val_topics": sorted({row["query_id"] for row in val_rows}),
            "test_topics": sorted({row["query_id"] for row in test_rows}),
        },
        "methods": METHODS,
        "all_baselines_comparable": False,
        "comparison": {
            "val": val_eval,
            "test": test_eval,
        },
        "baseline_recovery": {
            "bm25": "Recovered from V6.0.2 baseline_rankings metadata with partial coverage.",
            "v3_3_ltr": "Unavailable: exact V6.0.2 rank/score metadata is absent and runtime scoring was not invoked.",
            "v4_1_blend": "Unavailable: exact V6.0.2 rank/score metadata is absent and runtime scoring was not invoked.",
            "v4_9_guarded_text_blend": "Unavailable: exact V6.0.2 rank/score metadata is absent and runtime scoring was not invoked.",
        },
        "topic_win_loss_tie_vs_v4_9": {
            "val": win_loss_tie_vs_v49(val_eval),
            "test": win_loss_tie_vs_v49(test_eval),
        },
        "validation_diagnosis": validation_diagnosis,
        "model_selection_diagnosis": model_selection_recommendation(val_eval, test_eval),
        "leave_one_topic_out": leave_one_topic_out_note(val_rows + test_rows),
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    val_neural = report["comparison"]["val"][NEURAL_METHOD]["metrics"]
    test_neural = report["comparison"]["test"][NEURAL_METHOD]["metrics"]
    bm25_val = report["comparison"]["val"]["bm25"]
    bm25_test = report["comparison"]["test"]["bm25"]
    lines = [
        "# V6.1.1 Neural Vs Baselines Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- All baselines comparable: `False`",
        f"- Validation topics: `{report['dataset']['val_topics']}`",
        f"- Test topics: `{report['dataset']['test_topics']}`",
        f"- Model-selection recommendation: `{report['model_selection_diagnosis']['recommendation']}`",
        "",
        "## Neural Metrics",
        "",
        f"- Val Reading NDCG@10: `{val_neural['reading_ndcg_at_10']:.4f}`",
        f"- Val Topic NDCG@10: `{val_neural['topic_ndcg_at_10']:.4f}`",
        f"- Val Hard-neg@10: `{val_neural['hard_neg_at_10']:.4f}`",
        f"- Test Reading NDCG@10: `{test_neural['reading_ndcg_at_10']:.4f}`",
        f"- Test Topic NDCG@10: `{test_neural['topic_ndcg_at_10']:.4f}`",
        f"- Test Hard-neg@10: `{test_neural['hard_neg_at_10']:.4f}`",
        "",
        "## BM25 Comparison",
        "",
        f"- BM25 val coverage: `{bm25_val['coverage']['available_rows']}` / `{bm25_val['coverage']['total_rows']}`",
        f"- BM25 val Reading NDCG@10: `{bm25_val['metrics']['reading_ndcg_at_10']:.4f}`",
        f"- BM25 test coverage: `{bm25_test['coverage']['available_rows']}` / `{bm25_test['coverage']['total_rows']}`",
        f"- BM25 test Reading NDCG@10: `{bm25_test['metrics']['reading_ndcg_at_10']:.4f}`",
        "",
        "## Missing Baselines",
        "",
    ]
    for method in ("v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"):
        lines.append(f"- `{method}`: {report['baseline_recovery'][method]}")
    lines.extend(
        [
            "",
            "## Validation Diagnosis",
            "",
            f"- Validation appears harder than test: `{report['validation_diagnosis']['validation_harder_assessment']['validation_appears_harder_than_test']}`",
            f"- Likely high variance: `{report['validation_diagnosis']['validation_harder_assessment']['likely_high_variance']}`",
            f"- Failure summary: `{report['validation_diagnosis']['failure_summary']}`",
            "",
            "## Protected Hashes",
            "",
        ]
    )
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default=str(DEFAULT_VAL))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--val-predictions", default=str(DEFAULT_VAL_PREDICTIONS))
    parser.add_argument("--test-predictions", default=str(DEFAULT_TEST_PREDICTIONS))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--failures-out", default=str(DEFAULT_FAILURES))
    parser.add_argument("--val-out", default=str(DEFAULT_VAL_OUT))
    parser.add_argument("--test-out", default=str(DEFAULT_TEST_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_rows = join_rows(load_jsonl(resolve_repo_path(args.val)), load_jsonl(resolve_repo_path(args.val_predictions)), "val")
    test_rows = join_rows(load_jsonl(resolve_repo_path(args.test)), load_jsonl(resolve_repo_path(args.test_predictions)), "test")
    val_eval = evaluate_methods(val_rows)
    test_eval = evaluate_methods(test_rows)
    failures = validation_failures(val_rows)
    report = build_report(
        val_rows=val_rows,
        test_rows=test_rows,
        val_eval=val_eval,
        test_eval=test_eval,
        failures=failures,
        args=args,
    )
    write_jsonl(resolve_repo_path(args.val_out), prediction_output_rows(val_rows))
    write_jsonl(resolve_repo_path(args.test_out), prediction_output_rows(test_rows))
    write_jsonl(resolve_repo_path(args.failures_out), failures)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.1.1 neural-vs-baselines report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
