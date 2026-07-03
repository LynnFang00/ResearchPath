import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
import train_v6_1_cross_encoder_reranker as v61  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cross_validation_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cross_validation_report.md"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_predictions.jsonl"
DEFAULT_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_failure_examples.jsonl"
DEFAULT_BASELINE_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_exact_baseline_scoring_report.json"

NEURAL_METHOD = "v6_1_2_cross_encoder"
METHODS = [NEURAL_METHOD, *v61.BASELINE_METHODS]
MISSING_SCORE = -1.0
ORIGINAL_V61_VAL_TOPICS = {"v2_ai_for_scientific_discovery", "v2_multimodal_learning"}
ORIGINAL_V61_TEST_TOPICS = {"v2_retrieval_augmented_generation", "v2_robot_learning"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def topic_stats(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats = {}
    for topic in sorted({row["query_id"] for row in examples}):
        rows = [row for row in examples if row["query_id"] == topic]
        stats[topic] = {
            "row_count": len(rows),
            "hard_negative_count": sum(1 for row in rows if row["hard_negative"]),
            "relevance_distribution": dict(sorted(Counter(str(row["relevance_bucket"]) for row in rows).items())),
        }
    return stats


def deterministic_topic_pairs(examples: list[dict[str, Any]]) -> list[list[str]]:
    stats = topic_stats(examples)
    ordered = sorted(stats, key=lambda topic: (-stats[topic]["hard_negative_count"], topic))
    pairs = []
    for index in range(len(ordered) // 2):
        pairs.append(sorted([ordered[index], ordered[-(index + 1)]]))
    pairs.sort(key=lambda pair: (sum(stats[topic]["hard_negative_count"] for topic in pair), pair))
    return pairs


def build_folds(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = deterministic_topic_pairs(examples)
    folds = []
    all_topics = sorted({row["query_id"] for row in examples})
    for fold_index in range(4):
        val_topics = pairs[fold_index * 2]
        test_topics = pairs[fold_index * 2 + 1]
        eval_topics = set(val_topics) | set(test_topics)
        folds.append(
            {
                "fold_id": f"fold_{fold_index + 1}",
                "train_topics": [topic for topic in all_topics if topic not in eval_topics],
                "val_topics": val_topics,
                "test_topics": test_topics,
            }
        )
    return folds


def rows_for_topics(examples: list[dict[str, Any]], topics: list[str]) -> list[dict[str, Any]]:
    topic_set = set(topics)
    return sorted([row for row in examples if row["query_id"] in topic_set], key=lambda row: (row["query_id"], int(row["paper_id"])))


def baseline_score(row: dict[str, Any], method: str) -> float | None:
    return v61.baseline_score(row, method)


def score_rows_with_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        for method in v61.BASELINE_METHODS:
            score = baseline_score(row, method)
            copied[f"{method}_score_for_eval"] = score if score is not None else MISSING_SCORE
            copied[f"{method}_score_available"] = score is not None
        output.append(copied)
    return output


def add_neural_predictions(rows: list[dict[str, Any]], scores: list[float], fold_id: str, split: str) -> list[dict[str, Any]]:
    output = []
    for row, score in zip(rows, scores, strict=True):
        copied = dict(row)
        copied["fold_id"] = fold_id
        copied["cv_split"] = split
        copied["neural_score"] = float(score)
        copied[f"{NEURAL_METHOD}_score_for_eval"] = float(score)
        output.append(copied)
    return score_rows_with_baselines(output)


def score_field(method: str) -> str:
    if method == NEURAL_METHOD:
        return f"{NEURAL_METHOD}_score_for_eval"
    return f"{method}_score_for_eval"


def method_coverage(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    if method == NEURAL_METHOD:
        return {"available_rows": len(rows), "total_rows": len(rows), "coverage": 1.0, "fully_comparable": True, "unavailable_reason": None}
    available = sum(1 for row in rows if row[f"{method}_score_available"])
    reason = None
    if available == 0:
        reason = "No exact per-row score metadata is available; backend runtime scoring was not invoked in this offline CV."
    elif available < len(rows):
        reason = "Partial per-row score metadata only; missing rows are ranked last with -1.0."
    return {
        "available_rows": available,
        "total_rows": len(rows),
        "coverage": available / len(rows) if rows else 0.0,
        "fully_comparable": available == len(rows),
        "unavailable_reason": reason,
    }


def evaluate_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    coverage = method_coverage(rows, method)
    if method != NEURAL_METHOD and coverage["available_rows"] == 0:
        return {"coverage": coverage, "metrics": None, "per_topic": None}
    metrics = v61.ranking_metrics(rows, score_field(method))
    return {"coverage": coverage, "metrics": metrics["aggregate"], "per_topic": metrics["per_topic"]}


def evaluate_all(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {method: evaluate_method(rows, method) for method in METHODS}


def train_and_predict_fold(
    *,
    fold: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    model_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    transformer_available, transformer_reason = v61.transformer_weights_available(v61.MODEL_NAME)
    if model_type == "auto":
        use_transformer = transformer_available
    else:
        use_transformer = model_type == "transformer"
    if use_transformer:
        train_result = v61.train_transformer_cross_encoder(train_rows, val_rows)
        val_scores = v61.predict_transformer_cross_encoder(train_result["model"], train_result["tokenizer"], val_rows)
        test_scores = v61.predict_transformer_cross_encoder(train_result["model"], train_result["tokenizer"], test_rows)
        model_used = "cross_encoder_transformer"
    else:
        train_result = v61.train_fallback_mlp(train_rows, val_rows)
        val_scores = v61.predict_fallback_mlp(train_result["model"], train_result["vectorizer"], val_rows)
        test_scores = v61.predict_fallback_mlp(train_result["model"], train_result["vectorizer"], test_rows)
        model_used = "fallback_neural_mlp"
    val_pred = add_neural_predictions(val_rows, val_scores, fold["fold_id"], "val")
    test_pred = add_neural_predictions(test_rows, test_scores, fold["fold_id"], "test")
    fold_report = {
        **fold,
        "model_type": model_used,
        "transformer_status": {
            "local_pretrained_weights_available": transformer_available,
            "reason": transformer_reason,
        },
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "best_validation_epoch": train_result["best_epoch"],
        "best_validation_mse": train_result["best_val_mse"],
        "training_history": train_result["history"],
        "validation": evaluate_all(val_pred),
        "test": evaluate_all(test_pred),
    }
    return fold_report, val_pred, test_pred


def summarize_metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_cv(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_names = ["reading_ndcg_at_10", "topic_ndcg_at_10", "mrr_at_10", "recall_at_10", "hard_neg_at_10", "judged_at_10", "unjudged_at_10"]
    for split in ("validation", "test"):
        output[split] = {}
        for method in METHODS:
            output[split][method] = {}
            for metric in metric_names:
                values = [
                    fold[split][method]["metrics"][metric]
                    for fold in fold_reports
                    if fold[split][method]["metrics"] is not None
                ]
                output[split][method][metric] = summarize_metric(values)
    output["delta_vs_bm25"] = {}
    for split in ("validation", "test"):
        deltas = []
        for fold in fold_reports:
            neural = fold[split][NEURAL_METHOD]["metrics"]
            bm25 = fold[split]["bm25"]["metrics"]
            if neural and bm25:
                deltas.append(neural["reading_ndcg_at_10"] - bm25["reading_ndcg_at_10"])
        output["delta_vs_bm25"][split] = {"reading_ndcg_at_10": summarize_metric(deltas)}
    output["delta_vs_v4_9_guarded_text_blend"] = {
        "available": False,
        "reason": "V4.9 exact per-row scores are unavailable for V6.0.2 examples.",
    }
    return output


def prediction_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "schema_version": "v6.1.2_neural_cv_prediction",
                "fold_id": row["fold_id"],
                "split": row["cv_split"],
                "query_id": row["query_id"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "reading_value_score": float(row["reading_value_score"]),
                "topic_match_score": float(row["topic_match_score"]),
                "relevance_bucket": int(row["relevance_bucket"]),
                "hard_negative": bool(row["hard_negative"]),
                "scores": {
                    NEURAL_METHOD: float(row["neural_score"]),
                    "bm25": baseline_score(row, "bm25"),
                    "v3_3_ltr": baseline_score(row, "v3_3_ltr"),
                    "v4_1_blend": baseline_score(row, "v4_1_blend"),
                    "v4_9_guarded_text_blend": baseline_score(row, "v4_9_guarded_text_blend"),
                },
            }
        )
    return output


def failure_examples(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for fold_id in sorted({row["fold_id"] for row in predictions}):
        for split in ("val", "test"):
            split_rows = [row for row in predictions if row["fold_id"] == fold_id and row["cv_split"] == split]
            for topic in sorted({row["query_id"] for row in split_rows}):
                topic_rows = [row for row in split_rows if row["query_id"] == topic]
                ranked = sorted(topic_rows, key=lambda row: row["neural_score"], reverse=True)
                for rank, row in enumerate(ranked[:10], start=1):
                    if row["hard_negative"] or int(row["relevance_bucket"]) == 0:
                        failures.append(failure_row(row, rank, "hard_or_negative_promoted"))
                for rank, row in enumerate(ranked, start=1):
                    if int(row["relevance_bucket"]) == 2 and rank > 10:
                        failures.append(failure_row(row, rank, "good_paper_demoted_below_top10"))
                        break
    return failures[:200]


def failure_row(row: dict[str, Any], rank: int, failure_type: str) -> dict[str, Any]:
    return {
        "schema_version": "v6.1.2_neural_cv_failure_example",
        "failure_type": failure_type,
        "fold_id": row["fold_id"],
        "split": row["cv_split"],
        "query_id": row["query_id"],
        "paper_id": int(row["paper_id"]),
        "title": row["title"],
        "rank": rank,
        "neural_score": float(row["neural_score"]),
        "reading_value_score": float(row["reading_value_score"]),
        "relevance_bucket": int(row["relevance_bucket"]),
        "hard_negative": bool(row["hard_negative"]),
    }


def topic_instability(predictions: list[dict[str, Any]], fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    per_topic_metrics: dict[str, list[float]] = defaultdict(list)
    hard_promotions: Counter[str] = Counter()
    demotions: Counter[str] = Counter()
    for fold in fold_reports:
        for split in ("validation", "test"):
            for topic, metrics in fold[split][NEURAL_METHOD]["per_topic"].items():
                per_topic_metrics[topic].append(metrics["reading_ndcg_at_10"])
    for row in failure_examples(predictions):
        if row["failure_type"] == "hard_or_negative_promoted":
            hard_promotions[row["query_id"]] += 1
        if row["failure_type"] == "good_paper_demoted_below_top10":
            demotions[row["query_id"]] += 1
    topic_summary = {
        topic: {
            "reading_ndcg_at_10": summarize_metric(values),
            "hard_negative_promotion_failures": hard_promotions.get(topic, 0),
            "good_paper_demotion_failures": demotions.get(topic, 0),
        }
        for topic, values in sorted(per_topic_metrics.items())
    }
    strong = [topic for topic, row in topic_summary.items() if row["reading_ndcg_at_10"]["mean"] is not None and row["reading_ndcg_at_10"]["mean"] >= 0.8]
    failing = [topic for topic, row in topic_summary.items() if row["reading_ndcg_at_10"]["mean"] is not None and row["reading_ndcg_at_10"]["mean"] < 0.65]
    original_val = [topic_summary[topic]["reading_ndcg_at_10"]["mean"] for topic in ORIGINAL_V61_VAL_TOPICS if topic in topic_summary]
    original_test = [topic_summary[topic]["reading_ndcg_at_10"]["mean"] for topic in ORIGINAL_V61_TEST_TOPICS if topic in topic_summary]
    all_means = [row["reading_ndcg_at_10"]["mean"] for row in topic_summary.values() if row["reading_ndcg_at_10"]["mean"] is not None]
    overall = sum(all_means) / len(all_means) if all_means else 0.0
    return {
        "topic_summary": topic_summary,
        "strong_topics": strong,
        "failing_topics": failing,
        "hard_negative_promotion_topics": sorted(hard_promotions),
        "good_paper_demotion_topics": sorted(demotions),
        "original_v6_1_validation_topics_unusually_difficult": bool(original_val and sum(original_val) / len(original_val) < overall - 0.05),
        "original_v6_1_test_topics_unusually_easy": bool(original_test and sum(original_test) / len(original_test) > overall + 0.05),
    }


def baseline_scoring_report(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = {split: {} for split in ("validation", "test")}
    for split in ("validation", "test"):
        for method in v61.BASELINE_METHODS:
            values = [fold[split][method]["coverage"]["coverage"] for fold in fold_reports]
            coverage[split][method] = summarize_metric(values)
    return {
        "schema_version": "v6.1.2_exact_baseline_scoring_report",
        "created_at": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "runtime_scoring_invoked": False,
        "coverage_summary": coverage,
        "comparability": {
            "bm25": "partially comparable from V6.0.2 metadata",
            "v3_3_ltr": "not comparable: exact score metadata absent and runtime scoring not invoked",
            "v4_1_blend": "not comparable: exact score metadata absent and runtime scoring not invoked",
            "v4_9_guarded_text_blend": "not comparable: exact score metadata absent and runtime scoring not invoked",
        },
    }


def recommendation(aggregate: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, str]:
    bm25_delta = aggregate["delta_vs_bm25"]["test"]["reading_ndcg_at_10"]["mean"]
    v49_coverage = baseline_report["coverage_summary"]["test"]["v4_9_guarded_text_blend"]["mean"]
    if not v49_coverage:
        return {
            "recommendation": "improve_baseline_scoring_first",
            "reason": "Neural beats partial BM25, but exact V4.9/V4.1/V3.3 comparisons are still unavailable on these rows.",
        }
    if bm25_delta is not None and bm25_delta > 0.1:
        return {"recommendation": "proceed_to_v6_2_neural_plus_v4_9_blend", "reason": "Neural shows consistent lift and V4.9 is comparable."}
    return {"recommendation": "add_more_targeted_labels_first", "reason": "Neural lift is not stable enough."}


def render_markdown(report: dict[str, Any]) -> str:
    val = report["aggregate"]["validation"][NEURAL_METHOD]["reading_ndcg_at_10"]
    test = report["aggregate"]["test"][NEURAL_METHOD]["reading_ndcg_at_10"]
    delta_bm25 = report["aggregate"]["delta_vs_bm25"]["test"]["reading_ndcg_at_10"]
    lines = [
        "# V6.1.2 Neural Cross-Validation Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Folds: `{len(report['folds'])}`",
        f"- Model type: `{report['model_type']}`",
        f"- Validation Reading NDCG@10 mean/std: `{val['mean']:.4f}` / `{val['std']:.4f}`",
        f"- Test Reading NDCG@10 mean/std: `{test['mean']:.4f}` / `{test['std']:.4f}`",
        f"- Test neural delta vs BM25 mean: `{delta_bm25['mean']:.4f}`",
        f"- Recommendation: `{report['recommendation']['recommendation']}`",
        "",
        "## Baseline Comparability",
        "",
    ]
    for method, note in report["baseline_scoring"]["comparability"].items():
        lines.append(f"- `{method}`: {note}")
    lines.extend(["", "## Fold Topics", ""])
    for fold in report["folds"]:
        lines.append(f"- `{fold['fold_id']}` val `{fold['val_topics']}`, test `{fold['test_topics']}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    examples: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    fold_reports: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    aggregate = aggregate_cv(fold_reports)
    instability = topic_instability(predictions, fold_reports)
    rec = recommendation(aggregate, baseline_report)
    return {
        "schema_version": "v6.1.2_neural_cross_validation_report",
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
        "production_candidate_trained": False,
        "large_checkpoints_saved": False,
        "model_type": fold_reports[0]["model_type"] if fold_reports else None,
        "dataset": {
            "example_count": len(examples),
            "topic_count": len({row["query_id"] for row in examples}),
        },
        "folds": folds,
        "fold_reports": fold_reports,
        "aggregate": aggregate,
        "baseline_scoring": baseline_report,
        "instability_diagnosis": instability,
        "recommendation": rec,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--failures-out", default=str(DEFAULT_FAILURES))
    parser.add_argument("--baseline-report-out", default=str(DEFAULT_BASELINE_REPORT))
    parser.add_argument("--model-type", choices=["auto", "transformer", "fallback"], default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(resolve_repo_path(args.examples))
    folds = build_folds(examples)
    v61.TRANSFORMER_EPOCHS = args.epochs
    v61.TRANSFORMER_PATIENCE = 1
    v61.TRANSFORMER_BATCH_SIZE = args.batch_size
    v61.TRANSFORMER_MAX_LENGTH = args.max_length
    fold_reports = []
    predictions = []
    for fold in folds:
        train_rows = rows_for_topics(examples, fold["train_topics"])
        val_rows = rows_for_topics(examples, fold["val_topics"])
        test_rows = rows_for_topics(examples, fold["test_topics"])
        fold_report, val_pred, test_pred = train_and_predict_fold(
            fold=fold,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            model_type=args.model_type,
        )
        fold_reports.append(fold_report)
        predictions.extend(val_pred)
        predictions.extend(test_pred)
    baseline_report = baseline_scoring_report(fold_reports)
    failures = failure_examples(predictions)
    report = build_report(
        examples=examples,
        folds=folds,
        fold_reports=fold_reports,
        predictions=predictions,
        baseline_report=baseline_report,
        args=args,
    )
    write_jsonl(resolve_repo_path(args.predictions_out), prediction_output_rows(predictions))
    write_jsonl(resolve_repo_path(args.failures_out), failures)
    write_json(resolve_repo_path(args.baseline_report_out), baseline_report)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.1.2 neural CV report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
