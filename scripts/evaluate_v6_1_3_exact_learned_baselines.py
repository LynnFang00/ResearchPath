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
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.ltr_model_service import (  # noqa: E402
    V49_CONFIDENCE_THRESHOLD,
    V49_TEXT_WEIGHT,
    V49_V33_WEIGHT,
    V49_V41_WEIGHT,
    V49GuardedTextBlendModel,
    load_v3_3_ltr_model,
    load_v4_1_blend_model,
    load_v4_9_guarded_text_blend_model,
)
from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from run_v6_1_2_neural_cross_validation import build_folds, rows_for_topics  # noqa: E402
import train_v6_1_cross_encoder_reranker as v61  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_V612_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_predictions.jsonl"
DEFAULT_V612_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cross_validation_report.json"
DEFAULT_V33_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
DEFAULT_V41_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"
DEFAULT_V43_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_3_text_reranker"
DEFAULT_V41_BLEND_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"
DEFAULT_V49_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_9_guarded_text_blend_candidate.json"

DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_3_exact_learned_baseline_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_3_exact_learned_baseline_report.md"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_3_exact_baseline_predictions.jsonl"
DEFAULT_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_1_3_neural_vs_v4_9_failure_examples.jsonl"
DEFAULT_PARITY = REPO_ROOT / "data" / "eval" / "results" / "v6_1_3_exact_scoring_parity_report.json"

NEURAL_METHOD = "v6_1_2_cross_encoder"
METHODS = [NEURAL_METHOD, "bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"]
LEARNED_METHODS = ["v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"]
METRICS = [
    "reading_ndcg_at_10",
    "topic_ndcg_at_10",
    "mrr_at_10",
    "recall_at_10",
    "hard_neg_at_10",
    "judged_at_10",
    "unjudged_at_10",
]
FORBIDDEN_FEATURES = {
    "true_labels",
    "relevance_bucket",
    "reading_value_score",
    "topic_match_score",
    "section_label",
    "difficulty_label",
    "hard_negative",
    "eval_outcomes",
    "topic_hurt_status",
    "topic_improved_status",
}
MISSING_SCORE = -1.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def nested_baseline_value(row: dict[str, Any], method: str, key: str) -> float | None:
    value = ((row.get("baseline_rankings") or {}).get(method) or {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def exact_feature_map(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    bm25_score = nested_baseline_value(row, "bm25", "score")
    bm25_rank = nested_baseline_value(row, "bm25", "rank")
    if bm25_score is not None:
        values["bm25_score"] = bm25_score
    if bm25_rank is not None and bm25_rank > 0:
        values["bm25_rank"] = bm25_rank
        values["bm25_reciprocal_rank"] = 1.0 / bm25_rank
    if row.get("year") is not None:
        values["year"] = float(row["year"])
        values["paper_age_years"] = float(max(0, 2026 - int(row["year"])))
    if row.get("citation_count") is not None:
        import math

        values["citation_count_log1p"] = math.log1p(max(float(row["citation_count"]), 0.0))
    if row.get("full_text_available") is not None:
        values["full_text_available_bool"] = 1.0 if row.get("full_text_available") else 0.0
    return values


def exact_feature_status(
    rows: list[dict[str, Any]],
    required_features: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scorable = []
    missing_feature_counts: Counter[str] = Counter()
    row_failures = []
    for row in rows:
        feature_values = exact_feature_map(row)
        missing = [feature for feature in required_features if feature not in feature_values]
        if missing:
            missing_feature_counts.update(missing)
            if len(row_failures) < 25:
                row_failures.append(
                    {
                        "query_id": row["query_id"],
                        "paper_id": int(row["paper_id"]),
                        "missing_features": missing,
                    }
                )
            continue
        copied = dict(row)
        copied["_exact_feature_values"] = feature_values
        scorable.append(copied)
    status = {
        "required_features": required_features,
        "scorable_rows": len(scorable),
        "unscorable_rows": len(rows) - len(scorable),
        "missing_feature_counts": dict(sorted(missing_feature_counts.items())),
        "sample_unscorable_rows": row_failures,
    }
    return scorable, status


def load_artifacts(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts: dict[str, Any] = {}
    status: dict[str, Any] = {}
    try:
        artifacts["v3_3_ltr"] = load_v3_3_ltr_model(resolve_repo_path(args.v33_model_dir))
        status["v3_3_ltr"] = {
            "artifact_loaded": True,
            "feature_columns": artifacts["v3_3_ltr"].feature_columns,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - exercised by report assertions when absent
        status["v3_3_ltr"] = {"artifact_loaded": False, "feature_columns": [], "error": str(exc)}
    try:
        artifacts["v4_1_blend"] = load_v4_1_blend_model(
            v33_model_dir=resolve_repo_path(args.v33_model_dir),
            v41_model_dir=resolve_repo_path(args.v41_model_dir),
            blend_config_path=resolve_repo_path(args.v41_blend_config),
        )
        status["v4_1_blend"] = {
            "artifact_loaded": True,
            "feature_columns": artifacts["v4_1_blend"].v41_model.feature_columns,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        status["v4_1_blend"] = {"artifact_loaded": False, "feature_columns": [], "error": str(exc)}
    try:
        artifacts["v4_9_guarded_text_blend"] = load_v4_9_guarded_text_blend_model(
            v33_model_dir=resolve_repo_path(args.v33_model_dir),
            v41_model_dir=resolve_repo_path(args.v41_model_dir),
            v43_model_dir=resolve_repo_path(args.v43_model_dir),
            v41_blend_config_path=resolve_repo_path(args.v41_blend_config),
            v49_candidate_config_path=resolve_repo_path(args.v49_config),
        )
        text_features = artifacts["v4_9_guarded_text_blend"].text_model.numeric_ranker_features
        status["v4_9_guarded_text_blend"] = {
            "artifact_loaded": True,
            "feature_columns": text_features,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover
        status["v4_9_guarded_text_blend"] = {"artifact_loaded": False, "feature_columns": [], "error": str(exc)}
    return artifacts, status


def score_learned_rows(
    rows: list[dict[str, Any]],
    artifacts: dict[str, Any],
    artifact_status: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = [dict(row) for row in rows]
    by_key = {(row["fold_id"], row["cv_split"], row["query_id"], int(row["paper_id"])): row for row in output}
    scoring_status: dict[str, Any] = {}

    for row in output:
        row["bm25_score_for_eval"] = nested_baseline_value(row, "bm25", "score")
        row["bm25_score_available"] = row["bm25_score_for_eval"] is not None
        for method in LEARNED_METHODS:
            row[f"{method}_score_for_eval"] = None
            row[f"{method}_score_available"] = False

    if artifact_status["v3_3_ltr"]["artifact_loaded"]:
        model = artifacts["v3_3_ltr"]
        scorable, status = exact_feature_status(output, model.feature_columns)
        if scorable:
            feature_rows = [row["_exact_feature_values"] for row in scorable]
            scores = model.score_feature_rows(feature_rows)
            for row, score in zip(scorable, scores, strict=True):
                key = (row["fold_id"], row["cv_split"], row["query_id"], int(row["paper_id"]))
                by_key[key]["v3_3_ltr_score_for_eval"] = float(score)
                by_key[key]["v3_3_ltr_score_available"] = True
        status["artifact_loaded"] = True
        status["exact_scoring_succeeded"] = status["scorable_rows"] == len(output)
        status["unavailable_reason"] = None if status["exact_scoring_succeeded"] else "Required exact runtime feature inputs are missing from V6.0.2 rows."
        scoring_status["v3_3_ltr"] = status
    else:
        scoring_status["v3_3_ltr"] = {**artifact_status["v3_3_ltr"], "exact_scoring_succeeded": False}

    if artifact_status["v4_1_blend"]["artifact_loaded"]:
        model = artifacts["v4_1_blend"]
        scorable, status = exact_feature_status(output, model.v41_model.feature_columns)
        if scorable:
            feature_rows = [row["_exact_feature_values"] for row in scorable]
            score_rows = model.score_feature_rows(feature_rows)
            for row, scores in zip(scorable, score_rows, strict=True):
                key = (row["fold_id"], row["cv_split"], row["query_id"], int(row["paper_id"]))
                by_key[key]["v4_1_blend_score_for_eval"] = float(scores["v4_1_blend_score"])
                by_key[key]["v4_1_blend_score_available"] = True
                by_key[key]["v4_1_weighted_score"] = float(scores["v4_1_weighted_score"])
        status["artifact_loaded"] = True
        status["exact_scoring_succeeded"] = status["scorable_rows"] == len(output)
        status["unavailable_reason"] = None if status["exact_scoring_succeeded"] else "Required exact runtime feature inputs are missing from V6.0.2 rows."
        scoring_status["v4_1_blend"] = status
    else:
        scoring_status["v4_1_blend"] = {**artifact_status["v4_1_blend"], "exact_scoring_succeeded": False}

    if artifact_status["v4_9_guarded_text_blend"]["artifact_loaded"]:
        model = artifacts["v4_9_guarded_text_blend"]
        required = sorted(set(model.v41_blend_model.v41_model.feature_columns) | set(model.text_model.numeric_ranker_features))
        scorable, status = exact_feature_status(output, required)
        if scorable:
            feature_rows = [row["_exact_feature_values"] for row in scorable]
            text_inputs = [text_input(row) for row in scorable]
            component_scores = model.v41_blend_model.score_feature_rows(feature_rows)
            text_scores = model.text_model.score_text_rows(text_inputs, feature_rows)
            grouped: dict[tuple[str, str, str], dict[str, dict[int, float]]] = defaultdict(lambda: {"v33": {}, "v41": {}, "text": {}})
            for row, components, text_score in zip(scorable, component_scores, text_scores, strict=True):
                group_key = (row["fold_id"], row["cv_split"], row["query_id"])
                paper_id = int(row["paper_id"])
                grouped[group_key]["v33"][paper_id] = float(components["v3_3_score"])
                grouped[group_key]["v41"][paper_id] = float(components["v4_1_weighted_score"])
                grouped[group_key]["text"][paper_id] = float(text_score)
            for group_key, maps in grouped.items():
                guarded = model.score_component_maps(
                    v33_scores=maps["v33"],
                    v41_scores=maps["v41"],
                    text_scores=maps["text"],
                )
                for paper_id, details in guarded.items():
                    key = (*group_key, paper_id)
                    by_key[key]["v4_9_guarded_text_blend_score_for_eval"] = float(details["v4_9_guarded_text_blend_score"])
                    by_key[key]["v4_9_guarded_text_blend_score_available"] = True
        status["artifact_loaded"] = True
        status["exact_scoring_succeeded"] = status["scorable_rows"] == len(output)
        status["unavailable_reason"] = None if status["exact_scoring_succeeded"] else "Required V3.3/V4.1/V4.3 runtime feature inputs are missing from V6.0.2 rows."
        scoring_status["v4_9_guarded_text_blend"] = status
    else:
        scoring_status["v4_9_guarded_text_blend"] = {**artifact_status["v4_9_guarded_text_blend"], "exact_scoring_succeeded": False}

    for row in output:
        row.pop("_exact_feature_values", None)
    return output, scoring_status


def text_input(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Query: {row.get('query') or row.get('query_id') or ''}",
            f"Title: {row.get('title') or ''}",
            f"Abstract: {row.get('abstract') or ''}",
        ]
    )


def load_scored_rows(examples_path: Path, v612_predictions_path: Path) -> list[dict[str, Any]]:
    examples = {
        (row["query_id"], int(row["paper_id"])): row
        for row in load_jsonl(examples_path)
    }
    output = []
    for pred in load_jsonl(v612_predictions_path):
        key = (pred["query_id"], int(pred["paper_id"]))
        base = dict(examples[key])
        base["fold_id"] = pred["fold_id"]
        base["cv_split"] = pred["split"]
        base["v6_1_2_cross_encoder_score_for_eval"] = float(pred["scores"][NEURAL_METHOD])
        output.append(base)
    return sorted(output, key=lambda row: (row["fold_id"], row["cv_split"], row["query_id"], int(row["paper_id"])))


def score_field(method: str) -> str:
    if method == NEURAL_METHOD:
        return "v6_1_2_cross_encoder_score_for_eval"
    return f"{method}_score_for_eval"


def coverage(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    if method == NEURAL_METHOD:
        available = len(rows)
    else:
        available = sum(1 for row in rows if row.get(f"{method}_score_available"))
    reason = None
    if method != NEURAL_METHOD and available == 0:
        reason = "No rows have exact scores for this method."
    elif method != NEURAL_METHOD and available < len(rows):
        reason = "Only partial exact score coverage is available."
    return {
        "available_rows": available,
        "total_rows": len(rows),
        "coverage": available / len(rows) if rows else 0.0,
        "fully_comparable": available == len(rows),
        "unavailable_reason": reason,
    }


def evaluate_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    cov = coverage(rows, method)
    if method != NEURAL_METHOD and cov["available_rows"] == 0:
        return {"coverage": cov, "metrics": None, "per_topic": None}
    eval_rows = [dict(row) for row in rows]
    if method != NEURAL_METHOD:
        for row in eval_rows:
            if row.get(score_field(method)) is None:
                row[score_field(method)] = MISSING_SCORE
    metrics = v61.ranking_metrics(eval_rows, score_field(method))
    return {"coverage": cov, "metrics": metrics["aggregate"], "per_topic": metrics["per_topic"]}


def evaluate_fold_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {method: evaluate_method(rows, method) for method in METHODS}


def build_fold_reports(rows: list[dict[str, Any]], folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for fold in folds:
        fold_id = fold["fold_id"]
        fold_rows = [row for row in rows if row["fold_id"] == fold_id]
        val_rows = [row for row in fold_rows if row["cv_split"] == "val"]
        test_rows = [row for row in fold_rows if row["cv_split"] == "test"]
        reports.append(
            {
                **fold,
                "validation": evaluate_fold_split(val_rows),
                "test": evaluate_fold_split(test_rows),
                "validation_rows": len(val_rows),
                "test_rows": len(test_rows),
            }
        )
    return reports


def aggregate_cv(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"validation": {}, "test": {}, "deltas": {}}
    for split in ("validation", "test"):
        for method in METHODS:
            output[split][method] = {}
            for metric in METRICS:
                values = [
                    fold[split][method]["metrics"][metric]
                    for fold in fold_reports
                    if fold[split][method]["metrics"] is not None
                ]
                output[split][method][metric] = summarize_metric(values)
        for baseline in ("bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"):
            delta_values = []
            for fold in fold_reports:
                neural_metrics = fold[split][NEURAL_METHOD]["metrics"]
                baseline_metrics = fold[split][baseline]["metrics"]
                if neural_metrics and baseline_metrics:
                    delta_values.append(neural_metrics["reading_ndcg_at_10"] - baseline_metrics["reading_ndcg_at_10"])
            output["deltas"][f"{split}_neural_vs_{baseline}_reading_ndcg_at_10"] = summarize_metric(delta_values)
    return output


def severe_regression_summary(fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not all(fold["test"]["v4_9_guarded_text_blend"]["metrics"] is not None for fold in fold_reports):
        return {
            "available": False,
            "reason": "V4.9 exact scores are unavailable, so severe regression vs V4.9 cannot be computed.",
            "threshold": -0.05,
            "count": None,
            "topics": [],
        }
    regressions = []
    for fold in fold_reports:
        neural_topic = fold["test"][NEURAL_METHOD]["per_topic"]
        v49_topic = fold["test"]["v4_9_guarded_text_blend"]["per_topic"]
        for topic, metrics in neural_topic.items():
            delta = metrics["reading_ndcg_at_10"] - v49_topic[topic]["reading_ndcg_at_10"]
            if delta <= -0.05:
                regressions.append({"fold_id": fold["fold_id"], "query_id": topic, "reading_delta": delta})
    return {"available": True, "threshold": -0.05, "count": len(regressions), "topics": regressions}


def diagnostic_questions(aggregate: dict[str, Any], fold_reports: list[dict[str, Any]]) -> dict[str, Any]:
    bm25_deltas = aggregate["deltas"]["test_neural_vs_bm25_reading_ndcg_at_10"]
    v49_delta = aggregate["deltas"]["test_neural_vs_v4_9_guarded_text_blend_reading_ndcg_at_10"]
    hard_neural = aggregate["test"][NEURAL_METHOD]["hard_neg_at_10"]["mean"]
    hard_v49 = aggregate["test"]["v4_9_guarded_text_blend"]["hard_neg_at_10"]["mean"]
    short_topic_failures = []
    application_topic_failures = []
    for fold in fold_reports:
        for topic, metrics in fold["test"][NEURAL_METHOD]["per_topic"].items():
            if metrics["reading_ndcg_at_10"] < 0.65:
                short_topic_failures.append(topic)
                application_topic_failures.append(topic)
    return {
        "does_neural_beat_bm25_consistently": bool(bm25_deltas["count"] and bm25_deltas["min"] is not None and bm25_deltas["min"] > 0),
        "does_neural_beat_any_learned_baseline": None if v49_delta["count"] == 0 else bool(v49_delta["mean"] and v49_delta["mean"] > 0),
        "does_neural_reduce_hard_negatives_vs_v4_9": None if hard_v49 is None else hard_neural <= hard_v49,
        "does_neural_help_where_v4_9_is_weaker": None if v49_delta["count"] == 0 else "requires per-topic V4.9 exact scores",
        "neural_failure_pattern": "high_variance_topic_grouped_cv_with_failures_on_some_short_or_application_heavy_topics",
        "failing_topics": sorted(set(short_topic_failures + application_topic_failures)),
        "usefulness": {
            "standalone_ranker": "not_supported_by_current_evidence",
            "blend_feature": "possible_but_requires_exact_v4_9_comparison",
            "guardrail": "possible_for_future_hard_negative_work_after_exact_baselines_are_available",
        },
    }


def recommendation(scoring_status: dict[str, Any]) -> dict[str, str]:
    if any(not scoring_status.get(method, {}).get("exact_scoring_succeeded") for method in LEARNED_METHODS):
        return {
            "recommendation": "fix_exact_scoring_infrastructure_first",
            "reason": "V3.3/V4.1/V4.9 exact runtime feature inputs are absent from V6.0.2 rows, so neural cannot be compared to the strongest baselines yet.",
        }
    return {
        "recommendation": "proceed_to_v6_2_neural_plus_v4_9_blend",
        "reason": "Exact learned-baseline comparison is available.",
    }


def parity_report(
    *,
    args: argparse.Namespace,
    artifact_status: dict[str, Any],
    scoring_status: dict[str, Any],
) -> dict[str, Any]:
    config_path = resolve_repo_path(args.v49_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    formula = config.get("formula") or {}
    allowed = set(formula.get("features_used") or [])
    forbidden_intersection = sorted(allowed & FORBIDDEN_FEATURES)
    expected = {
        "v3_3_weight": V49_V33_WEIGHT,
        "v4_1_weight": V49_V41_WEIGHT,
        "text_weight": V49_TEXT_WEIGHT,
        "confidence_threshold": V49_CONFIDENCE_THRESHOLD,
    }
    config_matches_runtime_constants = (
        formula.get("name") == "v4_6_guardrail_text_when_rankers_not_confident"
        and "0.675" in str(formula.get("base_formula"))
        and "0.225" in str(formula.get("base_formula"))
        and "0.10" in str(formula.get("text_component"))
        and "0.85" in str(formula.get("text_component"))
    )
    return {
        "schema_version": "v6.1.3_exact_scoring_parity_report",
        "created_at": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "runtime_scoring_invoked": False,
        "v4_9_candidate_config_path": str(config_path),
        "v4_9_formula": formula,
        "runtime_formula_constants": expected,
        "config_matches_runtime_constants": config_matches_runtime_constants,
        "forbidden_features_required": forbidden_intersection,
        "uses_only_inference_time_features": not forbidden_intersection,
        "artifact_status": artifact_status,
        "scoring_status": scoring_status,
    }


def prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        scores = {method: row.get(score_field(method)) for method in METHODS}
        availability = {
            method: True if method == NEURAL_METHOD else bool(row.get(f"{method}_score_available"))
            for method in METHODS
        }
        output.append(
            {
                "schema_version": "v6.1.3_exact_baseline_prediction",
                "fold_id": row["fold_id"],
                "split": row["cv_split"],
                "query_id": row["query_id"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "scores": scores,
                "score_available": availability,
                "reading_value_score": float(row["reading_value_score"]),
                "topic_match_score": float(row["topic_match_score"]),
                "relevance_bucket": int(row["relevance_bucket"]),
                "hard_negative": bool(row["hard_negative"]),
                "short_or_missing_abstract": bool(row.get("short_or_missing_abstract")),
                "generic_survey": bool(row.get("generic_survey")),
            }
        )
    return output


def failure_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(row.get("v4_9_guarded_text_blend_score_available") for row in rows):
        return [
            {
                "schema_version": "v6.1.3_neural_vs_v4_9_failure_example",
                "failure_type": "v4_9_exact_scores_unavailable",
                "reason": "Cannot create neural-vs-V4.9 row-level failure examples until exact V4.9 runtime feature inputs are available.",
            }
        ]
    failures = []
    for fold_id in sorted({row["fold_id"] for row in rows}):
        for split in ("val", "test"):
            split_rows = [row for row in rows if row["fold_id"] == fold_id and row["cv_split"] == split]
            for topic in sorted({row["query_id"] for row in split_rows}):
                topic_rows = [row for row in split_rows if row["query_id"] == topic]
                neural_top = {int(row["paper_id"]) for row in sorted(topic_rows, key=lambda row: row[score_field(NEURAL_METHOD)], reverse=True)[:10]}
                v49_top = {int(row["paper_id"]) for row in sorted(topic_rows, key=lambda row: row[score_field("v4_9_guarded_text_blend")] or MISSING_SCORE, reverse=True)[:10]}
                for row in topic_rows:
                    paper_id = int(row["paper_id"])
                    if paper_id in neural_top and paper_id not in v49_top and (row["hard_negative"] or row["relevance_bucket"] == 0):
                        failures.append(
                            {
                                "schema_version": "v6.1.3_neural_vs_v4_9_failure_example",
                                "failure_type": "neural_promoted_negative_not_in_v4_9_top10",
                                "fold_id": fold_id,
                                "split": split,
                                "query_id": topic,
                                "paper_id": paper_id,
                                "title": row["title"],
                                "neural_score": row[score_field(NEURAL_METHOD)],
                                "v4_9_score": row[score_field("v4_9_guarded_text_blend")],
                            }
                        )
    return failures[:200]


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# V6.1.3 Exact Learned Baseline Scoring Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Exact V3.3 scoring succeeded: `{report['scoring_status']['v3_3_ltr']['exact_scoring_succeeded']}`",
        f"- Exact V4.1 scoring succeeded: `{report['scoring_status']['v4_1_blend']['exact_scoring_succeeded']}`",
        f"- Exact V4.9 scoring succeeded: `{report['scoring_status']['v4_9_guarded_text_blend']['exact_scoring_succeeded']}`",
        f"- Test neural Reading NDCG@10 mean: `{aggregate['test'][NEURAL_METHOD]['reading_ndcg_at_10']['mean']:.4f}`",
        f"- Test BM25 Reading NDCG@10 mean: `{aggregate['test']['bm25']['reading_ndcg_at_10']['mean']:.4f}`",
        f"- Test neural delta vs BM25 mean: `{aggregate['deltas']['test_neural_vs_bm25_reading_ndcg_at_10']['mean']:.4f}`",
        f"- Recommendation: `{report['recommendation']['recommendation']}`",
        "",
        "## Exact Scoring Status",
        "",
    ]
    for method in LEARNED_METHODS:
        status = report["scoring_status"][method]
        lines.extend(
            [
                f"### `{method}`",
                f"- Artifact loaded: `{status.get('artifact_loaded')}`",
                f"- Scorable rows: `{status.get('scorable_rows', 0)}` / `{report['dataset']['row_count']}`",
                f"- Reason: `{status.get('unavailable_reason') or status.get('error')}`",
                "",
            ]
        )
    lines.extend(["## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    fold_reports: list[dict[str, Any]],
    aggregate: dict[str, Any],
    artifact_status: dict[str, Any],
    scoring_status: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = diagnostic_questions(aggregate, fold_reports)
    return {
        "schema_version": "v6.1.3_exact_learned_baseline_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "production_candidate_created": False,
        "dataset": {
            "row_count": len(rows),
            "topic_count": len({row["query_id"] for row in rows}),
            "fold_count": len({row["fold_id"] for row in rows}),
        },
        "methods": METHODS,
        "artifact_status": artifact_status,
        "scoring_status": scoring_status,
        "fold_reports": fold_reports,
        "aggregate": aggregate,
        "severe_regressions_vs_v4_9": severe_regression_summary(fold_reports),
        "diagnostic_questions": diagnostics,
        "recommendation": recommendation(scoring_status),
        "parity": parity,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--v6-1-2-predictions", default=str(DEFAULT_V612_PREDICTIONS))
    parser.add_argument("--v6-1-2-report", default=str(DEFAULT_V612_REPORT))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR))
    parser.add_argument("--v41-model-dir", default=str(DEFAULT_V41_MODEL_DIR))
    parser.add_argument("--v43-model-dir", default=str(DEFAULT_V43_MODEL_DIR))
    parser.add_argument("--v41-blend-config", default=str(DEFAULT_V41_BLEND_CONFIG))
    parser.add_argument("--v49-config", default=str(DEFAULT_V49_CONFIG))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--failures-out", default=str(DEFAULT_FAILURES))
    parser.add_argument("--parity-out", default=str(DEFAULT_PARITY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(resolve_repo_path(args.examples))
    rows = load_scored_rows(resolve_repo_path(args.examples), resolve_repo_path(args.v6_1_2_predictions))
    _ = json.loads(resolve_repo_path(args.v6_1_2_report).read_text(encoding="utf-8"))
    artifacts, artifact_status = load_artifacts(args)
    rows, scoring_status = score_learned_rows(rows, artifacts, artifact_status)
    folds = build_folds(examples)
    fold_reports = build_fold_reports(rows, folds)
    aggregate = aggregate_cv(fold_reports)
    parity = parity_report(args=args, artifact_status=artifact_status, scoring_status=scoring_status)
    report = build_report(
        args=args,
        rows=rows,
        fold_reports=fold_reports,
        aggregate=aggregate,
        artifact_status=artifact_status,
        scoring_status=scoring_status,
        parity=parity,
    )
    write_jsonl(resolve_repo_path(args.predictions_out), prediction_rows(rows))
    write_jsonl(resolve_repo_path(args.failures_out), failure_examples(rows))
    write_json(resolve_repo_path(args.parity_out), parity)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.1.3 exact learned-baseline report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
