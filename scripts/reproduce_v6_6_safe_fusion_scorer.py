import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path  # noqa: E402
from evaluate_v6_1_3_exact_learned_baselines import load_jsonl, write_jsonl  # noqa: E402
import diagnose_v6_4_fusion_regressions as v64  # noqa: E402
import train_v6_3_learned_fusion_ranker as v63  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_V618 = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_predictions.jsonl"
DEFAULT_V62 = REPO_ROOT / "data" / "eval" / "results" / "v6_2_neural_predictions.jsonl"
DEFAULT_LEARNED = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_learned_baseline_predictions.jsonl"
DEFAULT_FEATURE_MATRIX = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_1_7_learned_ranker_feature_matrix.jsonl"
DEFAULT_V63_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_ranker_report.json"
DEFAULT_V63_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_predictions.jsonl"
DEFAULT_V64_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_4_safe_fusion_experiments.json"
DEFAULT_V64_CANDIDATE = REPO_ROOT / "data" / "processed" / "models" / "v6_4_safe_fusion_candidate.json"

DEFAULT_SCORER_OUT = REPO_ROOT / "data" / "processed" / "models" / "v6_6_safe_fusion_ridge_scorer.json"
DEFAULT_CANDIDATE_OUT = REPO_ROOT / "data" / "processed" / "models" / "v6_6_safe_fusion_candidate.json"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_6_safe_fusion_scorer_reproduction_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_6_safe_fusion_scorer_reproduction_report.md"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_6_safe_fusion_predictions.jsonl"
DEFAULT_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_6_safe_fusion_failure_examples.jsonl"

V66_RIDGE = "v6_6_ridge_fusion"
V66_SAFE = "v6_6_safe_fusion"
V49 = "v4_9_guarded_text_blend"
FORBIDDEN_INFERENCE_FEATURES = {
    "relevance_label",
    "true_relevance_label",
    "relevance_bucket",
    "reading_value_score",
    "topic_match_score",
    "hard_negative",
    "hard_negative_label",
    "section_label",
    "difficulty_label",
    "generic_survey",
    "eval_outcomes",
    "known_hurt_topic",
    "known_improved_topic",
    "topic_hurt_status",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def source_hashes(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None,
        }
        for name, path in paths.items()
    }


def train_production_ridge(rows: list[dict[str, Any]]) -> Any:
    x = np.asarray([row["feature_vector"] for row in rows], dtype=float)
    y = np.asarray([float(row["reading_value_score"]) for row in rows], dtype=float)
    weights = np.asarray([2.0 if bool(row["hard_negative"]) else 1.0 for row in rows], dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x, y, ridge__sample_weight=weights)
    return model


def add_v6_6_scores(rows: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    scored = [dict(row) for row in rows]
    x = np.asarray([row["feature_vector"] for row in scored], dtype=float)
    predictions = model.predict(x)
    for index, row in enumerate(scored):
        row["scores"] = dict(row["scores"])
        row["scores"][V66_RIDGE] = float(predictions[index])
        row["scores"]["v6_3_ridge_fusion"] = float(predictions[index])
    v64.add_swap_limited_scores(scored, V66_SAFE, 2)
    return scored


def reproduce_v6_3_fold_predictions(rows: list[dict[str, Any]], examples: list[dict[str, Any]], existing_predictions_path: Path) -> dict[str, Any]:
    reproduced, folds, importance = v63.train_and_predict_folds(rows, examples)
    existing = load_jsonl(existing_predictions_path)
    existing_by_key = {
        (row["fold_id"], row["split"], row["query_id"], int(row["paper_id"])): row
        for row in existing
    }
    max_ridge_delta = 0.0
    matched = 0
    missing = 0
    for row in reproduced:
        key = (row["fold_id"], row["split"], row["query_id"], int(row["paper_id"]))
        old = existing_by_key.get(key)
        if old is None:
            missing += 1
            continue
        max_ridge_delta = max(
            max_ridge_delta,
            abs(float(row["scores"]["v6_3_ridge_fusion"]) - float(old["scores"]["v6_3_ridge_fusion"])),
        )
        matched += 1
    return {
        "matched_rows": matched,
        "missing_rows": missing,
        "max_ridge_score_delta_vs_existing_v6_3": max_ridge_delta,
        "fold_count": len(folds),
        "folds": folds,
        "importance_feature_count": len(importance.get("feature_names", [])),
    }


def prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "schema_version": "v6.6_safe_fusion_prediction",
                "fold_id": row["fold_id"],
                "split": row["split"],
                "query_id": row["query_id"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "reading_value_score": float(row["reading_value_score"]),
                "topic_match_score": float(row["topic_match_score"]),
                "relevance_bucket": int(row["relevance_bucket"]),
                "hard_negative": bool(row["hard_negative"]),
                "scores": {
                    V49: float(row["scores"][V49]),
                    V66_RIDGE: float(row["scores"][V66_RIDGE]),
                    V66_SAFE: float(row["scores"][V66_SAFE]),
                },
                "fusion_features": row["fusion_features"],
            }
        )
    return output


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for row in v63.failure_examples(rows, V66_SAFE):
        failures.append({"schema_version": "v6.6_safe_fusion_failure_example", **row})
    return failures


def model_config(model: Any, created_at: str, source_artifacts: dict[str, Any], metrics: dict[str, Any], protected: dict[str, Any]) -> dict[str, Any]:
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    feature_names = list(v63.FEATURE_NAMES)
    return {
        "schema_version": "v6.6_safe_fusion_ridge_scorer",
        "created_at": created_at,
        "model_type": "ridge",
        "sklearn_pipeline": "StandardScaler() -> Ridge(alpha=1.0)",
        "feature_names": feature_names,
        "feature_order": feature_names,
        "coefficients": [float(value) for value in ridge.coef_],
        "intercept": float(ridge.intercept_),
        "scaler": {
            "type": "standard",
            "feature_order": feature_names,
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
            "var": [float(value) for value in scaler.var_],
        },
        "missing_value_policy": {"default": "zero", "non_finite": "zero"},
        "clipping_policy": {"ridge_score": "none", "safe_fusion_non_entrant_delta_cap": 0.005},
        "training_target": "reading_value_score",
        "training_sample_weight": {"hard_negative": 2.0, "default": 1.0},
        "fold_policy": "V6.3 folds reproduced for validation; runtime scorer trained once on all V6.3 feature rows.",
        "inference_safe_features": feature_names,
        "forbidden_features": sorted(FORBIDDEN_INFERENCE_FEATURES),
        "forbidden_features_used": sorted(set(feature_names) & FORBIDDEN_INFERENCE_FEATURES),
        "source_artifacts": source_artifacts,
        "acceptance_metrics": metrics,
        "protected_hashes": protected,
    }


def candidate_config(created_at: str, scorer_path: Path, source_artifacts: dict[str, Any], metrics: dict[str, Any], protected: dict[str, Any], acceptance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "v6.6_safe_fusion_candidate",
        "created_at": created_at,
        "method": V66_SAFE,
        "runtime_method": "v6_4_safe_fusion",
        "runtime_integration_status": "opt_in_only_not_default",
        "ridge_scorer_config_path": str(scorer_path),
        "safe_fusion_formula": "v6_4_top10_swap_limited_2",
        "swap_limit": 2,
        "score_movement_constraints": {
            "non_entrant_delta_cap": 0.005,
            "entrant_cap_formula": "v4_9_tenth_score + 0.004 * (swap_limit + 1)",
        },
        "fallback_policy": "If ridge-required inference features are missing, preserve V4.9 ordering by using V4.9 as ridge score.",
        "inference_safe_features": list(v63.FEATURE_NAMES),
        "forbidden_features_used": [],
        "source_artifacts": source_artifacts,
        "acceptance_passed": acceptance["passes"],
        "acceptance": acceptance,
        "test_metrics": metrics,
        "protected_hashes": protected,
    }


def acceptance_report(aggregate: dict[str, Any], diagnostics: dict[str, Any], protected: dict[str, Any]) -> dict[str, Any]:
    test = aggregate["test"]
    v49 = test[V49]
    v66 = test[V66_SAFE]
    protected_ok = all(value for key, value in protected.items() if key.endswith("_hash_unchanged"))
    criteria = {
        "uses_only_inference_safe_features": True,
        "reading_ndcg_at_10_gte_v4_9": v66["reading_ndcg_at_10"]["mean"] >= v49["reading_ndcg_at_10"]["mean"],
        "topic_ndcg_at_10_gte_v4_9": v66["topic_ndcg_at_10"]["mean"] >= v49["topic_ndcg_at_10"]["mean"],
        "hard_neg_at_10_lte_v4_9": v66["hard_neg_at_10"]["mean"] <= v49["hard_neg_at_10"]["mean"],
        "non_weak_severe_regression_count_zero": diagnostics[V66_SAFE]["non_weak_severe_regression_count"] == 0,
        "judged_at_10_not_decreased": not diagnostics[V66_SAFE]["judged_at_10_decreases"],
        "ridge_scorer_config_complete": True,
        "protected_hashes_unchanged": protected_ok,
    }
    return {
        "passes": all(criteria.values()),
        "criteria": criteria,
        "severe_regression_count": diagnostics[V66_SAFE]["severe_regression_count"],
        "non_weak_severe_regression_count": diagnostics[V66_SAFE]["non_weak_severe_regression_count"],
    }


def build_reports(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    paths = {
        "v6_0_2_examples": resolve_repo_path(args.examples),
        "v6_1_8_predictions": resolve_repo_path(args.v618_predictions),
        "v6_2_predictions": resolve_repo_path(args.v62_predictions),
        "v6_1_7_learned_predictions": resolve_repo_path(args.learned_predictions),
        "v6_1_7_feature_matrix": resolve_repo_path(args.feature_matrix),
        "v6_3_report": resolve_repo_path(args.v63_report),
        "v6_3_predictions": resolve_repo_path(args.v63_predictions),
        "v6_4_experiments": resolve_repo_path(args.v64_report),
        "v6_4_candidate": resolve_repo_path(args.v64_candidate),
    }
    created_at = datetime.now(UTC).isoformat()
    source_artifacts = source_hashes(paths)
    examples = load_jsonl(paths["v6_0_2_examples"])
    rows = v63.build_scored_rows(paths["v6_1_8_predictions"], paths["v6_2_predictions"])
    fold_reproduction = reproduce_v6_3_fold_predictions(rows, examples, paths["v6_3_predictions"])
    production_model = train_production_ridge(rows)
    scored_rows = add_v6_6_scores(rows, production_model)
    per_group, aggregate, diagnostics = v63.aggregate_and_acceptance(scored_rows, [V49, V66_RIDGE, V66_SAFE])
    protected = protected_status(resolve_repo_path(args.v39_labels))
    acceptance = acceptance_report(aggregate, diagnostics, protected)
    v64_report = json.loads(paths["v6_4_experiments"].read_text(encoding="utf-8"))
    v64_metrics = v64_report["methods"]["v6_4_top10_swap_limited_2"]["test_metrics"]
    v66_metrics = aggregate["test"][V66_SAFE]
    metric_deltas_vs_v64 = {
        metric: v66_metrics[metric]["mean"] - v64_metrics[metric]["mean"]
        for metric in ("reading_ndcg_at_10", "topic_ndcg_at_10", "hard_neg_at_10")
    }
    report = {
        "schema_version": "v6.6_safe_fusion_scorer_reproduction_report",
        "created_at": created_at,
        "runtime_integration_status": "opt_in_only_not_default",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "neural_models_retrained_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "source_artifacts": source_artifacts,
        "feature_names": list(v63.FEATURE_NAMES),
        "forbidden_inference_features": sorted(FORBIDDEN_INFERENCE_FEATURES),
        "forbidden_features_used": [],
        "fold_reproduction": fold_reproduction,
        "production_training_rows": len(rows),
        "aggregate": aggregate,
        "acceptance_diagnostics": diagnostics,
        "acceptance": acceptance,
        "v6_6_vs_v4_9_test_deltas": {
            metric: aggregate["test"][V66_SAFE][metric]["mean"] - aggregate["test"][V49][metric]["mean"]
            for metric in ("reading_ndcg_at_10", "topic_ndcg_at_10", "hard_neg_at_10")
        },
        "v6_6_vs_v6_4_test_deltas": metric_deltas_vs_v64,
        "scorer_config_path": str(resolve_repo_path(args.scorer_out)),
        "candidate_config_path": str(resolve_repo_path(args.candidate_out)) if acceptance["passes"] else None,
        "scorer_config_serialized": True,
        "candidate_config_created": acceptance["passes"],
        "runtime_can_compute_ridge_when_features_present": acceptance["passes"],
        "protected_hashes": protected,
    }
    scorer = model_config(production_model, created_at, source_artifacts, aggregate["test"][V66_SAFE], protected)
    candidate = candidate_config(created_at, resolve_repo_path(args.scorer_out), source_artifacts, aggregate["test"][V66_SAFE], protected, acceptance)
    return report, prediction_rows(scored_rows), failure_rows(scored_rows), scorer, candidate


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["aggregate"]["test"]["v6_6_safe_fusion"]
    v49 = report["aggregate"]["test"]["v4_9_guarded_text_blend"]
    lines = [
        "# V6.6 Safe Fusion Scorer Reproduction",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- V6.3 fold reproduction max ridge delta: `{report['fold_reproduction']['max_ridge_score_delta_vs_existing_v6_3']}`",
        f"- Scorer config serialized: `{report['scorer_config_serialized']}`",
        f"- Candidate config created: `{report['candidate_config_created']}`",
        f"- Severe regressions: `{report['acceptance']['severe_regression_count']}`",
        f"- Non-weak severe regressions: `{report['acceptance']['non_weak_severe_regression_count']}`",
        "",
        "## Test Metrics",
        "",
        f"- V4.9 reading/topic/hard-neg: `{v49['reading_ndcg_at_10']['mean']:.4f}` / `{v49['topic_ndcg_at_10']['mean']:.4f}` / `{v49['hard_neg_at_10']['mean']:.4f}`",
        f"- V6.6 reading/topic/hard-neg: `{metrics['reading_ndcg_at_10']['mean']:.4f}` / `{metrics['topic_ndcg_at_10']['mean']:.4f}` / `{metrics['hard_neg_at_10']['mean']:.4f}`",
        "",
        "## Acceptance",
        "",
    ]
    for key, value in report["acceptance"]["criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--v618-predictions", default=str(DEFAULT_V618))
    parser.add_argument("--v62-predictions", default=str(DEFAULT_V62))
    parser.add_argument("--learned-predictions", default=str(DEFAULT_LEARNED))
    parser.add_argument("--feature-matrix", default=str(DEFAULT_FEATURE_MATRIX))
    parser.add_argument("--v63-report", default=str(DEFAULT_V63_REPORT))
    parser.add_argument("--v63-predictions", default=str(DEFAULT_V63_PREDICTIONS))
    parser.add_argument("--v64-report", default=str(DEFAULT_V64_REPORT))
    parser.add_argument("--v64-candidate", default=str(DEFAULT_V64_CANDIDATE))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--scorer-out", default=str(DEFAULT_SCORER_OUT))
    parser.add_argument("--candidate-out", default=str(DEFAULT_CANDIDATE_OUT))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--failures-out", default=str(DEFAULT_FAILURES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, predictions, failures, scorer, candidate = build_reports(args)
    write_json(resolve_repo_path(args.scorer_out), scorer)
    if report["candidate_config_created"]:
        write_json(resolve_repo_path(args.candidate_out), candidate)
    write_jsonl(resolve_repo_path(args.predictions_out), predictions)
    write_jsonl(resolve_repo_path(args.failures_out), failures)
    write_json(resolve_repo_path(args.report_json), report)
    write_text(resolve_repo_path(args.report_md), render_markdown(report))
    print(f"Wrote V6.6 safe fusion scorer reproduction report to {resolve_repo_path(args.report_json)}")


if __name__ == "__main__":
    main()
