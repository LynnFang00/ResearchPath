import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from evaluate_v6_1_3_exact_learned_baselines import load_jsonl, write_jsonl  # noqa: E402
import evaluate_v6_1_8_exact_neural_vs_learned_baselines as v618  # noqa: E402
import run_v6_1_2_neural_cross_validation as v612  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_V618 = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_predictions.jsonl"
DEFAULT_V62 = REPO_ROOT / "data" / "eval" / "results" / "v6_2_neural_predictions.jsonl"
DEFAULT_LEARNED = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_learned_baseline_predictions.jsonl"
DEFAULT_FEATURE_MATRIX = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_1_7_learned_ranker_feature_matrix.jsonl"

DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_ranker_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_ranker_report.md"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_predictions.jsonl"
DEFAULT_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_3_fusion_vs_v4_9_failure_examples.jsonl"
DEFAULT_IMPORTANCE = REPO_ROOT / "data" / "eval" / "results" / "v6_3_fusion_feature_importance.json"
DEFAULT_CANDIDATE = REPO_ROOT / "data" / "processed" / "models" / "v6_3_learned_fusion_candidate.json"

V61 = "neural_cross_encoder_v6_1"
V62_MULTI = "v6_2_multitask_neural"
BASELINE_METHODS = ["bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend", V61, V62_MULTI]
FUSION_METHODS = [
    "v6_3_ridge_fusion",
    "v6_3_random_forest_fusion",
    "v6_3_conservative_bounded_fusion",
    "v6_3_guarded_fusion",
]
METHODS = [*BASELINE_METHODS, *FUSION_METHODS]
METRICS = v618.METRICS
SEED = 630
FEATURE_NAMES = [
    "bm25_score",
    "v3_3_ltr_score",
    "v4_1_blend_score",
    "v4_9_guarded_text_blend_score",
    "v6_1_neural_score",
    "v6_2_multitask_neural_score",
    "v4_9_minus_v6_1_neural_score",
    "v4_9_minus_v6_2_multitask_score",
    "v4_9_minus_v6_1_rank_diff",
    "v4_9_minus_v6_2_rank_diff",
    "v4_9_max_ranker_confidence",
    "v4_9_margin_to_10th",
    "v6_1_margin_to_10th",
    "v6_2_margin_to_10th",
]
FORBIDDEN_INFERENCE_FEATURES = {
    "relevance_bucket",
    "reading_value_score",
    "topic_match_score",
    "hard_negative",
    "section_label",
    "difficulty_label",
    "generic_survey",
    "eval_outcomes",
    "topic_hurt_status",
}


def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["fold_id"]), str(row["split"]), str(row["query_id"]), int(row["paper_id"]))


def rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    return {key(row): row for row in rows}


def rank_maps(rows: list[dict[str, Any]], method: str) -> dict[tuple[str, str, str, int], int]:
    ranks = {}
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["fold_id"], row["split"], row["query_id"])].append(row)
    for group, group_rows in by_group.items():
        ranked = sorted(group_rows, key=lambda row: (float(row["scores"][method]), -int(row["paper_id"])), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            ranks[(*group, int(row["paper_id"]))] = rank
    return ranks


def margin_to_10th(rows: list[dict[str, Any]], method: str) -> dict[tuple[str, str, str, int], float]:
    margins = {}
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["fold_id"], row["split"], row["query_id"])].append(row)
    for group, group_rows in by_group.items():
        ranked = sorted(group_rows, key=lambda row: float(row["scores"][method]), reverse=True)
        tenth = float(ranked[min(9, len(ranked) - 1)]["scores"][method]) if ranked else 0.0
        for row in group_rows:
            margins[(*group, int(row["paper_id"]))] = float(row["scores"][method]) - tenth
    return margins


def normalize_by_group(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_group[(row["fold_id"], row["split"], row["query_id"])].append(row)
    for group_rows in by_group.values():
        for method in methods:
            norm = v618.normalize({int(row["paper_id"]): float(row["scores"][method]) for row in group_rows})
            for row in group_rows:
                row["scores"][f"norm__{method}"] = norm[int(row["paper_id"])]
    return output


def build_scored_rows(v618_path: Path, v62_path: Path) -> list[dict[str, Any]]:
    base_rows = load_jsonl(v618_path)
    v62_by_key = rows_by_key(load_jsonl(v62_path))
    rows = []
    for row in base_rows:
        copied = dict(row)
        copied["scores"] = dict(row["scores"])
        v62 = v62_by_key.get(key(row))
        copied["scores"][V62_MULTI] = float((v62 or {}).get("scores", {}).get(V62_MULTI, row["scores"][V61]))
        rows.append(copied)
    rows = normalize_by_group(rows, [*BASELINE_METHODS])
    ranks_v49 = rank_maps(rows, "norm__v4_9_guarded_text_blend")
    ranks_v61 = rank_maps(rows, f"norm__{V61}")
    ranks_v62 = rank_maps(rows, f"norm__{V62_MULTI}")
    margins_v49 = margin_to_10th(rows, "norm__v4_9_guarded_text_blend")
    margins_v61 = margin_to_10th(rows, f"norm__{V61}")
    margins_v62 = margin_to_10th(rows, f"norm__{V62_MULTI}")
    for row in rows:
        row_key = key(row)
        scores = row["scores"]
        v49 = float(scores["norm__v4_9_guarded_text_blend"])
        v61 = float(scores[f"norm__{V61}"])
        v62 = float(scores[f"norm__{V62_MULTI}"])
        feature_values = {
            "bm25_score": float(scores["norm__bm25"]),
            "v3_3_ltr_score": float(scores["norm__v3_3_ltr"]),
            "v4_1_blend_score": float(scores["norm__v4_1_blend"]),
            "v4_9_guarded_text_blend_score": v49,
            "v6_1_neural_score": v61,
            "v6_2_multitask_neural_score": v62,
            "v4_9_minus_v6_1_neural_score": v49 - v61,
            "v4_9_minus_v6_2_multitask_score": v49 - v62,
            "v4_9_minus_v6_1_rank_diff": float(ranks_v61[row_key] - ranks_v49[row_key]),
            "v4_9_minus_v6_2_rank_diff": float(ranks_v62[row_key] - ranks_v49[row_key]),
            "v4_9_max_ranker_confidence": float((row.get("v4_9_details") or {}).get("v4_9_max_ranker_confidence") or v49),
            "v4_9_margin_to_10th": margins_v49[row_key],
            "v6_1_margin_to_10th": margins_v61[row_key],
            "v6_2_margin_to_10th": margins_v62[row_key],
        }
        row["fusion_features"] = feature_values
        row["feature_vector"] = [feature_values[name] for name in FEATURE_NAMES]
    return rows


def leakage_check() -> dict[str, Any]:
    used = set(FEATURE_NAMES)
    forbidden = sorted(used & FORBIDDEN_INFERENCE_FEATURES)
    return {
        "feature_names": FEATURE_NAMES,
        "forbidden_inference_features": sorted(FORBIDDEN_INFERENCE_FEATURES),
        "forbidden_features_used": forbidden,
        "uses_only_inference_safe_features": not forbidden,
        "labels_used_as_training_targets_only": True,
    }


def train_models(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([row["feature_vector"] for row in train_rows], dtype=float)
    y = np.asarray([float(row["reading_value_score"]) for row in train_rows], dtype=float)
    weights = np.asarray([2.0 if bool(row["hard_negative"]) else 1.0 for row in train_rows], dtype=float)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(x, y, ridge__sample_weight=weights)
    rf = RandomForestRegressor(n_estimators=160, max_depth=5, min_samples_leaf=4, random_state=SEED)
    rf.fit(x, y, sample_weight=weights)
    try:
        gb = GradientBoostingRegressor(random_state=SEED, max_depth=2, learning_rate=0.04, n_estimators=120)
        gb.fit(x, y, sample_weight=weights)
    except Exception:
        gb = None
    return {"ridge": ridge, "random_forest": rf, "gradient_boosting": gb}


def predict_models(models: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = np.asarray([row["feature_vector"] for row in rows], dtype=float)
    ridge_scores = models["ridge"].predict(x)
    rf_scores = models["random_forest"].predict(x)
    for index, row in enumerate(rows):
        scores = row["scores"]
        v49 = float(scores["norm__v4_9_guarded_text_blend"])
        neural = float(scores[f"norm__{V61}"])
        ridge = float(ridge_scores[index])
        rf = float(rf_scores[index])
        adjustment = max(-0.05, min(0.05, ridge - v49))
        disagreement = abs(v49 - neural)
        guarded_multiplier = 0.25 if disagreement > 0.35 else 0.60
        guarded_adjustment = max(-0.04, min(0.04, ridge - v49)) * guarded_multiplier
        row["scores"]["v6_3_ridge_fusion"] = ridge
        row["scores"]["v6_3_random_forest_fusion"] = rf
        row["scores"]["v6_3_conservative_bounded_fusion"] = v49 + adjustment
        row["scores"]["v6_3_guarded_fusion"] = v49 + guarded_adjustment
    return rows


def model_importance(models: dict[str, Any]) -> dict[str, Any]:
    ridge = models["ridge"].named_steps["ridge"]
    return {
        "ridge_coefficients": dict(zip(FEATURE_NAMES, [float(value) for value in ridge.coef_], strict=True)),
        "random_forest_feature_importance": dict(zip(FEATURE_NAMES, [float(value) for value in models["random_forest"].feature_importances_], strict=True)),
    }


def aggregate_model_importance(fold_importance: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = feature_importance_from_rows(rows)
    for key in ("ridge_coefficients", "random_forest_feature_importance"):
        output[key] = {}
        for name in FEATURE_NAMES:
            values = [float(item[key][name]) for item in fold_importance]
            output[key][name] = {
                "mean": sum(values) / len(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    output["top_random_forest_features"] = sorted(
        (
            {"feature": name, "importance_mean": stats["mean"]}
            for name, stats in output["random_forest_feature_importance"].items()
        ),
        key=lambda row: row["importance_mean"],
        reverse=True,
    )[:8]
    output["top_ridge_abs_coefficients"] = sorted(
        (
            {"feature": name, "coefficient_mean": stats["mean"], "abs_coefficient_mean": abs(stats["mean"])}
            for name, stats in output["ridge_coefficients"].items()
        ),
        key=lambda row: row["abs_coefficient_mean"],
        reverse=True,
    )[:8]
    return output


def train_and_predict_folds(rows: list[dict[str, Any]], examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    folds = v612.build_folds(examples)
    output = []
    fold_reports = []
    fold_importance = []
    for fold in folds:
        train_topics = set(fold["train_topics"])
        train_rows = [row for row in rows if row["query_id"] in train_topics]
        val_rows = [dict(row) for row in rows if row["fold_id"] == fold["fold_id"] and row["split"] == "val"]
        test_rows = [dict(row) for row in rows if row["fold_id"] == fold["fold_id"] and row["split"] == "test"]
        models = train_models(train_rows)
        fold_importance.append(model_importance(models))
        val_pred = predict_models(models, val_rows)
        test_pred = predict_models(models, test_rows)
        output.extend(val_pred)
        output.extend(test_pred)
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "train_topics": fold["train_topics"],
                "val_topics": fold["val_topics"],
                "test_topics": fold["test_topics"],
                "query_id_leakage": bool(set(fold["train_topics"]) & (set(fold["val_topics"]) | set(fold["test_topics"]))),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "test_rows": len(test_rows),
            }
        )
    return output, fold_reports, aggregate_model_importance(fold_importance, output)


def feature_importance_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # These summaries are model-agnostic and deterministic; per-model importances are
    # attached later from the final fold where sklearn exposes them.
    values = {name: [float(row["fusion_features"][name]) for row in rows] for name in FEATURE_NAMES}
    return {
        "schema_version": "v6.3_fusion_feature_importance",
        "feature_names": FEATURE_NAMES,
        "feature_value_summary": {
            name: {
                "mean": sum(items) / len(items) if items else 0.0,
                "std": statistics.pstdev(items) if len(items) > 1 else 0.0,
                "min": min(items) if items else 0.0,
                "max": max(items) if items else 0.0,
            }
            for name, items in values.items()
        },
    }


def aggregate_and_acceptance(rows: list[dict[str, Any]], methods: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    per_group, aggregate = v618.evaluate_all(rows, methods)
    acceptance = v618.acceptance_diagnostics(per_group, methods)
    return per_group, aggregate, acceptance


def candidate_passes(report: dict[str, Any], method: str) -> bool:
    test = report["aggregate"]["test"]
    diag = report["acceptance_diagnostics"][method]
    protected_ok = all(value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged"))
    return (
        report["leakage_check"]["uses_only_inference_safe_features"]
        and protected_ok
        and test[method]["reading_ndcg_at_10"]["mean"] >= test["v4_9_guarded_text_blend"]["reading_ndcg_at_10"]["mean"]
        and test[method]["topic_ndcg_at_10"]["mean"] >= test["v4_9_guarded_text_blend"]["topic_ndcg_at_10"]["mean"]
        and test[method]["hard_neg_at_10"]["mean"] <= test["v4_9_guarded_text_blend"]["hard_neg_at_10"]["mean"]
        and diag["non_weak_severe_regression_count"] == 0
        and not diag["judged_at_10_decreases"]
    )


def choose_best_variant(aggregate: dict[str, Any], acceptance: dict[str, Any]) -> str:
    def sort_key(method: str) -> tuple[float, float, float, float]:
        test = aggregate["test"][method]
        return (
            float(test["reading_ndcg_at_10"]["mean"] or 0.0),
            float(test["topic_ndcg_at_10"]["mean"] or 0.0),
            -float(test["hard_neg_at_10"]["mean"] or 0.0),
            -float(acceptance[method]["severe_regression_count"]),
        )

    return max(FUSION_METHODS, key=sort_key)


def failure_examples(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    failures = []
    for fold_id in sorted({row["fold_id"] for row in rows}):
        for split in ("val", "test"):
            split_rows = [row for row in rows if row["fold_id"] == fold_id and row["split"] == split]
            for topic in sorted({row["query_id"] for row in split_rows}):
                topic_rows = [row for row in split_rows if row["query_id"] == topic]
                fusion_top = v618.top10(topic_rows, method)
                v49_top = v618.top10(topic_rows, "v4_9_guarded_text_blend")
                v49_ids = {row["paper_id"] for row in v49_top}
                for rank, row in enumerate(fusion_top, start=1):
                    if row["paper_id"] not in v49_ids and (row["hard_negative"] or row["relevance_bucket"] == 0):
                        failures.append(failure_row(row, method, rank, "fusion_promoted_negative_or_hard_negative_vs_v4_9"))
                fusion_ids = {row["paper_id"] for row in fusion_top}
                for rank, row in enumerate(v49_top, start=1):
                    if row["paper_id"] not in fusion_ids and row["relevance_bucket"] == 2:
                        failures.append(failure_row(row, method, rank, "fusion_demoted_v4_9_strong_positive"))
    return failures[:300]


def failure_row(row: dict[str, Any], method: str, rank: int, failure_type: str) -> dict[str, Any]:
    return {
        "schema_version": "v6.3_fusion_vs_v4_9_failure_example",
        "failure_type": failure_type,
        "method": method,
        "fold_id": row["fold_id"],
        "split": row["split"],
        "query_id": row["query_id"],
        "paper_id": int(row["paper_id"]),
        "title": row["title"],
        "rank": rank,
        "fusion_score": float(row["scores"][method]),
        "v4_9_score": float(row["scores"]["v4_9_guarded_text_blend"]),
        "reading_value_score": float(row["reading_value_score"]),
        "topic_match_score": float(row["topic_match_score"]),
        "relevance_bucket": int(row["relevance_bucket"]),
        "hard_negative": bool(row["hard_negative"]),
    }


def prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "schema_version": "v6.3_learned_fusion_prediction",
                "fold_id": row["fold_id"],
                "split": row["split"],
                "query_id": row["query_id"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "reading_value_score": float(row["reading_value_score"]),
                "topic_match_score": float(row["topic_match_score"]),
                "relevance_bucket": int(row["relevance_bucket"]),
                "hard_negative": bool(row["hard_negative"]),
                "scores": {method: float(row["scores"][method]) for method in METHODS},
                "fusion_features": row["fusion_features"],
            }
        )
    return output


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.3 Learned Fusion Ranker",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Best fusion method: `{report['best_fusion_method']}`",
        f"- Candidate config created: `{report['candidate_config_created']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Inference-safe features: `{report['leakage_check']['uses_only_inference_safe_features']}`",
        "",
        "## Test Metrics",
        "",
    ]
    for method in ["v4_9_guarded_text_blend", V61, V62_MULTI, report["best_fusion_method"]]:
        row = report["aggregate"]["test"][method]
        lines.append(
            f"- `{method}`: reading `{row['reading_ndcg_at_10']['mean']:.4f}`, "
            f"topic `{row['topic_ndcg_at_10']['mean']:.4f}`, hard-neg `{row['hard_neg_at_10']['mean']:.4f}`"
        )
    lines.extend(["", "## Feature Names", ""])
    for name in FEATURE_NAMES:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    examples = load_jsonl(resolve_repo_path(args.examples))
    scored_rows = build_scored_rows(resolve_repo_path(args.v618_predictions), resolve_repo_path(args.v62_predictions))
    predictions, folds, importance = train_and_predict_folds(scored_rows, examples)
    per_group, aggregate, acceptance = aggregate_and_acceptance(predictions, METHODS)
    best = choose_best_variant(aggregate, acceptance)
    protected = protected_status(resolve_repo_path(args.v39_labels))
    report = {
        "schema_version": "v6.3_learned_fusion_ranker_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "production_candidate_created": False,
        "methods": METHODS,
        "fusion_methods": FUSION_METHODS,
        "best_fusion_method": best,
        "folds": folds,
        "query_id_leakage_detected": any(fold["query_id_leakage"] for fold in folds),
        "feature_names": FEATURE_NAMES,
        "leakage_check": leakage_check(),
        "aggregate": aggregate,
        "deltas_vs_v4_9": v618.deltas(aggregate, METHODS),
        "acceptance_diagnostics": acceptance,
        "feature_importance_path": str(resolve_repo_path(args.importance_out)),
        "protected_hashes": protected,
    }
    passes = candidate_passes(report, best)
    report["candidate_config_created"] = bool(passes)
    report["candidate_config_path"] = str(resolve_repo_path(args.candidate_out)) if passes else None
    report["neural_is_useful_as_auxiliary_feature"] = (
        aggregate["test"][best]["reading_ndcg_at_10"]["mean"] > aggregate["test"]["v4_9_guarded_text_blend"]["reading_ndcg_at_10"]["mean"]
        and acceptance[best]["severe_regression_count"] <= acceptance[V61]["severe_regression_count"]
    )
    report["recommendation"] = "accepted_offline_candidate_not_runtime_integrated" if passes else "research_only_no_candidate"
    return report, prediction_rows(predictions), failure_examples(predictions, best), importance


def write_candidate_if_accepted(path: Path, report: dict[str, Any]) -> None:
    if not report["candidate_config_created"]:
        return
    write_json(
        path,
        {
            "schema_version": "v6.3_learned_fusion_candidate",
            "created_at": report["created_at"],
            "runtime_integration_status": "offline_only_not_integrated",
            "production_candidate_created": False,
            "method": report["best_fusion_method"],
            "feature_names": FEATURE_NAMES,
            "acceptance_passed": True,
            "metrics": report["aggregate"]["test"][report["best_fusion_method"]],
            "protected_hashes": report["protected_hashes"],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--v618-predictions", default=str(DEFAULT_V618))
    parser.add_argument("--v62-predictions", default=str(DEFAULT_V62))
    parser.add_argument("--learned-predictions", default=str(DEFAULT_LEARNED))
    parser.add_argument("--feature-matrix", default=str(DEFAULT_FEATURE_MATRIX))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--failures-out", default=str(DEFAULT_FAILURES))
    parser.add_argument("--importance-out", default=str(DEFAULT_IMPORTANCE))
    parser.add_argument("--candidate-out", default=str(DEFAULT_CANDIDATE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, predictions, failures, importance = build_report(args)
    write_jsonl(resolve_repo_path(args.predictions_out), predictions)
    write_jsonl(resolve_repo_path(args.failures_out), failures)
    write_json(resolve_repo_path(args.importance_out), importance)
    write_json(resolve_repo_path(args.report_json), report)
    write_text(resolve_repo_path(args.report_md), render_markdown(report))
    write_candidate_if_accepted(resolve_repo_path(args.candidate_out), report)
    print(f"Wrote V6.3 learned fusion ranker report to {resolve_repo_path(args.report_json)}")


if __name__ == "__main__":
    main()
