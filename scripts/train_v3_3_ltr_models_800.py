import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    file_sha256,
    load_json,
    resolve_repo_path,
    write_json,
    write_text,
)
from evaluate_v2_7_score_blends import EXPECTED_V25_LABELS_SHA256  # noqa: E402
from evaluate_v2_9_score_columns import duplicate_rows, evaluate_score_columns, load_jsonl, rank_rows, validate_splits  # noqa: E402
from train_v2_6_production_aware_learned_hybrid import EXPECTED_V21_LABELS_SHA256  # noqa: E402
from train_v3_0_ltr_models import (  # noqa: E402
    METRIC_KEYS,
    ModelSpec,
    TrainedModel,
    build_estimator,
    feature_matrix,
    fit_model,
    numeric_feature_value,
    pairwise_training_matrix,
    rows_for_topics,
    target_vector,
)


DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v3_3_ltr_dataset_800.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "data" / "eval" / "training" / "v3_3_splits_800.json"
DEFAULT_V30_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v3_0_ltr_training_report.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_3_ltr_800_training_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_3_ltr_800_training_report.md"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V32_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"

BASELINE_COLUMNS = [
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
    "v2_7_score",
]
RAW_SCORE_FEATURES = {
    "bm25_raw_score",
    "tfidf_raw_score",
    "embedding_raw_score",
    "faiss_embedding_raw_score",
    "hybrid_raw_score",
}
RANK_FEATURES = [
    "bm25_rank",
    "tfidf_rank",
    "embedding_rank",
    "faiss_embedding_rank",
    "hybrid_rank",
    "bm25_reciprocal_rank",
    "tfidf_reciprocal_rank",
    "embedding_reciprocal_rank",
    "faiss_embedding_reciprocal_rank",
    "hybrid_reciprocal_rank",
]
NORMALIZED_SCORE_FEATURES = [
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
]
METADATA_FEATURES = [
    "year",
    "paper_age_years",
    "citation_count_log1p",
    "full_text_available_bool",
]
FEATURE_VARIANTS = {
    "rank_features_only": RANK_FEATURES,
    "normalized_scores_only": NORMALIZED_SCORE_FEATURES,
    "compact_combined": NORMALIZED_SCORE_FEATURES + RANK_FEATURES + METADATA_FEATURES,
    "no_v27_compact": NORMALIZED_SCORE_FEATURES + RANK_FEATURES + METADATA_FEATURES,
    "with_v27_compact": NORMALIZED_SCORE_FEATURES + ["v2_7_score"] + RANK_FEATURES + METADATA_FEATURES,
}
LEAKAGE_COLUMNS = {
    "reading_value_score",
    "topic_match_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
    "intent_scores",
    "primary_role",
    "secondary_roles",
    "duplicate_status",
    "duplicate_of_paper_id",
    "label_confidence",
    "label_notes",
    "relevance_label",
    "hard_negative",
    "hidden_positive",
    "positive",
    "judged_source",
}


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    score_column: str
    metrics: dict[str, float]


def model_score_column(model_name: str) -> str:
    return f"v3_3_{model_name}_score"


def build_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec("ridge_rank_features_only", "pointwise_ridge_regression", FEATURE_VARIANTS["rank_features_only"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_normalized_scores_only", "pointwise_ridge_regression", FEATURE_VARIANTS["normalized_scores_only"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_compact_combined", "pointwise_ridge_regression", FEATURE_VARIANTS["compact_combined"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_no_v27_compact", "pointwise_ridge_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_with_v27_compact", "pointwise_ridge_regression", FEATURE_VARIANTS["with_v27_compact"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("elasticnet_no_v27_compact", "pointwise_elasticnet_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"alpha": 0.005, "l1_ratio": 0.2, "random_state": 17, "max_iter": 10000}),
        ModelSpec("logistic_positive_no_v27_compact", "pointwise_logistic_regression", FEATURE_VARIANTS["no_v27_compact"], "positive", {"C": 1.0, "random_state": 17, "max_iter": 1000}),
        ModelSpec("random_forest_no_v27_compact", "pointwise_random_forest_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 3, "random_state": 17}),
        ModelSpec("gradient_boosting_no_v27_compact", "pointwise_gradient_boosting_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"n_estimators": 140, "learning_rate": 0.035, "max_depth": 2, "random_state": 17}),
        ModelSpec("hist_gradient_boosting_no_v27_compact", "pointwise_hist_gradient_boosting_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"max_iter": 140, "learning_rate": 0.035, "max_leaf_nodes": 8, "random_state": 17}),
        ModelSpec("pairwise_logistic_no_v27_compact", "pairwise_logistic_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label_pair_order", {"C": 0.5, "random_state": 17, "max_iter": 1000}),
    ]
    for spec in specs:
        raw_overlap = sorted(set(spec.feature_columns) & RAW_SCORE_FEATURES)
        if raw_overlap:
            raise ValueError(f"V3.3 compact feature spec unexpectedly includes raw score features: {spec.name} {raw_overlap}")
    return specs


def train_models_for_topics(
    rows: list[dict[str, Any]],
    train_topics: set[str],
    model_specs: list[ModelSpec] | None = None,
) -> list[TrainedModel]:
    train_rows = rows_for_topics(rows, train_topics)
    if not train_rows:
        raise ValueError("No training rows available for requested train topics.")
    observed_topics = {str(row["query_id"]) for row in train_rows}
    if not observed_topics <= train_topics:
        raise ValueError("Training rows include topics outside the requested train topic set.")
    return [fit_model(spec, train_rows, train_topics) for spec in (model_specs or build_model_specs())]


def predict_scores(model: TrainedModel, rows: list[dict[str, Any]]) -> list[float]:
    x_values = feature_matrix(rows, model.spec.feature_columns)
    if model.spec.model_type == "pointwise_logistic_regression":
        return [float(value) for value in model.estimator.predict_proba(x_values)[:, 1].tolist()]
    if model.spec.model_type == "pairwise_logistic_regression":
        return [float(value) for value in model.estimator.decision_function(x_values).tolist()]
    return [float(value) for value in model.estimator.predict(x_values).tolist()]


def scored_rows(rows: list[dict[str, Any]], trained_models: list[TrainedModel]) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    for model in trained_models:
        predictions = predict_scores(model, rows)
        score_column = model_score_column(model.spec.name)
        for row, score in zip(copied, predictions, strict=True):
            row[score_column] = score
    return copied


def evaluate_rows_with_scores(rows: list[dict[str, Any]], trained_models: list[TrainedModel], *, k: int) -> dict[str, Any]:
    scored = scored_rows(rows, trained_models)
    score_columns = BASELINE_COLUMNS + [model_score_column(model.spec.name) for model in trained_models]
    return evaluate_score_columns(scored, score_columns=score_columns, k=k)


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in fold_metrics:
        for key in METRIC_KEYS:
            by_method[str(row["method"])][key].append(float(row[key]))
    output: dict[str, dict[str, float]] = {}
    for method, metrics in sorted(by_method.items()):
        output[method] = {}
        for key, values in metrics.items():
            output[method][f"mean_{key}"] = sum(values) / len(values)
            output[method][f"median_{key}"] = statistics.median(values)
            output[method][f"stdev_{key}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return output


def train_dev_test_evaluation(rows: list[dict[str, Any]], split: dict[str, Any], *, model_specs: list[ModelSpec], k: int) -> dict[str, Any]:
    train_topics = set(split["train_topics"])
    dev_topics = set(split["dev_topics"])
    test_topics = set(split["test_topics"])
    trained = train_models_for_topics(rows, train_topics, model_specs)
    scopes: dict[str, Any] = {}
    for scope_name, topics in [("train", train_topics), ("dev", dev_topics), ("test", test_topics)]:
        scope_rows = rows_for_topics(rows, topics)
        scopes[scope_name] = {
            "topic_count": len(topics),
            "row_count": len(scope_rows),
            "topics": sorted(topics),
            "evaluation": evaluate_rows_with_scores(scope_rows, trained, k=k),
        }
    return {
        "split_strategy": "train/dev/test topics from v3_3_splits_800.json",
        "train_topics": sorted(train_topics),
        "dev_topics": sorted(dev_topics),
        "test_topics": sorted(test_topics),
        "training_row_count": len(rows_for_topics(rows, train_topics)),
        "model_training_diagnostics": model_training_diagnostics(trained),
        "scopes": scopes,
    }


def leave_topic_out_evaluation(rows: list[dict[str, Any]], folds: list[dict[str, Any]], *, model_specs: list[ModelSpec], k: int) -> dict[str, Any]:
    fold_reports: list[dict[str, Any]] = []
    aggregate_inputs: list[dict[str, Any]] = []
    for fold in folds:
        train_topics = set(fold["train_topics"])
        test_topics = set(fold["test_topics"])
        trained = train_models_for_topics(rows, train_topics, model_specs)
        test_rows = rows_for_topics(rows, test_topics)
        scored_test_rows = scored_rows(test_rows, trained)
        score_columns = BASELINE_COLUMNS + [model_score_column(model.spec.name) for model in trained]
        evaluation = evaluate_score_columns(scored_test_rows, score_columns=score_columns, k=k)
        combined_metrics = evaluation["combined"]["method_averages"]
        for method, metrics in combined_metrics.items():
            aggregate_inputs.append({"fold_id": fold["fold_id"], "method": method, **metrics})
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "train_topics": sorted(train_topics),
                "test_topics": sorted(test_topics),
                "training_row_count": len(rows_for_topics(rows, train_topics)),
                "test_row_count": len(test_rows),
                "model_training_diagnostics": model_training_diagnostics(trained),
                "method_averages": combined_metrics,
                "per_topic": evaluation["combined"]["per_topic"],
            }
        )
    return {
        "split_strategy": "leave_one_topic_out",
        "fold_count": len(folds),
        "folds": fold_reports,
        "aggregate_mean_median": aggregate_fold_metrics(aggregate_inputs),
    }


def model_training_diagnostics(trained_models: list[TrainedModel]) -> dict[str, Any]:
    return {
        model.spec.name: {
            "model_type": model.spec.model_type,
            "feature_count": len(model.spec.feature_columns),
            "uses_v2_7_feature": "v2_7_score" in model.spec.feature_columns,
            "training_topic_count": len(model.training_topics),
            "training_row_count": model.training_row_count,
            "pairwise_training_pairs": model.pairwise_training_pairs,
        }
        for model in trained_models
    }


def best_trained_model(loto_report: dict[str, Any], model_specs: list[ModelSpec]) -> ModelResult:
    model_names = {model_score_column(spec.name): spec.name for spec in model_specs}
    aggregates = loto_report["aggregate_mean_median"]
    best_column = max(
        model_names,
        key=lambda column: (
            aggregates[column]["mean_reading_value_ndcg"],
            aggregates[column]["mean_topic_match_ndcg"],
            aggregates[column]["mean_positive_recall_at_10"],
        ),
    )
    return ModelResult(model_names[best_column], best_column, aggregates[best_column])


def best_fixed_test_model(tdt_report: dict[str, Any], model_specs: list[ModelSpec]) -> ModelResult:
    model_names = {model_score_column(spec.name): spec.name for spec in model_specs}
    averages = tdt_report["scopes"]["test"]["evaluation"]["combined"]["method_averages"]
    best_column = max(
        model_names,
        key=lambda column: (
            averages[column]["reading_value_ndcg"],
            averages[column]["topic_match_ndcg"],
            averages[column]["positive_recall_at_10"],
        ),
    )
    return ModelResult(model_names[best_column], best_column, averages[best_column])


def coefficient_or_importance_metadata(model: TrainedModel, artifact_path: Path) -> dict[str, Any]:
    estimator_info: dict[str, Any] = {}
    final_estimator = model.estimator
    if hasattr(model.estimator, "named_steps"):
        final_estimator = list(model.estimator.named_steps.values())[-1]
        scaler = model.estimator.named_steps.get("standardscaler")
        if scaler is not None:
            estimator_info["standard_scaler_mean"] = [float(value) for value in scaler.mean_.tolist()]
            estimator_info["standard_scaler_scale"] = [float(value) for value in scaler.scale_.tolist()]
    if hasattr(final_estimator, "coef_"):
        coefficients = np.ravel(final_estimator.coef_).tolist()
        estimator_info["coefficients"] = [
            {"feature": feature, "weight": float(weight)}
            for feature, weight in zip(model.spec.feature_columns, coefficients, strict=False)
        ]
    if hasattr(final_estimator, "intercept_"):
        estimator_info["intercept"] = [float(value) for value in np.ravel(final_estimator.intercept_).tolist()]
    if hasattr(final_estimator, "feature_importances_"):
        estimator_info["feature_importances"] = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(model.spec.feature_columns, final_estimator.feature_importances_, strict=False)
        ]
    return {
        "schema_version": "v3.3_ltr_800_model_metadata",
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": model.spec.name,
        "score_column": model_score_column(model.spec.name),
        "model_type": model.spec.model_type,
        "target": model.spec.target,
        "feature_columns": model.spec.feature_columns,
        "uses_v2_7_feature": "v2_7_score" in model.spec.feature_columns,
        "raw_score_feature_intersection": sorted(set(model.spec.feature_columns) & RAW_SCORE_FEATURES),
        "leakage_feature_intersection": sorted(set(model.spec.feature_columns) & LEAKAGE_COLUMNS),
        "hyperparameters": model.spec.hyperparameters,
        "training_topics": model.training_topics,
        "training_row_count": model.training_row_count,
        "pairwise_training_pairs": model.pairwise_training_pairs,
        "artifact_path": str(artifact_path),
        "estimator_info": estimator_info,
    }


def write_model_artifacts(rows: list[dict[str, Any]], model_specs: list[ModelSpec], model_dir: Path, *, best_model_name: str) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    topics = {str(row["query_id"]) for row in rows}
    trained = train_models_for_topics(rows, topics, model_specs)
    artifacts: dict[str, Any] = {}
    for model in trained:
        artifact_path = model_dir / f"{model.spec.name}.joblib"
        metadata_path = model_dir / f"{model.spec.name}.metadata.json"
        joblib.dump(model.estimator, artifact_path)
        metadata = coefficient_or_importance_metadata(model, artifact_path)
        write_json(metadata_path, metadata)
        artifacts[model.spec.name] = {
            "artifact_path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "score_column": model_score_column(model.spec.name),
        }
    index_path = model_dir / "index.json"
    index = {
        "schema_version": "v3.3_ltr_800_model_artifact_index",
        "created_at": datetime.now(UTC).isoformat(),
        "index_path": str(index_path),
        "best_model_name_by_loto_reading_ndcg": best_model_name,
        "trained_on_all_v3_3_800_rows": True,
        "runtime_integration_status": "offline_only_not_integrated",
        "artifacts": artifacts,
    }
    write_json(index_path, index)
    return index


def protected_hashes(v21_labels: Path, v25_labels: Path, v32_labels: Path) -> dict[str, Any]:
    v21_hash = file_sha256(v21_labels)
    v25_hash = file_sha256(v25_labels)
    selected_hash = file_sha256(SELECTED_240)
    return {
        "v2_1_labels_sha256": v21_hash,
        "v2_1_labels_hash_unchanged": v21_hash == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": v25_hash,
        "v2_5_labels_hash_unchanged": v25_hash == EXPECTED_V25_LABELS_SHA256,
        "v3_2_labels_sha256": file_sha256(v32_labels),
        "selected_240_sha256": selected_hash,
        "selected_240_hash_unchanged": selected_hash == EXPECTED_SELECTED_240_SHA256,
    }


def dependency_report() -> dict[str, bool]:
    return {
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
    }


def expansion_comparison(report: dict[str, Any], v30_report: dict[str, Any], best_loto: ModelResult, best_test: ModelResult) -> dict[str, Any]:
    v30_best = v30_report["best_model_by_loto_reading_ndcg"]
    v30_col = v30_best["score_column"]
    v30_test = v30_report["train_dev_test"]["scopes"]["test"]["evaluation"]["combined"]["method_averages"][v30_col]
    v33_test = report["train_dev_test"]["scopes"]["test"]["evaluation"]["combined"]["method_averages"][best_test.score_column]
    return {
        "v3_0_row_count": v30_report["dataset"]["row_count"],
        "v3_3_row_count": report["dataset"]["row_count"],
        "v3_0_best_loto_model": v30_best["model_name"],
        "v3_3_best_loto_model": best_loto.model_name,
        "loto_reading_ndcg_delta": best_loto.metrics["mean_reading_value_ndcg"] - v30_best["metrics"]["mean_reading_value_ndcg"],
        "loto_topic_ndcg_delta": best_loto.metrics["mean_topic_match_ndcg"] - v30_best["metrics"]["mean_topic_match_ndcg"],
        "loto_hard_negative_rate_delta": best_loto.metrics["mean_hard_negative_promotion_rate_at_10"] - v30_best["metrics"]["mean_hard_negative_promotion_rate_at_10"],
        "loto_reading_stdev_delta": best_loto.metrics["stdev_reading_value_ndcg"] - v30_best["metrics"].get("stdev_reading_value_ndcg", 0.0),
        "fixed_test_reading_ndcg_delta": v33_test["reading_value_ndcg"] - v30_test["reading_value_ndcg"],
        "fixed_test_topic_ndcg_delta": v33_test["topic_match_ndcg"] - v30_test["topic_match_ndcg"],
        "fixed_test_hard_negative_rate_delta": v33_test["hard_negative_promotion_rate_at_10"] - v30_test["hard_negative_promotion_rate_at_10"],
    }


def success_summary(loto_report: dict[str, Any], best_model: ModelResult) -> dict[str, Any]:
    aggregates = loto_report["aggregate_mean_median"]
    best = best_model.metrics
    hybrid = aggregates["hybrid_score"]
    v27 = aggregates["v2_7_score"]
    v26 = aggregates["v2_6_score"]
    return {
        "best_model": best_model.model_name,
        "beats_hybrid_reading_ndcg": best["mean_reading_value_ndcg"] > hybrid["mean_reading_value_ndcg"],
        "beats_hybrid_topic_ndcg": best["mean_topic_match_ndcg"] > hybrid["mean_topic_match_ndcg"],
        "beats_or_matches_v2_7_reading_ndcg": best["mean_reading_value_ndcg"] >= v27["mean_reading_value_ndcg"],
        "beats_or_matches_v2_7_topic_ndcg": best["mean_topic_match_ndcg"] >= v27["mean_topic_match_ndcg"],
        "beats_v2_6_reading_ndcg": best["mean_reading_value_ndcg"] > v26["mean_reading_value_ndcg"],
        "beats_v2_6_topic_ndcg": best["mean_topic_match_ndcg"] > v26["mean_topic_match_ndcg"],
        "reading_delta_vs_hybrid": best["mean_reading_value_ndcg"] - hybrid["mean_reading_value_ndcg"],
        "topic_delta_vs_hybrid": best["mean_topic_match_ndcg"] - hybrid["mean_topic_match_ndcg"],
        "reading_delta_vs_v2_7": best["mean_reading_value_ndcg"] - v27["mean_reading_value_ndcg"],
        "topic_delta_vs_v2_7": best["mean_topic_match_ndcg"] - v27["mean_topic_match_ndcg"],
        "reading_delta_vs_v2_6": best["mean_reading_value_ndcg"] - v26["mean_reading_value_ndcg"],
        "topic_delta_vs_v2_6": best["mean_topic_match_ndcg"] - v26["mean_topic_match_ndcg"],
        "safe_to_present_as_v2_7_upgrade": (
            best["mean_reading_value_ndcg"] >= v27["mean_reading_value_ndcg"]
            and best["mean_topic_match_ndcg"] >= v27["mean_topic_match_ndcg"]
        ),
    }


def topic_losses(report: dict[str, Any], best_model: ModelResult) -> dict[str, Any]:
    losses_v27: list[dict[str, Any]] = []
    losses_v26: list[dict[str, Any]] = []
    wins_v27: list[dict[str, Any]] = []
    for fold in report["leave_topic_out"]["folds"]:
        topic = fold["test_topics"][0]
        best = fold["method_averages"][best_model.score_column]
        v27 = fold["method_averages"]["v2_7_score"]
        v26 = fold["method_averages"]["v2_6_score"]
        delta_v27 = best["reading_value_ndcg"] - v27["reading_value_ndcg"]
        delta_v26 = best["reading_value_ndcg"] - v26["reading_value_ndcg"]
        target = wins_v27 if delta_v27 >= 0 else losses_v27
        target.append({"topic": topic, "reading_delta": delta_v27, "best_reading_ndcg": best["reading_value_ndcg"], "v2_7_reading_ndcg": v27["reading_value_ndcg"]})
        if delta_v26 < 0:
            losses_v26.append({"topic": topic, "reading_delta": delta_v26, "best_reading_ndcg": best["reading_value_ndcg"], "v2_6_reading_ndcg": v26["reading_value_ndcg"]})
    return {
        "topics_winning_or_matching_v2_7": sorted(wins_v27, key=lambda row: row["reading_delta"], reverse=True),
        "topics_losing_to_v2_7": sorted(losses_v27, key=lambda row: row["reading_delta"]),
        "topics_losing_to_v2_6": sorted(losses_v26, key=lambda row: row["reading_delta"]),
    }


def build_report(
    *,
    dataset_path: Path,
    splits_path: Path,
    v30_report_path: Path,
    model_dir: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    eval_k: int,
) -> dict[str, Any]:
    rows = load_jsonl(dataset_path)
    splits = load_json(splits_path)
    v30_report = load_json(v30_report_path)
    duplicates = duplicate_rows(rows)
    split_validation = validate_splits(splits)
    if duplicates:
        raise ValueError(f"Duplicate query_id/paper_id rows found: {duplicates[:5]}")
    if not split_validation["train_dev_test_no_topic_leakage"] or not split_validation["leave_topic_out_no_topic_leakage"]:
        raise ValueError("Topic split leakage detected.")
    hashes = protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path)
    if not hashes["v2_1_labels_hash_unchanged"] or not hashes["v2_5_labels_hash_unchanged"] or not hashes["selected_240_hash_unchanged"]:
        raise ValueError("Protected label or selected-candidate hash changed.")
    model_specs = build_model_specs()
    tdt = train_dev_test_evaluation(rows, splits["train_dev_test"], model_specs=model_specs, k=eval_k)
    loto = leave_topic_out_evaluation(rows, splits["leave_topic_out"], model_specs=model_specs, k=eval_k)
    best_loto = best_trained_model(loto, model_specs)
    best_test = best_fixed_test_model(tdt, model_specs)
    partial_report = {
        "dataset": {
            "path": str(dataset_path),
            "row_count": len(rows),
            "topic_count": len({str(row["query_id"]) for row in rows}),
            "rows_by_judged_source": dict(Counter(str(row["judged_source"]) for row in rows)),
            "rows_per_topic": dict(sorted(Counter(str(row["query_id"]) for row in rows).items())),
            "duplicate_query_paper_rows": duplicates,
        },
        "train_dev_test": tdt,
        "leave_topic_out": loto,
    }
    artifact_index = write_model_artifacts(rows, model_specs, model_dir, best_model_name=best_loto.model_name)
    report = {
        "schema_version": "v3.3_ltr_800_training",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        **partial_report,
        "splits": {
            "path": str(splits_path),
            "split_strategy": splits.get("split_strategy"),
            "validation": split_validation,
            "train_dev_test": splits["train_dev_test"],
            "leave_topic_out_fold_count": len(splits["leave_topic_out"]),
        },
        "dependencies": dependency_report(),
        "baseline_columns": BASELINE_COLUMNS,
        "raw_score_features_dropped_from_compact_variants": sorted(RAW_SCORE_FEATURES),
        "feature_variants": FEATURE_VARIANTS,
        "model_specs": [
            {
                "name": spec.name,
                "score_column": model_score_column(spec.name),
                "model_type": spec.model_type,
                "target": spec.target,
                "feature_columns": spec.feature_columns,
                "uses_v2_7_feature": "v2_7_score" in spec.feature_columns,
                "raw_score_feature_intersection": sorted(set(spec.feature_columns) & RAW_SCORE_FEATURES),
                "leakage_feature_intersection": sorted(set(spec.feature_columns) & LEAKAGE_COLUMNS),
                "hyperparameters": spec.hyperparameters,
            }
            for spec in model_specs
        ],
        "best_model_by_loto_reading_ndcg": {
            "model_name": best_loto.model_name,
            "score_column": best_loto.score_column,
            "metrics": best_loto.metrics,
        },
        "best_model_by_fixed_test_reading_ndcg": {
            "model_name": best_test.model_name,
            "score_column": best_test.score_column,
            "metrics": best_test.metrics,
        },
        "success_summary": success_summary(loto, best_loto),
        "label_expansion_comparison_vs_v3_0": expansion_comparison(partial_report, v30_report, best_loto, best_test),
        "per_topic_comparison": topic_losses(partial_report, best_loto),
        "protected_hashes": hashes,
        "model_artifact_index": artifact_index,
    }
    return report


def markdown_metric_table(averages: dict[str, dict[str, float]], *, metric_prefix: str = "mean_") -> list[str]:
    lines = [
        "| method | reading NDCG@10 | topic NDCG@10 | MRR@10 | Recall@10 | hard-neg rate@10 | reading stdev |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sorted_items = sorted(averages.items(), key=lambda item: item[1][f"{metric_prefix}reading_value_ndcg"], reverse=True)
    for method, row in sorted_items:
        stdev = row.get(f"stdev_reading_value_ndcg", 0.0)
        lines.append(
            f"| `{method}` | {row[f'{metric_prefix}reading_value_ndcg']:.3f} | "
            f"{row[f'{metric_prefix}topic_match_ndcg']:.3f} | {row[f'{metric_prefix}mrr_at_10']:.3f} | "
            f"{row[f'{metric_prefix}positive_recall_at_10']:.3f} | "
            f"{row[f'{metric_prefix}hard_negative_promotion_rate_at_10']:.3f} | {stdev:.3f} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    success = report["success_summary"]
    expansion = report["label_expansion_comparison_vs_v3_0"]
    best_loto = report["best_model_by_loto_reading_ndcg"]
    best_test = report["best_model_by_fixed_test_reading_ndcg"]
    lines = [
        "# V3.3 800-Row Supervised LTR Training Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "V3.3 trains non-neural supervised LTR models on the full 800-row judged table. It is offline-only and does not change runtime or frontend defaults.",
        "",
        "## Dataset and Safety",
        "",
        f"- Rows: `{report['dataset']['row_count']}`",
        f"- Topics: `{report['dataset']['topic_count']}`",
        f"- Rows by source: `{report['dataset']['rows_by_judged_source']}`",
        f"- Duplicate `(query_id, paper_id)` rows: `{len(report['dataset']['duplicate_query_paper_rows'])}`",
        f"- Train/dev/test topic leakage: `{not report['splits']['validation']['train_dev_test_no_topic_leakage']}`",
        f"- Leave-topic-out leakage: `{not report['splits']['validation']['leave_topic_out_no_topic_leakage']}`",
        f"- V2.1 labels unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
        f"- V2.5 labels unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
        f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
        "",
        "## Feature Variants",
        "",
        f"- Raw score features dropped from compact variants: `{report['raw_score_features_dropped_from_compact_variants']}`",
    ]
    for name, columns in report["feature_variants"].items():
        lines.append(f"- `{name}`: `{len(columns)}` features, uses V2.7 `{('v2_7_score' in columns)}`")
    lines.extend(["", "## Models", ""])
    for spec in report["model_specs"]:
        lines.append(f"- `{spec['name']}`: `{spec['model_type']}`, features `{len(spec['feature_columns'])}`, uses V2.7 `{spec['uses_v2_7_feature']}`")
    lines.extend(["", "## Leave-Topic-Out Mean Metrics", ""])
    lines.extend(markdown_metric_table(report["leave_topic_out"]["aggregate_mean_median"], metric_prefix="mean_"))
    lines.extend(["", "## Fixed Test Metrics", ""])
    test_avg = report["train_dev_test"]["scopes"]["test"]["evaluation"]["combined"]["method_averages"]
    lines.extend(markdown_metric_table({key: {f"mean_{metric}": value for metric, value in row.items()} for key, row in test_avg.items()}))
    lines.extend(
        [
            "",
            "## Best Models",
            "",
            f"- Best by LOTO mean reading NDCG@10: `{best_loto['model_name']}` (`{best_loto['score_column']}`)",
            f"- Best by fixed test reading NDCG@10: `{best_test['model_name']}` (`{best_test['score_column']}`)",
            f"- Beats hybrid reading NDCG@10: `{success['beats_hybrid_reading_ndcg']}` (`{success['reading_delta_vs_hybrid']:+.3f}`)",
            f"- Beats V2.7 reading NDCG@10: `{success['beats_or_matches_v2_7_reading_ndcg']}` (`{success['reading_delta_vs_v2_7']:+.3f}`)",
            f"- Beats V2.6 reading NDCG@10: `{success['beats_v2_6_reading_ndcg']}` (`{success['reading_delta_vs_v2_6']:+.3f}`)",
            f"- Safe to present as V2.7 upgrade: `{success['safe_to_present_as_v2_7_upgrade']}`",
            "",
            "## 800-Label Expansion vs V3.0",
            "",
            f"- V3.0 rows: `{expansion['v3_0_row_count']}`",
            f"- V3.3 rows: `{expansion['v3_3_row_count']}`",
            f"- LOTO reading NDCG delta: `{expansion['loto_reading_ndcg_delta']:+.3f}`",
            f"- LOTO topic NDCG delta: `{expansion['loto_topic_ndcg_delta']:+.3f}`",
            f"- Fixed test reading NDCG delta: `{expansion['fixed_test_reading_ndcg_delta']:+.3f}`",
            f"- Fixed test topic NDCG delta: `{expansion['fixed_test_topic_ndcg_delta']:+.3f}`",
            f"- Hard-negative rate delta: `{expansion['loto_hard_negative_rate_delta']:+.3f}`",
            f"- LOTO reading stdev delta: `{expansion['loto_reading_stdev_delta']:+.3f}`",
            "",
            "## Remaining Failures",
            "",
            f"- LOTO topics losing reading NDCG@10 to V2.7: `{len(report['per_topic_comparison']['topics_losing_to_v2_7'])}`",
            f"- LOTO topics losing reading NDCG@10 to V2.6: `{len(report['per_topic_comparison']['topics_losing_to_v2_6'])}`",
            "",
            "## Artifacts",
            "",
            f"- Model directory: `{DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)}`",
            f"- Artifact index: `{Path(report['model_artifact_index']['index_path']).relative_to(REPO_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V3.3 non-neural supervised LTR models on the 800-row dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS.relative_to(REPO_ROOT)))
    parser.add_argument("--v30-report", default=str(DEFAULT_V30_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        dataset_path=resolve_repo_path(args.dataset),
        splits_path=resolve_repo_path(args.splits),
        v30_report_path=resolve_repo_path(args.v30_report),
        model_dir=resolve_repo_path(args.model_dir),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        eval_k=args.eval_k,
    )
    report["command"] = " ".join(sys.argv)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    print("V3.3 800-row supervised LTR training complete")
    print(f"Rows: {report['dataset']['row_count']}")
    print(f"Best LOTO model: {report['best_model_by_loto_reading_ndcg']['model_name']}")
    print(f"Best fixed-test model: {report['best_model_by_fixed_test_reading_ndcg']['model_name']}")
    print(f"LOTO reading delta vs hybrid: {report['success_summary']['reading_delta_vs_hybrid']:+.3f}")
    print(f"LOTO reading delta vs V2.7: {report['success_summary']['reading_delta_vs_v2_7']:+.3f}")
    print(f"V3.3 vs V3.0 LOTO reading delta: {report['label_expansion_comparison_vs_v3_0']['loto_reading_ndcg_delta']:+.3f}")
    print(f"JSON report: {resolve_repo_path(args.json_out)}")
    print(f"Markdown report: {resolve_repo_path(args.md_out)}")
    print(f"Model directory: {resolve_repo_path(args.model_dir)}")


if __name__ == "__main__":
    main()
