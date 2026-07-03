import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
from evaluate_v2_9_score_columns import (  # noqa: E402
    DEFAULT_SCORE_COLUMNS,
    duplicate_rows,
    evaluate_score_columns,
    load_jsonl,
    rank_rows,
    validate_splits,
)
from train_v2_6_production_aware_learned_hybrid import EXPECTED_V21_LABELS_SHA256  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v2_9_ltr_dataset.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "data" / "eval" / "training" / "v2_9_splits.json"
DEFAULT_BASELINES = REPO_ROOT / "data" / "eval" / "results" / "v2_9_score_column_baselines.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_0_ltr_training_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_0_ltr_training_report.md"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_0_ltr"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"

BASELINE_COLUMNS = DEFAULT_SCORE_COLUMNS
DIRECT_FEATURE_COLUMNS_NO_V27 = [
    "bm25_score",
    "bm25_raw_score",
    "bm25_rank",
    "bm25_reciprocal_rank",
    "tfidf_score",
    "tfidf_raw_score",
    "tfidf_rank",
    "tfidf_reciprocal_rank",
    "embedding_score",
    "embedding_raw_score",
    "embedding_rank",
    "embedding_reciprocal_rank",
    "faiss_embedding_score",
    "faiss_embedding_raw_score",
    "faiss_embedding_rank",
    "faiss_embedding_reciprocal_rank",
    "hybrid_score",
    "hybrid_raw_score",
    "hybrid_rank",
    "hybrid_reciprocal_rank",
    "old_v2_2b_score",
    "v2_6_score",
]
DERIVED_FEATURE_COLUMNS = [
    "year",
    "paper_age_years",
    "citation_count_log1p",
    "full_text_available_bool",
]
FEATURE_COLUMNS_NO_V27 = DIRECT_FEATURE_COLUMNS_NO_V27 + DERIVED_FEATURE_COLUMNS
FEATURE_COLUMNS_WITH_V27 = FEATURE_COLUMNS_NO_V27 + ["v2_7_score"]
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
METRIC_KEYS = [
    "reading_value_ndcg",
    "topic_match_ndcg",
    "mrr_at_10",
    "positive_recall_at_10",
    "judged_at_10",
    "unjudged_at_10",
    "hard_negative_promotion_rate_at_10",
    "duplicate_penalty_at_10",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    feature_columns: list[str]
    target: str
    hyperparameters: dict[str, Any]


@dataclass
class TrainedModel:
    spec: ModelSpec
    estimator: Any
    training_topics: list[str]
    training_row_count: int
    pairwise_training_pairs: int = 0


def validate_no_duplicate_query_paper(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return duplicate_rows(rows)


def validate_topic_splits(splits: dict[str, Any]) -> dict[str, Any]:
    return validate_splits(splits)


def rows_for_topics(rows: list[dict[str, Any]], topics: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["query_id"]) in topics]


def numeric_feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "citation_count_log1p":
        return math.log1p(max(float(row.get("citation_count") or 0.0), 0.0))
    if feature == "paper_age_years":
        year = row.get("year")
        return float(max(0, 2026 - int(year))) if year else 0.0
    if feature == "full_text_available_bool":
        return 1.0 if row.get("full_text_available") else 0.0
    value = row.get(feature)
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def feature_matrix(rows: list[dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    return np.array(
        [[numeric_feature_value(row, feature) for feature in feature_columns] for row in rows],
        dtype=float,
    )


def target_vector(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    if target == "relevance_label":
        return np.array([float(row.get("relevance_label") or 0.0) for row in rows], dtype=float)
    if target == "positive":
        return np.array([1.0 if row.get("positive") else 0.0 for row in rows], dtype=float)
    raise ValueError(f"Unsupported target: {target}")


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="ridge_no_v27",
            model_type="pointwise_ridge_regression",
            feature_columns=FEATURE_COLUMNS_NO_V27,
            target="relevance_label",
            hyperparameters={"alpha": 1.0, "random_state": 17},
        ),
        ModelSpec(
            name="elasticnet_no_v27",
            model_type="pointwise_elasticnet_regression",
            feature_columns=FEATURE_COLUMNS_NO_V27,
            target="relevance_label",
            hyperparameters={"alpha": 0.005, "l1_ratio": 0.2, "random_state": 17, "max_iter": 10000},
        ),
        ModelSpec(
            name="logistic_positive_no_v27",
            model_type="pointwise_logistic_regression",
            feature_columns=FEATURE_COLUMNS_NO_V27,
            target="positive",
            hyperparameters={"C": 1.0, "random_state": 17, "max_iter": 1000},
        ),
        ModelSpec(
            name="random_forest_no_v27",
            model_type="pointwise_random_forest_regression",
            feature_columns=FEATURE_COLUMNS_NO_V27,
            target="relevance_label",
            hyperparameters={"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 3, "random_state": 17},
        ),
        ModelSpec(
            name="gradient_boosting_no_v27",
            model_type="pointwise_gradient_boosting_regression",
            feature_columns=FEATURE_COLUMNS_NO_V27,
            target="relevance_label",
            hyperparameters={"n_estimators": 120, "learning_rate": 0.035, "max_depth": 2, "random_state": 17},
        ),
        ModelSpec(
            name="hist_gradient_boosting_no_v27",
            model_type="pointwise_hist_gradient_boosting_regression",
            feature_columns=FEATURE_COLUMNS_NO_V27,
            target="relevance_label",
            hyperparameters={"max_iter": 120, "learning_rate": 0.035, "max_leaf_nodes": 8, "random_state": 17},
        ),
        ModelSpec(
            name="pairwise_logistic_no_v27",
            model_type="pairwise_logistic_regression",
            feature_columns=FEATURE_COLUMNS_NO_V27,
            target="relevance_label_pair_order",
            hyperparameters={"C": 0.5, "random_state": 17, "max_iter": 1000},
        ),
        ModelSpec(
            name="ridge_with_v27_feature",
            model_type="pointwise_ridge_regression",
            feature_columns=FEATURE_COLUMNS_WITH_V27,
            target="relevance_label",
            hyperparameters={"alpha": 1.0, "random_state": 17},
        ),
        ModelSpec(
            name="elasticnet_with_v27_feature",
            model_type="pointwise_elasticnet_regression",
            feature_columns=FEATURE_COLUMNS_WITH_V27,
            target="relevance_label",
            hyperparameters={"alpha": 0.005, "l1_ratio": 0.2, "random_state": 17, "max_iter": 10000},
        ),
    ]


def build_estimator(spec: ModelSpec) -> Any:
    if spec.model_type == "pointwise_ridge_regression":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(spec.hyperparameters["alpha"]), random_state=int(spec.hyperparameters["random_state"])),
        )
    if spec.model_type == "pointwise_elasticnet_regression":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=float(spec.hyperparameters["alpha"]),
                l1_ratio=float(spec.hyperparameters["l1_ratio"]),
                random_state=int(spec.hyperparameters["random_state"]),
                max_iter=int(spec.hyperparameters["max_iter"]),
            ),
        )
    if spec.model_type == "pointwise_logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(spec.hyperparameters["C"]),
                random_state=int(spec.hyperparameters["random_state"]),
                max_iter=int(spec.hyperparameters["max_iter"]),
            ),
        )
    if spec.model_type == "pointwise_random_forest_regression":
        return RandomForestRegressor(**spec.hyperparameters)
    if spec.model_type == "pointwise_gradient_boosting_regression":
        return GradientBoostingRegressor(**spec.hyperparameters)
    if spec.model_type == "pointwise_hist_gradient_boosting_regression":
        return HistGradientBoostingRegressor(**spec.hyperparameters)
    if spec.model_type == "pairwise_logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(spec.hyperparameters["C"]),
                random_state=int(spec.hyperparameters["random_state"]),
                max_iter=int(spec.hyperparameters["max_iter"]),
            ),
        )
    raise ValueError(f"Unsupported model type: {spec.model_type}")


def pairwise_training_matrix(rows: list[dict[str, Any]], feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    x_pairs: list[np.ndarray] = []
    y_pairs: list[float] = []
    for topic_rows in grouped.values():
        features = feature_matrix(topic_rows, feature_columns)
        targets = target_vector(topic_rows, "relevance_label")
        for left in range(len(topic_rows)):
            for right in range(left + 1, len(topic_rows)):
                delta = targets[left] - targets[right]
                if abs(delta) < 1e-12:
                    continue
                diff = features[left] - features[right]
                if delta > 0:
                    x_pairs.append(diff)
                    y_pairs.append(1.0)
                    x_pairs.append(-diff)
                    y_pairs.append(0.0)
                else:
                    x_pairs.append(-diff)
                    y_pairs.append(1.0)
                    x_pairs.append(diff)
                    y_pairs.append(0.0)
    if not x_pairs:
        raise ValueError("No within-topic judged pairs with different relevance labels were available.")
    return np.vstack(x_pairs), np.array(y_pairs, dtype=float)


def fit_model(spec: ModelSpec, train_rows: list[dict[str, Any]], train_topics: set[str]) -> TrainedModel:
    estimator = build_estimator(spec)
    pair_count = 0
    if spec.model_type == "pairwise_logistic_regression":
        x_train, y_train = pairwise_training_matrix(train_rows, spec.feature_columns)
        pair_count = len(y_train)
    else:
        x_train = feature_matrix(train_rows, spec.feature_columns)
        y_train = target_vector(train_rows, spec.target)
    estimator.fit(x_train, y_train)
    return TrainedModel(
        spec=spec,
        estimator=estimator,
        training_topics=sorted(train_topics),
        training_row_count=len(train_rows),
        pairwise_training_pairs=pair_count,
    )


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
    if model.spec.model_type in {"pointwise_logistic_regression"}:
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


def model_score_column(model_name: str) -> str:
    return f"v3_0_{model_name}_score"


def evaluate_rows_with_scores(
    rows: list[dict[str, Any]],
    trained_models: list[TrainedModel],
    *,
    k: int,
    include_baselines: bool = True,
) -> dict[str, Any]:
    scored = scored_rows(rows, trained_models)
    score_columns = [model_score_column(model.spec.name) for model in trained_models]
    if include_baselines:
        score_columns = BASELINE_COLUMNS + score_columns
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
    return output


def train_dev_test_evaluation(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    model_specs: list[ModelSpec],
    k: int,
) -> dict[str, Any]:
    train_topics = set(split["train_topics"])
    dev_topics = set(split["dev_topics"])
    test_topics = set(split["test_topics"])
    trained = train_models_for_topics(rows, train_topics, model_specs)
    scopes: dict[str, Any] = {}
    for scope_name, topics in [
        ("train", train_topics),
        ("dev", dev_topics),
        ("test", test_topics),
    ]:
        scope_rows = rows_for_topics(rows, topics)
        scopes[scope_name] = {
            "topic_count": len(topics),
            "row_count": len(scope_rows),
            "topics": sorted(topics),
            "evaluation": evaluate_rows_with_scores(scope_rows, trained, k=k),
        }
    return {
        "split_strategy": "train/dev/test topics from v2_9_splits.json",
        "train_topics": sorted(train_topics),
        "dev_topics": sorted(dev_topics),
        "test_topics": sorted(test_topics),
        "training_row_count": len(rows_for_topics(rows, train_topics)),
        "model_training_diagnostics": model_training_diagnostics(trained),
        "scopes": scopes,
    }


def leave_topic_out_evaluation(
    rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    *,
    model_specs: list[ModelSpec],
    k: int,
) -> dict[str, Any]:
    fold_reports: list[dict[str, Any]] = []
    aggregate_inputs: list[dict[str, Any]] = []
    scored_test_rows_for_harm: list[dict[str, Any]] = []
    for fold in folds:
        train_topics = set(fold["train_topics"])
        test_topics = set(fold["test_topics"])
        trained = train_models_for_topics(rows, train_topics, model_specs)
        test_rows = rows_for_topics(rows, test_topics)
        scored_test_rows = scored_rows(test_rows, trained)
        scored_test_rows_for_harm.extend(scored_test_rows)
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
        "hidden_positive_harm_vs_v2_7": hidden_positive_harm_diagnostics(
            scored_test_rows_for_harm,
            model_columns=[model_score_column(spec.name) for spec in model_specs],
        ),
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


def hidden_positive_harm_diagnostics(rows: list[dict[str, Any]], *, model_columns: list[str]) -> dict[str, Any]:
    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_topic[str(row["query_id"])].append(row)
    diagnostics: dict[str, Any] = {}
    for model_column in model_columns:
        harmed: list[dict[str, Any]] = []
        improved = 0
        unchanged = 0
        total_delta = 0.0
        count = 0
        drops_from_top10 = 0
        for topic_rows in rows_by_topic.values():
            v27_ranks = {
                int(row["paper_id"]): index
                for index, row in enumerate(rank_rows(topic_rows, "v2_7_score"), start=1)
            }
            model_ranks = {
                int(row["paper_id"]): index
                for index, row in enumerate(rank_rows(topic_rows, model_column), start=1)
            }
            for row in topic_rows:
                if not row.get("hidden_positive"):
                    continue
                paper_id = int(row["paper_id"])
                v27_rank = v27_ranks[paper_id]
                model_rank = model_ranks[paper_id]
                delta = model_rank - v27_rank
                total_delta += delta
                count += 1
                if delta > 0:
                    harmed.append(
                        {
                            "query_id": row["query_id"],
                            "paper_id": paper_id,
                            "title": row.get("title"),
                            "v2_7_rank": v27_rank,
                            "model_rank": model_rank,
                            "rank_delta_positive_is_worse": delta,
                            "reading_value_score": row.get("reading_value_score"),
                            "topic_match_score": row.get("topic_match_score"),
                        }
                    )
                    if v27_rank <= 10 < model_rank:
                        drops_from_top10 += 1
                elif delta < 0:
                    improved += 1
                else:
                    unchanged += 1
        diagnostics[model_column] = {
            "hidden_positive_count": count,
            "harmed_count": len(harmed),
            "improved_count": improved,
            "unchanged_count": unchanged,
            "drops_from_v2_7_top10_count": drops_from_top10,
            "mean_rank_delta_positive_is_worse": total_delta / count if count else 0.0,
            "worst_harms": sorted(
                harmed,
                key=lambda row: row["rank_delta_positive_is_worse"],
                reverse=True,
            )[:20],
        }
    return diagnostics


def model_metadata(model: TrainedModel, artifact_path: Path) -> dict[str, Any]:
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
        "schema_version": "v3.0_ltr_model_metadata",
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": model.spec.name,
        "score_column": model_score_column(model.spec.name),
        "model_type": model.spec.model_type,
        "target": model.spec.target,
        "feature_columns": model.spec.feature_columns,
        "uses_v2_7_feature": "v2_7_score" in model.spec.feature_columns,
        "leakage_feature_intersection": sorted(set(model.spec.feature_columns) & LEAKAGE_COLUMNS),
        "hyperparameters": model.spec.hyperparameters,
        "training_topics": model.training_topics,
        "training_row_count": model.training_row_count,
        "pairwise_training_pairs": model.pairwise_training_pairs,
        "artifact_path": str(artifact_path),
        "estimator_info": estimator_info,
    }


def write_model_artifacts(
    rows: list[dict[str, Any]],
    model_specs: list[ModelSpec],
    model_dir: Path,
    *,
    best_model_name: str,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    topics = {str(row["query_id"]) for row in rows}
    trained = train_models_for_topics(rows, topics, model_specs)
    artifacts: dict[str, Any] = {}
    for model in trained:
        artifact_path = model_dir / f"{model.spec.name}.joblib"
        metadata_path = model_dir / f"{model.spec.name}.metadata.json"
        joblib.dump(model.estimator, artifact_path)
        metadata = model_metadata(model, artifact_path)
        write_json(metadata_path, metadata)
        artifacts[model.spec.name] = {
            "artifact_path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "score_column": model_score_column(model.spec.name),
        }
    index_path = model_dir / "index.json"
    index = {
        "schema_version": "v3.0_ltr_model_artifact_index",
        "created_at": datetime.now(UTC).isoformat(),
        "index_path": str(index_path),
        "best_model_name_by_loto_reading_ndcg": best_model_name,
        "trained_on_all_v2_9_rows": True,
        "runtime_integration_status": "offline_only_not_integrated",
        "artifacts": artifacts,
    }
    write_json(index_path, index)
    return index


def dependency_report() -> dict[str, bool]:
    return {
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
    }


def protected_hashes(v21_labels: Path, v25_labels: Path) -> dict[str, Any]:
    v21_hash = file_sha256(v21_labels)
    v25_hash = file_sha256(v25_labels)
    selected_hash = file_sha256(SELECTED_240)
    return {
        "v2_1_labels_sha256": v21_hash,
        "v2_1_labels_hash_unchanged": v21_hash == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": v25_hash,
        "v2_5_labels_hash_unchanged": v25_hash == EXPECTED_V25_LABELS_SHA256,
        "selected_240_sha256": selected_hash,
        "selected_240_hash_unchanged": selected_hash == EXPECTED_SELECTED_240_SHA256,
    }


def best_trained_model(loto_report: dict[str, Any], model_specs: list[ModelSpec]) -> dict[str, Any]:
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
    return {
        "model_name": model_names[best_column],
        "score_column": best_column,
        "metrics": aggregates[best_column],
    }


def success_summary(loto_report: dict[str, Any], best_model: dict[str, Any]) -> dict[str, Any]:
    aggregates = loto_report["aggregate_mean_median"]
    best = best_model["metrics"]
    hybrid = aggregates["hybrid_score"]
    v27 = aggregates["v2_7_score"]
    v26 = aggregates["v2_6_score"]
    return {
        "best_model": best_model["model_name"],
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


def failure_summary(report: dict[str, Any], best_model: dict[str, Any]) -> dict[str, Any]:
    best_column = best_model["score_column"]
    test_averages = report["train_dev_test"]["scopes"]["test"]["evaluation"]["combined"]["method_averages"]
    best_test = test_averages[best_column]
    v27_test = test_averages["v2_7_score"]
    v26_test = test_averages["v2_6_score"]
    loto_topic_losses_vs_v27: list[dict[str, Any]] = []
    loto_topic_losses_vs_v26: list[dict[str, Any]] = []
    for fold in report["leave_topic_out"]["folds"]:
        topic = fold["test_topics"][0] if fold["test_topics"] else fold["fold_id"]
        best_metrics = fold["method_averages"][best_column]
        v27_metrics = fold["method_averages"]["v2_7_score"]
        v26_metrics = fold["method_averages"]["v2_6_score"]
        delta_v27 = best_metrics["reading_value_ndcg"] - v27_metrics["reading_value_ndcg"]
        delta_v26 = best_metrics["reading_value_ndcg"] - v26_metrics["reading_value_ndcg"]
        if delta_v27 < 0:
            loto_topic_losses_vs_v27.append(
                {
                    "topic": topic,
                    "reading_delta": delta_v27,
                    "best_model_reading_ndcg": best_metrics["reading_value_ndcg"],
                    "v2_7_reading_ndcg": v27_metrics["reading_value_ndcg"],
                }
            )
        if delta_v26 < 0:
            loto_topic_losses_vs_v26.append(
                {
                    "topic": topic,
                    "reading_delta": delta_v26,
                    "best_model_reading_ndcg": best_metrics["reading_value_ndcg"],
                    "v2_6_reading_ndcg": v26_metrics["reading_value_ndcg"],
                }
            )
    return {
        "fixed_test_reading_delta_vs_v2_7": best_test["reading_value_ndcg"] - v27_test["reading_value_ndcg"],
        "fixed_test_topic_delta_vs_v2_7": best_test["topic_match_ndcg"] - v27_test["topic_match_ndcg"],
        "fixed_test_reading_delta_vs_v2_6": best_test["reading_value_ndcg"] - v26_test["reading_value_ndcg"],
        "fixed_test_topic_delta_vs_v2_6": best_test["topic_match_ndcg"] - v26_test["topic_match_ndcg"],
        "loto_topics_losing_reading_ndcg_vs_v2_7_count": len(loto_topic_losses_vs_v27),
        "loto_topics_losing_reading_ndcg_vs_v2_6_count": len(loto_topic_losses_vs_v26),
        "loto_topics_losing_reading_ndcg_vs_v2_7": sorted(
            loto_topic_losses_vs_v27,
            key=lambda row: row["reading_delta"],
        ),
        "loto_topics_losing_reading_ndcg_vs_v2_6": sorted(
            loto_topic_losses_vs_v26,
            key=lambda row: row["reading_delta"],
        ),
    }


def build_report(
    *,
    dataset_path: Path,
    splits_path: Path,
    baselines_path: Path,
    model_dir: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    eval_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_jsonl(dataset_path)
    splits = load_json(splits_path)
    baseline_report = load_json(baselines_path)
    duplicates = validate_no_duplicate_query_paper(rows)
    split_validation = validate_topic_splits(splits)
    if duplicates:
        raise ValueError(f"Duplicate query_id/paper_id rows found: {duplicates[:5]}")
    if not split_validation["train_dev_test_no_topic_leakage"] or not split_validation["leave_topic_out_no_topic_leakage"]:
        raise ValueError("Topic split leakage detected.")
    hashes = protected_hashes(v21_labels_path, v25_labels_path)
    if not all(
        [
            hashes["v2_1_labels_hash_unchanged"],
            hashes["v2_5_labels_hash_unchanged"],
            hashes["selected_240_hash_unchanged"],
        ]
    ):
        raise ValueError("Protected label or selected-candidate hash changed.")

    model_specs = build_model_specs()
    tdt = train_dev_test_evaluation(rows, splits["train_dev_test"], model_specs=model_specs, k=eval_k)
    loto = leave_topic_out_evaluation(rows, splits["leave_topic_out"], model_specs=model_specs, k=eval_k)
    best_model = best_trained_model(loto, model_specs)
    artifact_index = write_model_artifacts(rows, model_specs, model_dir, best_model_name=best_model["model_name"])
    report = {
        "schema_version": "v3.0_ltr_training",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "v2_9_dataset_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "dataset": {
            "path": str(dataset_path),
            "row_count": len(rows),
            "topic_count": len({str(row["query_id"]) for row in rows}),
            "rows_by_judged_source": dict(Counter(str(row["judged_source"]) for row in rows)),
            "duplicate_query_paper_rows": duplicates,
        },
        "splits": {
            "path": str(splits_path),
            "split_strategy": splits.get("split_strategy"),
            "validation": split_validation,
            "train_dev_test": splits["train_dev_test"],
            "leave_topic_out_fold_count": len(splits["leave_topic_out"]),
        },
        "dependencies": dependency_report(),
        "baseline_report_path": str(baselines_path),
        "baseline_columns": BASELINE_COLUMNS,
        "v2_9_baseline_combined_v2_7": baseline_report["scopes"]["combined"]["method_averages"]["v2_7_score"],
        "model_specs": [
            {
                "name": spec.name,
                "score_column": model_score_column(spec.name),
                "model_type": spec.model_type,
                "target": spec.target,
                "feature_columns": spec.feature_columns,
                "uses_v2_7_feature": "v2_7_score" in spec.feature_columns,
                "leakage_feature_intersection": sorted(set(spec.feature_columns) & LEAKAGE_COLUMNS),
                "hyperparameters": spec.hyperparameters,
            }
            for spec in model_specs
        ],
        "train_dev_test": tdt,
        "leave_topic_out": loto,
        "best_model_by_loto_reading_ndcg": best_model,
        "success_summary": success_summary(loto, best_model),
        "protected_hashes": hashes,
        "model_artifact_index": artifact_index,
    }
    report["failure_summary"] = failure_summary(report, best_model)
    return report, artifact_index


def markdown_metric_table(averages: dict[str, dict[str, float]], *, metric_prefix: str = "mean_") -> list[str]:
    lines = [
        "| method | reading NDCG@10 | topic NDCG@10 | MRR@10 | Recall@10 | hard-neg rate@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    sorted_items = sorted(
        averages.items(),
        key=lambda item: item[1][f"{metric_prefix}reading_value_ndcg"],
        reverse=True,
    )
    for method, row in sorted_items:
        lines.append(
            f"| `{method}` | {row[f'{metric_prefix}reading_value_ndcg']:.3f} | "
            f"{row[f'{metric_prefix}topic_match_ndcg']:.3f} | {row[f'{metric_prefix}mrr_at_10']:.3f} | "
            f"{row[f'{metric_prefix}positive_recall_at_10']:.3f} | "
            f"{row[f'{metric_prefix}hard_negative_promotion_rate_at_10']:.3f} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    success = report["success_summary"]
    failures = report["failure_summary"]
    best = report["best_model_by_loto_reading_ndcg"]
    lines = [
        "# V3.0 Supervised LTR Training Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "V3.0 trains non-neural supervised learning-to-rank models on the V2.9 judged LTR table. It is offline-only and does not change runtime or frontend defaults.",
        "",
        "## Dataset and Safety",
        "",
        f"- Rows: `{report['dataset']['row_count']}`",
        f"- Topics: `{report['dataset']['topic_count']}`",
        f"- Duplicate `(query_id, paper_id)` rows: `{len(report['dataset']['duplicate_query_paper_rows'])}`",
        f"- Train/dev/test topic leakage: `{not report['splits']['validation']['train_dev_test_no_topic_leakage']}`",
        f"- Leave-topic-out leakage: `{not report['splits']['validation']['leave_topic_out_no_topic_leakage']}`",
        f"- V2.1 labels unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
        f"- V2.5 labels unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
        f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
        "",
        "## Models",
        "",
    ]
    for spec in report["model_specs"]:
        lines.append(
            f"- `{spec['name']}`: `{spec['model_type']}`, features `{len(spec['feature_columns'])}`, "
            f"uses V2.7 feature `{spec['uses_v2_7_feature']}`"
        )
    lines.extend(
        [
            "",
            "## Leave-Topic-Out Mean Metrics",
            "",
        ]
    )
    lines.extend(markdown_metric_table(report["leave_topic_out"]["aggregate_mean_median"], metric_prefix="mean_"))
    lines.extend(
        [
            "",
            "## Train/Dev/Test Topic-Held-Out Metrics",
            "",
            "### Dev",
            "",
        ]
    )
    dev_avg = report["train_dev_test"]["scopes"]["dev"]["evaluation"]["combined"]["method_averages"]
    lines.extend(markdown_metric_table({key: {f"mean_{metric}": value for metric, value in row.items()} for key, row in dev_avg.items()}))
    lines.extend(["", "### Test", ""])
    test_avg = report["train_dev_test"]["scopes"]["test"]["evaluation"]["combined"]["method_averages"]
    lines.extend(markdown_metric_table({key: {f"mean_{metric}": value for metric, value in row.items()} for key, row in test_avg.items()}))
    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"- Best by leave-topic-out mean reading NDCG@10: `{best['model_name']}`",
            f"- Score column: `{best['score_column']}`",
            f"- Beats hybrid reading NDCG@10: `{success['beats_hybrid_reading_ndcg']}` (`{success['reading_delta_vs_hybrid']:+.3f}`)",
            f"- Beats hybrid topic NDCG@10: `{success['beats_hybrid_topic_ndcg']}` (`{success['topic_delta_vs_hybrid']:+.3f}`)",
            f"- Beats or matches V2.7 reading NDCG@10: `{success['beats_or_matches_v2_7_reading_ndcg']}` (`{success['reading_delta_vs_v2_7']:+.3f}`)",
            f"- Beats or matches V2.7 topic NDCG@10: `{success['beats_or_matches_v2_7_topic_ndcg']}` (`{success['topic_delta_vs_v2_7']:+.3f}`)",
            f"- Beats V2.6 reading NDCG@10: `{success['beats_v2_6_reading_ndcg']}` (`{success['reading_delta_vs_v2_6']:+.3f}`)",
            f"- Beats V2.6 topic NDCG@10: `{success['beats_v2_6_topic_ndcg']}` (`{success['topic_delta_vs_v2_6']:+.3f}`)",
            f"- Safe to present as a V2.7 upgrade: `{success['safe_to_present_as_v2_7_upgrade']}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if success["safe_to_present_as_v2_7_upgrade"]:
        lines.append("The best V3.0 model meets the grouped-evaluation bar against V2.7 on both reading and topic NDCG@10.")
    else:
        lines.append(
            "The best V3.0 model does not clear the grouped-evaluation upgrade bar over V2.7 on both reading and topic NDCG@10. Treat it as an offline diagnostic, not a replacement."
        )
    if not success["beats_v2_6_reading_ndcg"] or not success["beats_v2_6_topic_ndcg"]:
        lines.append(
            "However, the trained V3.0 models do not beat the existing V2.6 score column on the judged-table leave-topic-out aggregate, so V3.0 should not be described as the strongest overall judged-table signal."
        )
    lines.extend(
        [
            "",
            "## Failure Summary",
            "",
            f"- Fixed test reading delta vs V2.7: `{failures['fixed_test_reading_delta_vs_v2_7']:+.3f}`",
            f"- Fixed test topic delta vs V2.7: `{failures['fixed_test_topic_delta_vs_v2_7']:+.3f}`",
            f"- Fixed test reading delta vs V2.6: `{failures['fixed_test_reading_delta_vs_v2_6']:+.3f}`",
            f"- Fixed test topic delta vs V2.6: `{failures['fixed_test_topic_delta_vs_v2_6']:+.3f}`",
            f"- LOTO topics losing reading NDCG@10 vs V2.7: `{failures['loto_topics_losing_reading_ndcg_vs_v2_7_count']}`",
            f"- LOTO topics losing reading NDCG@10 vs V2.6: `{failures['loto_topics_losing_reading_ndcg_vs_v2_6_count']}`",
        ]
    )
    lines.extend(
        [
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
    parser = argparse.ArgumentParser(description="Train V3.0 non-neural supervised LTR models.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS.relative_to(REPO_ROOT)))
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    splits_path = resolve_repo_path(args.splits)
    baselines_path = resolve_repo_path(args.baselines)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    model_dir = resolve_repo_path(args.model_dir)
    report, _ = build_report(
        dataset_path=dataset_path,
        splits_path=splits_path,
        baselines_path=baselines_path,
        model_dir=model_dir,
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        eval_k=args.eval_k,
    )
    report["command"] = " ".join(sys.argv)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    success = report["success_summary"]
    print("V3.0 supervised LTR training complete")
    print(f"Rows: {report['dataset']['row_count']}")
    print(f"Best model: {success['best_model']}")
    print(f"LOTO reading delta vs hybrid: {success['reading_delta_vs_hybrid']:+.3f}")
    print(f"LOTO reading delta vs V2.7: {success['reading_delta_vs_v2_7']:+.3f}")
    print(f"Safe V2.7 upgrade: {success['safe_to_present_as_v2_7_upgrade']}")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")
    print(f"Model directory: {model_dir}")


if __name__ == "__main__":
    main()
