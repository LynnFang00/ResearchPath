import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
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

from build_v4_0_ltr_dataset_2400 import DEFAULT_DATASET_JSONL, DEFAULT_SPLITS_OUT  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v2_9_score_columns import evaluate_score_columns, load_jsonl, validate_splits  # noqa: E402
from train_v3_0_ltr_models import (  # noqa: E402
    METRIC_KEYS,
    ModelSpec,
    TrainedModel,
    feature_matrix,
    fit_model,
    rows_for_topics,
)


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_0_ltr_2400_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_0_ltr_2400_report.md"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_0_ltr_2400"
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
METADATA_FEATURES = ["year", "paper_age_years", "citation_count_log1p", "full_text_available_bool"]
FEATURE_VARIANTS = {
    "rank_features_only": RANK_FEATURES,
    "normalized_scores_only": NORMALIZED_SCORE_FEATURES,
    "compact_combined": NORMALIZED_SCORE_FEATURES + RANK_FEATURES + METADATA_FEATURES,
    "no_v27_compact": NORMALIZED_SCORE_FEATURES + RANK_FEATURES + METADATA_FEATURES,
    "with_v27_compact": NORMALIZED_SCORE_FEATURES + ["v2_7_score"] + RANK_FEATURES + METADATA_FEATURES,
    "v3_3_feature_compatible": NORMALIZED_SCORE_FEATURES + RANK_FEATURES + METADATA_FEATURES,
}


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    score_column: str
    metrics: dict[str, float]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_score_column(model_name: str) -> str:
    return f"v4_0_{model_name}_score"


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("ridge_rank_features_only", "pointwise_ridge_regression", FEATURE_VARIANTS["rank_features_only"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_normalized_scores_only", "pointwise_ridge_regression", FEATURE_VARIANTS["normalized_scores_only"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_compact_combined", "pointwise_ridge_regression", FEATURE_VARIANTS["compact_combined"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_no_v27_compact", "pointwise_ridge_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("ridge_with_v27_compact", "pointwise_ridge_regression", FEATURE_VARIANTS["with_v27_compact"], "relevance_label", {"alpha": 1.0, "random_state": 17}),
        ModelSpec("elasticnet_no_v27_compact", "pointwise_elasticnet_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"alpha": 0.005, "l1_ratio": 0.2, "random_state": 17, "max_iter": 10000}),
        ModelSpec("logistic_positive_no_v27_compact", "pointwise_logistic_regression", FEATURE_VARIANTS["no_v27_compact"], "positive", {"C": 1.0, "random_state": 17, "max_iter": 1000}),
        ModelSpec("random_forest_no_v27_compact", "pointwise_random_forest_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"n_estimators": 240, "max_depth": 7, "min_samples_leaf": 3, "random_state": 17}),
        ModelSpec("gradient_boosting_no_v27_compact", "pointwise_gradient_boosting_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"n_estimators": 180, "learning_rate": 0.035, "max_depth": 2, "random_state": 17}),
        ModelSpec("hist_gradient_boosting_no_v27_compact", "pointwise_hist_gradient_boosting_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label", {"max_iter": 180, "learning_rate": 0.035, "max_leaf_nodes": 8, "random_state": 17}),
        ModelSpec("pairwise_logistic_no_v27_compact", "pairwise_logistic_regression", FEATURE_VARIANTS["no_v27_compact"], "relevance_label_pair_order", {"C": 0.5, "random_state": 17, "max_iter": 1000}),
        ModelSpec("random_forest_with_v27_compact", "pointwise_random_forest_regression", FEATURE_VARIANTS["with_v27_compact"], "relevance_label", {"n_estimators": 240, "max_depth": 7, "min_samples_leaf": 3, "random_state": 17}),
    ]


def train_models_for_topics(rows: list[dict[str, Any]], train_topics: set[str], model_specs: list[ModelSpec]) -> list[TrainedModel]:
    train_rows = rows_for_topics(rows, train_topics)
    return [fit_model(spec, train_rows, train_topics) for spec in model_specs]


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
    output = {}
    for method, metrics in sorted(by_method.items()):
        output[method] = {}
        for key, values in metrics.items():
            output[method][f"mean_{key}"] = sum(values) / len(values)
            output[method][f"median_{key}"] = statistics.median(values)
            output[method][f"stdev_{key}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return output


def train_dev_test_evaluation(rows: list[dict[str, Any]], split: dict[str, Any], *, model_specs: list[ModelSpec], k: int) -> dict[str, Any]:
    train_topics = set(split["train_topics"])
    trained = train_models_for_topics(rows, train_topics, model_specs)
    scopes = {}
    for scope_name, topics in [("train", set(split["train_topics"])), ("dev", set(split["dev_topics"])), ("test", set(split["test_topics"]))]:
        scope_rows = rows_for_topics(rows, topics)
        scopes[scope_name] = {
            "topic_count": len(topics),
            "row_count": len(scope_rows),
            "topics": sorted(topics),
            "evaluation": evaluate_rows_with_scores(scope_rows, trained, k=k),
        }
    return {"training_row_count": len(rows_for_topics(rows, train_topics)), "scopes": scopes}


def leave_topic_out_evaluation(rows: list[dict[str, Any]], folds: list[dict[str, Any]], *, model_specs: list[ModelSpec], k: int) -> dict[str, Any]:
    fold_reports = []
    aggregate_inputs = []
    for fold in folds:
        train_topics = set(fold["train_topics"])
        test_topics = set(fold["test_topics"])
        trained = train_models_for_topics(rows, train_topics, model_specs)
        test_rows = rows_for_topics(rows, test_topics)
        evaluation = evaluate_rows_with_scores(test_rows, trained, k=k)
        averages = evaluation["combined"]["method_averages"]
        for method, metrics in averages.items():
            aggregate_inputs.append({"fold_id": fold["fold_id"], "method": method, **metrics})
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "train_topics": sorted(train_topics),
                "test_topics": sorted(test_topics),
                "training_row_count": len(rows_for_topics(rows, train_topics)),
                "test_row_count": len(test_rows),
                "method_averages": averages,
                "per_topic": evaluation["combined"]["per_topic"],
            }
        )
    return {"fold_count": len(folds), "folds": fold_reports, "aggregate_mean_median": aggregate_fold_metrics(aggregate_inputs)}


def best_model(loto_report: dict[str, Any], model_specs: list[ModelSpec], *, require_no_v27: bool = False) -> ModelResult:
    candidates = {
        model_score_column(spec.name): spec.name
        for spec in model_specs
        if not require_no_v27 or "v2_7_score" not in spec.feature_columns
    }
    aggregates = loto_report["aggregate_mean_median"]
    best_column = max(
        candidates,
        key=lambda column: (
            aggregates[column]["mean_reading_value_ndcg"],
            aggregates[column]["mean_topic_match_ndcg"],
            aggregates[column]["mean_positive_recall_at_10"],
        ),
    )
    return ModelResult(candidates[best_column], best_column, aggregates[best_column])


def model_metadata(model: TrainedModel, artifact_path: Path) -> dict[str, Any]:
    final_estimator = model.estimator
    info: dict[str, Any] = {}
    if hasattr(model.estimator, "named_steps"):
        final_estimator = list(model.estimator.named_steps.values())[-1]
        scaler = model.estimator.named_steps.get("standardscaler")
        if scaler is not None:
            info["standard_scaler_mean"] = [float(value) for value in scaler.mean_.tolist()]
            info["standard_scaler_scale"] = [float(value) for value in scaler.scale_.tolist()]
    if hasattr(final_estimator, "coef_"):
        info["coefficients"] = [
            {"feature": feature, "weight": float(weight)}
            for feature, weight in zip(model.spec.feature_columns, np.ravel(final_estimator.coef_).tolist(), strict=False)
        ]
    if hasattr(final_estimator, "feature_importances_"):
        info["feature_importances"] = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(model.spec.feature_columns, final_estimator.feature_importances_, strict=False)
        ]
    return {
        "schema_version": "v4.0_ltr_2400_model_metadata",
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": model.spec.name,
        "score_column": model_score_column(model.spec.name),
        "model_type": model.spec.model_type,
        "target": model.spec.target,
        "feature_columns": model.spec.feature_columns,
        "uses_v2_7_feature": "v2_7_score" in model.spec.feature_columns,
        "raw_score_feature_intersection": sorted(set(model.spec.feature_columns) & RAW_SCORE_FEATURES),
        "hyperparameters": model.spec.hyperparameters,
        "training_topics": model.training_topics,
        "training_row_count": model.training_row_count,
        "pairwise_training_pairs": model.pairwise_training_pairs,
        "artifact_path": str(artifact_path),
        "estimator_info": info,
    }


def write_model_artifacts(rows: list[dict[str, Any]], model_specs: list[ModelSpec], model_dir: Path, *, best_model_name: str, best_no_v27_model_name: str) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    trained = train_models_for_topics(rows, {str(row["query_id"]) for row in rows}, model_specs)
    artifacts = {}
    for model in trained:
        artifact_path = model_dir / f"{model.spec.name}.joblib"
        metadata_path = model_dir / f"{model.spec.name}.metadata.json"
        joblib.dump(model.estimator, artifact_path)
        metadata = model_metadata(model, artifact_path)
        write_json(metadata_path, metadata)
        artifacts[model.spec.name] = {"artifact_path": str(artifact_path), "metadata_path": str(metadata_path), "score_column": model_score_column(model.spec.name)}
    index = {
        "schema_version": "v4.0_ltr_2400_model_artifact_index",
        "created_at": datetime.now(UTC).isoformat(),
        "best_model_name_by_loto_reading_ndcg": best_model_name,
        "best_no_v27_model_name_by_loto_reading_ndcg": best_no_v27_model_name,
        "trained_on_all_v4_0_2400_rows": True,
        "runtime_integration_status": "offline_only_not_integrated",
        "artifacts": artifacts,
    }
    write_json(model_dir / "index.json", index)
    return index


def feature_importance_changes(v4_index: dict[str, Any], v33_model_dir: Path, best_no_v27_name: str) -> dict[str, Any]:
    v4_meta = json.loads(Path(v4_index["artifacts"][best_no_v27_name]["metadata_path"]).read_text(encoding="utf-8"))
    v33_path = v33_model_dir / "random_forest_no_v27_compact.metadata.json"
    if not v33_path.exists():
        return {"available": False}
    v33_meta = json.loads(v33_path.read_text(encoding="utf-8"))
    def importances(meta: dict[str, Any]) -> dict[str, float]:
        return {row["feature"]: float(row["importance"]) for row in meta.get("estimator_info", {}).get("feature_importances", [])}
    v4_imp = importances(v4_meta)
    v33_imp = importances(v33_meta)
    return {
        "available": bool(v4_imp and v33_imp),
        "v4_model": best_no_v27_name,
        "v3_3_model": "random_forest_no_v27_compact",
        "top_v4_importances": sorted(v4_imp.items(), key=lambda item: item[1], reverse=True)[:10],
        "top_v3_3_importances": sorted(v33_imp.items(), key=lambda item: item[1], reverse=True)[:10],
        "importance_deltas": sorted(
            [{"feature": feature, "delta": v4_imp.get(feature, 0.0) - v33_imp.get(feature, 0.0), "v4": v4_imp.get(feature, 0.0), "v3_3": v33_imp.get(feature, 0.0)} for feature in sorted(set(v4_imp) | set(v33_imp))],
            key=lambda row: abs(row["delta"]),
            reverse=True,
        )[:10],
    }


def build_report(dataset_path: Path, splits_path: Path, model_dir: Path, v33_model_dir: Path, eval_k: int) -> dict[str, Any]:
    rows = load_jsonl(dataset_path)
    splits = load_json(splits_path)
    split_validation = validate_splits(splits)
    model_specs = build_model_specs()
    tdt = train_dev_test_evaluation(rows, splits["train_dev_test"], model_specs=model_specs, k=eval_k)
    loto = leave_topic_out_evaluation(rows, splits["leave_topic_out"], model_specs=model_specs, k=eval_k)
    best_any = best_model(loto, model_specs)
    best_no_v27 = best_model(loto, model_specs, require_no_v27=True)
    index = write_model_artifacts(rows, model_specs, model_dir, best_model_name=best_any.model_name, best_no_v27_model_name=best_no_v27.model_name)
    report = {
        "schema_version": "v4.0_ltr_2400_training_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "dataset": {
            "path": str(dataset_path),
            "row_count": len(rows),
            "rows_by_judged_source": dict(sorted(Counter(str(row["judged_source"]) for row in rows).items())),
            "rows_per_topic": dict(sorted(Counter(str(row["query_id"]) for row in rows).items())),
        },
        "splits": {"path": str(splits_path), "validation": split_validation, "leave_topic_out_fold_count": len(splits["leave_topic_out"])},
        "feature_variants": FEATURE_VARIANTS,
        "model_specs": [
            {"name": spec.name, "score_column": model_score_column(spec.name), "model_type": spec.model_type, "feature_columns": spec.feature_columns, "uses_v2_7_feature": "v2_7_score" in spec.feature_columns}
            for spec in model_specs
        ],
        "train_dev_test": tdt,
        "leave_topic_out": loto,
        "best_model_by_loto_reading_ndcg": {"model_name": best_any.model_name, "score_column": best_any.score_column, "metrics": best_any.metrics},
        "best_no_v27_model_by_loto_reading_ndcg": {"model_name": best_no_v27.model_name, "score_column": best_no_v27.score_column, "metrics": best_no_v27.metrics},
        "model_artifact_index": index,
        "feature_importance_changes_vs_v3_3": feature_importance_changes(index, v33_model_dir, best_no_v27.model_name),
    }
    return report


def metric_table(averages: dict[str, dict[str, float]], prefix: str = "mean_") -> list[str]:
    lines = ["| method | reading | topic | MRR | recall | hard-neg |", "|---|---:|---:|---:|---:|---:|"]
    for method, row in sorted(averages.items(), key=lambda item: item[1][f"{prefix}reading_value_ndcg"], reverse=True):
        lines.append(
            f"| `{method}` | {row[f'{prefix}reading_value_ndcg']:.3f} | {row[f'{prefix}topic_match_ndcg']:.3f} | "
            f"{row[f'{prefix}mrr_at_10']:.3f} | {row[f'{prefix}positive_recall_at_10']:.3f} | {row[f'{prefix}hard_negative_promotion_rate_at_10']:.3f} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.0 LTR 2400 Training Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Rows: `{report['dataset']['row_count']}`",
        f"- Best model: `{report['best_model_by_loto_reading_ndcg']['model_name']}`",
        f"- Best no-V2.7 model: `{report['best_no_v27_model_by_loto_reading_ndcg']['model_name']}`",
        f"- Runtime integration: `{report['runtime_integration_status']}`",
        "",
        "## Leave-One-Topic-Out Metrics",
        "",
    ]
    lines.extend(metric_table(report["leave_topic_out"]["aggregate_mean_median"]))
    lines.extend(["", "## Feature Importance Changes", "", f"`{report['feature_importance_changes_vs_v3_3']}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4.0 feature-based LTR models on the 2,400-label dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSONL.relative_to(REPO_ROOT)))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-model-dir", default="data/processed/models/v3_3_ltr_800")
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        dataset_path=resolve_repo_path(args.dataset),
        splits_path=resolve_repo_path(args.splits),
        model_dir=resolve_repo_path(args.model_dir),
        v33_model_dir=resolve_repo_path(args.v33_model_dir),
        eval_k=args.eval_k,
    )
    report["command"] = " ".join(sys.argv)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    print("V4.0 LTR training complete")
    print(f"Best model: {report['best_model_by_loto_reading_ndcg']['model_name']}")
    print(f"Best no-V2.7 model: {report['best_no_v27_model_by_loto_reading_ndcg']['model_name']}")
    print(f"Report: {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
