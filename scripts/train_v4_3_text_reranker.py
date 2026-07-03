import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from build_v4_3_text_reranker_dataset import DEFAULT_DATASET_JSONL, NUMERIC_RANKER_FEATURES  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v2_9_score_columns import evaluate_score_columns, load_jsonl  # noqa: E402
from validate_v4_3_text_reranker_dataset import validate_rows  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_3_text_reranker"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_training_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_training_report.md"
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
MODEL_SPECS = [
    {"name": "text_only_ridge", "model_type": "tfidf_ridge_regression", "target": "relevance_label", "uses_ranker_features": False},
    {"name": "text_positive_logistic", "model_type": "tfidf_logistic_positive", "target": "positive", "uses_ranker_features": False},
    {"name": "text_plus_ranker_features_ridge", "model_type": "tfidf_plus_ranker_ridge_regression", "target": "relevance_label", "uses_ranker_features": True},
]
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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_column(model_name: str) -> str:
    return f"v4_3_{model_name}_score"


def topics(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["query_id"]) for row in rows})


def rows_for_topics(rows: list[dict[str, Any]], topic_set: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["query_id"]) in topic_set]


def row_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("text_input") or "") for row in rows]


def target_vector(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.array([float(row.get(target) or 0.0) for row in rows], dtype=float)


def numeric_matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        values = row.get("numeric_ranker_features") or {}
        matrix.append([float(values.get(feature) or 0.0) for feature in features])
    return np.array(matrix, dtype=float)


def fit_model(spec: dict[str, Any], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    x_text = vectorizer.fit_transform(row_texts(train_rows))
    scaler = None
    if spec["uses_ranker_features"]:
        scaler = StandardScaler()
        x_numeric = sparse.csr_matrix(scaler.fit_transform(numeric_matrix(train_rows, NUMERIC_RANKER_FEATURES)))
        x_train = sparse.hstack([x_text, x_numeric], format="csr")
    else:
        x_train = x_text

    if spec["model_type"] == "tfidf_logistic_positive":
        estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=17)
        estimator.fit(x_train, target_vector(train_rows, "positive"))
    else:
        estimator = Ridge(alpha=5.0, random_state=17)
        estimator.fit(x_train, target_vector(train_rows, "relevance_label"))

    return {
        "model_name": spec["name"],
        "model_type": spec["model_type"],
        "target": spec["target"],
        "uses_ranker_features": spec["uses_ranker_features"],
        "numeric_ranker_features": NUMERIC_RANKER_FEATURES if spec["uses_ranker_features"] else [],
        "vectorizer": vectorizer,
        "scaler": scaler,
        "estimator": estimator,
    }


def predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    x_text = model["vectorizer"].transform(row_texts(rows))
    if model["uses_ranker_features"]:
        x_numeric = sparse.csr_matrix(model["scaler"].transform(numeric_matrix(rows, model["numeric_ranker_features"])))
        x_values = sparse.hstack([x_text, x_numeric], format="csr")
    else:
        x_values = x_text
    estimator = model["estimator"]
    if model["model_type"] == "tfidf_logistic_positive":
        return [float(value) for value in estimator.predict_proba(x_values)[:, 1]]
    return [float(value) for value in estimator.predict(x_values)]


def add_model_scores(rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    for model in models:
        scores = predict(model, rows)
        column = score_column(model["model_name"])
        for row, score in zip(copied, scores, strict=True):
            row[column] = score
    return copied


def evaluate_models(rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> dict[str, Any]:
    scored = add_model_scores(rows, models)
    columns = BASELINE_COLUMNS + [score_column(model["model_name"]) for model in models]
    return evaluate_score_columns(scored, score_columns=columns, k=10)


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in fold_metrics:
        for key in METRIC_KEYS:
            values[str(row["method"])][key].append(float(row[key]))
    output: dict[str, dict[str, float]] = {}
    for method, metric_values in sorted(values.items()):
        output[method] = {}
        for key, rows in sorted(metric_values.items()):
            output[method][f"mean_{key}"] = sum(rows) / len(rows)
            output[method][f"median_{key}"] = statistics.median(rows)
            output[method][f"stdev_{key}"] = statistics.pstdev(rows) if len(rows) > 1 else 0.0
    return output


def split_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    models = [fit_model(spec, train_rows) for spec in MODEL_SPECS]
    scopes = {}
    for split in ["train", "dev", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        scopes[split] = {
            "row_count": len(split_rows),
            "topic_count": len(topics(split_rows)),
            "topics": topics(split_rows),
            "evaluation": evaluate_models(split_rows, models),
        }
    return {"training_row_count": len(train_rows), "scopes": scopes}


def leave_topic_out_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fold_reports = []
    aggregate_inputs = []
    all_topics = topics(rows)
    for index, test_topic in enumerate(all_topics, start=1):
        train_rows = rows_for_topics(rows, set(all_topics) - {test_topic})
        test_rows = rows_for_topics(rows, {test_topic})
        models = [fit_model(spec, train_rows) for spec in MODEL_SPECS]
        evaluation = evaluate_models(test_rows, models)
        averages = evaluation["combined"]["method_averages"]
        for method, metrics in averages.items():
            aggregate_inputs.append({"fold_id": f"loto_{index:02d}", "method": method, **metrics})
        fold_reports.append(
            {
                "fold_id": f"loto_{index:02d}",
                "train_topics": sorted(set(all_topics) - {test_topic}),
                "test_topics": [test_topic],
                "training_row_count": len(train_rows),
                "test_row_count": len(test_rows),
                "method_averages": averages,
                "per_topic": evaluation["combined"]["per_topic"],
            }
        )
    return {"fold_count": len(fold_reports), "folds": fold_reports, "aggregate_mean_median": aggregate_fold_metrics(aggregate_inputs)}


def disagreement_pair_accuracy(rows: list[dict[str, Any]], models: list[dict[str, Any]]) -> dict[str, Any]:
    scoped = [
        row for row in add_model_scores(rows, models)
        if row.get("large_score_disagreement") or row.get("v3_3_promoted_v4_demoted") or row.get("v4_fixed_v3_hard_negative")
    ]
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        by_topic[str(row["query_id"])].append(row)
    output = {}
    for model in models:
        column = score_column(model["model_name"])
        total_pairs = 0
        correct_pairs = 0
        for topic_rows in by_topic.values():
            positives = [row for row in topic_rows if row.get("positive")]
            negatives = [row for row in topic_rows if row.get("hard_negative")]
            for positive in positives:
                for negative in negatives:
                    total_pairs += 1
                    if float(positive.get(column) or 0.0) > float(negative.get(column) or 0.0):
                        correct_pairs += 1
        output[column] = {
            "scope_row_count": len(scoped),
            "pair_count": total_pairs,
            "pair_accuracy": correct_pairs / total_pairs if total_pairs else None,
        }
    return output


def save_final_models(rows: list[dict[str, Any]], model_dir: Path) -> list[dict[str, Any]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for spec in MODEL_SPECS:
        model = fit_model(spec, rows)
        artifact_path = model_dir / f"{spec['name']}.joblib"
        metadata_path = model_dir / f"{spec['name']}.metadata.json"
        joblib.dump(model, artifact_path)
        metadata = {
            "schema_version": "v4.3_text_reranker_model_metadata",
            "created_at": datetime.now(UTC).isoformat(),
            "model_name": spec["name"],
            "score_column": score_column(spec["name"]),
            "model_type": spec["model_type"],
            "target": spec["target"],
            "training_scope": "all_v4_3_text_reranker_rows",
            "training_row_count": len(rows),
            "topics": topics(rows),
            "uses_ranker_features": spec["uses_ranker_features"],
            "numeric_ranker_features": NUMERIC_RANKER_FEATURES if spec["uses_ranker_features"] else [],
            "excludes_v2_7_feature": True,
            "excludes_raw_score_features": True,
            "existing_models_retrained_by_this_script": False,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append({"model_name": spec["name"], "artifact_path": str(artifact_path), "metadata_path": str(metadata_path), **metadata})
    index = {
        "schema_version": "v4.3_text_reranker_model_index",
        "created_at": datetime.now(UTC).isoformat(),
        "models": artifacts,
    }
    (model_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifacts


def best_model(loto: dict[str, Any]) -> dict[str, Any]:
    aggregates = loto["aggregate_mean_median"]
    candidates = [score_column(spec["name"]) for spec in MODEL_SPECS]
    best = max(
        candidates,
        key=lambda column: (
            aggregates[column]["mean_reading_value_ndcg"],
            aggregates[column]["mean_topic_match_ndcg"],
            -aggregates[column]["mean_hard_negative_promotion_rate_at_10"],
        ),
    )
    return {"score_column": best, "model_name": best.removeprefix("v4_3_").removesuffix("_score"), "metrics": aggregates[best]}


def build_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    validation = validate_rows(rows, None, resolve_repo_path(args.v39_labels))
    split_report = split_evaluation(rows)
    loto = leave_topic_out_evaluation(rows)
    final_models = [fit_model(spec, rows) for spec in MODEL_SPECS]
    artifacts = save_final_models(rows, resolve_repo_path(args.model_dir))
    return {
        "schema_version": "v4.3_text_reranker_training_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "dataset": {
            "path": str(resolve_repo_path(args.dataset)),
            "row_count": len(rows),
            "topic_count": len(topics(rows)),
            "validation_valid": validation["valid"],
            "disagreement_marked_rows": validation["disagreement_marked_rows"],
        },
        "model_specs": MODEL_SPECS,
        "train_dev_test": split_report,
        "leave_topic_out": loto,
        "disagreement_pair_accuracy_on_all_rows": disagreement_pair_accuracy(rows, final_models),
        "best_model_by_loto_reading_ndcg": best_model(loto),
        "artifacts": artifacts,
        "skipped_model_families": [
            {
                "family": "sentence_transformer_embedding_regression",
                "reason": "Skipped to keep V4.3 lightweight and avoid introducing model downloads or corpus/index changes.",
            },
            {
                "family": "cross_encoder",
                "reason": "Skipped for this pass because previous V3.8-style neural reranking was not promoted and this task requested lightweight experiments.",
            },
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    best = report["best_model_by_loto_reading_ndcg"]
    lines = [
        "# V4.3 Text Reranker Training Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Dataset rows: `{report['dataset']['row_count']}`",
        f"- Topics: `{report['dataset']['topic_count']}`",
        f"- Best LOTO model: `{best['score_column']}`",
        f"- Best LOTO reading NDCG@10: `{best['metrics']['mean_reading_value_ndcg']:.3f}`",
        f"- Best LOTO topic NDCG@10: `{best['metrics']['mean_topic_match_ndcg']:.3f}`",
        "",
        "## Leave-Topic-Out Metrics",
        "",
        "| Method | Reading NDCG@10 | Topic NDCG@10 | Hard-neg@10 |",
        "|---|---:|---:|---:|",
    ]
    for method, metrics in sorted(report["leave_topic_out"]["aggregate_mean_median"].items()):
        lines.append(
            f"| `{method}` | `{metrics['mean_reading_value_ndcg']:.3f}` | "
            f"`{metrics['mean_topic_match_ndcg']:.3f}` | `{metrics['mean_hard_negative_promotion_rate_at_10']:.3f}` |"
        )
    lines.extend(["", "## Artifacts", ""])
    for artifact in report["artifacts"]:
        lines.append(f"- `{artifact['model_name']}`: `{artifact['artifact_path']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSONL))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(resolve_repo_path(args.dataset))
    report = build_report(rows, args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V4.3 text reranker training report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
