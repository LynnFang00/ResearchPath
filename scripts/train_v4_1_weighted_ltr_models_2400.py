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
from sklearn.ensemble import RandomForestRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from build_v4_0_ltr_dataset_2400 import DEFAULT_DATASET_JSONL, DEFAULT_SPLITS_OUT  # noqa: E402
from evaluate_v2_9_score_columns import evaluate_score_columns, load_jsonl, validate_splits  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
    build_method_scores,
    evaluate_rankings,
    load_json,
    production_candidate_rows,
    resolve_repo_path,
    score_rows,
    write_json,
    write_text,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes,
)
from evaluate_v4_0_production_pool_2400 import (  # noqa: E402
    DEFAULT_MODEL_DIR as DEFAULT_V40_MODEL_DIR,
    V33_METHOD,
    labels_with_source,
    load_model,
    score_v4_model,
    weak_topic_deltas,
)
from train_v3_0_ltr_models import METRIC_KEYS, feature_matrix, numeric_feature_value, rows_for_topics  # noqa: E402
from train_v4_0_ltr_models_2400 import FEATURE_VARIANTS  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_weighted_ltr_2400_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_weighted_ltr_2400_report.md"
V40_METHOD = "v4_0_random_forest_no_v27_compact"
FEATURE_COLUMNS = FEATURE_VARIANTS["no_v27_compact"]
RF_PARAMS = {"n_estimators": 80, "max_depth": 7, "min_samples_leaf": 3, "random_state": 17, "n_jobs": -1}
BASELINE_COLUMNS = ["bm25_score", "tfidf_score", "embedding_score", "faiss_embedding_score", "hybrid_score", "old_v2_2b_score", "v2_6_score", "v2_7_score"]
STRATEGIES = [
    "uniform_all_labels",
    "downweight_v3_9_active_learning",
    "upweight_original_800",
    "upweight_human_high_confidence",
    "balanced_by_topic",
    "balanced_by_source",
    "hard_negative_preserving",
    "weak_topic_boost",
    "hybrid_original_800_1_5_v3_9_0_75",
    "hybrid_positive_reading_weight_negatives_1_0",
]
WEAK_TOPICS = {
    "v2_ai_for_scientific_discovery",
    "v2_causal_representation_learning",
    "v2_large_language_model_agents",
    "v2_llm_evaluation",
    "v2_retrieval_augmented_generation",
    "v2_transformer_architecture",
}


@dataclass
class WeightedModel:
    strategy: str
    estimator: Any
    training_topics: list[str]
    training_row_count: int
    weight_summary: dict[str, Any]


def target_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([float(row.get("relevance_label") or 0.0) for row in rows], dtype=float)


def is_original_800(row: dict[str, Any]) -> bool:
    return str(row.get("judged_source")) in {"v2_1", "v2_5", "v3_2"}


def source_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row.get("judged_source")) for row in rows)
    mean = len(rows) / max(len(counts), 1)
    return {source: mean / count for source, count in counts.items()}


def topic_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row.get("query_id")) for row in rows)
    mean = len(rows) / max(len(counts), 1)
    return {topic: mean / count for topic, count in counts.items()}


def sample_weights(rows: list[dict[str, Any]], strategy: str) -> np.ndarray:
    weights = np.ones(len(rows), dtype=float)
    by_source = source_weights(rows)
    by_topic = topic_weights(rows)
    for index, row in enumerate(rows):
        source = str(row.get("judged_source"))
        topic = str(row.get("query_id"))
        if strategy == "downweight_v3_9_active_learning" and source == "v3_9":
            weights[index] *= 0.65
        elif strategy == "upweight_original_800" and is_original_800(row):
            weights[index] *= 1.5
        elif strategy == "upweight_human_high_confidence" and str(row.get("label_confidence")) == "high":
            weights[index] *= 1.35
        elif strategy == "balanced_by_topic":
            weights[index] *= by_topic[topic]
        elif strategy == "balanced_by_source":
            weights[index] *= by_source[source]
        elif strategy == "hard_negative_preserving" and row.get("hard_negative"):
            weights[index] *= 1.75
        elif strategy == "weak_topic_boost" and topic in WEAK_TOPICS:
            weights[index] *= 1.35
        elif strategy == "hybrid_original_800_1_5_v3_9_0_75":
            if is_original_800(row):
                weights[index] *= 1.5
            elif source == "v3_9":
                weights[index] *= 0.75
        elif strategy == "hybrid_positive_reading_weight_negatives_1_0":
            if row.get("positive"):
                weights[index] *= 1.0 + float(row.get("reading_value_score") or 0.0)
            elif row.get("hard_negative"):
                weights[index] *= 1.0
        elif strategy == "uniform_all_labels":
            weights[index] *= 1.0
    return weights


def weight_summary(rows: list[dict[str, Any]], weights: np.ndarray, strategy: str) -> dict[str, Any]:
    by_source: dict[str, float] = defaultdict(float)
    by_topic: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        by_source[str(row.get("judged_source"))] += float(weight)
        by_topic[str(row.get("query_id"))] += float(weight)
    return {
        "strategy": strategy,
        "count": len(weights),
        "min": float(weights.min()) if len(weights) else 0.0,
        "max": float(weights.max()) if len(weights) else 0.0,
        "mean": float(weights.mean()) if len(weights) else 0.0,
        "by_source": dict(sorted(by_source.items())),
        "by_topic": dict(sorted(by_topic.items())),
    }


def fit_weighted_model(rows: list[dict[str, Any]], train_topics: set[str], strategy: str) -> WeightedModel:
    train_rows = rows_for_topics(rows, train_topics)
    weights = sample_weights(train_rows, strategy)
    estimator = RandomForestRegressor(**RF_PARAMS)
    estimator.fit(feature_matrix(train_rows, FEATURE_COLUMNS), target_vector(train_rows), sample_weight=weights)
    return WeightedModel(strategy=strategy, estimator=estimator, training_topics=sorted(train_topics), training_row_count=len(train_rows), weight_summary=weight_summary(train_rows, weights, strategy))


def score_column(strategy: str) -> str:
    return f"v4_1_{strategy}_score"


def method_name(strategy: str) -> str:
    return f"v4_1_{strategy}"


def predict(model: WeightedModel, rows: list[dict[str, Any]]) -> list[float]:
    return [float(value) for value in model.estimator.predict(feature_matrix(rows, FEATURE_COLUMNS)).tolist()]


def scored_rows(rows: list[dict[str, Any]], models: list[WeightedModel]) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    for model in models:
        scores = predict(model, rows)
        for row, score in zip(copied, scores, strict=True):
            row[score_column(model.strategy)] = score
    return copied


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


def train_dev_test_evaluation(rows: list[dict[str, Any]], split: dict[str, Any], *, strategies: list[str], k: int) -> dict[str, Any]:
    train_topics = set(split["train_topics"])
    models = [fit_weighted_model(rows, train_topics, strategy) for strategy in strategies]
    scopes = {}
    for scope_name, topics in [("train", set(split["train_topics"])), ("dev", set(split["dev_topics"])), ("test", set(split["test_topics"]))]:
        scope_rows = rows_for_topics(rows, topics)
        evaluation = evaluate_score_columns(scored_rows(scope_rows, models), score_columns=BASELINE_COLUMNS + [score_column(strategy) for strategy in strategies], k=k)
        scopes[scope_name] = {"topic_count": len(topics), "row_count": len(scope_rows), "topics": sorted(topics), "evaluation": evaluation}
    return {"training_row_count": len(rows_for_topics(rows, train_topics)), "scopes": scopes}


def leave_topic_out_evaluation(rows: list[dict[str, Any]], folds: list[dict[str, Any]], *, strategies: list[str], k: int) -> dict[str, Any]:
    fold_reports = []
    aggregate_inputs = []
    for fold in folds:
        train_topics = set(fold["train_topics"])
        test_topics = set(fold["test_topics"])
        models = [fit_weighted_model(rows, train_topics, strategy) for strategy in strategies]
        test_rows = rows_for_topics(rows, test_topics)
        evaluation = evaluate_score_columns(scored_rows(test_rows, models), score_columns=BASELINE_COLUMNS + [score_column(strategy) for strategy in strategies], k=k)
        averages = evaluation["combined"]["method_averages"]
        for method, metrics in averages.items():
            aggregate_inputs.append({"fold_id": fold["fold_id"], "method": method, **metrics})
        fold_reports.append({"fold_id": fold["fold_id"], "train_topics": sorted(train_topics), "test_topics": sorted(test_topics), "training_row_count": len(rows_for_topics(rows, train_topics)), "test_row_count": len(test_rows), "method_averages": averages, "per_topic": evaluation["combined"]["per_topic"]})
    return {"fold_count": len(folds), "folds": fold_reports, "aggregate_mean_median": aggregate_fold_metrics(aggregate_inputs)}


def metadata(model: WeightedModel, artifact_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "v4.1_weighted_ltr_2400_model_metadata",
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": model.strategy,
        "method_name": method_name(model.strategy),
        "score_column": score_column(model.strategy),
        "model_family": "random_forest_no_v27_compact",
        "model_type": "pointwise_random_forest_regression",
        "target": "relevance_label",
        "feature_columns": FEATURE_COLUMNS,
        "uses_v2_7_feature": False,
        "hyperparameters": RF_PARAMS,
        "training_topics": model.training_topics,
        "training_row_count": model.training_row_count,
        "weight_summary": model.weight_summary,
        "artifact_path": str(artifact_path),
        "estimator_info": {
            "feature_importances": [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(FEATURE_COLUMNS, model.estimator.feature_importances_, strict=False)
            ]
        },
    }


def write_model_artifacts(rows: list[dict[str, Any]], strategies: list[str], model_dir: Path, best_strategy: str | None) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    topics = {str(row["query_id"]) for row in rows}
    artifacts = {}
    for strategy in strategies:
        model = fit_weighted_model(rows, topics, strategy)
        artifact_path = model_dir / f"{strategy}.joblib"
        metadata_path = model_dir / f"{strategy}.metadata.json"
        joblib.dump(model.estimator, artifact_path)
        write_json(metadata_path, metadata(model, artifact_path))
        artifacts[strategy] = {"artifact_path": str(artifact_path), "metadata_path": str(metadata_path), "score_column": score_column(strategy), "method_name": method_name(strategy)}
    index = {
        "schema_version": "v4.1_weighted_ltr_2400_model_artifact_index",
        "created_at": datetime.now(UTC).isoformat(),
        "best_candidate_strategy": best_strategy,
        "trained_on_all_v4_0_2400_rows": True,
        "runtime_integration_status": "offline_only_not_integrated",
        "artifacts": artifacts,
    }
    write_json(model_dir / "index.json", index)
    return index


def load_labels(paths_by_source: dict[str, Path]) -> list[dict[str, Any]]:
    labels = []
    for source, path in paths_by_source.items():
        labels.extend(labels_with_source(path, source))
    return labels


def score_artifact(model_dir: Path, strategy: str, production_rows_by_q: dict[str, list[dict[str, Any]]], scored: dict[str, dict[int, dict[str, float]]]) -> dict[str, dict[int, float]]:
    estimator = joblib.load(model_dir / f"{strategy}.joblib")
    output: dict[str, dict[int, float]] = {}
    for query_id, rows in sorted(production_rows_by_q.items()):
        query_scores = {}
        for row in rows:
            paper_id = int(row["paper_id"])
            merged = dict(row)
            merged.update(scored[query_id][paper_id])
            x_values = np.array([[numeric_feature_value(merged, feature) for feature in FEATURE_COLUMNS]], dtype=float)
            query_scores[paper_id] = float(estimator.predict(x_values)[0])
        output[query_id] = query_scores
    return output


def production_pool_evaluation(args: argparse.Namespace, labels: list[dict[str, Any]], strategies: list[str], model_dir: Path) -> dict[str, Any]:
    packet_rows = load_jsonl(resolve_repo_path(args.v21_packet))
    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=packet_rows, top_k=args.top_k)
    scored = score_rows(candidate_rows_by_q=production_rows_by_q, context_rows_by_q=production_rows_by_q, old_model_path=resolve_repo_path(args.old_model), v26_model_path=resolve_repo_path(args.v26_model))
    method_scores, model_diagnostics = build_method_scores(production_rows_by_q=production_rows_by_q, scored=scored, model_dir=resolve_repo_path(args.v33_model_dir))
    v40_payload = load_model(resolve_repo_path(args.v40_model_dir), "random_forest_no_v27_compact")
    method_scores[V40_METHOD] = score_v4_model(model_payload=v40_payload, production_rows_by_q=production_rows_by_q, scored=scored)
    model_diagnostics[V40_METHOD] = {"metadata_path": str(resolve_repo_path(args.v40_model_dir) / "random_forest_no_v27_compact.metadata.json"), "feature_columns": v40_payload["metadata"]["feature_columns"]}
    for strategy in strategies:
        method_scores[method_name(strategy)] = score_artifact(model_dir, strategy, production_rows_by_q, scored)
        model_diagnostics[method_name(strategy)] = {"metadata_path": str(model_dir / f"{strategy}.metadata.json"), "artifact_path": str(model_dir / f"{strategy}.joblib"), "feature_columns": FEATURE_COLUMNS, "uses_v2_7_feature": False}
    evaluation = evaluate_rankings(candidate_rows_by_q=production_rows_by_q, labels=labels, method_scores=method_scores, k=K)
    return {"production_diagnostics": production_diagnostics, "model_diagnostics": model_diagnostics, "evaluation": evaluation}


def choose_best_candidate(production_evaluation: dict[str, Any], strategies: list[str]) -> dict[str, Any]:
    averages = production_evaluation["evaluation"]["method_averages"]
    v33 = averages[V33_METHOD]
    candidates = []
    for strategy in strategies:
        method = method_name(strategy)
        row = averages[method]
        deltas = weak_topic_deltas(production_evaluation["evaluation"], method, V33_METHOD)
        severe_losses = [delta for delta in deltas if delta["reading_delta"] < -0.12]
        eligible = (
            row["reading_value_ndcg"] >= v33["reading_value_ndcg"] - 0.01
            and row["hard_negative_rate_at_10"] < v33["hard_negative_rate_at_10"]
            and row["judged_at_10"] >= 9.75
            and row["unjudged_at_10"] <= 0.25
            and len(severe_losses) <= 2
        )
        candidates.append({"strategy": strategy, "method": method, "eligible_for_replacement": eligible, "reading_delta_vs_v3_3": row["reading_value_ndcg"] - v33["reading_value_ndcg"], "topic_delta_vs_v3_3": row["topic_match_ndcg"] - v33["topic_match_ndcg"], "hard_negative_delta_vs_v3_3": row["hard_negative_rate_at_10"] - v33["hard_negative_rate_at_10"], "severe_weak_topic_losses": severe_losses})
    best = max(candidates, key=lambda row: (row["eligible_for_replacement"], row["reading_delta_vs_v3_3"], row["topic_delta_vs_v3_3"], -len(row["severe_weak_topic_losses"])))
    return {"selected": best if best["eligible_for_replacement"] else None, "ranked_candidates": sorted(candidates, key=lambda row: (row["eligible_for_replacement"], row["reading_delta_vs_v3_3"], row["topic_delta_vs_v3_3"]), reverse=True)}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(resolve_repo_path(args.dataset))
    splits = load_json(resolve_repo_path(args.splits))
    split_validation = validate_splits(splits)
    strategies = STRATEGIES
    tdt = train_dev_test_evaluation(rows, splits["train_dev_test"], strategies=strategies, k=args.eval_k)
    loto = leave_topic_out_evaluation(rows, splits["leave_topic_out"], strategies=strategies, k=args.eval_k)
    labels = load_labels({"v2_1": resolve_repo_path(args.v21_labels), "v2_5": resolve_repo_path(args.v25_labels), "v3_2": resolve_repo_path(args.v32_labels), "v3_5": resolve_repo_path(args.v35_labels), "v3_9": resolve_repo_path(args.v39_labels)})
    # Write artifacts before production scoring so the same persisted estimators are evaluated.
    temp_index = write_model_artifacts(rows, strategies, resolve_repo_path(args.model_dir), best_strategy=None)
    production = production_pool_evaluation(args, labels, strategies, resolve_repo_path(args.model_dir))
    selection = choose_best_candidate(production, strategies)
    best_strategy = selection["selected"]["strategy"] if selection["selected"] else selection["ranked_candidates"][0]["strategy"]
    temp_index["best_candidate_strategy"] = best_strategy
    write_json(resolve_repo_path(args.model_dir) / "index.json", temp_index)
    return {
        "schema_version": "v4.1_weighted_ltr_2400_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "dataset": {"path": str(resolve_repo_path(args.dataset)), "row_count": len(rows), "rows_by_judged_source": dict(sorted(Counter(str(row["judged_source"]) for row in rows).items())), "rows_per_topic": dict(sorted(Counter(str(row["query_id"]) for row in rows).items()))},
        "splits": {"path": str(resolve_repo_path(args.splits)), "validation": split_validation, "leave_topic_out_fold_count": len(splits["leave_topic_out"])},
        "model_family": "random_forest_no_v27_compact",
        "feature_columns": FEATURE_COLUMNS,
        "weighting_strategies": strategies,
        "train_dev_test": tdt,
        "leave_topic_out": loto,
        "production_pool": production,
        "selection": selection,
        "model_artifact_index": temp_index,
        "protected_hashes": protected_hashes(resolve_repo_path(args.v21_labels), resolve_repo_path(args.v25_labels), resolve_repo_path(args.v32_labels), resolve_repo_path(args.v35_labels), resolve_repo_path(args.selected_240)),
    }


def metric_table(averages: dict[str, dict[str, float]], methods: list[str]) -> list[str]:
    lines = ["| method | reading | topic | judged@10 | unjudged@10 | hard-neg | MRR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        row = averages[method]
        lines.append(f"| `{method}` | {row['reading_value_ndcg']:.3f} | {row['topic_match_ndcg']:.3f} | {row['judged_at_10']:.2f} | {row['unjudged_at_10']:.2f} | {row['hard_negative_rate_at_10']:.3f} | {row['mrr_at_10']:.3f} |")
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    averages = report["production_pool"]["evaluation"]["method_averages"]
    methods = ["hybrid", "v2_6", "v2_7", V33_METHOD, V40_METHOD] + [method_name(strategy) for strategy in report["weighting_strategies"]]
    selected = report["selection"]["selected"]
    lines = [
        "# V4.1 Weighted LTR 2400 Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Model family: `{report['model_family']}`",
        f"- Replacement-eligible candidate: `{selected['method'] if selected else None}`",
        f"- Best ranked strategy: `{report['selection']['ranked_candidates'][0]['strategy']}`",
        f"- Runtime integration: `{report['runtime_integration_status']}`",
        "",
        "## Production-Pool Metrics",
        "",
    ]
    lines.extend(metric_table(averages, methods))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V4.1 source-aware weighted RF no-V2.7 LTR variants.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSONL.relative_to(REPO_ROOT)))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v40-model-dir", default=str(DEFAULT_V40_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    report["command"] = " ".join(sys.argv)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    selected = report["selection"]["selected"]
    print("V4.1 weighted LTR training complete")
    print(f"Replacement-eligible candidate: {selected['method'] if selected else None}")
    print(f"Best ranked strategy: {report['selection']['ranked_candidates'][0]['strategy']}")
    print(f"Report: {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
