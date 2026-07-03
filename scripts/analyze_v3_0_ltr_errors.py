import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
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
    load_jsonl,
    resolve_repo_path,
    write_json,
    write_text,
)
from evaluate_v2_7_score_blends import (  # noqa: E402
    EXPECTED_V25_LABELS_SHA256,
    fixed_and_penalty_scores,
    score_rows,
)
from evaluate_v2_9_score_columns import duplicate_rows, evaluate_score_columns, rank_rows, validate_splits  # noqa: E402
from train_v2_6_production_aware_learned_hybrid import (  # noqa: E402
    EXPECTED_V21_LABELS_SHA256,
    production_candidate_rows,
    rows_by_query,
)
from train_v3_0_ltr_models import (  # noqa: E402
    FEATURE_COLUMNS_NO_V27,
    feature_matrix,
    numeric_feature_value,
)


DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v2_9_ltr_dataset.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "data" / "eval" / "training" / "v2_9_splits.json"
DEFAULT_BASELINES = REPO_ROOT / "data" / "eval" / "results" / "v2_9_score_column_baselines.json"
DEFAULT_V30_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v3_0_ltr_training_report.json"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_0_ltr"
DEFAULT_V21_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_OLD_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_V26_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_1_ltr_error_analysis.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_1_ltr_error_analysis.md"
DEFAULT_PACKET_OUT = REPO_ROOT / "data" / "eval" / "labeling" / "v3_1_active_learning_candidates.jsonl"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"

SELECTED_V27_METHOD = "blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding"
RIDGE_METHOD = "ridge_no_v27"
METHOD_SCORE_COLUMNS = {
    "hybrid": "hybrid_score",
    "v2_6_score": "v2_6_score",
    "v2_7_score": "v2_7_score",
    "ridge_no_v27": "v3_0_ridge_no_v27_score",
}
BASELINE_SCORE_FIELDS = [
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
    "v2_7_score",
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_model_bundle(model_dir: Path, model_name: str) -> dict[str, Any]:
    metadata_path = model_dir / f"{model_name}.metadata.json"
    metadata = load_json(metadata_path)
    artifact_path = Path(metadata["artifact_path"])
    if not artifact_path.is_absolute():
        artifact_path = model_dir / artifact_path
    return {
        "model_name": model_name,
        "metadata": metadata,
        "estimator": joblib.load(artifact_path),
    }


def best_fixed_test_model(v30_report: dict[str, Any]) -> str:
    test_averages = v30_report["train_dev_test"]["scopes"]["test"]["evaluation"]["combined"]["method_averages"]
    model_columns = [column for column in test_averages if column.startswith("v3_0_")]
    best_column = max(
        model_columns,
        key=lambda column: (
            test_averages[column]["reading_value_ndcg"],
            test_averages[column]["topic_match_ndcg"],
            test_averages[column]["positive_recall_at_10"],
        ),
    )
    return best_column.removeprefix("v3_0_").removesuffix("_score")


def score_with_model(rows: list[dict[str, Any]], bundle: dict[str, Any]) -> list[float]:
    metadata = bundle["metadata"]
    feature_columns = metadata["feature_columns"]
    matrix = feature_matrix(rows, feature_columns)
    estimator = bundle["estimator"]
    model_type = metadata["model_type"]
    if model_type == "pointwise_logistic_regression":
        return [float(value) for value in estimator.predict_proba(matrix)[:, 1].tolist()]
    if model_type == "pairwise_logistic_regression":
        return [float(value) for value in estimator.decision_function(matrix).tolist()]
    return [float(value) for value in estimator.predict(matrix).tolist()]


def add_model_scores(rows: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [dict(row) for row in rows]
    for bundle in bundles:
        column = f"v3_0_{bundle['model_name']}_score"
        for row, score in zip(scored, score_with_model(rows, bundle), strict=True):
            row[column] = score
    return scored


def positive_demotion_count(topic_rows: list[dict[str, Any]], score_column: str, *, k: int) -> int:
    ranked = rank_rows(topic_rows, score_column)
    ranks = {int(row["paper_id"]): index for index, row in enumerate(ranked, start=1)}
    return sum(1 for row in topic_rows if row.get("positive") and ranks[int(row["paper_id"])] > k)


def top_k_ids(topic_rows: list[dict[str, Any]], score_column: str, *, k: int) -> list[int]:
    return [int(row["paper_id"]) for row in rank_rows(topic_rows, score_column)[:k]]


def top_k_overlap(topic_rows: list[dict[str, Any]], left_column: str, right_column: str, *, k: int) -> dict[str, Any]:
    left = set(top_k_ids(topic_rows, left_column, k=k))
    right = set(top_k_ids(topic_rows, right_column, k=k))
    return {
        "left_only": sorted(left - right),
        "right_only": sorted(right - left),
        "intersection_count": len(left & right),
        "jaccard": len(left & right) / len(left | right) if left | right else 0.0,
    }


def per_topic_diagnostics(
    scored_rows: list[dict[str, Any]],
    *,
    method_columns: dict[str, str],
    k: int,
) -> dict[str, Any]:
    evaluation = evaluate_score_columns(scored_rows, score_columns=list(method_columns.values()), k=k)
    metrics_by_column = {
        column: {row["query_id"]: row for row in rows}
        for column, rows in evaluation["combined"]["per_topic"].items()
    }
    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        rows_by_topic[str(row["query_id"])].append(row)

    topics: dict[str, Any] = {}
    ridge_column = method_columns["ridge_no_v27"]
    for query_id, topic_rows in sorted(rows_by_topic.items()):
        method_metrics: dict[str, Any] = {}
        for method, column in method_columns.items():
            metric = dict(metrics_by_column[column][query_id])
            metric["positive_demotion_count"] = positive_demotion_count(topic_rows, column, k=k)
            method_metrics[method] = metric
        overlaps = {}
        for method, column in method_columns.items():
            if method == "ridge_no_v27":
                continue
            overlaps[f"ridge_no_v27_vs_{method}"] = top_k_overlap(topic_rows, ridge_column, column, k=k)
        topics[query_id] = {
            "query": topic_rows[0].get("query"),
            "row_count": len(topic_rows),
            "metrics": method_metrics,
            "top_10_overlap": overlaps,
            "ridge_vs_v2_7_reading_delta": (
                method_metrics["ridge_no_v27"]["reading_value_ndcg"]
                - method_metrics["v2_7_score"]["reading_value_ndcg"]
            ),
            "ridge_vs_v2_6_reading_delta": (
                method_metrics["ridge_no_v27"]["reading_value_ndcg"]
                - method_metrics["v2_6_score"]["reading_value_ndcg"]
            ),
        }
    return {
        "topics": topics,
        "ridge_beats_v2_7_topics": sorted(
            [topic for topic, row in topics.items() if row["ridge_vs_v2_7_reading_delta"] > 0]
        ),
        "ridge_loses_to_v2_7_topics": sorted(
            [topic for topic, row in topics.items() if row["ridge_vs_v2_7_reading_delta"] < 0]
        ),
        "ridge_loses_to_v2_6_topics": sorted(
            [topic for topic, row in topics.items() if row["ridge_vs_v2_6_reading_delta"] < 0]
        ),
    }


def rank_maps(rows: list[dict[str, Any]], columns: dict[str, str]) -> dict[str, dict[tuple[str, int], int]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_topic[str(row["query_id"])].append(row)
    output: dict[str, dict[tuple[str, int], int]] = {method: {} for method in columns}
    for method, column in columns.items():
        for query_id, topic_rows in by_topic.items():
            for rank, row in enumerate(rank_rows(topic_rows, column), start=1):
                output[method][(query_id, int(row["paper_id"]))] = rank
    return output


def compact_example(row: dict[str, Any], ranks: dict[str, int] | None = None) -> dict[str, Any]:
    payload = {
        "query_id": row["query_id"],
        "query": row.get("query"),
        "paper_id": int(row["paper_id"]),
        "title": row.get("title"),
        "reading_value_score": row.get("reading_value_score"),
        "topic_match_score": row.get("topic_match_score"),
        "positive": row.get("positive"),
        "hard_negative": row.get("hard_negative"),
        "hidden_positive": row.get("hidden_positive"),
        "scores": {
            "hybrid": row.get("hybrid_score"),
            "v2_6": row.get("v2_6_score"),
            "v2_7": row.get("v2_7_score"),
            "ridge_no_v27": row.get("v3_0_ridge_no_v27_score"),
        },
    }
    if ranks is not None:
        payload["ranks"] = ranks
    return payload


def concrete_failure_examples(
    scored_rows: list[dict[str, Any]],
    *,
    method_columns: dict[str, str],
    k: int,
) -> dict[str, Any]:
    ranks_by_method = rank_maps(scored_rows, method_columns)
    row_by_key = {(str(row["query_id"]), int(row["paper_id"])): row for row in scored_rows}
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_topic[str(row["query_id"])].append(row)

    ridge_promoted_not_v27: list[dict[str, Any]] = []
    v27_promoted_not_ridge: list[dict[str, Any]] = []
    hard_negatives_top10: list[dict[str, Any]] = []
    positives_below10: list[dict[str, Any]] = []
    high_disagreement: list[dict[str, Any]] = []
    near_boundary: list[dict[str, Any]] = []

    for query_id, topic_rows in sorted(by_topic.items()):
        ridge_top = set(top_k_ids(topic_rows, method_columns["ridge_no_v27"], k=k))
        v27_top = set(top_k_ids(topic_rows, method_columns["v2_7_score"], k=k))
        for paper_id in sorted(ridge_top - v27_top):
            key = (query_id, paper_id)
            ridge_promoted_not_v27.append(
                compact_example(
                    row_by_key[key],
                    {
                        "ridge_no_v27": ranks_by_method["ridge_no_v27"][key],
                        "v2_7_score": ranks_by_method["v2_7_score"][key],
                    },
                )
            )
        for paper_id in sorted(v27_top - ridge_top):
            key = (query_id, paper_id)
            v27_promoted_not_ridge.append(
                compact_example(
                    row_by_key[key],
                    {
                        "ridge_no_v27": ranks_by_method["ridge_no_v27"][key],
                        "v2_7_score": ranks_by_method["v2_7_score"][key],
                    },
                )
            )
        for row in topic_rows:
            key = (query_id, int(row["paper_id"]))
            ranks = {method: ranks_by_method[method][key] for method in method_columns}
            if row.get("hard_negative") and min(ranks.values()) <= k:
                hard_negatives_top10.append(compact_example(row, ranks))
            if row.get("positive") and ranks["ridge_no_v27"] > k:
                positives_below10.append(compact_example(row, ranks))
            spread = max(ranks.values()) - min(ranks.values())
            if spread >= 8:
                item = compact_example(row, ranks)
                item["rank_spread"] = spread
                high_disagreement.append(item)
            if any(8 <= rank <= 12 for rank in ranks.values()):
                item = compact_example(row, ranks)
                item["boundary_methods"] = [method for method, rank in ranks.items() if 8 <= rank <= 12]
                near_boundary.append(item)

    return {
        "ridge_promoted_not_v2_7": ridge_promoted_not_v27[:40],
        "v2_7_promoted_not_ridge": v27_promoted_not_ridge[:40],
        "known_hard_negatives_ranked_top10": sorted(
            hard_negatives_top10,
            key=lambda row: min(row["ranks"].values()),
        )[:40],
        "judged_positives_ranked_below10_by_ridge": sorted(
            positives_below10,
            key=lambda row: row["ranks"]["ridge_no_v27"],
            reverse=True,
        )[:40],
        "high_disagreement_query_paper_pairs": sorted(
            high_disagreement,
            key=lambda row: row["rank_spread"],
            reverse=True,
        )[:50],
        "near_top10_boundary_examples": near_boundary[:50],
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return 0.0
    return numerator / (x_den * y_den)


def feature_correlations(rows: list[dict[str, Any]], feature_columns: list[str]) -> list[dict[str, Any]]:
    y = [float(row.get("relevance_label") or 0.0) for row in rows]
    correlations = []
    for feature in feature_columns:
        values = [numeric_feature_value(row, feature) for row in rows]
        correlations.append({"feature": feature, "pearson_relevance_label": pearson(values, y)})
    return sorted(correlations, key=lambda row: abs(row["pearson_relevance_label"]), reverse=True)


def load_feature_importances(model_dir: Path, names: list[str]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name in names:
        path = model_dir / f"{name}.metadata.json"
        if not path.exists():
            continue
        metadata = load_json(path)
        diagnostics[name] = {
            "model_type": metadata["model_type"],
            "feature_importances": sorted(
                metadata.get("estimator_info", {}).get("feature_importances", []),
                key=lambda row: row.get("importance", 0.0),
                reverse=True,
            )[:15],
        }
    return diagnostics


def model_diagnostics(
    scored_rows: list[dict[str, Any]],
    *,
    model_dir: Path,
    ridge_metadata: dict[str, Any],
) -> dict[str, Any]:
    coefficients = sorted(
        ridge_metadata["estimator_info"]["coefficients"],
        key=lambda row: abs(row["weight"]),
        reverse=True,
    )
    top_abs_weight = abs(coefficients[0]["weight"]) if coefficients else 0.0
    total_abs_weight = sum(abs(row["weight"]) for row in coefficients)
    over_reliance = {
        "top_feature": coefficients[0]["feature"] if coefficients else None,
        "top_abs_weight_share": top_abs_weight / total_abs_weight if total_abs_weight else 0.0,
        "appears_over_reliant_on_single_feature": bool(total_abs_weight and top_abs_weight / total_abs_weight > 0.5),
    }
    disagreement_rows = []
    for row in scored_rows:
        ridge = float(row.get("v3_0_ridge_no_v27_score") or 0.0)
        hybrid = float(row.get("hybrid_score") or 0.0)
        v26 = float(row.get("v2_6_score") or 0.0)
        if abs(ridge - hybrid) >= 0.35 or abs(ridge - v26) >= 0.35:
            item = compact_example(row)
            item["ridge_minus_hybrid"] = ridge - hybrid
            item["ridge_minus_v2_6"] = ridge - v26
            disagreement_rows.append(item)
    return {
        "ridge_no_v27": {
            "top_positive_coefficients": sorted(
                ridge_metadata["estimator_info"]["coefficients"],
                key=lambda row: row["weight"],
                reverse=True,
            )[:12],
            "top_negative_coefficients": sorted(
                ridge_metadata["estimator_info"]["coefficients"],
                key=lambda row: row["weight"],
            )[:12],
            "top_absolute_coefficients": coefficients[:12],
            "over_reliance_check": over_reliance,
        },
        "tree_feature_importances": load_feature_importances(
            model_dir,
            ["random_forest_no_v27", "gradient_boosting_no_v27"],
        ),
        "feature_relevance_correlations": feature_correlations(scored_rows, FEATURE_COLUMNS_NO_V27)[:20],
        "retrieval_model_signal_disagreements": sorted(
            disagreement_rows,
            key=lambda row: max(abs(row["ridge_minus_hybrid"]), abs(row["ridge_minus_v2_6"])),
            reverse=True,
        )[:40],
    }


def row_from_candidate(candidate: dict[str, Any], scored_values: dict[str, Any], v27_score: float) -> dict[str, Any]:
    row = {
        "query_id": str(candidate["query_id"]),
        "query": candidate.get("query"),
        "topic": candidate.get("query"),
        "paper_id": int(candidate["paper_id"]),
        "title": candidate.get("title"),
        "abstract": candidate.get("abstract"),
        "year": candidate.get("year"),
        "authors": candidate.get("authors") or [],
        "venue": candidate.get("venue"),
        "citation_count": candidate.get("citation_count"),
        "identifiers": candidate.get("identifiers") or {},
        "source_url": candidate.get("source_url"),
        "pdf_url": candidate.get("pdf_url"),
        "sources_provenance": candidate.get("sources_provenance") or [],
        "full_text_available": (candidate.get("evidence_availability") or {}).get("fulltext_available"),
        "v2_7_score": float(v27_score),
        "old_v2_2b_score": float(scored_values.get("old_v2_2b_score", 0.0)),
        "v2_6_score": float(scored_values.get("v2_6_score", 0.0)),
    }
    for method in ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]:
        row[f"{method}_score"] = float(scored_values.get(f"{method}_score", 0.0))
        row[f"{method}_raw_score"] = float(scored_values.get(f"{method}_raw_score", 0.0))
        row[f"{method}_rank"] = int(float(scored_values.get(f"{method}_rank", 999999)))
        row[f"{method}_reciprocal_rank"] = float(scored_values.get(f"{method}_reciprocal_rank", 0.0))
    return row


def build_unjudged_candidate_rows(
    *,
    packet_rows: list[dict[str, Any]],
    labeled_keys: set[tuple[str, int]],
    old_model_path: Path,
    v26_model_path: Path,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    production_rows, production_diagnostics = production_candidate_rows(packet_rows=packet_rows, top_k=top_k)
    scores = score_rows(
        candidate_rows_by_q=production_rows,
        context_rows_by_q=production_rows,
        old_model_path=old_model_path,
        v26_model_path=v26_model_path,
    )
    v27_scores = fixed_and_penalty_scores(scores)[SELECTED_V27_METHOD]
    rows: list[dict[str, Any]] = []
    for query_id, candidates in sorted(production_rows.items()):
        for candidate in candidates:
            paper_id = int(candidate["paper_id"])
            key = (str(query_id), paper_id)
            if key in labeled_keys:
                continue
            scored_values = scores.get(str(query_id), {}).get(paper_id)
            if scored_values is None:
                continue
            rows.append(
                row_from_candidate(
                    candidate,
                    scored_values,
                    v27_scores[str(query_id)][paper_id],
                )
            )
    return rows, {
        "production_candidate_rows_before_label_exclusion": sum(len(rows) for rows in production_rows.values()),
        "unjudged_candidate_rows_after_label_exclusion": len(rows),
        "production_diagnostics": production_diagnostics,
    }


def source_methods(row: dict[str, Any]) -> list[str]:
    return [
        method
        for method in ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
        if int(row.get(f"{method}_rank") or 999999) < 999999 or float(row.get(f"{method}_score") or 0.0) > 0.0
    ]


def add_candidate_ranks(rows: list[dict[str, Any]], columns: dict[str, str]) -> dict[tuple[str, int], dict[str, int]]:
    ranks = rank_maps(rows, columns)
    output: dict[tuple[str, int], dict[str, int]] = {}
    for method, method_ranks in ranks.items():
        for key, rank in method_ranks.items():
            output.setdefault(key, {})[method] = rank
    return output


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def topic_priority_weights(v30_report: dict[str, Any]) -> dict[str, float]:
    weights = defaultdict(lambda: 1.0)
    for row in v30_report.get("failure_summary", {}).get("loto_topics_losing_reading_ndcg_vs_v2_7", []):
        weights[str(row["topic"])] += min(1.0, abs(float(row["reading_delta"])) * 10.0)
    for row in v30_report.get("failure_summary", {}).get("loto_topics_losing_reading_ndcg_vs_v2_6", []):
        weights[str(row["topic"])] += min(1.0, abs(float(row["reading_delta"])) * 10.0)
    for topic in v30_report["splits"]["train_dev_test"]["test_topics"]:
        weights[str(topic)] += 0.75
    return dict(weights)


def select_active_learning_candidates(
    candidate_rows: list[dict[str, Any]],
    *,
    v30_report: dict[str, Any],
    per_topic_min: int,
    per_topic_max: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    columns = {
        "hybrid": "hybrid_score",
        "v2_6_score": "v2_6_score",
        "v2_7_score": "v2_7_score",
        "ridge_no_v27": "v3_0_ridge_no_v27_score",
    }
    ranks = add_candidate_ranks(candidate_rows, columns)
    ridge_scores = normalize([float(row.get("v3_0_ridge_no_v27_score") or 0.0) for row in candidate_rows])
    v27_scores = normalize([float(row.get("v2_7_score") or 0.0) for row in candidate_rows])
    v26_scores = normalize([float(row.get("v2_6_score") or 0.0) for row in candidate_rows])
    hybrid_scores = normalize([float(row.get("hybrid_score") or 0.0) for row in candidate_rows])
    priority = topic_priority_weights(v30_report)

    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_rows):
        key = (str(row["query_id"]), int(row["paper_id"]))
        row_ranks = ranks[key]
        score_values = [ridge_scores[index], v27_scores[index], v26_scores[index], hybrid_scores[index]]
        score_disagreement = max(score_values) - min(score_values)
        rank_disagreement = max(row_ranks.values()) - min(row_ranks.values())
        boundary = any(8 <= rank <= 14 for rank in row_ranks.values())
        suspected_hard_negative = row_ranks["ridge_no_v27"] <= 10 and row_ranks["v2_7_score"] > 20
        suspected_hidden_positive = (
            min(row_ranks["v2_7_score"], row_ranks["v2_6_score"], row_ranks["hybrid"]) <= 12
            and row_ranks["ridge_no_v27"] > 15
        )
        reasons = []
        if row_ranks["ridge_no_v27"] <= 12:
            reasons.append("high_scoring_unjudged_by_ridge_no_v27")
        if score_disagreement >= 0.35 or rank_disagreement >= 20:
            reasons.append("high_disagreement_between_ridge_v2_7_v2_6_hybrid")
        if boundary:
            reasons.append("near_top10_boundary")
        if suspected_hard_negative:
            reasons.append("suspected_hard_negative")
        if suspected_hidden_positive:
            reasons.append("suspected_hidden_positive")
        if priority.get(str(row["query_id"]), 1.0) > 1.0:
            reasons.append("underperforming_or_unstable_topic")
        if not reasons:
            reasons.append("coverage_candidate")
        selection_score = (
            0.45 * ridge_scores[index]
            + 0.25 * score_disagreement
            + 0.15 * min(rank_disagreement / 50.0, 1.0)
            + 0.10 * (1.0 if boundary else 0.0)
            + 0.05 * min(priority.get(str(row["query_id"]), 1.0) - 1.0, 1.0)
        )
        enriched.append(
            {
                **row,
                "selection_score": selection_score,
                "selection_reasons": reasons,
                "method_ranks": row_ranks,
                "score_disagreement": score_disagreement,
                "rank_disagreement": rank_disagreement,
                "already_labeled": False,
                "diagnostic_only": False,
            }
        )

    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_topic[str(row["query_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for query_id, rows in sorted(by_topic.items()):
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                row["selection_score"],
                -row["method_ranks"]["ridge_no_v27"],
                -int(row["paper_id"]),
            ),
            reverse=True,
        )
        selected.extend(sorted_rows[:per_topic_max])
        if len(sorted_rows) < per_topic_min:
            continue
    packet_rows = [active_learning_packet_row(row) for row in selected]
    counts = dict(Counter(row["query_id"] for row in packet_rows))
    return packet_rows, {
        "candidate_pool_size": len(candidate_rows),
        "exported_count": len(packet_rows),
        "per_topic_counts": counts,
        "topics_represented": sorted(counts),
        "topics_with_fewer_than_min_candidates": sorted(
            topic for topic, count in counts.items() if count < per_topic_min
        ),
    }


def active_learning_packet_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "v3.1_active_learning_candidate",
        "query_id": row["query_id"],
        "query": row.get("query"),
        "topic": row.get("topic") or row.get("query"),
        "paper_id": int(row["paper_id"]),
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "year": row.get("year"),
        "authors": row.get("authors") or [],
        "venue": row.get("venue"),
        "citation_count": row.get("citation_count"),
        "identifiers": row.get("identifiers") or {},
        "source_url": row.get("source_url"),
        "pdf_url": row.get("pdf_url"),
        "sources_provenance": row.get("sources_provenance") or [],
        "source_methods": source_methods(row),
        "already_labeled": False,
        "diagnostic_only": False,
        "selection_score": row["selection_score"],
        "why_selected": row["selection_reasons"],
        "scores": {
            "bm25": row.get("bm25_score"),
            "tfidf": row.get("tfidf_score"),
            "embedding": row.get("embedding_score"),
            "faiss_embedding": row.get("faiss_embedding_score"),
            "hybrid": row.get("hybrid_score"),
            "old_v2_2b": row.get("old_v2_2b_score"),
            "v2_6": row.get("v2_6_score"),
            "v2_7": row.get("v2_7_score"),
            "ridge_no_v27": row.get("v3_0_ridge_no_v27_score"),
        },
        "ranks": row["method_ranks"],
        "retrieval_ranks": {
            method: row.get(f"{method}_rank")
            for method in ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
        },
        "score_disagreement": row["score_disagreement"],
        "rank_disagreement": row["rank_disagreement"],
        "labeling_instruction": "Manual label only. Do not infer labels from selection reasons.",
    }


def validate_packet(
    packet_rows: list[dict[str, Any]],
    *,
    labeled_keys: set[tuple[str, int]],
    expected_topics: set[str],
) -> dict[str, Any]:
    duplicates = duplicate_rows(packet_rows)
    labeled_included = [
        {"query_id": row["query_id"], "paper_id": row["paper_id"]}
        for row in packet_rows
        if (str(row["query_id"]), int(row["paper_id"])) in labeled_keys and not row.get("diagnostic_only")
    ]
    topics = {str(row["query_id"]) for row in packet_rows}
    return {
        "duplicate_query_paper_rows": duplicates,
        "no_duplicate_query_paper_rows": not duplicates,
        "already_labeled_non_diagnostic_rows": labeled_included,
        "no_already_labeled_non_diagnostic_rows": not labeled_included,
        "expected_topic_count": len(expected_topics),
        "represented_topic_count": len(topics),
        "all_expected_topics_represented": expected_topics <= topics,
        "missing_topics": sorted(expected_topics - topics),
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


def build_markdown(report: dict[str, Any]) -> str:
    packet = report["active_learning_packet"]
    diagnostics = report["per_topic_diagnostics"]
    model = report["model_diagnostics"]["ridge_no_v27"]
    examples = report["failure_examples"]
    lines = [
        "# V3.1 LTR Error Analysis",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "V3.1 analyzes V3.0 offline LTR failures and exports an active-learning packet. It does not modify labels, selected candidates, runtime defaults, or frontend defaults.",
        "",
        "## Active-Learning Packet",
        "",
        f"- Exported candidates: `{packet['exported_count']}`",
        f"- Topics represented: `{len(packet['per_topic_counts'])}`",
        f"- Duplicate query-paper rows: `{len(report['packet_validation']['duplicate_query_paper_rows'])}`",
        f"- Already-labeled non-diagnostic rows: `{len(report['packet_validation']['already_labeled_non_diagnostic_rows'])}`",
        "",
        "| topic | candidates |",
        "|---|---:|",
    ]
    for topic, count in sorted(packet["per_topic_counts"].items()):
        lines.append(f"| `{topic}` | {count} |")
    lines.extend(
        [
            "",
            "## Where Ridge Improves or Fails",
            "",
            f"- Ridge beats V2.7 topics: `{len(diagnostics['ridge_beats_v2_7_topics'])}`",
            f"- Ridge loses to V2.7 topics: `{len(diagnostics['ridge_loses_to_v2_7_topics'])}`",
            f"- Ridge loses to V2.6 topics: `{len(diagnostics['ridge_loses_to_v2_6_topics'])}`",
            "",
            "Ridge loses to V2.7:",
        ]
    )
    for topic in diagnostics["ridge_loses_to_v2_7_topics"][:12]:
        delta = diagnostics["topics"][topic]["ridge_vs_v2_7_reading_delta"]
        lines.append(f"- `{topic}`: `{delta:+.3f}` reading NDCG@10")
    lines.extend(["", "Ridge loses to V2.6:"])
    for topic in diagnostics["ridge_loses_to_v2_6_topics"][:12]:
        delta = diagnostics["topics"][topic]["ridge_vs_v2_6_reading_delta"]
        lines.append(f"- `{topic}`: `{delta:+.3f}` reading NDCG@10")
    lines.extend(
        [
            "",
            "## Top Ridge Coefficients",
            "",
            "Positive:",
        ]
    )
    for row in model["top_positive_coefficients"][:8]:
        lines.append(f"- `{row['feature']}`: `{row['weight']:.4f}`")
    lines.append("")
    lines.append("Negative:")
    for row in model["top_negative_coefficients"][:8]:
        lines.append(f"- `{row['feature']}`: `{row['weight']:.4f}`")
    lines.extend(
        [
            "",
            "## Failure Examples",
            "",
            f"- Ridge promoted but V2.7 missed examples: `{len(examples['ridge_promoted_not_v2_7'])}`",
            f"- V2.7 promoted but ridge missed examples: `{len(examples['v2_7_promoted_not_ridge'])}`",
            f"- Known hard negatives in a top 10: `{len(examples['known_hard_negatives_ranked_top10'])}`",
            f"- Judged positives below ridge top 10: `{len(examples['judged_positives_ranked_below10_by_ridge'])}`",
            f"- High-disagreement pairs: `{len(examples['high_disagreement_query_paper_pairs'])}`",
            "",
            "## Recommendation",
            "",
            "Label the V3.1 active-learning packet before further V3.0 tuning. The main uncertainty is not another model family yet; it is whether the high-disagreement and boundary candidates are hidden positives or hard negatives in the expanded production pool.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(
    *,
    dataset_path: Path,
    splits_path: Path,
    baselines_path: Path,
    v30_report_path: Path,
    model_dir: Path,
    v21_packet_path: Path,
    old_model_path: Path,
    v26_model_path: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    top_k: int,
    eval_k: int,
    per_topic_min: int,
    per_topic_max: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_rows = load_jsonl(dataset_path)
    splits = load_json(splits_path)
    baselines = load_json(baselines_path)
    v30_report = load_json(v30_report_path)
    split_validation = validate_splits(splits)
    duplicates = duplicate_rows(dataset_rows)
    if duplicates:
        raise ValueError(f"Duplicate V2.9 query-paper rows found: {duplicates[:5]}")
    if not split_validation["train_dev_test_no_topic_leakage"] or not split_validation["leave_topic_out_no_topic_leakage"]:
        raise ValueError("Topic leakage detected in V2.9 splits.")

    fixed_test_model = best_fixed_test_model(v30_report)
    model_names = [RIDGE_METHOD]
    if fixed_test_model != RIDGE_METHOD:
        model_names.append(fixed_test_model)
    bundles = [load_model_bundle(model_dir, name) for name in model_names]
    scored_judged_rows = add_model_scores(dataset_rows, bundles)
    method_columns = dict(METHOD_SCORE_COLUMNS)
    if fixed_test_model != RIDGE_METHOD:
        method_columns[f"fixed_test_best_{fixed_test_model}"] = f"v3_0_{fixed_test_model}_score"

    diagnostics = per_topic_diagnostics(scored_judged_rows, method_columns=method_columns, k=eval_k)
    failures = concrete_failure_examples(scored_judged_rows, method_columns=method_columns, k=eval_k)
    ridge_bundle = bundles[0]
    model_diag = model_diagnostics(
        scored_judged_rows,
        model_dir=model_dir,
        ridge_metadata=ridge_bundle["metadata"],
    )

    labeled_keys = {(str(row["query_id"]), int(row["paper_id"])) for row in dataset_rows}
    candidate_rows, candidate_diagnostics = build_unjudged_candidate_rows(
        packet_rows=load_jsonl(v21_packet_path),
        labeled_keys=labeled_keys,
        old_model_path=old_model_path,
        v26_model_path=v26_model_path,
        top_k=top_k,
    )
    scored_candidates = add_model_scores(candidate_rows, [ridge_bundle])
    packet_rows, packet_summary = select_active_learning_candidates(
        scored_candidates,
        v30_report=v30_report,
        per_topic_min=per_topic_min,
        per_topic_max=per_topic_max,
    )
    packet_validation = validate_packet(
        packet_rows,
        labeled_keys=labeled_keys,
        expected_topics={str(row["query_id"]) for row in dataset_rows},
    )
    hashes = protected_hashes(v21_labels_path, v25_labels_path)
    if not all(
        [
            hashes["v2_1_labels_hash_unchanged"],
            hashes["v2_5_labels_hash_unchanged"],
            hashes["selected_240_hash_unchanged"],
        ]
    ):
        raise ValueError("Protected hash check failed.")
    if not packet_validation["no_duplicate_query_paper_rows"]:
        raise ValueError("Active-learning packet contains duplicate query-paper rows.")
    if not packet_validation["no_already_labeled_non_diagnostic_rows"]:
        raise ValueError("Active-learning packet contains already-labeled non-diagnostic rows.")

    report = {
        "schema_version": "v3.1_ltr_error_analysis",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "v2_9_dataset_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_changed_by_this_script": False,
        "auto_labeled_by_this_script": False,
        "inputs": {
            "dataset": str(dataset_path),
            "splits": str(splits_path),
            "baselines": str(baselines_path),
            "v3_0_report": str(v30_report_path),
            "model_dir": str(model_dir),
        },
        "dataset": {
            "row_count": len(dataset_rows),
            "topic_count": len({str(row["query_id"]) for row in dataset_rows}),
            "duplicate_query_paper_rows": duplicates,
            "rows_by_judged_source": dict(Counter(str(row["judged_source"]) for row in dataset_rows)),
        },
        "split_validation": split_validation,
        "methods_compared": list(method_columns),
        "fixed_test_best_model": fixed_test_model,
        "v2_9_baseline_combined": baselines["scopes"]["combined"]["method_averages"],
        "v3_0_failure_summary": v30_report.get("failure_summary", {}),
        "per_topic_diagnostics": diagnostics,
        "failure_examples": failures,
        "model_diagnostics": model_diag,
        "candidate_pool_diagnostics": candidate_diagnostics,
        "active_learning_packet": packet_summary,
        "packet_validation": packet_validation,
        "protected_hashes": hashes,
        "recommendation": (
            "Label the V3.1 active-learning packet before further V3.0 tuning; it targets high-disagreement, "
            "boundary, suspected hard-negative, and suspected hidden-positive candidates."
        ),
    }
    return report, packet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze V3.0 LTR errors and export a V3.1 active-learning packet.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS.relative_to(REPO_ROOT)))
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES.relative_to(REPO_ROOT)))
    parser.add_argument("--v30-report", default=str(DEFAULT_V30_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--packet-out", default=str(DEFAULT_PACKET_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--per-topic-min", type=int, default=8)
    parser.add_argument("--per-topic-max", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, packet_rows = build_report(
        dataset_path=resolve_repo_path(args.dataset),
        splits_path=resolve_repo_path(args.splits),
        baselines_path=resolve_repo_path(args.baselines),
        v30_report_path=resolve_repo_path(args.v30_report),
        model_dir=resolve_repo_path(args.model_dir),
        v21_packet_path=resolve_repo_path(args.v21_packet),
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        top_k=args.top_k,
        eval_k=args.eval_k,
        per_topic_min=args.per_topic_min,
        per_topic_max=args.per_topic_max,
    )
    report["command"] = " ".join(sys.argv)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    packet_out = resolve_repo_path(args.packet_out)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    write_jsonl(packet_out, packet_rows)
    print("V3.1 LTR error analysis complete")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")
    print(f"Active-learning packet: {packet_out}")
    print(f"Candidates exported: {report['active_learning_packet']['exported_count']}")
    print(f"Topics represented: {len(report['active_learning_packet']['per_topic_counts'])}")
    print(f"Protected hashes unchanged: {report['protected_hashes']}")


if __name__ == "__main__":
    main()
