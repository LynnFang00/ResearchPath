import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
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
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.v2_2_learned_ranker import build_feature_context, load_v2_2_ridge_ranker  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    BASELINE_METHODS,
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    aggregate,
    clamp_score,
    file_sha256,
    labels_by_query,
    load_json,
    load_jsonl,
    resolve_repo_path,
    target_gain,
    write_json,
    write_text,
)
from train_v2_6_production_aware_learned_hybrid import (  # noqa: E402
    AUDIENCE_SCORE_FIELDS,
    EXPECTED_V21_LABELS_SHA256,
    combined_label_diagnostics,
    evaluate_ranked_ids,
    flatten_rows,
    per_topic_winners,
    production_candidate_rows,
    rows_by_key,
    rows_by_query,
)


DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V21_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_V25_PACKET = REPO_ROOT / "data" / "eval" / "v2_5_hard_negative_labeling_packet.jsonl"
DEFAULT_V26_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v2_6_production_aware_learned_hybrid_report.json"
DEFAULT_OLD_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_V26_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"
DEFAULT_MODEL_OUT = REPO_ROOT / "data" / "processed" / "models" / "v2_7_score_blend.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_7_score_blend_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_7_score_blend_report.md"

EXPECTED_V25_LABELS_SHA256 = "F3CEFD7ED5C89D79796AD487C255879A8246DE87692289C9A3A6C67157F7453C"

SCORE_METHODS = [
    "old_only",
    "v2_6_only",
    "blend_0_9_old_0_1_v26",
    "blend_0_8_old_0_2_v26",
    "blend_0_7_old_0_3_v26",
    "blend_0_8_old_0_1_v26_0_1_hybrid",
    "blend_0_7_old_0_2_v26_0_1_embedding",
    "blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding",
    "penalty_lambda_0_05",
    "penalty_lambda_0_10",
    "penalty_lambda_0_15",
    "penalty_lambda_0_20",
    "penalty_lambda_0_30",
    "stacked_ridge_loto_cv",
    "stacked_ridge_full_data_apparent",
]

REFERENCE_METHODS = [
    "bm25",
    "tfidf",
    "embedding",
    "faiss_embedding",
    "hybrid",
    "old_v2_2b_learned_hybrid",
    "v2_6_learned_hybrid",
]

ALL_METHODS = REFERENCE_METHODS + SCORE_METHODS

FIXED_BLEND_WEIGHTS = {
    "blend_0_9_old_0_1_v26": {"old": 0.9, "v26": 0.1},
    "blend_0_8_old_0_2_v26": {"old": 0.8, "v26": 0.2},
    "blend_0_7_old_0_3_v26": {"old": 0.7, "v26": 0.3},
    "blend_0_8_old_0_1_v26_0_1_hybrid": {"old": 0.8, "v26": 0.1, "hybrid": 0.1},
    "blend_0_7_old_0_2_v26_0_1_embedding": {"old": 0.7, "v26": 0.2, "embedding": 0.1},
    "blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding": {
        "old": 0.6,
        "v26": 0.2,
        "hybrid": 0.1,
        "embedding": 0.1,
    },
}

PENALTY_LAMBDAS = {
    "penalty_lambda_0_05": 0.05,
    "penalty_lambda_0_10": 0.10,
    "penalty_lambda_0_15": 0.15,
    "penalty_lambda_0_20": 0.20,
    "penalty_lambda_0_30": 0.30,
}

STACKED_FEATURE_NAMES = [
    "old_v2_2b_score",
    "v2_6_score",
    "hybrid_score",
    "embedding_score",
    "faiss_embedding_score",
    "bm25_score",
    "tfidf_score",
    "old_minus_v26",
    "old_minus_v26_positive",
    "v26_minus_old_positive",
    "hybrid_reciprocal_rank",
    "embedding_reciprocal_rank",
    "faiss_embedding_reciprocal_rank",
    "bm25_reciprocal_rank",
    "tfidf_reciprocal_rank",
]


def labels_by_key(labels: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["query_id"]), int(row["paper_id"])): row for row in labels}


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def score_rows(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    context_rows_by_q: dict[str, list[dict[str, Any]]],
    old_model_path: Path,
    v26_model_path: Path,
) -> dict[str, dict[int, dict[str, float]]]:
    old_ranker = load_v2_2_ridge_ranker(old_model_path)
    v26_ranker = load_v2_2_ridge_ranker(v26_model_path)
    context = build_feature_context(flatten_rows(context_rows_by_q))
    scored: dict[str, dict[int, dict[str, float]]] = {}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        per_query: dict[int, dict[str, float]] = {}
        raw_baselines: dict[str, dict[int, float]] = {method: {} for method in BASELINE_METHODS}
        ranks_by_method: dict[str, dict[int, int]] = {method: {} for method in BASELINE_METHODS}
        for row in rows:
            paper_id = int(row["paper_id"])
            per_query[paper_id] = {
                "old_v2_2b_score": float(old_ranker.score_packet_row(row, context)),
                "v2_6_score": float(v26_ranker.score_packet_row(row, context)),
            }
            for method in BASELINE_METHODS:
                raw_score = float((row.get("retrieval_scores_by_method") or {}).get(method, 0.0))
                rank = int((row.get("retrieval_ranks_by_method") or {}).get(method, 999999))
                raw_baselines[method][paper_id] = raw_score
                ranks_by_method[method][paper_id] = rank

        normalized_baselines = {method: minmax(values) for method, values in raw_baselines.items()}
        for paper_id, values in per_query.items():
            for method in BASELINE_METHODS:
                rank = ranks_by_method[method].get(paper_id, 999999)
                values[f"{method}_score"] = normalized_baselines[method].get(paper_id, 0.0)
                values[f"{method}_raw_score"] = raw_baselines[method].get(paper_id, 0.0)
                values[f"{method}_rank"] = float(rank)
                values[f"{method}_reciprocal_rank"] = 1.0 / max(1.0, float(rank))
            old_score = values["old_v2_2b_score"]
            v26_score = values["v2_6_score"]
            values["old_minus_v26"] = old_score - v26_score
            values["old_minus_v26_positive"] = max(0.0, old_score - v26_score)
            values["v26_minus_old_positive"] = max(0.0, v26_score - old_score)
        scored[query_id] = per_query
    return scored


def fixed_and_penalty_scores(scored: dict[str, dict[int, dict[str, float]]]) -> dict[str, dict[str, dict[int, float]]]:
    method_scores: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for query_id, rows in scored.items():
        method_scores["old_v2_2b_learned_hybrid"][query_id] = {
            paper_id: values["old_v2_2b_score"] for paper_id, values in rows.items()
        }
        method_scores["v2_6_learned_hybrid"][query_id] = {
            paper_id: values["v2_6_score"] for paper_id, values in rows.items()
        }
        method_scores["old_only"][query_id] = dict(method_scores["old_v2_2b_learned_hybrid"][query_id])
        method_scores["v2_6_only"][query_id] = dict(method_scores["v2_6_learned_hybrid"][query_id])
        for method in BASELINE_METHODS:
            method_scores[method][query_id] = {
                paper_id: values[f"{method}_score"] for paper_id, values in rows.items()
            }
        for method, weights in FIXED_BLEND_WEIGHTS.items():
            method_scores[method][query_id] = {
                paper_id: (
                    weights.get("old", 0.0) * values["old_v2_2b_score"]
                    + weights.get("v26", 0.0) * values["v2_6_score"]
                    + weights.get("hybrid", 0.0) * values["hybrid_score"]
                    + weights.get("embedding", 0.0) * values["embedding_score"]
                )
                for paper_id, values in rows.items()
            }
        for method, penalty_lambda in PENALTY_LAMBDAS.items():
            method_scores[method][query_id] = {
                paper_id: values["old_v2_2b_score"] - penalty_lambda * values["old_minus_v26_positive"]
                for paper_id, values in rows.items()
            }
    return dict(method_scores)


def feature_vector(values: dict[str, float]) -> list[float]:
    return [float(values[name]) for name in STACKED_FEATURE_NAMES]


def fit_stacked_ridge(training_rows: list[dict[str, Any]], *, alpha: float) -> Any:
    x_train = np.array([row["features"] for row in training_rows], dtype=float)
    y_train = np.array([row["target"] for row in training_rows], dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=17))
    model.fit(x_train, y_train)
    return model


def build_stacked_training_rows(
    *,
    v21_labels: list[dict[str, Any]],
    v25_labels: list[dict[str, Any]],
    scope_scores: dict[str, dict[str, dict[int, dict[str, float]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ("v2_1", v21_labels, "v2_1_packet"),
        ("v2_5", v25_labels, "v2_5_hard_negative"),
    ]
    for source, labels, scope_name in sources:
        scored = scope_scores[scope_name]
        for label in labels:
            query_id = str(label["query_id"])
            paper_id = int(label["paper_id"])
            values = scored.get(query_id, {}).get(paper_id)
            if values is None:
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "paper_id": paper_id,
                    "label_source": source,
                    "features": feature_vector(values),
                    "target": target_gain(label),
                }
            )
    return rows


def add_stacked_scores(
    *,
    method_scores_by_scope: dict[str, dict[str, dict[str, dict[int, float]]]],
    scope_scores: dict[str, dict[str, dict[int, dict[str, float]]]],
    training_rows: list[dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    topics = sorted({str(row["query_id"]) for row in training_rows})
    cv_diagnostics: list[dict[str, Any]] = []
    for heldout in topics:
        train_rows = [row for row in training_rows if row["query_id"] != heldout]
        model = fit_stacked_ridge(train_rows, alpha=alpha)
        for scope_name, scored_by_q in scope_scores.items():
            heldout_rows = scored_by_q.get(heldout, {})
            method_scores_by_scope[scope_name].setdefault("stacked_ridge_loto_cv", {})[heldout] = {
                paper_id: float(model.predict(np.array([feature_vector(values)], dtype=float))[0])
                for paper_id, values in heldout_rows.items()
            }
        cv_diagnostics.append(
            {
                "heldout_query_id": heldout,
                "train_example_count": len(train_rows),
                "test_scope_candidate_counts": {
                    scope_name: len(scored_by_q.get(heldout, {})) for scope_name, scored_by_q in scope_scores.items()
                },
            }
        )

    full_model = fit_stacked_ridge(training_rows, alpha=alpha)
    for scope_name, scored_by_q in scope_scores.items():
        method_scores_by_scope[scope_name].setdefault("stacked_ridge_full_data_apparent", {})
        for query_id, rows in scored_by_q.items():
            method_scores_by_scope[scope_name]["stacked_ridge_full_data_apparent"][query_id] = {
                paper_id: float(full_model.predict(np.array([feature_vector(values)], dtype=float))[0])
                for paper_id, values in rows.items()
            }

    scaler = full_model.named_steps["standardscaler"]
    ridge = full_model.named_steps["ridge"]
    coefficients = [
        {"feature": name, "weight": float(weight)}
        for name, weight in zip(STACKED_FEATURE_NAMES, ridge.coef_.tolist(), strict=True)
    ]
    return {
        "alpha": alpha,
        "training_example_count": len(training_rows),
        "training_examples_by_source": dict(Counter(row["label_source"] for row in training_rows)),
        "folds": cv_diagnostics,
        "full_data_apparent_model": {
            "feature_names": STACKED_FEATURE_NAMES,
            "weights": coefficients,
            "intercept": float(ridge.intercept_),
            "standard_scaler_mean": [float(value) for value in scaler.mean_.tolist()],
            "standard_scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        },
    }


def rankings_from_scores(method_scores: dict[str, dict[str, dict[int, float]]]) -> dict[str, dict[str, list[int]]]:
    by_query: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for method, query_scores in method_scores.items():
        for query_id, scores in query_scores.items():
            by_query[query_id][method] = [
                paper_id
                for paper_id, _ in sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
            ]
    return dict(by_query)


def evaluate_scope_from_scores(
    *,
    scope_name: str,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    k: int,
) -> dict[str, Any]:
    label_lookup = labels_by_query(labels)
    rankings = rankings_from_scores(method_scores)
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in ALL_METHODS}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        candidate_ids = {int(row["paper_id"]) for row in rows}
        query_labels = label_lookup.get(query_id, {})
        judged_in_pool = {paper_id: label for paper_id, label in query_labels.items() if paper_id in candidate_ids}
        for method in ALL_METHODS:
            rows_by_method[method].append(
                evaluate_ranked_ids(
                    method=method,
                    query_id=query_id,
                    all_query_labels=query_labels,
                    judged_in_pool=judged_in_pool,
                    candidate_ids=candidate_ids,
                    retrieved_ids=rankings.get(query_id, {}).get(method, []),
                    k=k,
                )
            )
    return {
        "scope": scope_name,
        "method_averages": {method: aggregate(rows) for method, rows in rows_by_method.items()},
        "per_topic": rows_by_method,
        "per_topic_winners": per_topic_winners(rows_by_method),
    }


def label_is_negative(label: dict[str, Any]) -> bool:
    return clamp_score(label.get("reading_value_score")) <= 0.25 or str(label.get("primary_role")) == "negative"


def label_is_hidden_positive(label: dict[str, Any]) -> bool:
    return clamp_score(label.get("reading_value_score")) >= 0.75 and clamp_score(label.get("topic_match_score")) >= 0.5


def rank_lookup(scope: dict[str, Any], method: str) -> dict[tuple[str, int], int]:
    lookup: dict[tuple[str, int], int] = {}
    for row in scope["per_topic"].get(method, []):
        query_id = str(row["query_id"])
        for rank, paper_id in enumerate(row.get("retrieved_ids_at_k") or [], start=1):
            lookup[(query_id, int(paper_id))] = rank
    return lookup


def hard_negative_blend_diagnostics(
    *,
    v25_labels: list[dict[str, Any]],
    production_scope: dict[str, Any],
    selected_method: str,
) -> dict[str, Any]:
    v25_lookup = labels_by_key(v25_labels)
    old_ranks = rank_lookup(production_scope, "old_v2_2b_learned_hybrid")
    new_ranks = rank_lookup(production_scope, selected_method)
    fixed: list[dict[str, Any]] = []
    harmed: list[dict[str, Any]] = []
    still_promoted_negatives: list[dict[str, Any]] = []
    for key, label in v25_lookup.items():
        old_rank = old_ranks.get(key)
        new_rank = new_ranks.get(key)
        rank_delta = (99 if new_rank is None else new_rank) - (99 if old_rank is None else old_rank)
        base = {
            "query_id": key[0],
            "paper_id": key[1],
            "old_rank_at_10": old_rank,
            "v2_7_rank_at_10": new_rank,
            "rank_delta_positive_is_worse": rank_delta,
            "reading_value_score": label.get("reading_value_score"),
            "topic_match_score": label.get("topic_match_score"),
            "primary_role": label.get("primary_role"),
            "title": label.get("title"),
        }
        if label_is_negative(label):
            if old_rank is not None and (new_rank is None or new_rank > old_rank):
                fixed.append(base)
            if new_rank is not None:
                still_promoted_negatives.append(base)
        if label_is_hidden_positive(label) and old_rank is not None and (new_rank is None or new_rank > old_rank):
            harmed.append(base)
    fixed.sort(key=lambda row: (row["v2_7_rank_at_10"] is not None, row["old_rank_at_10"] or 99))
    harmed.sort(key=lambda row: (row["old_rank_at_10"] or 99, row["v2_7_rank_at_10"] or 99))
    still_promoted_negatives.sort(key=lambda row: row["v2_7_rank_at_10"] or 99)
    return {
        "selected_method": selected_method,
        "top_10_hard_negatives_fixed_vs_old_v2_2b": fixed[:10],
        "top_hidden_positives_harmed_vs_old_v2_2b": harmed[:10],
        "still_promoted_negative_rows": still_promoted_negatives[:20],
        "fixed_negative_count": len(fixed),
        "harmed_hidden_positive_count": len(harmed),
    }


def per_topic_deltas(scope: dict[str, Any], method: str) -> list[dict[str, Any]]:
    old = {row["query_id"]: row for row in scope["per_topic"]["old_v2_2b_learned_hybrid"]}
    new = {row["query_id"]: row for row in scope["per_topic"][method]}
    rows: list[dict[str, Any]] = []
    for query_id in sorted(old):
        old_row = old[query_id]
        new_row = new[query_id]
        rows.append(
            {
                "query_id": query_id,
                "reading_value_delta_vs_old": new_row["reading_value_ndcg"] - old_row["reading_value_ndcg"],
                "topic_match_delta_vs_old": new_row["topic_match_ndcg"] - old_row["topic_match_ndcg"],
                "judged_at_10_delta_vs_old": new_row["judged_retrieved_at_k"] - old_row["judged_retrieved_at_k"],
                "unjudged_at_10_delta_vs_old": new_row["unjudged_retrieved_at_k"] - old_row["unjudged_retrieved_at_k"],
            }
        )
    rows.sort(key=lambda row: row["reading_value_delta_vs_old"])
    return rows


def negative_top10_count(scope: dict[str, Any], method: str, v25_labels: list[dict[str, Any]]) -> int:
    negative_keys = {
        (str(label["query_id"]), int(label["paper_id"]))
        for label in v25_labels
        if label_is_negative(label)
    }
    return sum(1 for key in rank_lookup(scope, method) if key in negative_keys)


def select_best_methods(report_scopes: dict[str, Any], *, v25_labels: list[dict[str, Any]]) -> dict[str, Any]:
    production = report_scopes["true_production_candidate_pool"]
    v25 = report_scopes["v2_5_hard_negative"]
    prod_avg = production["method_averages"]
    v25_avg = v25["method_averages"]
    candidates = SCORE_METHODS
    best_production = max(candidates, key=lambda method: prod_avg[method]["reading_value_ndcg"])
    best_v25 = max(candidates, key=lambda method: v25_avg[method]["reading_value_ndcg"])
    old_prod = prod_avg["old_v2_2b_learned_hybrid"]
    old_v25 = v25_avg["old_v2_2b_learned_hybrid"]
    bm25_prod = prod_avg["bm25"]
    old_negative_top10_count = negative_top10_count(production, "old_v2_2b_learned_hybrid", v25_labels)
    useful_methods = []
    for method in candidates:
        row = prod_avg[method]
        hard = v25_avg[method]
        method_negative_top10_count = negative_top10_count(production, method, v25_labels)
        criteria = {
            "production_reading_within_0_01_of_old": (
                row["reading_value_ndcg"] >= old_prod["reading_value_ndcg"] - 0.01
            ),
            "v2_5_reading_better_than_old": (
                hard["reading_value_ndcg"] > old_v25["reading_value_ndcg"] + 1e-12
            ),
            "production_negative_top10_count_lower_than_old": (
                method_negative_top10_count < old_negative_top10_count
            ),
            "production_unjudged_not_worse_than_old": (
                row["unjudged_retrieved_at_k"] <= old_prod["unjudged_retrieved_at_k"]
            ),
            "production_topic_not_more_than_0_01_below_old_or_bm25": (
                row["topic_match_ndcg"] >= max(old_prod["topic_match_ndcg"], bm25_prod["topic_match_ndcg"]) - 0.01
            ),
        }
        if all(criteria.values()):
            useful_methods.append(
                {
                    "method": method,
                    "criteria": criteria,
                    "production_reading_value_ndcg": row["reading_value_ndcg"],
                    "v2_5_reading_value_ndcg": hard["reading_value_ndcg"],
                    "production_negative_top10_count": method_negative_top10_count,
                }
            )
    useful_methods.sort(key=lambda item: item["production_reading_value_ndcg"], reverse=True)
    selected = useful_methods[0]["method"] if useful_methods else best_production
    return {
        "best_method_by_production_reading_ndcg": best_production,
        "best_method_by_v2_5_hard_negative_reading_ndcg": best_v25,
        "same_method_wins_production_and_v2_5": best_production == best_v25,
        "old_v2_2b_negative_top10_count": old_negative_top10_count,
        "methods_satisfying_all_success_criteria": useful_methods,
        "selected_method_for_diagnostics": selected,
        "safe_to_present_as_best_production_style_learned_ranker": bool(useful_methods),
    }


def build_report(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v21_packet_path: Path,
    v25_packet_path: Path,
    v26_report_path: Path,
    old_model_path: Path,
    v26_model_path: Path,
    top_k: int,
    eval_k: int,
    stacked_alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    v21_labels = load_jsonl(v21_labels_path)
    v25_labels = load_jsonl(v25_labels_path)
    combined_labels = v21_labels + v25_labels
    v21_packet_rows = load_jsonl(v21_packet_path)
    v25_packet_rows = load_jsonl(v25_packet_path)
    v26_report = load_json(v26_report_path)

    v21_rows_by_q = rows_by_query(v21_packet_rows)
    v25_rows_by_q = rows_by_query(v25_packet_rows)
    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=v21_packet_rows, top_k=top_k)

    scope_rows = {
        "v2_1_packet": v21_rows_by_q,
        "v2_5_hard_negative": v25_rows_by_q,
        "true_production_candidate_pool": production_rows_by_q,
    }
    context_rows = {
        "v2_1_packet": v21_rows_by_q,
        "v2_5_hard_negative": production_rows_by_q,
        "true_production_candidate_pool": production_rows_by_q,
    }
    scope_labels = {
        "v2_1_packet": v21_labels,
        "v2_5_hard_negative": v25_labels,
        "true_production_candidate_pool": combined_labels,
    }

    scope_scores = {
        scope_name: score_rows(
            candidate_rows_by_q=rows,
            context_rows_by_q=context_rows[scope_name],
            old_model_path=old_model_path,
            v26_model_path=v26_model_path,
        )
        for scope_name, rows in scope_rows.items()
    }
    method_scores_by_scope = {
        scope_name: fixed_and_penalty_scores(scored) for scope_name, scored in scope_scores.items()
    }

    training_rows = build_stacked_training_rows(
        v21_labels=v21_labels,
        v25_labels=v25_labels,
        scope_scores=scope_scores,
    )
    stacked_diagnostics = add_stacked_scores(
        method_scores_by_scope=method_scores_by_scope,
        scope_scores=scope_scores,
        training_rows=training_rows,
        alpha=stacked_alpha,
    )

    scopes = {
        scope_name: evaluate_scope_from_scores(
            scope_name=scope_name,
            candidate_rows_by_q=scope_rows[scope_name],
            labels=scope_labels[scope_name],
            method_scores=method_scores_by_scope[scope_name],
            k=eval_k,
        )
        for scope_name in scope_rows
    }

    selection = select_best_methods(scopes, v25_labels=v25_labels)
    selected_method = selection["selected_method_for_diagnostics"]
    diagnostics = hard_negative_blend_diagnostics(
        v25_labels=v25_labels,
        production_scope=scopes["true_production_candidate_pool"],
        selected_method=selected_method,
    )
    production_deltas = per_topic_deltas(scopes["true_production_candidate_pool"], selected_method)

    label_diagnostics = combined_label_diagnostics(
        v21_labels=v21_labels,
        v25_labels=v25_labels,
        v21_labels_path=v21_labels_path,
        v25_labels_path=v25_labels_path,
    )
    label_diagnostics["v2_5_labels_hash_unchanged"] = file_sha256(v25_labels_path) == EXPECTED_V25_LABELS_SHA256

    prod_avg = scopes["true_production_candidate_pool"]["method_averages"]
    v25_avg = scopes["v2_5_hard_negative"]["method_averages"]
    old_prod = prod_avg["old_v2_2b_learned_hybrid"]
    selected_prod = prod_avg[selected_method]
    old_v25 = v25_avg["old_v2_2b_learned_hybrid"]
    selected_v25 = v25_avg[selected_method]
    success_criteria = {
        "selected_method": selected_method,
        "production_reading_delta_vs_old_v2_2b": (
            selected_prod["reading_value_ndcg"] - old_prod["reading_value_ndcg"]
        ),
        "production_topic_delta_vs_old_v2_2b": selected_prod["topic_match_ndcg"] - old_prod["topic_match_ndcg"],
        "production_unjudged_delta_vs_old_v2_2b": (
            selected_prod["unjudged_retrieved_at_k"] - old_prod["unjudged_retrieved_at_k"]
        ),
        "v2_5_reading_delta_vs_old_v2_2b": selected_v25["reading_value_ndcg"] - old_v25["reading_value_ndcg"],
        "v2_5_topic_delta_vs_old_v2_2b": selected_v25["topic_match_ndcg"] - old_v25["topic_match_ndcg"],
        "safe_to_present_as_best_production_style_learned_ranker": selection[
            "safe_to_present_as_best_production_style_learned_ranker"
        ],
    }

    report = {
        "schema_version": "v2.7",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_scope": "score_level_blending_and_calibration",
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "runtime_backend_integration": "none: offline score-level evaluation only",
        "candidate_pool_definition": {
            "scopes": ["v2_1_packet", "v2_5_hard_negative", "true_production_candidate_pool"],
            "true_production_candidate_pool_union": [
                "top_50_bm25",
                "top_50_tfidf",
                "top_50_embedding_exact_matrix",
                "top_50_faiss_embedding",
                "top_50_current_hybrid",
            ],
            "deduplicate_by": "paper_id",
        },
        "inputs": {
            "v2_1_labels": str(v21_labels_path),
            "v2_5_labels": str(v25_labels_path),
            "v2_1_packet": str(v21_packet_path),
            "v2_5_packet": str(v25_packet_path),
            "v2_6_report": str(v26_report_path),
            "old_model": str(old_model_path),
            "v2_6_model": str(v26_model_path),
        },
        "reference_v2_6_summary": {
            "safe_to_present_as_current_hybrid_improvement": v26_report.get(
                "safe_to_present_as_current_hybrid_improvement"
            ),
            "safe_to_present_as_production_style_learned_reranking_improvement": v26_report.get(
                "safe_to_present_as_production_style_learned_reranking_improvement"
            ),
        },
        "fixed_blends": FIXED_BLEND_WEIGHTS,
        "hard_negative_penalty_blends": PENALTY_LAMBDAS,
        "stacked_model": {
            "type": "Ridge regression over score-level features",
            "alpha": stacked_alpha,
            "cv": "leave-one-topic-out by query_id",
            "features": STACKED_FEATURE_NAMES,
            "diagnostics": stacked_diagnostics,
            "apparent_metrics_warning": (
                "stacked_ridge_full_data_apparent is trained and evaluated on the same labeled topics; "
                "use stacked_ridge_loto_cv for grouped-CV estimates."
            ),
        },
        "label_diagnostics": label_diagnostics,
        "production_diagnostics": production_diagnostics,
        "scopes": scopes,
        "method_selection": selection,
        "success_criteria": success_criteria,
        "diagnostics": {
            "hard_negative_and_hidden_positive": diagnostics,
            "production_per_topic_deltas_for_selected_method": production_deltas,
            "best_method_by_production_reading_ndcg": selection["best_method_by_production_reading_ndcg"],
            "best_method_by_v2_5_hard_negative_reading_ndcg": selection[
                "best_method_by_v2_5_hard_negative_reading_ndcg"
            ],
            "same_method_wins_both": selection["same_method_wins_production_and_v2_5"],
        },
    }

    model_artifact = {
        "schema_version": "v2.7_score_blend",
        "created_at": report["created_at"],
        "model_name": "v2_7_score_level_blend_candidate",
        "model_type": "offline_score_level_blend_and_stacked_ridge",
        "selected_method": selected_method,
        "safe_to_present_as_best_production_style_learned_ranker": selection[
            "safe_to_present_as_best_production_style_learned_ranker"
        ],
        "fixed_blends": FIXED_BLEND_WEIGHTS,
        "hard_negative_penalty_blends": PENALTY_LAMBDAS,
        "stacked_ridge_full_data_apparent": stacked_diagnostics["full_data_apparent_model"],
        "runtime_note": "Offline evaluation artifact only; no backend runtime integration has been implemented.",
        "training_note": "No labels were created or modified. Stacked Ridge uses only score-level features from existing rankers/retrievers.",
        "label_hashes": {
            "v2_1_labels_sha256": label_diagnostics["v2_1_labels_sha256"],
            "v2_5_labels_sha256": label_diagnostics["v2_5_labels_sha256"],
            "selected_240_sha256": label_diagnostics["selected_240_sha256"],
        },
    }
    return report, model_artifact


def compact_scope(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_averages": scope["method_averages"],
        "per_topic_winners": scope["per_topic_winners"],
    }


def method_table(averages: dict[str, dict[str, float]], methods: list[str]) -> list[str]:
    lines = [
        "| method | reading NDCG@10 | topic NDCG@10 | judged@10 | unjudged@10 | role cov | path cov |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['reading_value_ndcg']:.3f} | {row['topic_match_ndcg']:.3f} | "
            f"{row['judged_retrieved_at_k']:.2f} | {row['unjudged_retrieved_at_k']:.2f} | "
            f"{row['role_coverage_at_k']:.3f} | {row['path_level_coverage_at_k']:.3f} |"
        )
    return lines


def markdown_report(report: dict[str, Any], *, command: str, outputs: list[str]) -> str:
    prod_avg = report["scopes"]["true_production_candidate_pool"]["method_averages"]
    v25_avg = report["scopes"]["v2_5_hard_negative"]["method_averages"]
    selected = report["success_criteria"]["selected_method"]
    best_prod = report["method_selection"]["best_method_by_production_reading_ndcg"]
    best_v25 = report["method_selection"]["best_method_by_v2_5_hard_negative_reading_ndcg"]
    core_methods = [
        "hybrid",
        "embedding",
        "faiss_embedding",
        "old_v2_2b_learned_hybrid",
        "v2_6_learned_hybrid",
        best_prod,
        best_v25,
        "stacked_ridge_loto_cv",
        selected,
    ]
    ordered_methods = []
    for method in core_methods:
        if method not in ordered_methods:
            ordered_methods.append(method)

    lines = [
        "# V2.7 Score-Level Blend Evaluation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Files Created or Modified",
        "",
    ]
    lines.extend([f"- `{path}`" for path in outputs])
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- Labels created by this script: `{report['labels_created_by_this_script']}`",
            f"- Labels modified by this script: `{report['labels_modified_by_this_script']}`",
            f"- Selected 240 modified by this script: `{report['selected_candidates_modified_by_this_script']}`",
            f"- Neural models trained: `{report['neural_models_trained_by_this_script']}`",
            f"- Runtime backend integration: `{report['runtime_backend_integration']}`",
            f"- V2.1 labels hash unchanged: `{report['label_diagnostics']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels hash unchanged: `{report['label_diagnostics']['v2_5_labels_hash_unchanged']}`",
            f"- Selected 240 hash unchanged: `{report['label_diagnostics']['selected_240_hash_unchanged']}`",
            "",
            "## Selection Summary",
            "",
            f"- Best method by expanded production reading NDCG@10: `{best_prod}`",
            f"- Best method by V2.5 hard-negative reading NDCG@10: `{best_v25}`",
            "- Note: `stacked_ridge_full_data_apparent` is an apparent full-data fit; use `stacked_ridge_loto_cv` as the grouped-CV estimate.",
            f"- Same method wins both: `{report['method_selection']['same_method_wins_production_and_v2_5']}`",
            f"- Selected method for diagnostics: `{selected}`",
            f"- Safe to present as best production-style learned ranker: `{report['method_selection']['safe_to_present_as_best_production_style_learned_ranker']}`",
            f"- Old V2.2b V2.5-negative rows in production top 10: `{report['method_selection']['old_v2_2b_negative_top10_count']}`",
            "",
            "## V2.1 Packet",
            "",
        ]
    )
    lines.extend(method_table(report["scopes"]["v2_1_packet"]["method_averages"], ordered_methods))
    lines.extend(
        [
            "",
            "## Expanded Production Pool",
            "",
        ]
    )
    lines.extend(method_table(prod_avg, ordered_methods))
    lines.extend(["", "## V2.5 Hard-Negative Packet", ""])
    lines.extend(method_table(v25_avg, ordered_methods))
    lines.extend(
        [
            "",
            "## Success Criteria",
            "",
        ]
    )
    success = report["success_criteria"]
    lines.extend(
        [
            f"- Production reading delta vs old V2.2b: `{success['production_reading_delta_vs_old_v2_2b']:+.3f}`",
            f"- Production topic delta vs old V2.2b: `{success['production_topic_delta_vs_old_v2_2b']:+.3f}`",
            f"- Production unjudged@10 delta vs old V2.2b: `{success['production_unjudged_delta_vs_old_v2_2b']:+.2f}`",
            f"- V2.5 reading delta vs old V2.2b: `{success['v2_5_reading_delta_vs_old_v2_2b']:+.3f}`",
            f"- V2.5 topic delta vs old V2.2b: `{success['v2_5_topic_delta_vs_old_v2_2b']:+.3f}`",
            "",
            "## Diagnostics",
            "",
            "### Hard Negatives Fixed",
            "",
            "| query_id | paper_id | old rank | V2.7 rank | reading | topic | title |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["diagnostics"]["hard_negative_and_hidden_positive"]["top_10_hard_negatives_fixed_vs_old_v2_2b"]:
        lines.append(
            f"| `{row['query_id']}` | {row['paper_id']} | {row['old_rank_at_10']} | "
            f"{row['v2_7_rank_at_10']} | {row['reading_value_score']} | {row['topic_match_score']} | "
            f"{row['title']} |"
        )
    lines.extend(
        [
            "",
            "### Hidden Positives Harmed",
            "",
            "| query_id | paper_id | old rank | V2.7 rank | reading | topic | title |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["diagnostics"]["hard_negative_and_hidden_positive"]["top_hidden_positives_harmed_vs_old_v2_2b"]:
        lines.append(
            f"| `{row['query_id']}` | {row['paper_id']} | {row['old_rank_at_10']} | "
            f"{row['v2_7_rank_at_10']} | {row['reading_value_score']} | {row['topic_match_score']} | "
            f"{row['title']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if report["method_selection"]["safe_to_present_as_best_production_style_learned_ranker"]:
        lines.append(
            "A V2.7 blend satisfies the configured success criteria: it stays within 0.01 of old V2.2b on expanded production reading NDCG, improves V2.5 hard-negative reading NDCG, does not increase unjudged@10, and preserves topic NDCG within tolerance."
        )
    else:
        lines.append(
            "No V2.7 blend satisfies all configured success criteria. Treat the result as a calibration/ablation finding, not as a replacement production learned ranker."
        )
    lines.append(
        "The stacked full-data metrics are apparent metrics only; grouped leave-one-topic-out metrics are the unbiased estimate for the score-level stacked model."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET))
    parser.add_argument("--v25-packet", default=str(DEFAULT_V25_PACKET))
    parser.add_argument("--v26-report", default=str(DEFAULT_V26_REPORT))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--stacked-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v21_labels_path = resolve_repo_path(args.v21_labels)
    v25_labels_path = resolve_repo_path(args.v25_labels)
    v21_packet_path = resolve_repo_path(args.v21_packet)
    v25_packet_path = resolve_repo_path(args.v25_packet)
    v26_report_path = resolve_repo_path(args.v26_report)
    old_model_path = resolve_repo_path(args.old_model)
    v26_model_path = resolve_repo_path(args.v26_model)
    model_out = resolve_repo_path(args.model_out)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    report, model_artifact = build_report(
        v21_labels_path=v21_labels_path,
        v25_labels_path=v25_labels_path,
        v21_packet_path=v21_packet_path,
        v25_packet_path=v25_packet_path,
        v26_report_path=v26_report_path,
        old_model_path=old_model_path,
        v26_model_path=v26_model_path,
        top_k=args.top_k,
        eval_k=args.eval_k,
        stacked_alpha=args.stacked_alpha,
    )
    command = (
        f".\\backend\\.venv\\Scripts\\python.exe scripts\\evaluate_v2_7_score_blends.py "
        f"--v21-labels {args.v21_labels} --v25-labels {args.v25_labels} "
        f"--v21-packet {args.v21_packet} --v25-packet {args.v25_packet} "
        f"--v26-report {args.v26_report} --old-model {args.old_model} "
        f"--v26-model {args.v26_model} --model-out {args.model_out} "
        f"--json-out {args.json_out} --md-out {args.md_out} "
        f"--top-k {args.top_k} --eval-k {args.eval_k} --stacked-alpha {args.stacked_alpha}"
    )
    report["command"] = command
    write_json(model_out, model_artifact)
    write_json(json_out, report)
    write_text(
        md_out,
        markdown_report(
            report,
            command=command,
            outputs=[
                "scripts/evaluate_v2_7_score_blends.py",
                args.model_out,
                args.json_out,
                args.md_out,
            ],
        ),
    )
    selected = report["success_criteria"]["selected_method"]
    print("V2.7 score-level blend evaluation complete")
    print(f"Selected method: {selected}")
    print(f"Best production reading method: {report['method_selection']['best_method_by_production_reading_ndcg']}")
    print(f"Best V2.5 hard-negative method: {report['method_selection']['best_method_by_v2_5_hard_negative_reading_ndcg']}")
    print(
        "Safe as best production-style learned ranker: "
        f"{report['method_selection']['safe_to_present_as_best_production_style_learned_ranker']}"
    )
    print(f"Report JSON: {json_out}")
    print(f"Report Markdown: {md_out}")
    print(f"Model artifact: {model_out}")


if __name__ == "__main__":
    main()
