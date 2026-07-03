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

from app.services.v2_2_learned_ranker import (  # noqa: E402
    build_feature_context,
    build_feature_vector,
    build_model_payload,
    feature_names,
    load_v2_2_ridge_ranker,
    v2_2_ridge_ranker_from_payload,
)
from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    BASELINE_METHODS,
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    aggregate,
    build_candidate_rows,
    build_documents,
    build_retrieval_runs,
    clamp_score,
    duplicate_penalty_at_k,
    feature_availability,
    file_sha256,
    labels_by_query,
    load_db_assets,
    load_json,
    load_jsonl,
    load_vector_assets,
    ndcg_at_k,
    packet_by_query,
    path_level_coverage_at_k,
    resolve_repo_path,
    role_coverage_at_k,
    score_map,
    target_gain,
    write_json,
    write_text,
)


DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V21_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_V25_PACKET = REPO_ROOT / "data" / "eval" / "v2_5_hard_negative_labeling_packet.jsonl"
DEFAULT_V24_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v2_4_true_production_candidate_pool_report.json"
DEFAULT_OLD_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_MODEL_OUT = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_6_production_aware_learned_hybrid_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_6_production_aware_learned_hybrid_report.md"
EXPECTED_V21_LABELS_SHA256 = "0ECC4A0F638E77E76D44913281EC6431ADDF1873474C3B80A48531B9529359F7"
METHODS = BASELINE_METHODS + ["old_v2_2b_learned_hybrid", "v2_6_learned_hybrid"]
AUDIENCE_SCORE_FIELDS = {
    "beginner": "beginner_fit_score",
    "intermediate": "intermediate_fit_score",
    "advanced": "advanced_fit_score",
    "expert": "expert_fit_score",
}
INTENT_SCORE_FIELDS = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
PATH_ROLES = ["background", "foundational", "core_methods", "recent_frontier"]
DUPLICATE_PENALTIES = {"none": 0.0, "uncertain": 0.25, "near_duplicate": 0.5, "exact_duplicate": 1.0}


def labels_by_key(labels: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["query_id"]), int(row["paper_id"])): row for row in labels}


def rows_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    return dict(grouped)


def rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["query_id"]), int(row["paper_id"])): row for row in rows}


def flatten_rows(rows_by_q: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [row for query_rows in rows_by_q.values() for row in query_rows]


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, *, alpha: float) -> Any:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=17))
    model.fit(x_train, y_train)
    return model


def build_training_examples(
    *,
    v21_labels: list[dict[str, Any]],
    v25_labels: list[dict[str, Any]],
    v21_packet_rows: list[dict[str, Any]],
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v21_lookup = labels_by_key(v21_labels)
    v25_lookup = labels_by_key(v25_labels)
    v21_context = build_feature_context(v21_packet_rows)
    production_context = build_feature_context(flatten_rows(production_rows_by_q))
    production_lookup = rows_by_key(flatten_rows(production_rows_by_q))
    examples: list[dict[str, Any]] = []
    missing_v25_rows: list[dict[str, Any]] = []

    for row in v21_packet_rows:
        key = (str(row["query_id"]), int(row["paper_id"]))
        label = v21_lookup.get(key)
        if label is None:
            continue
        examples.append(
            {
                "query_id": key[0],
                "paper_id": key[1],
                "label_source": "v2_1",
                "features": build_feature_vector(row, names=names, context=v21_context),
                "target": target_gain(label),
            }
        )
    for label in v25_labels:
        key = (str(label["query_id"]), int(label["paper_id"]))
        row = production_lookup.get(key)
        if row is None:
            missing_v25_rows.append({"query_id": key[0], "paper_id": key[1]})
            continue
        examples.append(
            {
                "query_id": key[0],
                "paper_id": key[1],
                "label_source": "v2_5",
                "features": build_feature_vector(row, names=names, context=production_context),
                "target": target_gain(label),
            }
        )
    diagnostics = {
        "training_example_count": len(examples),
        "training_examples_by_source": dict(Counter(example["label_source"] for example in examples)),
        "missing_v2_5_rows_from_production_pool": missing_v25_rows,
        "v2_5_feature_context": "full V2.4-style production candidate pool",
        "v2_1_feature_context": "original V2.1 15-row packet universe",
    }
    return examples, diagnostics


def model_payload_from_pipeline(
    *,
    model: Any,
    names: list[str],
    alpha: float,
    training_example_count: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    return build_model_payload(
        feature_names=names,
        weights=[float(value) for value in ridge.coef_.tolist()],
        intercept=float(ridge.intercept_),
        standard_scaler_mean=[float(value) for value in scaler.mean_.tolist()],
        standard_scaler_scale=[float(value) for value in scaler.scale_.tolist()],
        target=target_payload(),
        hyperparameters={"alpha": alpha, "fit_intercept": True, "random_state": 17},
        training_example_count=training_example_count,
        version="v2_6_production_aware",
        metadata=metadata,
    )


def target_payload() -> dict[str, Any]:
    return {
        "name": "v2_1_beginner_path_gain",
        "formula": (
            "clamp01(0.40*reading_value_score + 0.25*topic_match_score + "
            "0.15*beginner_fit_score + 0.05*intermediate_fit_score + "
            "0.05*intent_path_score + 0.05*intent_application_or_benchmark_score - "
            "0.05*duplicate_penalty)"
        ),
    }


def baseline_rankings(candidate_rows_by_q: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[int]]]:
    ranked: dict[str, dict[str, list[int]]] = {}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        ranked[query_id] = {}
        for method in BASELINE_METHODS:
            ranked[query_id][method] = [
                int(row["paper_id"])
                for row in sorted(
                    rows,
                    key=lambda row: (
                        float((row.get("retrieval_scores_by_method") or {}).get(method, 0.0)),
                        -int((row.get("retrieval_ranks_by_method") or {}).get(method, 999999)),
                        -int(row["paper_id"]),
                    ),
                    reverse=True,
                )
            ]
    return ranked


def add_model_rankings(
    *,
    rankings: dict[str, dict[str, list[int]]],
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    context_rows_by_q: dict[str, list[dict[str, Any]]],
    ranker: Any,
    method_name: str,
) -> dict[str, dict[int, float]]:
    context = build_feature_context(flatten_rows(context_rows_by_q))
    scores_by_query: dict[str, dict[int, float]] = {}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        scores = {
            int(row["paper_id"]): float(ranker.score_packet_row(row, context))
            for row in rows
        }
        scores_by_query[query_id] = scores
        rankings.setdefault(query_id, {})[method_name] = [
            paper_id
            for paper_id, _ in sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)
        ]
    return scores_by_query


def evaluate_scope(
    *,
    scope_name: str,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    old_ranker: Any,
    new_ranker: Any,
    model_context_rows_by_q: dict[str, list[dict[str, Any]]] | None,
    k: int,
) -> dict[str, Any]:
    context_rows_by_q = model_context_rows_by_q or candidate_rows_by_q
    label_lookup = labels_by_query(labels)
    rankings = baseline_rankings(candidate_rows_by_q)
    old_scores = add_model_rankings(
        rankings=rankings,
        candidate_rows_by_q=candidate_rows_by_q,
        context_rows_by_q=context_rows_by_q,
        ranker=old_ranker,
        method_name="old_v2_2b_learned_hybrid",
    )
    new_scores = add_model_rankings(
        rankings=rankings,
        candidate_rows_by_q=candidate_rows_by_q,
        context_rows_by_q=context_rows_by_q,
        ranker=new_ranker,
        method_name="v2_6_learned_hybrid",
    )
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for query_id, by_method in sorted(rankings.items()):
        candidate_ids = {int(row["paper_id"]) for row in candidate_rows_by_q[query_id]}
        query_labels = label_lookup.get(query_id, {})
        judged_in_pool = {paper_id: label for paper_id, label in query_labels.items() if paper_id in candidate_ids}
        for method in METHODS:
            rows_by_method[method].append(
                evaluate_ranked_ids(
                    method=method,
                    query_id=query_id,
                    all_query_labels=query_labels,
                    judged_in_pool=judged_in_pool,
                    candidate_ids=candidate_ids,
                    retrieved_ids=by_method.get(method, []),
                    k=k,
                )
            )
    method_averages = {method: aggregate(rows) for method, rows in rows_by_method.items()}
    return {
        "scope": scope_name,
        "method_averages": method_averages,
        "per_topic": rows_by_method,
        "model_scores_by_query": {
            "old_v2_2b_learned_hybrid": old_scores,
            "v2_6_learned_hybrid": new_scores,
        },
        "per_topic_winners": per_topic_winners(rows_by_method),
    }


def evaluate_ranked_ids(
    *,
    method: str,
    query_id: str,
    all_query_labels: dict[int, dict[str, Any]],
    judged_in_pool: dict[int, dict[str, Any]],
    candidate_ids: set[int],
    retrieved_ids: list[int],
    k: int,
) -> dict[str, Any]:
    score_maps = {
        "topic_match_ndcg": score_map(all_query_labels, "topic_match_score"),
        "reading_value_ndcg": score_map(all_query_labels, "reading_value_score"),
    }
    for audience, field_name in AUDIENCE_SCORE_FIELDS.items():
        score_maps[f"{audience}_ndcg"] = score_map(all_query_labels, field_name)
    for intent in INTENT_SCORE_FIELDS:
        score_maps[f"intent_{intent}_ndcg"] = {
            paper_id: clamp_score((label.get("intent_scores") or {}).get(intent))
            for paper_id, label in all_query_labels.items()
        }
    metrics = {name: ndcg_at_k(retrieved_ids, relevance, k) for name, relevance in score_maps.items()}
    judged_top_k = sum(1 for paper_id in retrieved_ids[:k] if paper_id in all_query_labels)
    return {
        "query_id": query_id,
        "method": method,
        "retrieved_ids_at_k": retrieved_ids[:k],
        "candidate_pool_size": len(candidate_ids),
        "judged_candidates_in_pool": len(judged_in_pool),
        "packet_candidate_coverage": len(judged_in_pool) / max(len(all_query_labels), 1),
        "judged_retrieved_at_k": judged_top_k,
        "unjudged_retrieved_at_k": max(0, min(k, len(retrieved_ids)) - judged_top_k),
        "can_rank_all_candidates": len(retrieved_ids) == len(candidate_ids),
        **metrics,
        "role_coverage_at_k": role_coverage_at_k(retrieved_ids, all_query_labels, k)["coverage"],
        "path_level_coverage_at_k": path_level_coverage_at_k(retrieved_ids, all_query_labels, k)["coverage"],
        "duplicate_penalty_at_k": duplicate_penalty_at_k(retrieved_ids, all_query_labels, k),
    }


def per_topic_winners(rows_by_method: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in rows_by_method.items():
        for row in rows:
            by_query[str(row["query_id"])][method] = row
    winners: list[dict[str, Any]] = []
    for query_id, rows in sorted(by_query.items()):
        topic = max(rows.items(), key=lambda item: float(item[1]["topic_match_ndcg"]))
        reading = max(rows.items(), key=lambda item: float(item[1]["reading_value_ndcg"]))
        winners.append(
            {
                "query_id": query_id,
                "topic_match_winner": topic[0],
                "topic_match_ndcg": topic[1]["topic_match_ndcg"],
                "reading_value_winner": reading[0],
                "reading_value_ndcg": reading[1]["reading_value_ndcg"],
            }
        )
    return winners


def production_candidate_rows(
    *,
    packet_rows: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    packet_lookup = packet_by_query(packet_rows)
    db_assets = load_db_assets()
    vector_assets = load_vector_assets(db_assets["settings"], db_assets["paper_by_id"])
    documents = build_documents(db_assets["papers"])
    runs = build_retrieval_runs(
        packet_lookup=packet_lookup,
        paper_by_id=db_assets["paper_by_id"],
        documents=documents,
        vector_assets=vector_assets,
        top_k=top_k,
    )
    rows = build_candidate_rows(
        runs=runs,
        paper_by_id=db_assets["paper_by_id"],
        identifiers_by_id=db_assets["identifiers_by_id"],
        sources_by_id=db_assets["sources_by_id"],
        source_urls_by_id=db_assets["source_urls_by_id"],
    )
    diagnostics = {
        "db_diagnostics": db_assets["db_diagnostics"],
        "production_asset_diagnostics": vector_assets["diagnostics"],
    }
    return rows, diagnostics


def combined_label_diagnostics(
    *,
    v21_labels: list[dict[str, Any]],
    v25_labels: list[dict[str, Any]],
    v21_labels_path: Path,
    v25_labels_path: Path,
) -> dict[str, Any]:
    combined = v21_labels + v25_labels
    keys = [(str(row["query_id"]), int(row["paper_id"])) for row in combined]
    duplicate_keys = [
        {"query_id": query_id, "paper_id": paper_id}
        for (query_id, paper_id), count in Counter(keys).items()
        if count > 1
    ]
    topics = sorted({query_id for query_id, _ in keys})
    return {
        "v2_1_label_count": len(v21_labels),
        "v2_5_label_count": len(v25_labels),
        "combined_label_count": len(combined),
        "no_duplicate_query_paper": not duplicate_keys,
        "duplicate_query_paper_examples": duplicate_keys[:10],
        "topic_count": len(topics),
        "topics": topics,
        "v2_1_labels_sha256": file_sha256(v21_labels_path),
        "v2_1_labels_hash_unchanged": file_sha256(v21_labels_path) == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": file_sha256(v25_labels_path),
        "selected_240_sha256": file_sha256(SELECTED_240),
        "selected_240_hash_unchanged": file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256,
    }


def score_group_average(
    *,
    labels: list[dict[str, Any]],
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    context_rows_by_q: dict[str, list[dict[str, Any]]],
    old_ranker: Any,
    new_ranker: Any,
    predicate: Any,
) -> dict[str, Any]:
    context = build_feature_context(flatten_rows(context_rows_by_q))
    row_lookup = rows_by_key(flatten_rows(candidate_rows_by_q))
    old_scores: list[float] = []
    new_scores: list[float] = []
    keys: list[dict[str, Any]] = []
    for label in labels:
        if not predicate(label):
            continue
        key = (str(label["query_id"]), int(label["paper_id"]))
        row = row_lookup.get(key)
        if row is None:
            continue
        old_scores.append(float(old_ranker.score_packet_row(row, context)))
        new_scores.append(float(new_ranker.score_packet_row(row, context)))
        keys.append({"query_id": key[0], "paper_id": key[1]})
    return {
        "count": len(old_scores),
        "old_v2_2b_mean_predicted_score": sum(old_scores) / len(old_scores) if old_scores else 0.0,
        "v2_6_mean_predicted_score": sum(new_scores) / len(new_scores) if new_scores else 0.0,
        "examples": keys[:10],
    }


def hard_negative_diagnostics(
    *,
    v21_labels: list[dict[str, Any]],
    v25_labels: list[dict[str, Any]],
    v21_rows_by_q: dict[str, list[dict[str, Any]]],
    v25_rows_by_q: dict[str, list[dict[str, Any]]],
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    production_scope: dict[str, Any],
    old_ranker: Any,
    new_ranker: Any,
) -> dict[str, Any]:
    old_prod = {row["query_id"]: row for row in production_scope["per_topic"]["old_v2_2b_learned_hybrid"]}
    new_prod = {row["query_id"]: row for row in production_scope["per_topic"]["v2_6_learned_hybrid"]}
    v25_lookup = labels_by_key(v25_labels)
    v25_keys = set(v25_lookup)
    old_top10 = {
        (query_id, paper_id): rank
        for query_id, row in old_prod.items()
        for rank, paper_id in enumerate(row["retrieved_ids_at_k"], start=1)
        if (query_id, int(paper_id)) in v25_keys
    }
    new_top10 = {
        (query_id, paper_id): rank
        for query_id, row in new_prod.items()
        for rank, paper_id in enumerate(row["retrieved_ids_at_k"], start=1)
        if (query_id, int(paper_id)) in v25_keys
    }
    still_promoted = [
        {
            "query_id": query_id,
            "paper_id": paper_id,
            "new_rank_at_10": new_rank,
            "old_rank_at_10": old_top10.get((query_id, paper_id)),
            "reading_value_score": v25_lookup[(query_id, paper_id)].get("reading_value_score"),
            "topic_match_score": v25_lookup[(query_id, paper_id)].get("topic_match_score"),
            "title": v25_lookup[(query_id, paper_id)].get("title"),
        }
        for (query_id, paper_id), new_rank in sorted(new_top10.items(), key=lambda item: (item[0][0], item[1]))
    ]
    fixes = []
    for key, old_rank in old_top10.items():
        if key in new_top10:
            continue
        label = v25_lookup[key]
        if clamp_score(label.get("reading_value_score")) <= 0.25 or label.get("primary_role") == "negative":
            fixes.append(
                {
                    "query_id": key[0],
                    "paper_id": key[1],
                    "old_rank_at_10": old_rank,
                    "new_rank_at_10": None,
                    "reading_value_score": label.get("reading_value_score"),
                    "topic_match_score": label.get("topic_match_score"),
                    "title": label.get("title"),
                }
            )
    v25_negative = lambda label: label.get("primary_role") == "negative" or clamp_score(label.get("reading_value_score")) <= 0.25
    v25_hidden_positive = lambda label: clamp_score(label.get("reading_value_score")) >= 0.5 and clamp_score(label.get("topic_match_score")) >= 0.5
    v21_high_reading = lambda label: clamp_score(label.get("reading_value_score")) >= 0.75
    return {
        "predicted_score_groups": {
            "v2_5_negative_rows": score_group_average(
                labels=v25_labels,
                candidate_rows_by_q=v25_rows_by_q,
                context_rows_by_q=production_rows_by_q,
                old_ranker=old_ranker,
                new_ranker=new_ranker,
                predicate=v25_negative,
            ),
            "v2_5_hidden_positive_rows": score_group_average(
                labels=v25_labels,
                candidate_rows_by_q=v25_rows_by_q,
                context_rows_by_q=production_rows_by_q,
                old_ranker=old_ranker,
                new_ranker=new_ranker,
                predicate=v25_hidden_positive,
            ),
            "v2_1_high_reading_value_rows": score_group_average(
                labels=v21_labels,
                candidate_rows_by_q=v21_rows_by_q,
                context_rows_by_q=v21_rows_by_q,
                old_ranker=old_ranker,
                new_ranker=new_ranker,
                predicate=v21_high_reading,
            ),
        },
        "top_v2_5_candidates_still_promoted_by_v2_6": still_promoted[:20],
        "top_cases_where_v2_6_fixes_old_learned_hybrid": fixes[:20],
    }


def coefficient_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    pairs = list(zip(payload["feature_names"], payload["weights"], strict=True))
    top_positive = sorted(pairs, key=lambda item: item[1], reverse=True)[:15]
    top_negative = sorted(pairs, key=lambda item: item[1])[:15]
    watched = {
        name: weight
        for name, weight in pairs
        if name in {"log_citation_count", "source_openalex", "source_arxiv", "source_seed", "vector_best_rank"}
        or name.startswith("embedding_")
        or name.startswith("faiss_embedding_")
    }
    return {
        "top_positive_coefficients": [{"feature": name, "weight": weight} for name, weight in top_positive],
        "top_negative_coefficients": [{"feature": name, "weight": weight} for name, weight in top_negative],
        "watched_source_quality_citation_dense_coefficients": dict(sorted(watched.items())),
        "interpretation": {
            "overweights_citation_count": abs(float(watched.get("log_citation_count", 0.0))) > 0.2,
            "overweights_dense_rank": any(abs(float(weight)) > 0.2 for name, weight in watched.items() if name.startswith(("embedding_", "faiss_embedding_"))),
            "overweights_source_quality": any(abs(float(watched.get(name, 0.0))) > 0.2 for name in ["source_openalex", "source_arxiv", "source_seed"]),
        },
    }


def failure_cases(scope: dict[str, Any]) -> dict[str, Any]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in scope["per_topic"].items():
        for row in rows:
            by_query[str(row["query_id"])][method] = row
    learned_loses_hybrid = []
    learned_loses_bm25_topic = []
    for query_id, rows in sorted(by_query.items()):
        learned = rows["v2_6_learned_hybrid"]
        hybrid = rows["hybrid"]
        bm25 = rows["bm25"]
        if learned["reading_value_ndcg"] < hybrid["reading_value_ndcg"]:
            learned_loses_hybrid.append(
                {
                    "query_id": query_id,
                    "learned": learned["reading_value_ndcg"],
                    "hybrid": hybrid["reading_value_ndcg"],
                    "delta": learned["reading_value_ndcg"] - hybrid["reading_value_ndcg"],
                }
            )
        if learned["topic_match_ndcg"] < bm25["topic_match_ndcg"] - 0.01:
            learned_loses_bm25_topic.append(
                {
                    "query_id": query_id,
                    "learned": learned["topic_match_ndcg"],
                    "bm25": bm25["topic_match_ndcg"],
                    "delta": learned["topic_match_ndcg"] - bm25["topic_match_ndcg"],
                }
            )
    return {
        "v2_6_loses_to_hybrid_on_reading_value_ndcg": learned_loses_hybrid,
        "v2_6_loses_to_bm25_on_topic_match_ndcg_by_more_than_0_01": learned_loses_bm25_topic,
        "topics_where_v2_6_still_fails": sorted(
            {row["query_id"] for row in learned_loses_hybrid + learned_loses_bm25_topic}
        ),
    }


def build_report(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v21_packet_path: Path,
    v25_packet_path: Path,
    v24_report_path: Path,
    old_model_path: Path,
    model_out: Path,
    alpha: float,
    top_k: int,
    eval_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    v21_labels = load_jsonl(v21_labels_path)
    v25_labels = load_jsonl(v25_labels_path)
    combined_labels = v21_labels + v25_labels
    v21_packet_rows = load_jsonl(v21_packet_path)
    v25_packet_rows = load_jsonl(v25_packet_path)
    v24_report = load_json(v24_report_path)

    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=v21_packet_rows, top_k=top_k)
    names = feature_names()
    training_examples, training_diagnostics = build_training_examples(
        v21_labels=v21_labels,
        v25_labels=v25_labels,
        v21_packet_rows=v21_packet_rows,
        production_rows_by_q=production_rows_by_q,
        names=names,
    )
    if training_diagnostics["missing_v2_5_rows_from_production_pool"]:
        raise ValueError("Some V2.5 labels were not found in the reconstructed production candidate pool.")
    x_train = np.array([example["features"] for example in training_examples], dtype=float)
    y_train = np.array([example["target"] for example in training_examples], dtype=float)
    model = fit_ridge(x_train, y_train, alpha=alpha)
    label_diagnostics = combined_label_diagnostics(
        v21_labels=v21_labels,
        v25_labels=v25_labels,
        v21_labels_path=v21_labels_path,
        v25_labels_path=v25_labels_path,
    )
    payload = model_payload_from_pipeline(
        model=model,
        names=names,
        alpha=alpha,
        training_example_count=len(training_examples),
        metadata={
            "label_counts": {
                "v2_1": len(v21_labels),
                "v2_5": len(v25_labels),
                "combined": len(combined_labels),
            },
            "training_feature_contexts": {
                "v2_1": training_diagnostics["v2_1_feature_context"],
                "v2_5": training_diagnostics["v2_5_feature_context"],
            },
        },
    )
    old_ranker = load_v2_2_ridge_ranker(old_model_path)
    new_ranker = v2_2_ridge_ranker_from_payload(payload)

    v21_rows_by_q = rows_by_query(v21_packet_rows)
    v25_rows_by_q = rows_by_query(v25_packet_rows)
    v21_scope = evaluate_scope(
        scope_name="v2_1_packet_evaluation",
        candidate_rows_by_q=v21_rows_by_q,
        labels=v21_labels,
        old_ranker=old_ranker,
        new_ranker=new_ranker,
        model_context_rows_by_q=v21_rows_by_q,
        k=eval_k,
    )
    v25_scope = evaluate_scope(
        scope_name="v2_5_hard_negative_evaluation",
        candidate_rows_by_q=v25_rows_by_q,
        labels=v25_labels,
        old_ranker=old_ranker,
        new_ranker=new_ranker,
        model_context_rows_by_q=production_rows_by_q,
        k=eval_k,
    )
    production_scope = evaluate_scope(
        scope_name="true_production_candidate_pool_expanded_judgments",
        candidate_rows_by_q=production_rows_by_q,
        labels=combined_labels,
        old_ranker=old_ranker,
        new_ranker=new_ranker,
        model_context_rows_by_q=production_rows_by_q,
        k=eval_k,
    )

    prod_avg = production_scope["method_averages"]
    v25_avg = v25_scope["method_averages"]
    v24_old = v24_report["method_averages"]["learned_hybrid"]
    success = {
        "v2_6_vs_hybrid_reading_value_delta": (
            prod_avg["v2_6_learned_hybrid"]["reading_value_ndcg"] - prod_avg["hybrid"]["reading_value_ndcg"]
        ),
        "v2_6_vs_old_v2_2b_expanded_production_reading_value_delta": (
            prod_avg["v2_6_learned_hybrid"]["reading_value_ndcg"]
            - prod_avg["old_v2_2b_learned_hybrid"]["reading_value_ndcg"]
        ),
        "v2_6_vs_bm25_topic_match_delta": (
            prod_avg["v2_6_learned_hybrid"]["topic_match_ndcg"] - prod_avg["bm25"]["topic_match_ndcg"]
        ),
        "v2_6_unjudged_at_10": prod_avg["v2_6_learned_hybrid"]["unjudged_retrieved_at_k"],
        "v2_4_old_learned_unjudged_at_10": v24_old["unjudged_retrieved_at_k"],
        "old_expanded_learned_unjudged_at_10": prod_avg["old_v2_2b_learned_hybrid"]["unjudged_retrieved_at_k"],
        "v2_5_v2_6_vs_old_reading_value_delta": (
            v25_avg["v2_6_learned_hybrid"]["reading_value_ndcg"]
            - v25_avg["old_v2_2b_learned_hybrid"]["reading_value_ndcg"]
        ),
    }
    success |= {
        "beats_current_hybrid_on_production_reading_value_ndcg": success["v2_6_vs_hybrid_reading_value_delta"] > 0,
        "beats_old_v2_2b_on_expanded_production_reading_value_ndcg": (
            success["v2_6_vs_old_v2_2b_expanded_production_reading_value_delta"] > 0
        ),
        "does_not_lose_more_than_0_01_to_bm25_topic_match": success["v2_6_vs_bm25_topic_match_delta"] >= -0.01,
        "reduces_unjudged_at_10_vs_v2_4_old_learned": success["v2_6_unjudged_at_10"] < success["v2_4_old_learned_unjudged_at_10"],
        "improves_v2_5_hard_negative_reading_vs_old": success["v2_5_v2_6_vs_old_reading_value_delta"] > 0,
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2.6",
        "evaluation_scope": "production_aware_lightweight_learned_hybrid_retraining",
        "command": None,
        "model_type": "Ridge regression with StandardScaler",
        "model_name": "v2_6_ridge_production_aware_learned_hybrid",
        "hyperparameters": {"alpha": alpha, "fit_intercept": True, "random_state": 17},
        "target": target_payload(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "models_trained_by_this_script": ["Ridge regression with StandardScaler"],
        "label_diagnostics": label_diagnostics,
        "training_diagnostics": training_diagnostics,
        "production_diagnostics": production_diagnostics,
        "candidate_pool_definition": {
            "corpus": "backend Postgres papers table",
            "union": [
                f"top_{top_k}_bm25",
                f"top_{top_k}_tfidf",
                f"top_{top_k}_embedding_exact_matrix",
                f"top_{top_k}_faiss_embedding",
                f"top_{top_k}_current_hybrid",
            ],
            "deduplicate_by": "paper_id",
        },
        "scopes": {
            "v2_1_packet": without_scores(v21_scope),
            "v2_5_hard_negative": without_scores(v25_scope),
            "true_production_candidate_pool": without_scores(production_scope),
        },
        "hard_negative_diagnostics": hard_negative_diagnostics(
            v21_labels=v21_labels,
            v25_labels=v25_labels,
            v21_rows_by_q=v21_rows_by_q,
            v25_rows_by_q=v25_rows_by_q,
            production_rows_by_q=production_rows_by_q,
            production_scope=production_scope,
            old_ranker=old_ranker,
            new_ranker=new_ranker,
        ),
        "coefficient_diagnostics": coefficient_diagnostics(payload),
        "feature_availability": feature_availability(production_rows_by_q),
        "failure_cases": {
            "true_production_candidate_pool": failure_cases(production_scope),
            "v2_5_hard_negative": failure_cases(v25_scope),
        },
        "success_criteria": success,
        "safe_to_present_as_production_style_learned_reranking_improvement": bool(
            success["beats_current_hybrid_on_production_reading_value_ndcg"]
            and success["does_not_lose_more_than_0_01_to_bm25_topic_match"]
            and success["reduces_unjudged_at_10_vs_v2_4_old_learned"]
            and success["improves_v2_5_hard_negative_reading_vs_old"]
            and success["beats_old_v2_2b_on_expanded_production_reading_value_ndcg"]
        ),
        "safe_to_present_as_current_hybrid_improvement": bool(
            success["beats_current_hybrid_on_production_reading_value_ndcg"]
            and success["does_not_lose_more_than_0_01_to_bm25_topic_match"]
        ),
        "runtime_backend_integration": (
            "scaffold_only: V2.2/V2.6 feature scorer artifact is reusable, but production HybridRetriever "
            "still needs full shared-candidate-pool feature-context integration before runtime use."
        ),
        "reproducibility_checks": {
            "old_v2_2b_model_path": str(old_model_path),
            "new_model_path": str(model_out),
            "v2_1_labels_sha256": label_diagnostics["v2_1_labels_sha256"],
            "v2_1_labels_hash_unchanged": label_diagnostics["v2_1_labels_hash_unchanged"],
            "v2_5_labels_sha256": label_diagnostics["v2_5_labels_sha256"],
            "selected_240_sha256": label_diagnostics["selected_240_sha256"],
            "selected_240_hash_unchanged": label_diagnostics["selected_240_hash_unchanged"],
        },
    }
    return report, payload


def without_scores(scope: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in scope.items() if key != "model_scores_by_query"}


def markdown_report(report: dict[str, Any], *, command: str, outputs: list[str]) -> str:
    lines = [
        "# V2.6 Production-Aware Learned Hybrid",
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
    for output in outputs:
        lines.append(f"- `{output}`")
    labels = report["label_diagnostics"]
    lines.extend(
        [
            "",
            "## Labels",
            "",
            f"- V2.1 labels: `{labels['v2_1_label_count']}`",
            f"- V2.5 labels: `{labels['v2_5_label_count']}`",
            f"- Combined labels: `{labels['combined_label_count']}`",
            f"- Duplicate `(query_id, paper_id)`: `{not labels['no_duplicate_query_paper']}`",
            f"- Topics represented: `{labels['topic_count']}`",
            f"- V2.1 labels hash unchanged: `{labels['v2_1_labels_hash_unchanged']}`",
            f"- Selected 240 hash unchanged: `{labels['selected_240_hash_unchanged']}`",
            f"- V2.5 labels SHA256: `{labels['v2_5_labels_sha256']}`",
            "",
            "## Model",
            "",
            f"- Type: `{report['model_type']}`",
            f"- Name: `{report['model_name']}`",
            f"- Hyperparameters: `{json.dumps(report['hyperparameters'], sort_keys=True)}`",
            f"- Target formula: `{report['target']['formula']}`",
            "",
            "## V2.1 Packet Results",
            "",
        ]
    )
    lines.extend(method_table(report["scopes"]["v2_1_packet"]["method_averages"]))
    lines.extend(["", "## V2.5 Hard-Negative Results", ""])
    lines.extend(method_table(report["scopes"]["v2_5_hard_negative"]["method_averages"]))
    lines.extend(["", "## True Production Candidate-Pool Results", ""])
    lines.extend(method_table(report["scopes"]["true_production_candidate_pool"]["method_averages"]))
    success = report["success_criteria"]
    lines.extend(
        [
            "",
            "## Success Criteria",
            "",
            f"- V2.6 vs hybrid reading-value NDCG@10: `{success['v2_6_vs_hybrid_reading_value_delta']:+.3f}`",
            f"- V2.6 vs old V2.2b expanded-production reading-value NDCG@10: `{success['v2_6_vs_old_v2_2b_expanded_production_reading_value_delta']:+.3f}`",
            f"- V2.6 vs BM25 topic-match NDCG@10: `{success['v2_6_vs_bm25_topic_match_delta']:+.3f}`",
            f"- V2.6 unjudged@10: `{success['v2_6_unjudged_at_10']:.2f}`",
            f"- V2.4 old learned unjudged@10: `{success['v2_4_old_learned_unjudged_at_10']:.2f}`",
            f"- V2.5 hard-negative reading delta vs old: `{success['v2_5_v2_6_vs_old_reading_value_delta']:+.3f}`",
            f"- Safe to present as production-style learned reranking improvement: `{report['safe_to_present_as_production_style_learned_reranking_improvement']}`",
            f"- Safe to present as current-hybrid improvement: `{report['safe_to_present_as_current_hybrid_improvement']}`",
            "",
            "## Hard-Negative Diagnostics",
            "",
        ]
    )
    groups = report["hard_negative_diagnostics"]["predicted_score_groups"]
    for name, group in groups.items():
        lines.append(
            f"- `{name}`: count `{group['count']}`, old mean `{group['old_v2_2b_mean_predicted_score']:.3f}`, "
            f"V2.6 mean `{group['v2_6_mean_predicted_score']:.3f}`"
        )
    lines.extend(
        [
            "",
            "### Top V2.6 Still-Promoted V2.5 Candidates",
            "",
            "| query_id | paper_id | new rank | old rank | topic | reading | title |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["hard_negative_diagnostics"]["top_v2_5_candidates_still_promoted_by_v2_6"][:12]:
        lines.append(
            f"| `{row['query_id']}` | {row['paper_id']} | {row['new_rank_at_10']} | {row['old_rank_at_10']} | "
            f"{float(row['topic_match_score']):.2f} | {float(row['reading_value_score']):.2f} | {row['title']} |"
        )
    lines.extend(
        [
            "",
            "### V2.6 Fixes",
            "",
            "| query_id | paper_id | old rank | reading | topic | title |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["hard_negative_diagnostics"]["top_cases_where_v2_6_fixes_old_learned_hybrid"][:12]:
        lines.append(
            f"| `{row['query_id']}` | {row['paper_id']} | {row['old_rank_at_10']} | "
            f"{float(row['reading_value_score']):.2f} | {float(row['topic_match_score']):.2f} | {row['title']} |"
        )
    coef = report["coefficient_diagnostics"]["interpretation"]
    failures = report["failure_cases"]["true_production_candidate_pool"]
    lines.extend(
        [
            "",
            "## Coefficient Diagnostics",
            "",
            f"- Overweights citation count: `{coef['overweights_citation_count']}`",
            f"- Overweights dense rank: `{coef['overweights_dense_rank']}`",
            f"- Overweights source quality: `{coef['overweights_source_quality']}`",
            "",
            "Top positive coefficients:",
        ]
    )
    for row in report["coefficient_diagnostics"]["top_positive_coefficients"][:8]:
        lines.append(f"- `{row['feature']}`: `{row['weight']:.4f}`")
    lines.append("")
    lines.append("Top negative coefficients:")
    for row in report["coefficient_diagnostics"]["top_negative_coefficients"][:8]:
        lines.append(f"- `{row['feature']}`: `{row['weight']:.4f}`")
    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            f"- Topics where V2.6 still fails: `{failures['topics_where_v2_6_still_fails']}`",
            f"- Runtime backend integration: {report['runtime_backend_integration']}",
            "",
        ]
    )
    return "\n".join(lines)


def method_table(averages: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | topic NDCG@10 | reading NDCG@10 | beginner | intermediate | advanced | expert | path coverage | role coverage | dup penalty | judged@10 | unjudged@10 | judged in pool | coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(averages, key=lambda name: averages[name]["reading_value_ndcg"], reverse=True):
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['topic_match_ndcg']:.3f} | {row['reading_value_ndcg']:.3f} | "
            f"{row['beginner_ndcg']:.3f} | {row['intermediate_ndcg']:.3f} | {row['advanced_ndcg']:.3f} | "
            f"{row['expert_ndcg']:.3f} | {row['path_level_coverage_at_k']:.3f} | {row['role_coverage_at_k']:.3f} | "
            f"{row['duplicate_penalty_at_k']:.3f} | {row['judged_retrieved_at_k']:.2f} | "
            f"{row['unjudged_retrieved_at_k']:.2f} | {row['judged_candidates_in_pool']:.2f} | "
            f"{row['packet_candidate_coverage']:.3f} |"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate V2.6 production-aware learned hybrid.")
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-packet", default=str(DEFAULT_V25_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--v24-report", default=str(DEFAULT_V24_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v21_labels_path = resolve_repo_path(args.v21_labels)
    v25_labels_path = resolve_repo_path(args.v25_labels)
    v21_packet_path = resolve_repo_path(args.v21_packet)
    v25_packet_path = resolve_repo_path(args.v25_packet)
    v24_report_path = resolve_repo_path(args.v24_report)
    old_model_path = resolve_repo_path(args.old_model)
    model_out = resolve_repo_path(args.model_out)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    report, payload = build_report(
        v21_labels_path=v21_labels_path,
        v25_labels_path=v25_labels_path,
        v21_packet_path=v21_packet_path,
        v25_packet_path=v25_packet_path,
        v24_report_path=v24_report_path,
        old_model_path=old_model_path,
        model_out=model_out,
        alpha=args.alpha,
        top_k=args.top_k,
        eval_k=args.eval_k,
    )
    command = (
        f".\\backend\\.venv\\Scripts\\python.exe scripts\\train_v2_6_production_aware_learned_hybrid.py "
        f"--v21-labels {args.v21_labels} --v25-labels {args.v25_labels} "
        f"--v21-packet {args.v21_packet} --v25-packet {args.v25_packet} "
        f"--v24-report {args.v24_report} --old-model {args.old_model} "
        f"--model-out {args.model_out} --json-out {args.json_out} --md-out {args.md_out} "
        f"--alpha {args.alpha} --top-k {args.top_k} --eval-k {args.eval_k}"
    )
    report["command"] = command
    write_json(model_out, payload)
    write_json(json_out, report)
    write_text(
        md_out,
        markdown_report(
            report,
            command=command,
            outputs=[
                "scripts/train_v2_6_production_aware_learned_hybrid.py",
                args.model_out,
                args.json_out,
                args.md_out,
            ],
        ),
    )
    success = report["success_criteria"]
    print("V2.6 production-aware learned_hybrid training complete")
    print(f"Combined labels: {report['label_diagnostics']['combined_label_count']}")
    print(f"Model: {model_out}")
    print(f"Report JSON: {json_out}")
    print(f"Report Markdown: {md_out}")
    print(f"V2.6 vs hybrid reading delta: {success['v2_6_vs_hybrid_reading_value_delta']:+.3f}")
    print(f"V2.6 vs BM25 topic delta: {success['v2_6_vs_bm25_topic_match_delta']:+.3f}")
    print(f"Safe production-style improvement: {report['safe_to_present_as_production_style_learned_reranking_improvement']}")


if __name__ == "__main__":
    main()
