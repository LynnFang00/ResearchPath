import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from statistics import mean, pvariance
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.v2_2_learned_ranker import (  # noqa: E402
    MODEL_SCHEMA_VERSION,
    build_feature_context,
    build_feature_vector,
    build_model_payload,
    feature_names,
    v2_2_ridge_ranker_from_payload,
)


DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_OLD_V22 = REPO_ROOT / "data" / "eval" / "results" / "v2_2_learned_hybrid_cv_report.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_2b_fair_learned_hybrid_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_2b_fair_learned_hybrid_report.md"
DEFAULT_MODEL_OUT = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
SELECTED_240 = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"
EXPECTED_SELECTED_240_SHA256 = "6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F"

RANK_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
ALL_METHODS = ["learned_hybrid"] + RANK_METHODS
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
ROLE_VALUES_FOR_COVERAGE = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
DUPLICATE_PENALTIES = {"none": 0.0, "uncertain": 0.25, "near_duplicate": 0.5, "exact_duplicate": 1.0}
LEAKAGE_FIELD_NAMES = [
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
    "label_confidence",
    "notes",
    "selection_reasons",
    "likely_coverage",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def labels_by_query(labels: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for label in labels:
        grouped[str(label["query_id"])][int(label["paper_id"])] = label
    return dict(grouped)


def labels_by_key(labels: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(label["query_id"]), int(label["paper_id"])): label for label in labels}


def packet_by_query(packet_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packet_rows:
        grouped[str(row["query_id"])].append(row)
    return dict(grouped)


def target_gain(label: dict[str, Any]) -> float:
    intents = label.get("intent_scores") or {}
    intent_path_score = max(clamp_score(intents.get(name)) for name in PATH_ROLES)
    intent_application_or_benchmark_score = max(
        clamp_score(intents.get("evaluation_benchmark")),
        clamp_score(intents.get("application")),
    )
    duplicate_penalty = DUPLICATE_PENALTIES.get(str(label.get("duplicate_status", "none")), 0.0)
    value = (
        0.40 * clamp_score(label.get("reading_value_score"))
        + 0.25 * clamp_score(label.get("topic_match_score"))
        + 0.15 * clamp_score(label.get("beginner_fit_score"))
        + 0.05 * clamp_score(label.get("intermediate_fit_score"))
        + 0.05 * intent_path_score
        + 0.05 * intent_application_or_benchmark_score
        - 0.05 * duplicate_penalty
    )
    return clamp_score(value)


def build_examples(packet_rows: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = feature_names()
    context = build_feature_context(packet_rows)
    label_lookup = labels_by_key(labels)
    examples: list[dict[str, Any]] = []
    for row in packet_rows:
        key = (str(row["query_id"]), int(row["paper_id"]))
        label = label_lookup.get(key)
        if label is None:
            continue
        examples.append(
            {
                "query_id": key[0],
                "paper_id": key[1],
                "features": build_feature_vector(row, names=names, context=context),
                "target": target_gain(label),
            }
        )
    return examples


def grouped_folds(query_ids: list[str], fold_count: int) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        validation = [query_id for index, query_id in enumerate(query_ids) if index % fold_count == fold_index]
        training = [query_id for query_id in query_ids if query_id not in set(validation)]
        folds.append(
            {
                "fold_index": fold_index + 1,
                "train_query_ids": training,
                "validation_query_ids": validation,
            }
        )
    return folds


def fit_model(examples: list[dict[str, Any]], *, alpha: float) -> Any:
    x_train = np.array([example["features"] for example in examples], dtype=float)
    y_train = np.array([example["target"] for example in examples], dtype=float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=17))
    model.fit(x_train, y_train)
    return model


def evaluate_scope(
    *,
    scope_name: str,
    candidate_scope: str,
    packet_rows: list[dict[str, Any]],
    labels_by_q: dict[str, dict[int, dict[str, Any]]],
    examples: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    alpha: float,
    k: int,
) -> dict[str, Any]:
    rows_by_query = packet_by_query(packet_rows)
    examples_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        examples_by_query[example["query_id"]].append(example)

    method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in ALL_METHODS}
    folds_payload: list[dict[str, Any]] = []
    for fold in folds:
        train_examples = [
            example
            for query_id in fold["train_query_ids"]
            for example in examples_by_query.get(query_id, [])
        ]
        model = fit_model(train_examples, alpha=alpha)
        fold_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in ALL_METHODS}
        for query_id in fold["validation_query_ids"]:
            scoped_rows = candidate_rows_for_scope(rows_by_query[query_id], candidate_scope)
            learned_ids = learned_ranked_ids(scoped_rows, examples_by_query[query_id], model)
            learned_row = evaluate_ranked_ids(
                method="learned_hybrid",
                query_id=query_id,
                query_labels=labels_by_q[query_id],
                retrieved_ids=learned_ids,
                k=k,
                candidate_count=len(scoped_rows),
            )
            fold_rows["learned_hybrid"].append(learned_row)
            method_rows["learned_hybrid"].append(learned_row)
            for method in RANK_METHODS:
                baseline_ids = baseline_ranked_ids(scoped_rows, method)
                baseline_row = evaluate_ranked_ids(
                    method=method,
                    query_id=query_id,
                    query_labels=labels_by_q[query_id],
                    retrieved_ids=baseline_ids,
                    k=k,
                    candidate_count=len(scoped_rows),
                )
                fold_rows[method].append(baseline_row)
                method_rows[method].append(baseline_row)
        folds_payload.append(
            {
                **fold,
                "train_label_count": len(train_examples),
                "validation_candidate_count": sum(
                    len(candidate_rows_for_scope(rows_by_query[query_id], candidate_scope))
                    for query_id in fold["validation_query_ids"]
                ),
                "method_averages": {method: aggregate(rows) for method, rows in fold_rows.items()},
            }
        )

    method_averages = {method: aggregate(rows) for method, rows in method_rows.items()}
    return {
        "scope_name": scope_name,
        "candidate_scope": candidate_scope,
        "folds": folds_payload,
        "method_averages": method_averages,
        "method_variances_across_topics": {method: aggregate_variance(rows) for method, rows in method_rows.items()},
        "per_topic": method_rows,
        "per_topic_winners": build_per_topic_winners(method_rows),
        "failure_cases": build_failure_cases(method_rows),
        "success_criteria": success_criteria(method_averages),
    }


def candidate_rows_for_scope(rows: list[dict[str, Any]], candidate_scope: str) -> list[dict[str, Any]]:
    if candidate_scope == "same_packet_15":
        return list(rows)
    if candidate_scope == "current_hybrid_ranked_candidates":
        return [row for row in rows if "hybrid" in (row.get("retrieval_ranks_by_method") or {})]
    raise ValueError(f"Unknown candidate scope: {candidate_scope}")


def learned_ranked_ids(scoped_rows: list[dict[str, Any]], query_examples: list[dict[str, Any]], model: Any) -> list[int]:
    features_by_id = {int(example["paper_id"]): example["features"] for example in query_examples}
    scored: list[tuple[float, int]] = []
    for row in scoped_rows:
        paper_id = int(row["paper_id"])
        features = features_by_id[paper_id]
        score = float(np.clip(model.predict(np.array([features], dtype=float))[0], 0.0, 1.0))
        scored.append((score, paper_id))
    return [paper_id for _, paper_id in sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)]


def baseline_ranked_ids(scoped_rows: list[dict[str, Any]], method: str) -> list[int]:
    scored: list[tuple[int, float, float, int, int]] = []
    for original_index, row in enumerate(scoped_rows):
        paper_id = int(row["paper_id"])
        ranks = row.get("retrieval_ranks_by_method") or {}
        scores = row.get("retrieval_scores_by_method") or {}
        rank = int(ranks.get(method, 999999))
        has_method = method in ranks or method in scores
        score = float(scores.get(method, 0.0)) if isinstance(scores.get(method), (int, float)) else 0.0
        reciprocal_rank = 1.0 / rank if has_method and rank > 0 else 0.0
        scored.append((1 if has_method else 0, score, reciprocal_rank, -original_index, paper_id))
    return [paper_id for _, _, _, _, paper_id in sorted(scored, reverse=True)]


def evaluate_ranked_ids(
    *,
    method: str,
    query_id: str,
    query_labels: dict[int, dict[str, Any]],
    retrieved_ids: list[int],
    k: int,
    candidate_count: int,
) -> dict[str, Any]:
    score_maps = {
        "topic_match_ndcg": score_map(query_labels, "topic_match_score"),
        "reading_value_ndcg": score_map(query_labels, "reading_value_score"),
    }
    for audience, field_name in AUDIENCE_SCORE_FIELDS.items():
        score_maps[f"{audience}_ndcg"] = score_map(query_labels, field_name)
    for intent in INTENT_SCORE_FIELDS:
        score_maps[f"intent_{intent}_ndcg"] = intent_score_map(query_labels, intent)
    metrics = {
        metric_name: ndcg_at_k(retrieved_ids, relevance_by_id, k)
        for metric_name, relevance_by_id in score_maps.items()
    }
    role_coverage = role_coverage_at_k(retrieved_ids, query_labels, k)
    path_coverage = path_level_coverage_at_k(retrieved_ids, query_labels, k)
    return {
        "query_id": query_id,
        "method": method,
        "candidate_count": candidate_count,
        "retrieved_ids_at_k": retrieved_ids[:k],
        "judged_retrieved_at_k": sum(1 for paper_id in retrieved_ids[:k] if paper_id in query_labels),
        **metrics,
        "role_coverage_at_k": role_coverage["coverage"],
        "roles_covered_at_k": role_coverage["covered_roles"],
        "roles_available": role_coverage["available_roles"],
        "duplicate_penalty_at_k": duplicate_penalty_at_k(retrieved_ids, query_labels, k),
        "path_level_coverage_at_k": path_coverage["coverage"],
        "path_roles_covered_at_k": path_coverage["covered_roles"],
    }


def score_map(labels: dict[int, dict[str, Any]], field_name: str) -> dict[int, float]:
    return {paper_id: clamp_score(label.get(field_name)) for paper_id, label in labels.items()}


def intent_score_map(labels: dict[int, dict[str, Any]], intent: str) -> dict[int, float]:
    return {
        paper_id: clamp_score((label.get("intent_scores") or {}).get(intent))
        for paper_id, label in labels.items()
    }


def ndcg_at_k(retrieved_ids: list[int], relevance_by_id: dict[int, float], k: int) -> float:
    gains = [relevance_by_id.get(paper_id, 0.0) for paper_id in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / ideal if ideal > 0 else 0.0


def role_coverage_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> dict[str, Any]:
    available = topic_roles(labels.values(), threshold=0.3)
    covered = topic_roles((labels[paper_id] for paper_id in retrieved_ids[:k] if paper_id in labels), threshold=0.3)
    denominator = len(available) if available else len(ROLE_VALUES_FOR_COVERAGE)
    return {
        "coverage": len(covered & available) / denominator if denominator else 0.0,
        "covered_roles": sorted(covered),
        "available_roles": sorted(available),
    }


def topic_roles(rows: Any, threshold: float) -> set[str]:
    roles: set[str] = set()
    for row in rows:
        if clamp_score(row.get("reading_value_score")) < threshold:
            continue
        primary_role = row.get("primary_role")
        if primary_role in ROLE_VALUES_FOR_COVERAGE:
            roles.add(str(primary_role))
        secondary_roles = row.get("secondary_roles")
        if isinstance(secondary_roles, list):
            roles.update(str(role) for role in secondary_roles if role in ROLE_VALUES_FOR_COVERAGE)
    return roles


def path_level_coverage_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> dict[str, Any]:
    covered: set[str] = set()
    for paper_id in retrieved_ids[:k]:
        label = labels.get(paper_id)
        if not label or clamp_score(label.get("reading_value_score")) < 0.5:
            continue
        roles = [label.get("primary_role")]
        secondary_roles = label.get("secondary_roles")
        if isinstance(secondary_roles, list):
            roles.extend(secondary_roles)
        covered.update(str(role) for role in roles if role in PATH_ROLES)
    return {
        "coverage": len(covered) / len(PATH_ROLES),
        "covered_roles": sorted(covered),
    }


def duplicate_penalty_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> float:
    if k <= 0:
        return 0.0
    penalties = [
        DUPLICATE_PENALTIES.get(str(labels.get(paper_id, {}).get("duplicate_status", "none")), 0.0)
        for paper_id in retrieved_ids[:k]
    ]
    return sum(penalties) / k


def clamp_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def metric_fields() -> list[str]:
    return [
        "topic_match_ndcg",
        "reading_value_ndcg",
        "beginner_ndcg",
        "intermediate_ndcg",
        "advanced_ndcg",
        "expert_ndcg",
        "role_coverage_at_k",
        "duplicate_penalty_at_k",
        "path_level_coverage_at_k",
        "judged_retrieved_at_k",
        "candidate_count",
    ] + [f"intent_{intent}_ndcg" for intent in INTENT_SCORE_FIELDS]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {field: mean_metric(rows, field) for field in metric_fields()}


def aggregate_variance(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {field: variance_metric(rows, field) for field in metric_fields()}


def mean_metric(rows: list[dict[str, Any]], field_name: str) -> float:
    return sum(float(row[field_name]) for row in rows) / len(rows) if rows else 0.0


def variance_metric(rows: list[dict[str, Any]], field_name: str) -> float:
    values = [float(row[field_name]) for row in rows]
    return pvariance(values) if len(values) > 1 else 0.0


def build_per_topic_winners(method_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows_by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in method_rows.items():
        for row in rows:
            rows_by_query[str(row["query_id"])][method] = row
    winners: list[dict[str, Any]] = []
    for query_id, by_method in sorted(rows_by_query.items()):
        topic_method, topic_score = best_method(by_method, "topic_match_ndcg")
        reading_method, reading_score = best_method(by_method, "reading_value_ndcg")
        winners.append(
            {
                "query_id": query_id,
                "topic_match_winner": topic_method,
                "topic_match_ndcg": topic_score,
                "reading_value_winner": reading_method,
                "reading_value_ndcg": reading_score,
            }
        )
    return winners


def best_method(rows_by_method: dict[str, dict[str, Any]], metric: str) -> tuple[str, float]:
    candidates = [(method, float(row.get(metric, 0.0))) for method, row in rows_by_method.items()]
    return max(candidates, key=lambda item: item[1])


def build_failure_cases(method_rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in method_rows.items():
        for row in rows:
            by_query[str(row["query_id"])][method] = row
    reading_failures: list[dict[str, Any]] = []
    topic_failures: list[dict[str, Any]] = []
    for query_id, rows in sorted(by_query.items()):
        learned = rows.get("learned_hybrid")
        hybrid = rows.get("hybrid")
        bm25 = rows.get("bm25")
        if learned and hybrid:
            delta = float(learned["reading_value_ndcg"]) - float(hybrid["reading_value_ndcg"])
            if delta < 0:
                reading_failures.append(
                    {
                        "query_id": query_id,
                        "learned_reading_value_ndcg": float(learned["reading_value_ndcg"]),
                        "hybrid_reading_value_ndcg": float(hybrid["reading_value_ndcg"]),
                        "delta": delta,
                    }
                )
        if learned and bm25:
            delta = float(learned["topic_match_ndcg"]) - float(bm25["topic_match_ndcg"])
            if delta < 0:
                topic_failures.append(
                    {
                        "query_id": query_id,
                        "learned_topic_match_ndcg": float(learned["topic_match_ndcg"]),
                        "bm25_topic_match_ndcg": float(bm25["topic_match_ndcg"]),
                        "delta": delta,
                    }
                )
    return {
        "worse_than_hybrid_on_reading_value_ndcg": sorted(reading_failures, key=lambda row: row["delta"]),
        "worse_than_bm25_on_topic_match_ndcg": sorted(topic_failures, key=lambda row: row["delta"]),
    }


def success_criteria(method_averages: dict[str, dict[str, float]]) -> dict[str, Any]:
    learned = method_averages["learned_hybrid"]
    hybrid = method_averages["hybrid"]
    bm25 = method_averages["bm25"]
    reading_delta = learned["reading_value_ndcg"] - hybrid["reading_value_ndcg"]
    topic_delta = learned["topic_match_ndcg"] - bm25["topic_match_ndcg"]
    return {
        "beat_hybrid_on_reading_value_ndcg": reading_delta > 0,
        "reading_value_delta_vs_hybrid": reading_delta,
        "avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg": topic_delta >= -0.01,
        "topic_match_delta_vs_bm25": topic_delta,
        "overall_pass": reading_delta > 0 and topic_delta >= -0.01,
    }


def target_payload() -> dict[str, str]:
    return {
        "name": "v2_1_beginner_path_gain",
        "formula": (
            "clamp01(0.40*reading_value_score + 0.25*topic_match_score + "
            "0.15*beginner_fit_score + 0.05*intermediate_fit_score + "
            "0.05*intent_path_score + 0.05*intent_application_or_benchmark_score - "
            "0.05*duplicate_penalty)"
        ),
    }


def train_final_model_payload(examples: list[dict[str, Any]], *, alpha: float) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = fit_model(examples, alpha=alpha)
    x_all = np.array([example["features"] for example in examples], dtype=float)
    pipeline_predictions = np.clip(model.predict(x_all), 0.0, 1.0)
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    payload = build_model_payload(
        feature_names=feature_names(),
        weights=[float(value) for value in ridge.coef_],
        intercept=float(ridge.intercept_),
        standard_scaler_mean=[float(value) for value in scaler.mean_],
        standard_scaler_scale=[float(value) for value in scaler.scale_],
        target=target_payload(),
        hyperparameters={"alpha": alpha, "fit_intercept": True, "random_state": 17},
        training_example_count=len(examples),
        version=f"v2_2b_lightweight_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        metadata={"trained_on": "all_240_v2_1_labels_after_cv"},
    )
    artifact_model = v2_2_ridge_ranker_from_payload(payload)
    artifact_predictions = np.array(
        [artifact_model.score_feature_values(example["features"]) for example in examples],
        dtype=float,
    )
    return payload, pipeline_predictions, artifact_predictions


def reproducibility_checks(
    *,
    labels_path: Path,
    selected_path: Path,
    folds: list[dict[str, Any]],
    names: list[str],
    pipeline_predictions: np.ndarray,
    artifact_predictions: np.ndarray,
) -> dict[str, Any]:
    feature_text = "\n".join(names)
    train_validation_overlaps = [
        sorted(set(fold["train_query_ids"]) & set(fold["validation_query_ids"]))
        for fold in folds
    ]
    return {
        "grouped_cv_by_query_id": all(not overlap for overlap in train_validation_overlaps),
        "train_validation_overlaps": train_validation_overlaps,
        "label_fields_used_as_features": [name for name in LEAKAGE_FIELD_NAMES if name in feature_text],
        "selection_reasons_used": "selection_reasons" in feature_text,
        "likely_coverage_used": "likely_coverage" in feature_text,
        "labels_sha256": file_sha256(labels_path),
        "selected_240_sha256": file_sha256(selected_path),
        "selected_240_matches_expected_hash": file_sha256(selected_path) == EXPECTED_SELECTED_240_SHA256,
        "artifact_prediction_max_abs_diff": float(np.max(np.abs(pipeline_predictions - artifact_predictions))),
        "artifact_prediction_mean_abs_diff": float(np.mean(np.abs(pipeline_predictions - artifact_predictions))),
        "artifact_can_regenerate_all_240_predictions": bool(
            np.max(np.abs(pipeline_predictions - artifact_predictions)) < 1e-12
        ),
    }


def build_report(
    *,
    labels_path: Path,
    packet_path: Path,
    selected_path: Path,
    old_v22_path: Path,
    model_out: Path,
    alpha: float,
    fold_count: int,
    k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = load_jsonl(labels_path)
    packet_rows = load_jsonl(packet_path)
    old_v22 = load_json(old_v22_path)
    label_lookup = labels_by_query(labels)
    query_ids = sorted(label_lookup)
    folds = grouped_folds(query_ids, fold_count)
    examples = build_examples(packet_rows, labels)

    same_packet = evaluate_scope(
        scope_name="same_packet_reranking",
        candidate_scope="same_packet_15",
        packet_rows=packet_rows,
        labels_by_q=label_lookup,
        examples=examples,
        folds=folds,
        alpha=alpha,
        k=k,
    )
    restricted = evaluate_scope(
        scope_name="current_hybrid_candidate_restricted",
        candidate_scope="current_hybrid_ranked_candidates",
        packet_rows=packet_rows,
        labels_by_q=label_lookup,
        examples=examples,
        folds=folds,
        alpha=alpha,
        k=k,
    )
    model_payload, pipeline_predictions, artifact_predictions = train_final_model_payload(examples, alpha=alpha)
    checks = reproducibility_checks(
        labels_path=labels_path,
        selected_path=selected_path,
        folds=folds,
        names=feature_names(),
        pipeline_predictions=pipeline_predictions,
        artifact_predictions=artifact_predictions,
    )
    model_payload["reproducibility_checks"] = {
        "artifact_prediction_max_abs_diff": checks["artifact_prediction_max_abs_diff"],
        "artifact_prediction_mean_abs_diff": checks["artifact_prediction_mean_abs_diff"],
    }

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2.2b",
        "evaluation_scope": "corrected_fair_packet_and_restricted_candidate_evaluation",
        "old_v2_2_exploratory_report": {
            "path": str(old_v22_path),
            "classification": "exploratory_upper_bound_not_apples_to_apples",
            "reason": (
                "The original V2.2 learned_hybrid ranked all 15 packet candidates while baselines used "
                "pre-materialized method ranks with lower judged@10."
            ),
            "reported_learned_reading_value_ndcg": old_v22["method_averages"]["learned_hybrid"]["reading_value_ndcg"],
            "reported_learned_topic_match_ndcg": old_v22["method_averages"]["learned_hybrid"]["topic_match_ndcg"],
        },
        "model": {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "model_output": str(model_out),
            "model_type": "Ridge regression with StandardScaler",
            "hyperparameters": {"alpha": alpha, "fit_intercept": True, "random_state": 17},
            "target": target_payload(),
            "feature_names": feature_names(),
            "training_example_count": len(examples),
        },
        "candidate_score_policy": {
            "same_packet_reranking": (
                "Every method ranks the same 15 labeled packet candidates. Missing method scores/ranks are "
                "materialized as absent and sorted below candidates with a materialized score/rank."
            ),
            "restricted_candidate_evaluation": (
                "Every method ranks only candidates with a current hybrid materialized rank for the topic."
            ),
        },
        "same_packet_reranking": same_packet,
        "restricted_candidate_evaluation": restricted,
        "reproducibility_checks": checks,
        "safe_to_present_as_validated_method_improvement": bool(
            same_packet["success_criteria"]["overall_pass"]
            and restricted["success_criteria"]["overall_pass"]
            and checks["artifact_can_regenerate_all_240_predictions"]
            and checks["selected_240_matches_expected_hash"]
            and not checks["label_fields_used_as_features"]
            and not checks["selection_reasons_used"]
            and not checks["likely_coverage_used"]
        ),
        "inputs": {
            "labels": str(labels_path),
            "packet": str(packet_path),
            "old_v22": str(old_v22_path),
            "selected_240": str(selected_path),
        },
    }
    return report, model_payload


def markdown_report(report: dict[str, Any], *, command: str, outputs: list[str]) -> str:
    lines = [
        "# V2.2b Fair learned_hybrid Evaluation",
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
    lines.extend(
        [
            "",
            "## Original V2.2 Classification",
            "",
            "The original V2.2 result is preserved and classified as `exploratory_upper_bound_not_apples_to_apples`.",
            "",
            report["old_v2_2_exploratory_report"]["reason"],
            "",
            "## Same-Packet Reranking Comparison",
            "",
        ]
    )
    lines.extend(method_table(report["same_packet_reranking"]["method_averages"]))
    lines.extend(
        [
            "",
            "## Restricted Current-Hybrid Candidate Comparison",
            "",
        ]
    )
    lines.extend(method_table(report["restricted_candidate_evaluation"]["method_averages"]))

    for key, title in [
        ("same_packet_reranking", "Same-Packet Success Criteria"),
        ("restricted_candidate_evaluation", "Restricted-Candidate Success Criteria"),
    ]:
        criteria = report[key]["success_criteria"]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"- learned_hybrid vs hybrid reading-value NDCG@10: `{criteria['reading_value_delta_vs_hybrid']:+.3f}`",
                f"- learned_hybrid vs BM25 topic-match NDCG@10: `{criteria['topic_match_delta_vs_bm25']:+.3f}`",
                f"- Beat hybrid on reading-value NDCG@10: `{criteria['beat_hybrid_on_reading_value_ndcg']}`",
                f"- Avoid losing more than 0.01 to BM25 on topic-match NDCG@10: `{criteria['avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg']}`",
                f"- Overall pass: `{criteria['overall_pass']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Same-Packet Failure Cases",
            "",
            "### Worse Than Hybrid on Reading Value",
            "",
            "| query_id | learned | hybrid | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["same_packet_reranking"]["failure_cases"]["worse_than_hybrid_on_reading_value_ndcg"]:
        lines.append(
            f"| `{row['query_id']}` | {row['learned_reading_value_ndcg']:.3f} | "
            f"{row['hybrid_reading_value_ndcg']:.3f} | {row['delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "### Worse Than BM25 on Topic Match",
            "",
            "| query_id | learned | bm25 | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["same_packet_reranking"]["failure_cases"]["worse_than_bm25_on_topic_match_ndcg"]:
        lines.append(
            f"| `{row['query_id']}` | {row['learned_topic_match_ndcg']:.3f} | "
            f"{row['bm25_topic_match_ndcg']:.3f} | {row['delta']:+.3f} |"
        )

    checks = report["reproducibility_checks"]
    lines.extend(
        [
            "",
            "## Reproducibility Checks",
            "",
            f"- Grouped CV by query_id: `{checks['grouped_cv_by_query_id']}`",
            f"- Label fields used as features: `{checks['label_fields_used_as_features']}`",
            f"- `selection_reasons` used: `{checks['selection_reasons_used']}`",
            f"- `likely_coverage` used: `{checks['likely_coverage_used']}`",
            f"- Labels SHA256: `{checks['labels_sha256']}`",
            f"- Selected 240 SHA256: `{checks['selected_240_sha256']}`",
            f"- Selected 240 hash matches expected: `{checks['selected_240_matches_expected_hash']}`",
            f"- Artifact regenerates all-240 predictions: `{checks['artifact_can_regenerate_all_240_predictions']}`",
            f"- Max artifact prediction diff: `{checks['artifact_prediction_max_abs_diff']:.12f}`",
            "",
            "## Conclusion",
            "",
            f"Safe to present as validated method improvement: `{report['safe_to_present_as_validated_method_improvement']}`",
            "",
        ]
    )
    return "\n".join(lines)


def method_table(averages: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | topic NDCG@10 | reading NDCG@10 | beginner | intermediate | advanced | expert | role coverage | path coverage | dup penalty | judged@10 | candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(averages, key=lambda name: averages[name]["reading_value_ndcg"], reverse=True):
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['topic_match_ndcg']:.3f} | {row['reading_value_ndcg']:.3f} | "
            f"{row['beginner_ndcg']:.3f} | {row['intermediate_ndcg']:.3f} | "
            f"{row['advanced_ndcg']:.3f} | {row['expert_ndcg']:.3f} | "
            f"{row['role_coverage_at_k']:.3f} | {row['path_level_coverage_at_k']:.3f} | "
            f"{row['duplicate_penalty_at_k']:.3f} | {row['judged_retrieved_at_k']:.2f} | "
            f"{row['candidate_count']:.2f} |"
        )
    return lines


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected V2.2b fair learned_hybrid evaluation.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--old-v22", default=str(DEFAULT_OLD_V22))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = resolve_repo_path(args.labels)
    packet_path = resolve_repo_path(args.packet)
    old_v22_path = resolve_repo_path(args.old_v22)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    model_out = resolve_repo_path(args.model_out)
    report, model_payload = build_report(
        labels_path=labels_path,
        packet_path=packet_path,
        selected_path=SELECTED_240,
        old_v22_path=old_v22_path,
        model_out=model_out,
        alpha=args.alpha,
        fold_count=args.folds,
        k=args.k,
    )
    command = (
        f".\\backend\\.venv\\Scripts\\python.exe scripts\\evaluate_v2_2b_fair_learned_hybrid.py "
        f"--labels {args.labels} --packet {args.packet} --old-v22 {args.old_v22} "
        f"--json-out {args.json_out} --md-out {args.md_out} --model-out {args.model_out} "
        f"--alpha {args.alpha} --folds {args.folds} --k {args.k}"
    )
    report["command"] = command
    write_json(model_out, model_payload)
    write_json(json_out, report)
    write_text(
        md_out,
        markdown_report(
            report,
            command=command,
            outputs=[
                "backend/app/services/v2_2_learned_ranker.py",
                "backend/app/services/learned_ranker.py",
                "scripts/evaluate_v2_2b_fair_learned_hybrid.py",
                args.json_out,
                args.md_out,
                args.model_out,
            ],
        ),
    )

    same = report["same_packet_reranking"]["success_criteria"]
    restricted = report["restricted_candidate_evaluation"]["success_criteria"]
    print("V2.2b fair learned_hybrid evaluation complete")
    print(f"Report JSON: {json_out}")
    print(f"Report Markdown: {md_out}")
    print(f"Model artifact: {model_out}")
    print(f"same-packet reading delta vs hybrid: {same['reading_value_delta_vs_hybrid']:+.3f}")
    print(f"same-packet topic delta vs bm25: {same['topic_match_delta_vs_bm25']:+.3f}")
    print(f"restricted reading delta vs hybrid: {restricted['reading_value_delta_vs_hybrid']:+.3f}")
    print(f"restricted topic delta vs bm25: {restricted['topic_match_delta_vs_bm25']:+.3f}")
    print(f"safe to present: {report['safe_to_present_as_validated_method_improvement']}")


if __name__ == "__main__":
    main()
