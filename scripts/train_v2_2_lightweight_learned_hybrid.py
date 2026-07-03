import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
from statistics import mean, pvariance
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_BASELINE = REPO_ROOT / "data" / "eval" / "results" / "v2_1_benchmark_method_comparison.json"
DEFAULT_MILESTONE = REPO_ROOT / "data" / "eval" / "results" / "v2_1_learned_ranker_milestone.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_2_learned_hybrid_cv_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_2_learned_hybrid_cv_report.md"
DEFAULT_MODEL_OUT = REPO_ROOT / "data" / "processed" / "models" / "v2_2_lightweight_learned_hybrid.json"

RANK_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
BASELINE_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
ALL_METHODS = ["learned_hybrid"] + BASELINE_METHODS
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
CURRENT_YEAR = 2026


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


def labels_by_query(labels: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for label in labels:
        grouped[str(label["query_id"])][int(label["paper_id"])] = label
    return dict(grouped)


def packet_by_query(packet_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packet_rows:
        grouped[str(row["query_id"])].append(row)
    return dict(grouped)


def labels_by_key(labels: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["query_id"]), int(row["paper_id"])): row for row in labels}


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


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def token_overlap_features(query: str, text: str) -> tuple[float, float, float]:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    if not query_terms:
        return 0.0, 0.0, 0.0
    overlap = len(query_terms & text_terms)
    return float(overlap), overlap / len(query_terms), overlap / max(1, len(text_terms))


def phrase_hit(query: str, text: str) -> float:
    query_norm = " ".join(tokenize(query))
    text_norm = " ".join(tokenize(text))
    return 1.0 if query_norm and query_norm in text_norm else 0.0


def evidence_level(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("level") or "")
    return str(value or "")


def fulltext_available(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("fulltext_available"))
    return False


def identifier_value(row: dict[str, Any], key: str) -> Any:
    identifiers = row.get("identifiers") or {}
    if isinstance(identifiers, dict):
        return identifiers.get(key)
    return None


def title_duplicate_sizes(packet_rows: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    titles_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    normalized_by_key: dict[tuple[str, int], str] = {}
    for row in packet_rows:
        query_id = str(row["query_id"])
        paper_id = int(row["paper_id"])
        normalized = normalize_title(str(row.get("title") or ""))
        normalized_by_key[(query_id, paper_id)] = normalized
        if normalized:
            titles_by_query[query_id][normalized] += 1
    return {
        key: titles_by_query[key[0]][normalized] if normalized else 0
        for key, normalized in normalized_by_key.items()
    }


def score_stats_by_query(packet_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in packet_rows:
        query_id = str(row["query_id"])
        scores = row.get("retrieval_scores_by_method") or {}
        for method, value in scores.items():
            if isinstance(value, (int, float)):
                values[query_id][str(method)].append(float(value))

    stats: dict[str, dict[str, dict[str, float]]] = {}
    for query_id, by_method in values.items():
        stats[query_id] = {}
        for method, method_values in by_method.items():
            method_min = min(method_values)
            method_max = max(method_values)
            method_mean = mean(method_values)
            variance = pvariance(method_values) if len(method_values) > 1 else 0.0
            stats[query_id][method] = {
                "min": method_min,
                "max": method_max,
                "mean": method_mean,
                "std": math.sqrt(variance),
            }
    return stats


def feature_names() -> list[str]:
    names: list[str] = []
    for method in RANK_METHODS:
        names.extend(
            [
                f"{method}_rank_present",
                f"{method}_rank",
                f"{method}_reciprocal_rank",
                f"{method}_score_present",
                f"{method}_score_raw",
                f"{method}_score_minmax",
                f"{method}_score_z",
            ]
        )
    names.extend(
        [
            "rank_method_count",
            "score_method_count",
            "best_rank",
            "mean_rank",
            "mean_reciprocal_rank",
            "rank_std",
            "lexical_best_rank",
            "vector_best_rank",
            "lexical_vector_rank_gap",
            "query_title_overlap_count",
            "query_title_overlap_query_ratio",
            "query_title_overlap_title_ratio",
            "query_abstract_overlap_count",
            "query_abstract_overlap_query_ratio",
            "query_abstract_overlap_abstract_ratio",
            "query_phrase_in_title",
            "query_phrase_in_abstract",
            "title_char_length",
            "abstract_char_length",
            "title_token_length",
            "abstract_token_length",
            "abstract_missing",
            "abstract_short",
            "year",
            "paper_age",
            "log_citation_count",
            "venue_present",
            "author_count",
            "source_count",
            "source_arxiv",
            "source_openalex",
            "source_seed",
            "has_arxiv_id",
            "has_doi",
            "has_openalex_id",
            "has_source_url",
            "has_pdf_url",
            "fulltext_available",
            "evidence_title_abstract",
            "evidence_title_abstract_intro_conclusion",
            "duplicate_title_cluster_present",
            "duplicate_title_cluster_size",
            "normalized_title_duplicate",
        ]
    )
    return names


def build_feature_vector(
    row: dict[str, Any],
    *,
    names: list[str],
    score_stats: dict[str, dict[str, dict[str, float]]],
    duplicate_sizes: dict[tuple[str, int], int],
) -> list[float]:
    query_id = str(row["query_id"])
    paper_id = int(row["paper_id"])
    ranks = {str(method): int(rank) for method, rank in (row.get("retrieval_ranks_by_method") or {}).items()}
    scores = {
        str(method): float(score)
        for method, score in (row.get("retrieval_scores_by_method") or {}).items()
        if isinstance(score, (int, float))
    }
    features: dict[str, float] = {}
    observed_ranks: list[float] = []
    reciprocal_ranks: list[float] = []
    for method in RANK_METHODS:
        rank = ranks.get(method)
        score = scores.get(method)
        rank_present = rank is not None
        score_present = score is not None
        if rank_present:
            observed_ranks.append(float(rank))
            reciprocal_ranks.append(1.0 / max(1.0, float(rank)))
        stats = score_stats.get(query_id, {}).get(method, {})
        score_min = stats.get("min", 0.0)
        score_max = stats.get("max", 0.0)
        score_mean = stats.get("mean", 0.0)
        score_std = stats.get("std", 0.0)
        score_raw = float(score) if score_present else 0.0
        features.update(
            {
                f"{method}_rank_present": 1.0 if rank_present else 0.0,
                f"{method}_rank": float(rank) if rank_present else 999.0,
                f"{method}_reciprocal_rank": 1.0 / max(1.0, float(rank)) if rank_present else 0.0,
                f"{method}_score_present": 1.0 if score_present else 0.0,
                f"{method}_score_raw": score_raw,
                f"{method}_score_minmax": ((score_raw - score_min) / (score_max - score_min))
                if score_present and score_max > score_min
                else 0.0,
                f"{method}_score_z": ((score_raw - score_mean) / score_std)
                if score_present and score_std > 0
                else 0.0,
            }
        )

    lexical_ranks = [float(ranks[method]) for method in ["bm25", "tfidf"] if method in ranks]
    vector_ranks = [float(ranks[method]) for method in ["embedding", "faiss_embedding"] if method in ranks]
    lexical_best = min(lexical_ranks) if lexical_ranks else 999.0
    vector_best = min(vector_ranks) if vector_ranks else 999.0

    query = str(row.get("query") or "")
    title = str(row.get("title") or "")
    abstract = str(row.get("abstract") or "")
    title_tokens = tokenize(title)
    abstract_tokens = tokenize(abstract)
    title_overlap = token_overlap_features(query, title)
    abstract_overlap = token_overlap_features(query, abstract)
    year = float(row.get("year")) if isinstance(row.get("year"), (int, float)) else 0.0
    citation_count = float(row.get("citation_count")) if isinstance(row.get("citation_count"), (int, float)) else 0.0
    sources = {str(source).lower() for source in row.get("sources_provenance") or []}
    evidence = evidence_level(row.get("evidence_availability"))
    duplicate_cluster = row.get("duplicate_title_cluster")
    duplicate_cluster_size = (
        len(duplicate_cluster)
        if isinstance(duplicate_cluster, list)
        else int(duplicate_cluster.get("size", 0))
        if isinstance(duplicate_cluster, dict) and isinstance(duplicate_cluster.get("size"), int)
        else 0
    )
    normalized_duplicate_size = duplicate_sizes.get((query_id, paper_id), 0)

    features.update(
        {
            "rank_method_count": float(len(observed_ranks)),
            "score_method_count": float(len(scores)),
            "best_rank": min(observed_ranks) if observed_ranks else 999.0,
            "mean_rank": mean(observed_ranks) if observed_ranks else 999.0,
            "mean_reciprocal_rank": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            "rank_std": float(np.std(observed_ranks)) if observed_ranks else 0.0,
            "lexical_best_rank": lexical_best,
            "vector_best_rank": vector_best,
            "lexical_vector_rank_gap": lexical_best - vector_best,
            "query_title_overlap_count": title_overlap[0],
            "query_title_overlap_query_ratio": title_overlap[1],
            "query_title_overlap_title_ratio": title_overlap[2],
            "query_abstract_overlap_count": abstract_overlap[0],
            "query_abstract_overlap_query_ratio": abstract_overlap[1],
            "query_abstract_overlap_abstract_ratio": abstract_overlap[2],
            "query_phrase_in_title": phrase_hit(query, title),
            "query_phrase_in_abstract": phrase_hit(query, abstract),
            "title_char_length": float(len(title)),
            "abstract_char_length": float(len(abstract)),
            "title_token_length": float(len(title_tokens)),
            "abstract_token_length": float(len(abstract_tokens)),
            "abstract_missing": 1.0 if not abstract.strip() else 0.0,
            "abstract_short": 1.0 if 0 < len(abstract_tokens) < 40 else 0.0,
            "year": year,
            "paper_age": max(0.0, CURRENT_YEAR - year) if year else 0.0,
            "log_citation_count": math.log1p(max(0.0, citation_count)),
            "venue_present": 1.0 if row.get("venue") else 0.0,
            "author_count": float(len(row.get("authors") or [])),
            "source_count": float(len(sources)),
            "source_arxiv": 1.0 if "arxiv" in sources else 0.0,
            "source_openalex": 1.0 if "openalex" in sources else 0.0,
            "source_seed": 1.0 if "seed" in sources else 0.0,
            "has_arxiv_id": 1.0 if identifier_value(row, "arxiv_id") else 0.0,
            "has_doi": 1.0 if identifier_value(row, "doi") else 0.0,
            "has_openalex_id": 1.0 if identifier_value(row, "openalex_id") else 0.0,
            "has_source_url": 1.0 if row.get("source_url") else 0.0,
            "has_pdf_url": 1.0 if row.get("pdf_url") else 0.0,
            "fulltext_available": 1.0 if fulltext_available(row.get("evidence_availability")) else 0.0,
            "evidence_title_abstract": 1.0 if evidence == "title_abstract" else 0.0,
            "evidence_title_abstract_intro_conclusion": 1.0
            if evidence == "title_abstract_intro_conclusion"
            else 0.0,
            "duplicate_title_cluster_present": 1.0 if duplicate_cluster_size > 1 else 0.0,
            "duplicate_title_cluster_size": float(duplicate_cluster_size),
            "normalized_title_duplicate": 1.0 if normalized_duplicate_size > 1 else 0.0,
        }
    )
    return [float(features[name]) for name in names]


def build_examples(
    packet_rows: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    names: list[str],
) -> list[dict[str, Any]]:
    label_lookup = labels_by_key(labels)
    score_stats = score_stats_by_query(packet_rows)
    duplicate_sizes = title_duplicate_sizes(packet_rows)
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
                "features": build_feature_vector(
                    row,
                    names=names,
                    score_stats=score_stats,
                    duplicate_sizes=duplicate_sizes,
                ),
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


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, *, alpha: float) -> Any:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=17))
    model.fit(x_train, y_train)
    return model


def learned_rows_for_fold(
    *,
    fold: dict[str, Any],
    model: Any,
    examples: list[dict[str, Any]],
    labels_by_q: dict[str, dict[int, dict[str, Any]]],
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    examples_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        if example["query_id"] in set(fold["validation_query_ids"]):
            examples_by_query[example["query_id"]].append(example)
    for query_id in fold["validation_query_ids"]:
        query_examples = examples_by_query[query_id]
        if not query_examples:
            continue
        x_val = np.array([example["features"] for example in query_examples], dtype=float)
        predictions = np.clip(model.predict(x_val), 0.0, 1.0)
        ranked = sorted(
            zip(query_examples, predictions, strict=False),
            key=lambda item: (float(item[1]), -int(item[0]["paper_id"])),
            reverse=True,
        )
        retrieved_ids = [int(example["paper_id"]) for example, _ in ranked]
        rows.append(
            evaluate_ranked_ids(
                method="learned_hybrid",
                query_id=query_id,
                query_labels=labels_by_q[query_id],
                retrieved_ids=retrieved_ids,
                k=k,
                learned_scores_by_id={int(example["paper_id"]): float(score) for example, score in ranked},
            )
        )
    return rows


def baseline_rows_for_queries(
    baseline_report: dict[str, Any],
    query_ids: list[str],
    methods: list[str],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    per_query = baseline_report.get("per_query", {})
    for query_id in query_ids:
        for method in methods:
            row = per_query.get(query_id, {}).get(method)
            if row is not None:
                rows_by_method[method].append(row)
    return rows_by_method


def evaluate_ranked_ids(
    *,
    method: str,
    query_id: str,
    query_labels: dict[int, dict[str, Any]],
    retrieved_ids: list[int],
    k: int,
    learned_scores_by_id: dict[int, float] | None = None,
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
    row = {
        "query_id": query_id,
        "method": method,
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
    if learned_scores_by_id is not None:
        row["learned_scores_at_k"] = {
            str(paper_id): round(learned_scores_by_id.get(paper_id, 0.0), 6)
            for paper_id in retrieved_ids[:k]
        }
    return row


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


def mean_metric(rows: list[dict[str, Any]], field_name: str) -> float:
    return sum(float(row[field_name]) for row in rows) / len(rows) if rows else 0.0


def variance_metric(rows: list[dict[str, Any]], field_name: str) -> float:
    values = [float(row[field_name]) for row in rows]
    return pvariance(values) if len(values) > 1 else 0.0


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
    ] + [f"intent_{intent}_ndcg" for intent in INTENT_SCORE_FIELDS]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {field: mean_metric(rows, field) for field in metric_fields()}


def aggregate_variance(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {field: variance_metric(rows, field) for field in metric_fields()}


def train_and_evaluate(
    *,
    labels: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    milestone_report: dict[str, Any],
    alpha: float,
    fold_count: int,
    k: int,
) -> tuple[dict[str, Any], Any]:
    names = feature_names()
    examples = build_examples(packet_rows, labels, names=names)
    label_lookup = labels_by_query(labels)
    query_ids = sorted(label_lookup)
    folds = grouped_folds(query_ids, fold_count)

    learned_rows: list[dict[str, Any]] = []
    fold_payloads: list[dict[str, Any]] = []
    examples_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        examples_by_query[example["query_id"]].append(example)

    for fold in folds:
        train_examples = [
            example
            for query_id in fold["train_query_ids"]
            for example in examples_by_query.get(query_id, [])
        ]
        x_train = np.array([example["features"] for example in train_examples], dtype=float)
        y_train = np.array([example["target"] for example in train_examples], dtype=float)
        model = fit_ridge(x_train, y_train, alpha=alpha)
        fold_learned_rows = learned_rows_for_fold(
            fold=fold,
            model=model,
            examples=examples,
            labels_by_q=label_lookup,
            k=k,
        )
        learned_rows.extend(fold_learned_rows)
        fold_baselines = baseline_rows_for_queries(
            baseline_report,
            fold["validation_query_ids"],
            BASELINE_METHODS,
        )
        fold_payloads.append(
            {
                **fold,
                "train_label_count": len(train_examples),
                "validation_label_count": sum(
                    len(examples_by_query.get(query_id, [])) for query_id in fold["validation_query_ids"]
                ),
                "learned_hybrid": aggregate(fold_learned_rows),
                "baselines": {
                    method: aggregate(rows)
                    for method, rows in fold_baselines.items()
                },
            }
        )

    baseline_rows = baseline_rows_for_queries(baseline_report, query_ids, BASELINE_METHODS)
    method_rows = {"learned_hybrid": learned_rows} | baseline_rows
    method_averages = {method: aggregate(rows) for method, rows in method_rows.items()}
    method_variances = {method: aggregate_variance(rows) for method, rows in method_rows.items()}
    per_topic_winners = build_per_topic_winners(method_rows)
    failure_cases = build_failure_cases(method_rows)

    all_x = np.array([example["features"] for example in examples], dtype=float)
    all_y = np.array([example["target"] for example in examples], dtype=float)
    final_model = fit_ridge(all_x, all_y, alpha=alpha)

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2.2",
        "evaluation_scope": "v2_2_grouped_cv_lightweight_learned_hybrid",
        "k": k,
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "models_trained_by_this_script": True,
        "model_type": "Ridge regression with StandardScaler",
        "hyperparameters": {"alpha": alpha, "fit_intercept": True, "random_state": 17},
        "target": target_payload(milestone_report),
        "feature_names": names,
        "leakage_exclusions": [
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
        ],
        "fold_count": fold_count,
        "folds": fold_payloads,
        "method_averages": method_averages,
        "method_variances_across_topics": method_variances,
        "per_topic": {method: rows for method, rows in method_rows.items()},
        "per_topic_winners": per_topic_winners,
        "failure_cases": failure_cases,
        "satisfies_success_criteria": success_criteria(method_averages),
        "training_example_count": len(examples),
        "topics": query_ids,
    }
    return report, final_model


def target_payload(milestone_report: dict[str, Any]) -> dict[str, Any]:
    target = milestone_report.get("target_score_proposal") or {}
    return {
        "name": target.get("name", "v2_1_beginner_path_gain"),
        "formula": target.get(
            "formula",
            "clamp01(0.40*reading_value_score + 0.25*topic_match_score + 0.15*beginner_fit_score + "
            "0.05*intermediate_fit_score + 0.05*intent_path_score + "
            "0.05*intent_application_or_benchmark_score - 0.05*duplicate_penalty)",
        ),
    }


def build_per_topic_winners(method_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows_by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in method_rows.items():
        for row in rows:
            rows_by_query[str(row["query_id"])][method] = row
    winners: list[dict[str, Any]] = []
    for query_id, by_method in sorted(rows_by_query.items()):
        topic_method, topic_score = best_method(by_method, "topic_match_ndcg")
        reading_method, reading_score = best_method(by_method, "reading_value_ndcg")
        beginner_method, beginner_score = best_method(by_method, "beginner_ndcg")
        winners.append(
            {
                "query_id": query_id,
                "topic_match_winner": topic_method,
                "topic_match_ndcg": topic_score,
                "reading_value_winner": reading_method,
                "reading_value_ndcg": reading_score,
                "beginner_winner": beginner_method,
                "beginner_ndcg": beginner_score,
            }
        )
    return winners


def best_method(rows_by_method: dict[str, dict[str, Any]], metric: str) -> tuple[str, float]:
    candidates = [
        (method, float(row.get(metric, 0.0)))
        for method, row in rows_by_method.items()
    ]
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
    topic_delta_vs_bm25 = learned["topic_match_ndcg"] - bm25["topic_match_ndcg"]
    return {
        "beat_hybrid_on_reading_value_ndcg": reading_delta > 0,
        "reading_value_delta_vs_hybrid": reading_delta,
        "avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg": topic_delta_vs_bm25 >= -0.01,
        "topic_match_delta_vs_bm25": topic_delta_vs_bm25,
        "overall_pass": reading_delta > 0 and topic_delta_vs_bm25 >= -0.01,
    }


def save_model_artifact(path: Path, model: Any, report: dict[str, Any]) -> None:
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    payload = {
        "model_type": "v2_2_ridge_learned_hybrid",
        "version": f"v2_2_lightweight_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "target": report["target"],
        "feature_names": report["feature_names"],
        "standard_scaler_mean": [float(value) for value in scaler.mean_],
        "standard_scaler_scale": [float(value) for value in scaler.scale_],
        "weights": [float(value) for value in ridge.coef_],
        "intercept": float(ridge.intercept_),
        "hyperparameters": report["hyperparameters"],
        "training_example_count": report["training_example_count"],
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_note": (
            "Scores require the same V2.2 feature extraction schema. This artifact is not a neural model "
            "and was trained only on V2.1 manual labels."
        ),
    }
    write_json(path, payload)


def markdown_report(report: dict[str, Any], *, command: str, outputs: list[str]) -> str:
    avg = report["method_averages"]
    criteria = report["satisfies_success_criteria"]
    lines = [
        "# V2.2 Lightweight learned_hybrid CV Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Files Created",
        "",
    ]
    for output in outputs:
        lines.append(f"- `{output}`")

    lines.extend(
        [
            "",
            "## Model",
            "",
            f"- Type: `{report['model_type']}`",
            f"- Hyperparameters: `{json.dumps(report['hyperparameters'], sort_keys=True)}`",
            f"- Training examples: `{report['training_example_count']}`",
            f"- Grouped folds: `{report['fold_count']}`",
            "",
            "## Feature List Actually Used",
            "",
        ]
    )
    for name in report["feature_names"]:
        lines.append(f"- `{name}`")

    lines.extend(
        [
            "",
            "## Grouped CV Results",
            "",
            "| method | topic NDCG@10 | reading NDCG@10 | beginner | intermediate | advanced | expert | role coverage | path coverage | dup penalty | judged@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(avg, key=lambda name: avg[name]["reading_value_ndcg"], reverse=True):
        row = avg[method]
        lines.append(
            f"| `{method}` | {row['topic_match_ndcg']:.3f} | {row['reading_value_ndcg']:.3f} | "
            f"{row['beginner_ndcg']:.3f} | {row['intermediate_ndcg']:.3f} | "
            f"{row['advanced_ndcg']:.3f} | {row['expert_ndcg']:.3f} | "
            f"{row['role_coverage_at_k']:.3f} | {row['path_level_coverage_at_k']:.3f} | "
            f"{row['duplicate_penalty_at_k']:.3f} | {row['judged_retrieved_at_k']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Fold Mean and Variance",
            "",
            "| fold | validation topics | learned reading NDCG@10 | learned topic NDCG@10 |",
            "|---:|---|---:|---:|",
        ]
    )
    for fold in report["folds"]:
        learned = fold["learned_hybrid"]
        topics = ", ".join(f"`{query_id}`" for query_id in fold["validation_query_ids"])
        lines.append(
            f"| {fold['fold_index']} | {topics} | {learned['reading_value_ndcg']:.3f} | "
            f"{learned['topic_match_ndcg']:.3f} |"
        )
    learned_variance = report["method_variances_across_topics"]["learned_hybrid"]
    lines.append("")
    lines.append(
        f"Learned topic-level variance: reading `{learned_variance['reading_value_ndcg']:.6f}`, "
        f"topic `{learned_variance['topic_match_ndcg']:.6f}`."
    )

    lines.extend(
        [
            "",
            "## Intent-Specific NDCG@10",
            "",
            "| method | background | foundational | core_methods | recent_frontier | evaluation_benchmark | application |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(avg):
        row = avg[method]
        values = " | ".join(f"{row[f'intent_{intent}_ndcg']:.3f}" for intent in INTENT_SCORE_FIELDS)
        lines.append(f"| `{method}` | {values} |")

    lines.extend(
        [
            "",
            "## Success Criteria",
            "",
            f"- learned_hybrid vs current hybrid on reading-value NDCG@10: `{criteria['reading_value_delta_vs_hybrid']:+.3f}`",
            f"- learned_hybrid vs BM25 on topic-match NDCG@10: `{criteria['topic_match_delta_vs_bm25']:+.3f}`",
            f"- Beat hybrid on reading-value NDCG@10: `{criteria['beat_hybrid_on_reading_value_ndcg']}`",
            f"- Avoid losing more than 0.01 to BM25 on topic-match NDCG@10: `{criteria['avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg']}`",
            f"- Overall pass: `{criteria['overall_pass']}`",
            "",
            "## Per-Topic Winners",
            "",
            "| query_id | topic winner | topic NDCG@10 | reading winner | reading NDCG@10 | beginner winner | beginner NDCG@10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["per_topic_winners"]:
        lines.append(
            f"| `{row['query_id']}` | `{row['topic_match_winner']}` | {row['topic_match_ndcg']:.3f} | "
            f"`{row['reading_value_winner']}` | {row['reading_value_ndcg']:.3f} | "
            f"`{row['beginner_winner']}` | {row['beginner_ndcg']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            "### Worse Than Current Hybrid on Reading-Value NDCG@10",
            "",
            "| query_id | learned | hybrid | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["failure_cases"]["worse_than_hybrid_on_reading_value_ndcg"]:
        lines.append(
            f"| `{row['query_id']}` | {row['learned_reading_value_ndcg']:.3f} | "
            f"{row['hybrid_reading_value_ndcg']:.3f} | {row['delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "### Worse Than BM25 on Topic-Match NDCG@10",
            "",
            "| query_id | learned | bm25 | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["failure_cases"]["worse_than_bm25_on_topic_match_ndcg"]:
        lines.append(
            f"| `{row['query_id']}` | {row['learned_topic_match_ndcg']:.3f} | "
            f"{row['bm25_topic_match_ndcg']:.3f} | {row['delta']:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the V2.2 lightweight learned_hybrid reranker.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--milestone", default=str(DEFAULT_MILESTONE))
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
    baseline_path = resolve_repo_path(args.baseline)
    milestone_path = resolve_repo_path(args.milestone)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    model_out = resolve_repo_path(args.model_out)

    labels = load_jsonl(labels_path)
    packet_rows = load_jsonl(packet_path)
    baseline = load_json(baseline_path)
    milestone = load_json(milestone_path)
    report, final_model = train_and_evaluate(
        labels=labels,
        packet_rows=packet_rows,
        baseline_report=baseline,
        milestone_report=milestone,
        alpha=args.alpha,
        fold_count=args.folds,
        k=args.k,
    )
    save_model_artifact(model_out, final_model, report)
    report["model_output"] = str(model_out)
    report["input_files"] = {
        "labels": str(labels_path),
        "packet": str(packet_path),
        "baseline": str(baseline_path),
        "milestone": str(milestone_path),
    }
    command = (
        f".\\backend\\.venv\\Scripts\\python.exe scripts\\train_v2_2_lightweight_learned_hybrid.py "
        f"--labels {args.labels} --packet {args.packet} --baseline {args.baseline} "
        f"--milestone {args.milestone} --json-out {args.json_out} --md-out {args.md_out} "
        f"--model-out {args.model_out} --alpha {args.alpha} --folds {args.folds} --k {args.k}"
    )
    report["command"] = command
    outputs = [args.json_out, args.md_out, args.model_out]
    write_json(json_out, report)
    write_text(md_out, markdown_report(report, command=command, outputs=outputs))

    avg = report["method_averages"]["learned_hybrid"]
    criteria = report["satisfies_success_criteria"]
    print("V2.2 lightweight learned_hybrid complete")
    print(f"Report JSON: {json_out}")
    print(f"Report Markdown: {md_out}")
    print(f"Model artifact: {model_out}")
    print(f"learned_hybrid reading_value_ndcg@10={avg['reading_value_ndcg']:.3f}")
    print(f"learned_hybrid topic_match_ndcg@10={avg['topic_match_ndcg']:.3f}")
    print(f"delta vs hybrid reading={criteria['reading_value_delta_vs_hybrid']:+.3f}")
    print(f"delta vs bm25 topic={criteria['topic_match_delta_vs_bm25']:+.3f}")
    print(f"overall pass={criteria['overall_pass']}")


if __name__ == "__main__":
    main()
