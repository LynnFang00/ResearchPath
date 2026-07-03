from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from statistics import mean, pvariance
from typing import Any

import numpy as np


RANK_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
CURRENT_YEAR = 2026
MODEL_SCHEMA_VERSION = "v2.2b_ridge_packet_feature_model"


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


@dataclass(frozen=True)
class V22FeatureContext:
    score_stats: dict[str, dict[str, dict[str, float]]]
    duplicate_sizes: dict[tuple[str, int], int]


@dataclass(frozen=True)
class V22RidgeRanker:
    feature_names: list[str]
    weights: list[float]
    intercept: float
    standard_scaler_mean: list[float]
    standard_scaler_scale: list[float]
    version: str
    target: dict[str, Any]
    metadata: dict[str, Any]

    def score_feature_values(self, feature_values: list[float]) -> float:
        if len(feature_values) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(feature_values)}.")
        scaled = []
        for value, center, scale in zip(
            feature_values,
            self.standard_scaler_mean,
            self.standard_scaler_scale,
            strict=True,
        ):
            denominator = scale if scale else 1.0
            scaled.append((float(value) - center) / denominator)
        raw_score = self.intercept + sum(weight * value for weight, value in zip(self.weights, scaled, strict=True))
        return max(0.0, min(1.0, float(raw_score)))

    def score_packet_row(self, row: dict[str, Any], context: V22FeatureContext) -> float:
        values = build_feature_vector(row, names=self.feature_names, context=context)
        return self.score_feature_values(values)


def build_feature_context(packet_rows: list[dict[str, Any]]) -> V22FeatureContext:
    return V22FeatureContext(
        score_stats=score_stats_by_query(packet_rows),
        duplicate_sizes=title_duplicate_sizes(packet_rows),
    )


def load_v2_2_ridge_ranker(path: Path) -> V22RidgeRanker:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return v2_2_ridge_ranker_from_payload(payload)


def v2_2_ridge_ranker_from_payload(payload: dict[str, Any]) -> V22RidgeRanker:
    names = [str(name) for name in payload["feature_names"]]
    weights = [float(value) for value in payload["weights"]]
    scaler_mean = [float(value) for value in payload["standard_scaler_mean"]]
    scaler_scale = [float(value) for value in payload["standard_scaler_scale"]]
    expected_lengths = {len(names), len(weights), len(scaler_mean), len(scaler_scale)}
    if len(expected_lengths) != 1:
        raise ValueError("V2.2 ranker artifact has inconsistent feature/scaler/weight lengths.")
    return V22RidgeRanker(
        feature_names=names,
        weights=weights,
        intercept=float(payload["intercept"]),
        standard_scaler_mean=scaler_mean,
        standard_scaler_scale=scaler_scale,
        version=str(payload.get("version", "v2_2b")),
        target=dict(payload.get("target") or {}),
        metadata=dict(payload.get("metadata") or {}),
    )


def build_model_payload(
    *,
    feature_names: list[str],
    weights: list[float],
    intercept: float,
    standard_scaler_mean: list[float],
    standard_scaler_scale: list[float],
    target: dict[str, Any],
    hyperparameters: dict[str, Any],
    training_example_count: int,
    version: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "v2_2b_ridge_learned_hybrid",
        "version": version,
        "target": target,
        "feature_names": feature_names,
        "weights": [float(value) for value in weights],
        "intercept": float(intercept),
        "standard_scaler_mean": [float(value) for value in standard_scaler_mean],
        "standard_scaler_scale": [float(value) for value in standard_scaler_scale],
        "hyperparameters": hyperparameters,
        "training_example_count": int(training_example_count),
        "metadata": metadata or {},
        "runtime_note": (
            "This artifact requires V2.2 packet-style feature extraction before scoring. "
            "It is not a neural model."
        ),
    }


def build_feature_vector(
    row: dict[str, Any],
    *,
    names: list[str],
    context: V22FeatureContext,
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
        stats = context.score_stats.get(query_id, {}).get(method, {})
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
    duplicate_cluster_size = duplicate_title_cluster_size(row.get("duplicate_title_cluster"))
    normalized_duplicate_size = context.duplicate_sizes.get((query_id, paper_id), 0)

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


def title_duplicate_sizes(packet_rows: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    titles_by_query: dict[str, Counter[str]] = defaultdict(Counter)
    normalized_by_key: dict[tuple[str, int], str] = {}
    for row in packet_rows:
        key = (str(row["query_id"]), int(row["paper_id"]))
        normalized = normalize_title(str(row.get("title") or ""))
        normalized_by_key[key] = normalized
        if normalized:
            titles_by_query[key[0]][normalized] += 1
    return {
        key: titles_by_query[key[0]][normalized] if normalized else 0
        for key, normalized in normalized_by_key.items()
    }


def duplicate_title_cluster_size(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and isinstance(value.get("size"), int):
        return int(value["size"])
    return 0


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
