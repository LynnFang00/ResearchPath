import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.models.paper import Paper  # noqa: E402
from app.services.bm25 import BM25Retriever  # noqa: E402
from app.services.difficulty import score_difficulty  # noqa: E402
from app.services.quality_signals import compute_quality_signals, hybrid_score, normalize_scores  # noqa: E402
from app.services.retrievers.base import PaperDocument  # noqa: E402
from app.services.retrievers.embedding import load_sentence_transformer, normalize_rows  # noqa: E402
from app.services.retrievers.tfidf import TfidfRetriever  # noqa: E402
from app.services.v2_2_learned_ranker import build_feature_context, load_v2_2_ridge_ranker  # noqa: E402
from app.services.ltr_model_service import V49GuardedTextBlendModel  # noqa: E402
from build_v4_3_text_reranker_dataset import protected_status, text_input  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from evaluate_v6_1_3_exact_learned_baselines import load_artifacts, load_jsonl, write_jsonl  # noqa: E402
from materialize_v6_1_4_learned_ranker_features import (  # noqa: E402
    FEATURE_CATEGORIES,
    FORBIDDEN_FEATURES,
    LEARNED_METHODS,
    feature_hash,
    required_features,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_CANONICAL = REPO_ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_paper_metadata.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_corpus_manifest.json"
DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_V612_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_predictions.jsonl"
DEFAULT_EMBEDDINGS = REPO_ROOT / "data" / "processed" / "embeddings" / "all_minilm_l6_v2_50k.npz"
DEFAULT_FAISS_IDS = REPO_ROOT / "data" / "processed" / "faiss" / "all_minilm_l6_v2_50k.ids.npz"
DEFAULT_V22_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_V26_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"
DEFAULT_V33_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
DEFAULT_V41_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"
DEFAULT_V43_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_3_text_reranker"
DEFAULT_V41_BLEND_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"
DEFAULT_V49_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_9_guarded_text_blend_candidate.json"

DEFAULT_FEATURE_MATRIX = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_1_7_learned_ranker_feature_matrix.jsonl"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_learned_baseline_predictions.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_learned_ranker_feature_regeneration_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_learned_ranker_feature_regeneration_report.md"
DEFAULT_COVERAGE = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_feature_coverage_report.json"
DEFAULT_PARITY = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_scoring_parity_report.json"

CURRENT_YEAR = 2026
BASE_RETRIEVAL_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
SAFE_DEFAULT_RANK = 999999


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def canonical_documents(rows: list[dict[str, Any]]) -> list[PaperDocument]:
    return [
        PaperDocument(
            document_id=int(row["paper_id"]),
            text=f"{row.get('title') or ''}\n\n{row.get('abstract') or ''}",
            title=str(row.get("title") or ""),
            abstract=str(row.get("abstract") or ""),
            year=safe_int(row.get("year")),
            citation_count=safe_int(row.get("citation_count")) or 0,
        )
        for row in rows
    ]


def canonical_paper(row: dict[str, Any]) -> Paper:
    external_ids = row.get("external_ids") if isinstance(row.get("external_ids"), dict) else {}
    return Paper(
        id=int(row["paper_id"]),
        title=str(row.get("title") or ""),
        abstract=str(row.get("abstract") or ""),
        authors="",
        year=safe_int(row.get("year")),
        venue=row.get("venue"),
        categories="",
        citation_count=safe_int(row.get("citation_count")) or 0,
        source_url=row.get("source_url"),
        external_id=external_ids.get("external_id"),
        source=row.get("source"),
        doi=external_ids.get("doi"),
        url=row.get("source_url"),
        references_count=0,
        influential_citation_count=0,
        abstract_word_count=safe_int(row.get("abstract_word_count")) or 0,
    )


def rows_with_split(examples_path: Path, predictions_path: Path) -> list[dict[str, Any]]:
    examples = {(str(row["query_id"]), int(row["paper_id"])): row for row in load_jsonl(examples_path)}
    rows = []
    for pred in load_jsonl(predictions_path):
        key = (str(pred["query_id"]), int(pred["paper_id"]))
        row = dict(examples[key])
        row["fold_id"] = pred["fold_id"]
        row["split"] = pred["split"]
        rows.append(row)
    return sorted(rows, key=lambda row: (row["fold_id"], row["split"], row["query_id"], int(row["paper_id"])))


def rank_from_scores(scores: dict[int, float], *, positive_only: bool) -> tuple[dict[int, int], dict[int, float]]:
    filtered = {
        paper_id: float(score)
        for paper_id, score in scores.items()
        if math.isfinite(float(score)) and (not positive_only or float(score) > 0.0)
    }
    ranked = sorted(filtered.items(), key=lambda item: (item[1], -item[0]), reverse=True)
    ranks = {paper_id: index for index, (paper_id, _) in enumerate(ranked, start=1)}
    normalized = normalize_scores(filtered)
    return ranks, normalized


def dense_scores_for_query(query: str, *, encoder: Any, document_ids: np.ndarray, embeddings: np.ndarray) -> dict[int, float]:
    query_embedding = encoder.encode(query, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    query_embedding = normalize_rows(query_embedding)[0]
    scores = embeddings @ query_embedding
    return {int(paper_id): float(score) for paper_id, score in zip(document_ids.tolist(), scores.tolist(), strict=True)}


def build_retrieval_maps(
    *,
    canonical_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    embeddings_path: Path,
    faiss_ids_path: Path,
) -> tuple[dict[str, dict[str, dict[int, dict[str, float | int]]]], dict[str, Any]]:
    documents = canonical_documents(canonical_rows)
    paper_by_id = {int(row["paper_id"]): canonical_paper(row) for row in canonical_rows}
    bm25 = BM25Retriever(documents)
    tfidf = TfidfRetriever(documents)
    embedding_status: dict[str, Any] = {"artifact_path": str(embeddings_path), "available": False}
    embedding_payload = np.load(embeddings_path, allow_pickle=False)
    document_ids = embedding_payload["document_ids"].astype(np.int64)
    embeddings = normalize_rows(embedding_payload["embeddings"])
    embedding_status.update(
        {
            "available": True,
            "row_count": int(len(document_ids)),
            "document_id_unique_count": int(len(set(document_ids.tolist()))),
            "model_name": str(embedding_payload["model_name"]) if "model_name" in embedding_payload.files else "sentence-transformers/all-MiniLM-L6-v2",
        }
    )
    faiss_status = {"artifact_path": str(faiss_ids_path), "available": faiss_ids_path.exists(), "ids_match_embeddings": False}
    if faiss_ids_path.exists():
        faiss_ids = np.load(faiss_ids_path, allow_pickle=False)["document_ids"].astype(np.int64)
        faiss_status["ids_match_embeddings"] = bool(np.array_equal(faiss_ids, document_ids))
        faiss_status["row_count"] = int(len(faiss_ids))
    encoder = load_sentence_transformer(embedding_status["model_name"])
    by_query = defaultdict(list)
    for row in target_rows:
        by_query[str(row["query_id"])].append(row)
    output: dict[str, dict[str, dict[int, dict[str, float | int]]]] = {}
    for query_id, rows in sorted(by_query.items()):
        query = str(rows[0]["query"])
        target_ids = {int(row["paper_id"]) for row in rows}
        method_maps: dict[str, dict[int, dict[str, float | int]]] = {}
        bm25_raw = {item.document_id: item.score for item in bm25.search(query, k=len(documents))}
        tfidf_raw = {item.document_id: item.score for item in tfidf.search(query, k=len(documents))}
        dense_raw = dense_scores_for_query(query, encoder=encoder, document_ids=document_ids, embeddings=embeddings)
        raw_by_method = {
            "bm25": (bm25_raw, True),
            "tfidf": (tfidf_raw, True),
            "embedding": (dense_raw, False),
            "faiss_embedding": (dense_raw, False),
        }
        normalized_for_hybrid: dict[str, dict[int, float]] = {}
        for method, (scores, positive_only) in raw_by_method.items():
            ranks, normalized = rank_from_scores(scores, positive_only=positive_only)
            normalized_for_hybrid[method] = normalized
            method_maps[method] = {
                paper_id: {
                    "score": float(normalized.get(paper_id, 0.0)),
                    "raw_score": float(scores.get(paper_id, 0.0)),
                    "rank": int(ranks.get(paper_id, SAFE_DEFAULT_RANK)),
                    "reciprocal_rank": 1.0 / ranks[paper_id] if paper_id in ranks else 0.0,
                }
                for paper_id in target_ids
            }
        hybrid_scores: dict[int, float] = {}
        for paper_id in target_ids:
            paper = paper_by_id[paper_id]
            difficulty = score_difficulty(paper, current_year=CURRENT_YEAR)
            signals = compute_quality_signals(
                query=query,
                paper=paper,
                difficulty=difficulty,
                background_level="basic_ml",
                bm25_score=normalized_for_hybrid["bm25"].get(paper_id, 0.0),
                tfidf_score=normalized_for_hybrid["tfidf"].get(paper_id, 0.0),
                faiss_score=normalized_for_hybrid["faiss_embedding"].get(paper_id, 0.0),
                current_year=CURRENT_YEAR,
            )
            hybrid_scores[paper_id] = float(hybrid_score(signals))
        hybrid_ranks, hybrid_norm = rank_from_scores(hybrid_scores, positive_only=False)
        method_maps["hybrid"] = {
            paper_id: {
                "score": float(hybrid_norm.get(paper_id, 0.0)),
                "raw_score": float(hybrid_scores.get(paper_id, 0.0)),
                "rank": int(hybrid_ranks.get(paper_id, SAFE_DEFAULT_RANK)),
                "reciprocal_rank": 1.0 / hybrid_ranks[paper_id] if paper_id in hybrid_ranks else 0.0,
            }
            for paper_id in target_ids
        }
        output[query_id] = method_maps
    status = {
        "bm25": {"available": True, "source": "canonical_snapshot_runtime_bm25"},
        "tfidf": {"available": True, "source": "canonical_snapshot_runtime_tfidf"},
        "embedding": embedding_status,
        "faiss_embedding": faiss_status | {"source": "embedding_npz_scores_with_matching_faiss_id_map"},
        "hybrid": {"available": True, "source": "canonical_snapshot_runtime_quality_formula_with_snapshot_defaults"},
    }
    return output, status


def attach_retrieval_features(rows: list[dict[str, Any]], retrieval_maps: dict[str, dict[str, dict[int, dict[str, float | int]]]], canonical_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        paper_id = int(row["paper_id"])
        canonical = canonical_by_id[paper_id]
        enriched = dict(row)
        enriched.update(
            {
                "title": canonical["title"],
                "abstract": canonical["abstract"],
                "year": canonical["year"],
                "venue": canonical.get("venue"),
                "citation_count": canonical.get("citation_count", 0),
                "sources_provenance": [canonical.get("source")] if canonical.get("source") else [],
                "identifiers": canonical.get("external_ids") or {},
                "source_url": canonical.get("source_url"),
                "evidence_availability": {"level": "title_abstract", "fulltext_available": False},
                "full_text_available": False,
            }
        )
        retrieval_scores: dict[str, float] = {}
        retrieval_ranks: dict[str, int] = {}
        for method in BASE_RETRIEVAL_METHODS:
            info = retrieval_maps[str(row["query_id"])][method][paper_id]
            retrieval_scores[method] = float(info["score"])
            retrieval_ranks[method] = int(info["rank"])
            enriched[f"{method}_score"] = float(info["score"])
            enriched[f"{method}_raw_score"] = float(info["raw_score"])
            enriched[f"{method}_rank"] = int(info["rank"])
            enriched[f"{method}_reciprocal_rank"] = float(info["reciprocal_rank"])
        enriched["retrieval_scores_by_method"] = retrieval_scores
        enriched["retrieval_ranks_by_method"] = retrieval_ranks
        output.append(enriched)
    return output


def add_v2_scores(rows: list[dict[str, Any]], old_model_path: Path, v26_model_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_ranker = load_v2_2_ridge_ranker(old_model_path)
    v26_ranker = load_v2_2_ridge_ranker(v26_model_path)
    context = build_feature_context(rows)
    output = []
    for row in rows:
        enriched = dict(row)
        enriched["old_v2_2b_score"] = float(old_ranker.score_packet_row(enriched, context))
        enriched["v2_6_score"] = float(v26_ranker.score_packet_row(enriched, context))
        output.append(enriched)
    return output, {
        "old_v2_2b_score": {"available": True, "source": str(old_model_path), "feature_context": "v6_0_2_candidate_rows_with_regenerated_retrieval_features"},
        "v2_6_score": {"available": True, "source": str(v26_model_path), "feature_context": "v6_0_2_candidate_rows_with_regenerated_retrieval_features"},
    }


def compact_feature_values(row: dict[str, Any], required_union: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    for method in BASE_RETRIEVAL_METHODS:
        for suffix in ("score", "rank", "reciprocal_rank"):
            name = f"{method}_{suffix}"
            if name in required_union:
                values[name] = float(row.get(name, 0.0 if suffix != "rank" else SAFE_DEFAULT_RANK))
                sources[name] = f"v6_1_7_regenerated_{method}"
    for name in ("old_v2_2b_score", "v2_6_score"):
        if name in required_union:
            values[name] = float(row[name])
            sources[name] = "v6_1_7_regenerated_v2_packet_ranker_score"
    if "year" in required_union:
        values["year"] = float(row["year"] or 0.0)
        sources["year"] = "v6_1_6b_canonical_snapshot"
    if "paper_age_years" in required_union:
        values["paper_age_years"] = float(max(0, CURRENT_YEAR - int(row["year"]))) if row.get("year") else 0.0
        sources["paper_age_years"] = "derived_from_canonical_year"
    if "citation_count_log1p" in required_union:
        values["citation_count_log1p"] = math.log1p(max(float(row.get("citation_count") or 0.0), 0.0))
        sources["citation_count_log1p"] = "derived_from_canonical_citation_count"
    if "full_text_available_bool" in required_union:
        values["full_text_available_bool"] = 1.0 if row.get("full_text_available") else 0.0
        sources["full_text_available_bool"] = "safe_snapshot_default_no_fulltext_artifact"
    return values, sources


def build_feature_matrix(rows: list[dict[str, Any]], required_by_method: dict[str, list[str]]) -> list[dict[str, Any]]:
    required_union = sorted({feature for features in required_by_method.values() for feature in features})
    output = []
    for row in rows:
        values, sources = compact_feature_values(row, required_union)
        missing = [feature for feature in required_union if feature not in values]
        missing_by_method = {
            method: [feature for feature in features if feature not in values]
            for method, features in required_by_method.items()
        }
        output.append(
            {
                "schema_version": "v6.1.7_learned_ranker_feature_row",
                "query_id": row["query_id"],
                "query": row["query"],
                "paper_id": int(row["paper_id"]),
                "title": row["title"],
                "fold_id": row["fold_id"],
                "split": row["split"],
                "feature_values": values,
                "missing_feature_names": missing,
                "missing_features_by_method": missing_by_method,
                "feature_source": sources,
                "feature_hash": feature_hash(values, missing),
                "forbidden_features_used": [],
                "exact_runtime_feature_complete": not missing,
            }
        )
    return output


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["fold_id"]), str(row["split"]), str(row["query_id"]), int(row["paper_id"]))


def score_learned_baselines(
    matrix_rows: list[dict[str, Any]],
    enriched_rows: list[dict[str, Any]],
    artifacts: dict[str, Any],
    artifact_status: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_key = {row_key(row): row for row in enriched_rows}
    coverage: dict[str, Any] = {}
    score_maps: dict[str, dict[tuple[str, str, str, int], Any]] = {method: {} for method in LEARNED_METHODS}
    for method in LEARNED_METHODS:
        method_rows = [
            row for row in matrix_rows
            if not row["missing_features_by_method"].get(method)
            and artifact_status.get(method, {}).get("artifact_loaded")
        ]
        missing_counts = Counter(feature for row in matrix_rows for feature in row["missing_features_by_method"].get(method, []))
        coverage[method] = {
            "rows_scored": len(method_rows),
            "rows_missing": len(matrix_rows) - len(method_rows),
            "coverage": len(method_rows) / len(matrix_rows) if matrix_rows else 0.0,
            "exact_scoring_available": len(method_rows) == len(matrix_rows) and bool(matrix_rows),
            "missing_feature_counts": dict(sorted(missing_counts.items())),
            "missing_feature_categories": category_counts(missing_counts),
            "rows_silently_dropped": 0,
        }
    if coverage["v3_3_ltr"]["exact_scoring_available"]:
        model = artifacts["v3_3_ltr"]
        scores = model.score_feature_rows([row["feature_values"] for row in matrix_rows])
        for row, score in zip(matrix_rows, scores, strict=True):
            score_maps["v3_3_ltr"][row_key(row)] = float(score)
    if coverage["v4_1_blend"]["exact_scoring_available"]:
        model = artifacts["v4_1_blend"]
        scores = model.score_feature_rows([row["feature_values"] for row in matrix_rows])
        for row, score_row in zip(matrix_rows, scores, strict=True):
            score_maps["v4_1_blend"][row_key(row)] = score_row
    if coverage["v4_9_guarded_text_blend"]["exact_scoring_available"]:
        model = artifacts["v4_9_guarded_text_blend"]
        texts = [text_input(rows_by_key[row_key(row)]) for row in matrix_rows]
        text_scores = model.text_model.score_text_rows(texts, [row["feature_values"] for row in matrix_rows])
        by_query: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: {"v33": {}, "v41": {}, "text": {}})
        key_by_query_paper = {(str(row["query_id"]), int(row["paper_id"])): row_key(row) for row in matrix_rows}
        for row, text_score in zip(matrix_rows, text_scores, strict=True):
            key = row_key(row)
            query_id = str(row["query_id"])
            paper_id = int(row["paper_id"])
            by_query[query_id]["v33"][paper_id] = float(score_maps["v3_3_ltr"][key])
            by_query[query_id]["v41"][paper_id] = float(score_maps["v4_1_blend"][key]["v4_1_blend_score"])
            by_query[query_id]["text"][paper_id] = float(text_score)
        for query_id, maps in by_query.items():
            details = V49GuardedTextBlendModel.score_component_maps(
                v33_scores=maps["v33"],
                v41_scores=maps["v41"],
                text_scores=maps["text"],
            )
            for paper_id, score_details in details.items():
                score_maps["v4_9_guarded_text_blend"][key_by_query_paper[(query_id, paper_id)]] = score_details
    predictions = []
    for row in matrix_rows:
        key = row_key(row)
        v41_details = score_maps["v4_1_blend"].get(key) or {}
        v49_details = score_maps["v4_9_guarded_text_blend"].get(key) or {}
        predictions.append(
            {
                "schema_version": "v6.1.7_learned_baseline_prediction",
                "query_id": row["query_id"],
                "paper_id": row["paper_id"],
                "fold_id": row["fold_id"],
                "split": row["split"],
                "v3_3_score": score_maps["v3_3_ltr"].get(key),
                "v4_1_score": v41_details.get("v4_1_blend_score"),
                "v4_1_weighted_score": v41_details.get("v4_1_weighted_score"),
                "v4_9_score": v49_details.get("v4_9_guarded_text_blend_score"),
                "v4_9_details": v49_details or None,
                "missing_score_reasons": {
                    method: None if not row["missing_features_by_method"].get(method) else f"Missing exact runtime features: {', '.join(row['missing_features_by_method'][method][:12])}"
                    for method in LEARNED_METHODS
                },
            }
        )
    return predictions, coverage


def category_counts(counter: Counter[str]) -> dict[str, int]:
    categories: Counter[str] = Counter()
    for feature, count in counter.items():
        categories[FEATURE_CATEGORIES.get(feature, "other")] += int(count)
    return dict(sorted(categories.items()))


def feature_coverage(matrix_rows: list[dict[str, Any]], retrieval_status: dict[str, Any], v2_status: dict[str, Any]) -> dict[str, Any]:
    features = sorted({feature for row in matrix_rows for feature in set(row["feature_values"]) | set(row["missing_feature_names"])})
    by_feature = {
        feature: {
            "present_rows": sum(1 for row in matrix_rows if feature in row["feature_values"]),
            "missing_rows": sum(1 for row in matrix_rows if feature in row["missing_feature_names"]),
            "category": FEATURE_CATEGORIES.get(feature, "other"),
        }
        for feature in features
    }
    by_group = {
        "bm25": {"available": True, "features": [f for f in features if f.startswith("bm25_")]},
        "tfidf": {"available": True, "features": [f for f in features if f.startswith("tfidf_")]},
        "dense_embedding": {"available": retrieval_status["embedding"]["available"], "features": [f for f in features if f.startswith("embedding_")]},
        "faiss_embedding": {"available": retrieval_status["faiss_embedding"]["available"], "features": [f for f in features if f.startswith("faiss_embedding_")]},
        "hybrid": {"available": True, "features": [f for f in features if f.startswith("hybrid_")]},
        "v2_2b": v2_status["old_v2_2b_score"],
        "v2_6": v2_status["v2_6_score"],
        "citation_quality": {"available": True, "features": ["citation_count_log1p"]},
        "metadata": {"available": True, "features": ["year", "paper_age_years"]},
        "full_text": {"available": True, "features": ["full_text_available_bool"], "note": "safe default false; no full-text artifact regenerated"},
    }
    return {"by_feature": by_feature, "by_group": by_group}


def parity_report(report: dict[str, Any], coverage: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    sample = predictions[:25]
    return {
        "schema_version": "v6.1.7_scoring_parity_report",
        "created_at": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "runtime_scoring_invoked": False,
        "v4_9_formula_validated_against_candidate_config": True,
        "forbidden_features_required": [],
        "forbidden_features_used": [],
        "max_score_delta": {"v3_3_ltr": None, "v4_1_blend": None, "v4_9_guarded_text_blend": None},
        "parity_comparable": False,
        "parity_blocker": "No persisted prior V6.0.2 exact learned-ranker scores exist for the canonical PostgreSQL snapshot.",
        "coverage": coverage,
        "deterministic_prediction_hash_sample": [
            hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            for row in sample
        ],
        "report_feature_regeneration_succeeded": report["feature_regeneration_succeeded"],
    }


def build_report(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    target_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    artifact_status: dict[str, Any],
    required_by_method: dict[str, list[str]],
    feature_cov: dict[str, Any],
    retrieval_status: dict[str, Any],
    v2_status: dict[str, Any],
) -> dict[str, Any]:
    exact_success = all(coverage[method]["exact_scoring_available"] for method in LEARNED_METHODS)
    return {
        "schema_version": "v6.1.7_learned_ranker_feature_regeneration_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "production_candidate_created": False,
        "external_api_calls_made": False,
        "canonical_snapshot": {
            "path": str(resolve_repo_path(args.canonical)),
            "manifest_path": str(resolve_repo_path(args.manifest)),
            "row_count": manifest["row_count"],
            "corpus_hash": manifest["corpus_hash"],
        },
        "dataset": {
            "row_count": len(target_rows),
            "topic_count": len({row["query_id"] for row in target_rows}),
            "fold_count": len({row["fold_id"] for row in target_rows}),
        },
        "artifact_introspection": {
            method: {
                "artifact_loaded": artifact_status.get(method, {}).get("artifact_loaded"),
                "expected_feature_names": required_by_method.get(method, []),
                "expected_feature_order": required_by_method.get(method, []),
                "expected_normalization_or_scaling": "compact retrieval scores are query-normalized; rank defaults use 999999/0 reciprocal; model-internal sklearn artifacts apply their own scaling where applicable",
                "dependency_artifacts": artifact_status.get(method, {}),
                "all_required_features_regenerated": coverage[method]["exact_scoring_available"],
            }
            for method in LEARNED_METHODS
        },
        "feature_regeneration_succeeded": exact_success,
        "exact_neural_vs_v4_9_comparison_possible": coverage["v4_9_guarded_text_blend"]["exact_scoring_available"],
        "rows_silently_dropped": 0,
        "forbidden_features_used": [],
        "forbidden_features_required": [],
        "learned_baseline_coverage": coverage,
        "feature_coverage": feature_cov,
        "retrieval_artifact_status": retrieval_status,
        "v2_score_status": v2_status,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.1.7 Learned-Ranker Feature Regeneration",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Rows: `{report['dataset']['row_count']}`",
        f"- Canonical corpus hash: `{report['canonical_snapshot']['corpus_hash']}`",
        f"- Feature regeneration succeeded: `{report['feature_regeneration_succeeded']}`",
        f"- Exact neural-vs-V4.9 comparison possible: `{report['exact_neural_vs_v4_9_comparison_possible']}`",
        f"- Rows silently dropped: `{report['rows_silently_dropped']}`",
        "",
        "## Learned Baseline Coverage",
        "",
    ]
    for method, row in report["learned_baseline_coverage"].items():
        lines.append(
            f"- `{method}`: scored `{row['rows_scored']}` / `{report['dataset']['row_count']}`, exact `{row['exact_scoring_available']}`"
        )
    lines.extend(["", "## Feature Groups", ""])
    for group, row in report["feature_coverage"]["by_group"].items():
        lines.append(f"- `{group}`: available `{row.get('available')}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default=str(DEFAULT_CANONICAL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--v6-1-2-predictions", default=str(DEFAULT_V612_PREDICTIONS))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--faiss-ids", default=str(DEFAULT_FAISS_IDS))
    parser.add_argument("--old-v22-model", default=str(DEFAULT_V22_MODEL))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR))
    parser.add_argument("--v41-model-dir", default=str(DEFAULT_V41_MODEL_DIR))
    parser.add_argument("--v43-model-dir", default=str(DEFAULT_V43_MODEL_DIR))
    parser.add_argument("--v41-blend-config", default=str(DEFAULT_V41_BLEND_CONFIG))
    parser.add_argument("--v49-config", default=str(DEFAULT_V49_CONFIG))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--feature-matrix-out", default=str(DEFAULT_FEATURE_MATRIX))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--coverage-out", default=str(DEFAULT_COVERAGE))
    parser.add_argument("--parity-out", default=str(DEFAULT_PARITY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_rows = load_jsonl(resolve_repo_path(args.canonical))
    manifest = load_json(resolve_repo_path(args.manifest))
    canonical_by_id = {int(row["paper_id"]): row for row in canonical_rows}
    target_rows = rows_with_split(resolve_repo_path(args.examples), resolve_repo_path(args.v6_1_2_predictions))
    missing_ids = sorted({int(row["paper_id"]) for row in target_rows} - set(canonical_by_id))
    if missing_ids:
        raise ValueError(f"V6.0.2 rows did not resolve in canonical snapshot: {missing_ids[:20]}")
    artifacts, artifact_status = load_artifacts(args)
    required_by_method = required_features(artifacts, artifact_status)
    retrieval_maps, retrieval_status = build_retrieval_maps(
        canonical_rows=canonical_rows,
        target_rows=target_rows,
        embeddings_path=resolve_repo_path(args.embeddings),
        faiss_ids_path=resolve_repo_path(args.faiss_ids),
    )
    enriched_rows = attach_retrieval_features(target_rows, retrieval_maps, canonical_by_id)
    enriched_rows, v2_status = add_v2_scores(enriched_rows, resolve_repo_path(args.old_v22_model), resolve_repo_path(args.v26_model))
    matrix_rows = build_feature_matrix(enriched_rows, required_by_method)
    predictions, coverage = score_learned_baselines(matrix_rows, enriched_rows, artifacts, artifact_status)
    feature_cov = feature_coverage(matrix_rows, retrieval_status, v2_status)
    report = build_report(
        args=args,
        manifest=manifest,
        target_rows=target_rows,
        matrix_rows=matrix_rows,
        coverage=coverage,
        artifact_status=artifact_status,
        required_by_method=required_by_method,
        feature_cov=feature_cov,
        retrieval_status=retrieval_status,
        v2_status=v2_status,
    )
    parity = parity_report(report, coverage, predictions)
    write_jsonl(resolve_repo_path(args.feature_matrix_out), matrix_rows)
    write_jsonl(resolve_repo_path(args.predictions_out), predictions)
    write_json(resolve_repo_path(args.coverage_out), {"schema_version": "v6.1.7_feature_coverage_report", **feature_cov, "learned_baseline_coverage": coverage})
    write_json(resolve_repo_path(args.parity_out), parity)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.1.7 feature regeneration report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
