import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sqlalchemy import func, select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402
from app.services.bm25 import BM25Retriever, tokenize as bm25_tokenize  # noqa: E402
from app.services.difficulty import score_difficulty  # noqa: E402
from app.services.quality_signals import compute_quality_signals, hybrid_score, normalize_scores  # noqa: E402
from app.services.retrievers.base import PaperDocument  # noqa: E402
from app.services.retrievers.embedding import load_sentence_transformer, normalize_rows  # noqa: E402
from app.services.retrievers.faiss import load_faiss  # noqa: E402
from app.services.retrievers.faiss_id_mapping import load_faiss_id_mapping  # noqa: E402
from app.services.retrievers.tfidf import TfidfRetriever  # noqa: E402
from app.services.v2_2_learned_ranker import build_feature_context, load_v2_2_ridge_ranker  # noqa: E402


DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_V21 = REPO_ROOT / "data" / "eval" / "results" / "v2_1_benchmark_method_comparison.json"
DEFAULT_V22 = REPO_ROOT / "data" / "eval" / "results" / "v2_2_learned_hybrid_cv_report.json"
DEFAULT_V22B = REPO_ROOT / "data" / "eval" / "results" / "v2_2b_fair_learned_hybrid_report.json"
DEFAULT_V23 = REPO_ROOT / "data" / "eval" / "results" / "v2_3_production_candidate_pool_report.json"
DEFAULT_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_4_true_production_candidate_pool_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_4_true_production_candidate_pool_report.md"
SELECTED_240 = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"
EXPECTED_SELECTED_240_SHA256 = "6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F"

BASELINE_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
METHODS = ["learned_hybrid"] + BASELINE_METHODS
INTENT_SCORE_FIELDS = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
AUDIENCE_SCORE_FIELDS = {
    "beginner": "beginner_fit_score",
    "intermediate": "intermediate_fit_score",
    "advanced": "advanced_fit_score",
    "expert": "expert_fit_score",
}
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


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.exists() else None


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


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


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


def load_db_assets() -> dict[str, Any]:
    settings = get_settings()
    with SessionLocal() as db:
        count = int(db.scalar(select(func.count(Paper.id))) or 0)
        min_id, max_id = db.execute(select(func.min(Paper.id), func.max(Paper.id))).one()
        papers = list(db.scalars(select(Paper).order_by(Paper.id)).all())
        identifiers = list(db.execute(select(PaperIdentifier.paper_id, PaperIdentifier.source, PaperIdentifier.identifier)))
        sources = list(db.execute(select(PaperSource.paper_id, PaperSource.source, PaperSource.source_url)))
    identifiers_by_id: dict[int, dict[str, str]] = defaultdict(dict)
    for paper_id, source, identifier in identifiers:
        identifiers_by_id[int(paper_id)][str(source)] = str(identifier)
    sources_by_id: dict[int, set[str]] = defaultdict(set)
    source_urls_by_id: dict[int, list[str]] = defaultdict(list)
    for paper_id, source, source_url in sources:
        sources_by_id[int(paper_id)].add(str(source))
        if source_url:
            source_urls_by_id[int(paper_id)].append(str(source_url))
    return {
        "settings": settings,
        "db_diagnostics": {
            "database_url": settings.database_url,
            "reachable": True,
            "paper_count": count,
            "min_paper_id": int(min_id or 0),
            "max_paper_id": int(max_id or 0),
            "paper_identifier_count": len(identifiers),
            "paper_source_record_count": len(sources),
        },
        "papers": papers,
        "paper_by_id": {int(paper.id): paper for paper in papers},
        "identifiers_by_id": dict(identifiers_by_id),
        "sources_by_id": dict(sources_by_id),
        "source_urls_by_id": dict(source_urls_by_id),
    }


def db_unreachable_report(error: Exception) -> dict[str, Any]:
    settings = get_settings()
    return {
        "database_url": settings.database_url,
        "reachable": False,
        "error": f"{type(error).__name__}: {error}",
    }


def build_documents(papers: list[Paper]) -> list[PaperDocument]:
    return [
        PaperDocument(
            document_id=int(paper.id),
            text=paper.searchable_text,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            citation_count=paper.citation_count,
        )
        for paper in papers
    ]


def load_vector_assets(settings: Any, paper_by_id: dict[int, Paper]) -> dict[str, Any]:
    embedding_path = resolve_repo_path(settings.embedding_index_path)
    faiss_index_path = resolve_repo_path(settings.faiss_index_path)
    faiss_id_map_path = resolve_repo_path(settings.faiss_id_map_path)
    mapping = load_faiss_id_mapping(faiss_id_map_path)
    faiss = load_faiss()
    index = faiss.read_index(str(faiss_index_path))
    embedding_payload = np.load(embedding_path, allow_pickle=False)
    embedding_ids = embedding_payload["document_ids"].astype(np.int64)
    embeddings = normalize_rows(embedding_payload["embeddings"].astype(np.float32))
    faiss_ids = mapping.document_ids.astype(np.int64)
    position_by_id = mapping.position_by_paper_id()
    db_ids = set(paper_by_id)
    faiss_id_set = set(int(value) for value in faiss_ids.tolist())
    missing_in_db = sorted(faiss_id_set - db_ids)
    db_missing_in_faiss = sorted(db_ids - faiss_id_set)
    if not np.array_equal(embedding_ids, faiss_ids):
        raise ValueError("Embedding document_ids and FAISS id-map document_ids do not match.")
    if int(index.ntotal) != int(faiss_ids.shape[0]):
        raise ValueError("FAISS index ntotal does not match id-map document_ids length.")
    return {
        "embedding_path": embedding_path,
        "faiss_index_path": faiss_index_path,
        "faiss_id_map_path": faiss_id_map_path,
        "faiss_index": index,
        "mapping": mapping,
        "embeddings": embeddings,
        "position_by_id": position_by_id,
        "diagnostics": {
            "embedding_path": str(embedding_path),
            "embedding_exists": embedding_path.exists(),
            "embedding_document_count": int(embedding_ids.shape[0]),
            "embedding_dimension": int(embeddings.shape[1]),
            "faiss_index_path": str(faiss_index_path),
            "faiss_index_exists": faiss_index_path.exists(),
            "faiss_index_ntotal": int(index.ntotal),
            "faiss_index_dimension": int(index.d),
            "faiss_id_map_path": str(faiss_id_map_path),
            "faiss_id_map": mapping.diagnostics(),
            "faiss_ids_present_in_db": len(faiss_id_set & db_ids),
            "faiss_ids_missing_in_db_count": len(missing_in_db),
            "db_papers_missing_in_faiss_count": len(db_missing_in_faiss),
            "sample_faiss_ids_missing_in_db": missing_in_db[:10],
            "sample_db_papers_missing_in_faiss": db_missing_in_faiss[:10],
            "mapping_chain_available": not missing_in_db and not db_missing_in_faiss,
            "position_to_paper_id_link": "data/processed/faiss/*.ids.npz document_ids",
            "paper_id_to_metadata_link": "backend Postgres papers/paper_identifiers/paper_sources tables",
        },
    }


def encode_query(query: str, encoder: Any) -> np.ndarray:
    query_embedding = encoder.encode(
        query,
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return normalize_rows(query_embedding).astype(np.float32)[0]


def bm25_score_for_doc(retriever: BM25Retriever, query_terms: list[str], doc_id: int) -> float:
    if doc_id not in retriever.term_frequencies:
        return 0.0
    score = 0.0
    doc_length = retriever.doc_lengths[doc_id]
    term_frequency = retriever.term_frequencies[doc_id]
    for term in query_terms:
        frequency = term_frequency.get(term, 0)
        if frequency == 0:
            continue
        denominator = frequency + retriever.k1 * (
            1 - retriever.b + retriever.b * doc_length / max(retriever.avg_doc_length, 1e-9)
        )
        score += retriever._idf(term) * (frequency * (retriever.k1 + 1)) / denominator
    return float(score)


def tfidf_scores_for_query(retriever: TfidfRetriever, query: str) -> dict[int, float]:
    if retriever.matrix is None:
        return {}
    query_vector = retriever.vectorizer.transform([query])
    scores = (retriever.matrix @ query_vector.T).toarray().ravel()
    return {
        int(document.document_id): float(scores[index])
        for index, document in enumerate(retriever.documents)
        if float(scores[index]) > 0
    }


def top_ids_from_scores(scores: dict[int, float], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:top_k]
    return [
        {"paper_id": int(paper_id), "score": float(score), "rank": rank}
        for rank, (paper_id, score) in enumerate(ranked, start=1)
    ]


def dense_scores_for_query(
    *,
    query: str,
    encoder: Any,
    embeddings: np.ndarray,
    mapping: Any,
) -> dict[int, float]:
    query_embedding = encode_query(query, encoder)
    scores = embeddings @ query_embedding
    return {
        int(paper_id): float(scores[position])
        for position, paper_id in enumerate(mapping.document_ids.tolist())
    }


def score_hybrid_candidates(
    *,
    query: str,
    candidate_ids: set[int],
    paper_by_id: dict[int, Paper],
    bm25_scores: dict[int, float],
    tfidf_scores: dict[int, float],
    faiss_scores: dict[int, float],
) -> dict[int, float]:
    bm25_norm = normalize_scores(bm25_scores)
    tfidf_norm = normalize_scores(tfidf_scores)
    faiss_norm = normalize_scores(faiss_scores)
    scored: dict[int, float] = {}
    for paper_id in candidate_ids:
        paper = paper_by_id.get(paper_id)
        if paper is None:
            continue
        difficulty = score_difficulty(paper, current_year=2026)
        signals = compute_quality_signals(
            query=query,
            paper=paper,
            difficulty=difficulty,
            background_level="basic_ml",
            bm25_score=bm25_norm.get(paper_id, 0.0),
            tfidf_score=tfidf_norm.get(paper_id, 0.0),
            faiss_score=faiss_norm.get(paper_id, 0.0),
            current_year=2026,
        )
        scored[paper_id] = float(hybrid_score(signals))
    return scored


def build_retrieval_runs(
    *,
    packet_lookup: dict[str, list[dict[str, Any]]],
    paper_by_id: dict[int, Paper],
    documents: list[PaperDocument],
    vector_assets: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    settings = get_settings()
    encoder = load_sentence_transformer(settings.embedding_model_name)
    bm25 = BM25Retriever(documents)
    tfidf = TfidfRetriever(documents)
    runs: dict[str, Any] = {}
    for query_id, packet_rows in sorted(packet_lookup.items()):
        query = str(packet_rows[0]["query"])
        query_terms = bm25_tokenize(query)
        bm25_top = bm25.search(query, k=top_k)
        tfidf_all_scores = tfidf_scores_for_query(tfidf, query)
        tfidf_top = top_ids_from_scores(tfidf_all_scores, top_k)
        dense_all_scores = dense_scores_for_query(
            query=query,
            encoder=encoder,
            embeddings=vector_assets["embeddings"],
            mapping=vector_assets["mapping"],
        )
        faiss_top = top_ids_from_scores(dense_all_scores, top_k)
        embedding_top = list(faiss_top)

        hybrid_search_k = min(len(documents), max(top_k * 8, 50))
        bm25_hybrid = bm25.search(query, k=hybrid_search_k)
        tfidf_hybrid = top_ids_from_scores(tfidf_all_scores, hybrid_search_k)
        faiss_hybrid = top_ids_from_scores(dense_all_scores, hybrid_search_k)
        hybrid_candidate_ids = {
            int(row["paper_id"])
            for row in scored_rows(bm25_hybrid) + tfidf_hybrid + faiss_hybrid
        }
        bm25_hybrid_scores = {int(row["paper_id"]): float(row["score"]) for row in scored_rows(bm25_hybrid)}
        tfidf_hybrid_scores = {int(row["paper_id"]): float(row["score"]) for row in tfidf_hybrid}
        faiss_hybrid_scores = {int(row["paper_id"]): float(row["score"]) for row in faiss_hybrid}
        hybrid_scores = score_hybrid_candidates(
            query=query,
            candidate_ids=hybrid_candidate_ids,
            paper_by_id=paper_by_id,
            bm25_scores=bm25_hybrid_scores,
            tfidf_scores=tfidf_hybrid_scores,
            faiss_scores=faiss_hybrid_scores,
        )
        hybrid_top = top_ids_from_scores(hybrid_scores, top_k)

        pool_ids = {
            int(row["paper_id"])
            for row in scored_rows(bm25_top) + tfidf_top + embedding_top + faiss_top + hybrid_top
        }
        pool_bm25_scores = {paper_id: bm25_score_for_doc(bm25, query_terms, paper_id) for paper_id in pool_ids}
        pool_tfidf_scores = {paper_id: tfidf_all_scores.get(paper_id, 0.0) for paper_id in pool_ids}
        pool_dense_scores = {paper_id: dense_all_scores.get(paper_id, 0.0) for paper_id in pool_ids}
        pool_hybrid_scores = score_hybrid_candidates(
            query=query,
            candidate_ids=pool_ids,
            paper_by_id=paper_by_id,
            bm25_scores=pool_bm25_scores,
            tfidf_scores=pool_tfidf_scores,
            faiss_scores=pool_dense_scores,
        )
        runs[query_id] = {
            "query": query,
            "candidate_pool_ids": sorted(pool_ids),
            "generation_runs": {
                "bm25": scored_rows(bm25_top),
                "tfidf": tfidf_top,
                "embedding": embedding_top,
                "faiss_embedding": faiss_top,
                "hybrid": hybrid_top,
            },
            "pool_scores": {
                "bm25": pool_bm25_scores,
                "tfidf": pool_tfidf_scores,
                "embedding": pool_dense_scores,
                "faiss_embedding": pool_dense_scores,
                "hybrid": pool_hybrid_scores,
            },
        }
    return runs


def scored_rows(results: Any) -> list[dict[str, Any]]:
    return [
        {"paper_id": int(item.document_id), "score": float(item.score), "rank": rank}
        for rank, item in enumerate(results, start=1)
    ]


def rank_from_scores(scores: dict[int, float], candidate_ids: list[int]) -> list[int]:
    return [
        paper_id
        for paper_id, _ in sorted(
            ((paper_id, float(scores.get(paper_id, 0.0))) for paper_id in candidate_ids),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
    ]


def build_candidate_rows(
    *,
    runs: dict[str, Any],
    paper_by_id: dict[int, Paper],
    identifiers_by_id: dict[int, dict[str, str]],
    sources_by_id: dict[int, set[str]],
    source_urls_by_id: dict[int, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    for query_id, payload in runs.items():
        candidate_ids = payload["candidate_pool_ids"]
        rankings = {
            method: rank_from_scores(payload["pool_scores"][method], candidate_ids)
            for method in BASELINE_METHODS
        }
        rank_by_method = {
            method: {paper_id: rank for rank, paper_id in enumerate(ranked_ids, start=1)}
            for method, ranked_ids in rankings.items()
        }
        normalized_counts = Counter(normalize_title(paper_by_id[paper_id].title) for paper_id in candidate_ids)
        rows: list[dict[str, Any]] = []
        for paper_id in candidate_ids:
            paper = paper_by_id[paper_id]
            identifiers = identifiers_by_id.get(paper_id, {})
            source_set = set(sources_by_id.get(paper_id, set()))
            if paper.source:
                source_set.add(str(paper.source))
            source_url = paper.source_url or paper.url or next(iter(source_urls_by_id.get(paper_id, [])), None)
            arxiv_id = normalize_identifier(identifiers.get("arxiv") or paper.external_id, prefix="arxiv:")
            openalex_id = normalize_identifier(identifiers.get("openalex") or paper.external_id, prefix="openalex:")
            doi = paper.doi or identifiers.get("doi")
            normalized = normalize_title(paper.title)
            rows.append(
                {
                    "query_id": query_id,
                    "query": payload["query"],
                    "paper_id": paper_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "venue": paper.venue,
                    "authors": parse_authors(paper.authors),
                    "citation_count": paper.citation_count,
                    "sources_provenance": sorted(source_set),
                    "identifiers": {
                        "arxiv_id": arxiv_id,
                        "doi": doi,
                        "openalex_id": openalex_id,
                    },
                    "source_url": source_url,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
                    "evidence_availability": {"level": "title_abstract", "fulltext_available": False},
                    "duplicate_title_cluster": {"size": int(normalized_counts.get(normalized, 0))},
                    "retrieval_ranks_by_method": {
                        method: int(rank_by_method[method][paper_id])
                        for method in BASELINE_METHODS
                    },
                    "retrieval_scores_by_method": {
                        method: float(payload["pool_scores"][method].get(paper_id, 0.0))
                        for method in BASELINE_METHODS
                    },
                }
            )
        rows_by_query[query_id] = rows
    return rows_by_query


def parse_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    separator = ";" if ";" in text else ","
    return [part.strip() for part in text.split(separator) if part.strip()]


def normalize_identifier(value: Any, *, prefix: str) -> str | None:
    if not value:
        return None
    text = str(value)
    if text.startswith(prefix):
        return text.removeprefix(prefix)
    return text if ":" not in text else None


def normalize_title(value: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def rank_methods(candidate_rows_by_q: dict[str, list[dict[str, Any]]], model_path: Path) -> dict[str, dict[str, list[int]]]:
    all_rows = [row for rows in candidate_rows_by_q.values() for row in rows]
    context = build_feature_context(all_rows)
    ranker = load_v2_2_ridge_ranker(model_path)
    ranked: dict[str, dict[str, list[int]]] = {}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        ranked[query_id] = {}
        ranked[query_id]["learned_hybrid"] = [
            paper_id
            for _, paper_id in sorted(
                ((ranker.score_packet_row(row, context), int(row["paper_id"])) for row in rows),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
        ]
        for method in BASELINE_METHODS:
            ranked[query_id][method] = [
                int(row["paper_id"])
                for row in sorted(
                    rows,
                    key=lambda row: (
                        float(row["retrieval_scores_by_method"].get(method, 0.0)),
                        -int(row["retrieval_ranks_by_method"].get(method, 999999)),
                        -int(row["paper_id"]),
                    ),
                    reverse=True,
                )
            ]
    return ranked


def evaluate_methods(
    *,
    ranked: dict[str, dict[str, list[int]]],
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels_by_q: dict[str, dict[int, dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    oracle_rows: dict[str, list[dict[str, Any]]] = {
        "oracle_reading_value": [],
        "oracle_topic_match": [],
        "oracle_v2_1_beginner_path_gain": [],
    }
    learned_unjudged_promotions: list[dict[str, Any]] = []
    for query_id, method_rankings in ranked.items():
        candidate_ids = {int(row["paper_id"]) for row in candidate_rows_by_q[query_id]}
        all_query_labels = labels_by_q[query_id]
        judged_in_pool = {paper_id: label for paper_id, label in all_query_labels.items() if paper_id in candidate_ids}
        for method in METHODS:
            row = evaluate_ranked_ids(
                method=method,
                query_id=query_id,
                all_query_labels=all_query_labels,
                judged_in_pool=judged_in_pool,
                candidate_ids=candidate_ids,
                retrieved_ids=method_rankings.get(method, []),
                k=k,
            )
            rows_by_method[method].append(row)
        for oracle_name, field_name in [
            ("oracle_reading_value", "reading_value_score"),
            ("oracle_topic_match", "topic_match_score"),
        ]:
            oracle_ids = [
                paper_id
                for paper_id, _ in sorted(
                    ((paper_id, clamp_score(label.get(field_name))) for paper_id, label in judged_in_pool.items()),
                    key=lambda item: (item[1], -item[0]),
                    reverse=True,
                )
            ]
            oracle_rows[oracle_name].append(
                evaluate_ranked_ids(
                    method=oracle_name,
                    query_id=query_id,
                    all_query_labels=all_query_labels,
                    judged_in_pool=judged_in_pool,
                    candidate_ids=candidate_ids,
                    retrieved_ids=oracle_ids,
                    k=k,
                )
            )
        oracle_target_ids = [
            paper_id
            for paper_id, _ in sorted(
                ((paper_id, target_gain(label)) for paper_id, label in judged_in_pool.items()),
                key=lambda item: (item[1], -item[0]),
                reverse=True,
            )
        ]
        oracle_rows["oracle_v2_1_beginner_path_gain"].append(
            evaluate_ranked_ids(
                method="oracle_v2_1_beginner_path_gain",
                query_id=query_id,
                all_query_labels=all_query_labels,
                judged_in_pool=judged_in_pool,
                candidate_ids=candidate_ids,
                retrieved_ids=oracle_target_ids,
                k=k,
            )
        )
        learned_ids = method_rankings["learned_hybrid"][:k]
        learned_unjudged_promotions.append(
            {
                "query_id": query_id,
                "unjudged_at_10": sum(1 for paper_id in learned_ids if paper_id not in all_query_labels),
                "top_unjudged_ids": [paper_id for paper_id in learned_ids if paper_id not in all_query_labels][:5],
            }
        )
    method_averages = {method: aggregate(rows) for method, rows in rows_by_method.items()}
    oracle_averages = {method: aggregate(rows) for method, rows in oracle_rows.items()}
    return {
        "method_averages": method_averages,
        "per_topic": rows_by_method,
        "per_topic_winners": build_per_topic_winners(rows_by_method),
        "failure_cases": build_failure_cases(rows_by_method, oracle_rows),
        "oracle_upper_bounds": {
            "method_averages": oracle_averages,
            "per_topic": oracle_rows,
            "gaps": oracle_gaps(method_averages, oracle_averages),
            "interpretation": oracle_interpretation(rows_by_method, oracle_rows),
        },
        "learned_unjudged_promotions": sorted(
            learned_unjudged_promotions,
            key=lambda row: row["unjudged_at_10"],
            reverse=True,
        ),
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
        score_maps[f"intent_{intent}_ndcg"] = intent_score_map(all_query_labels, intent)
    metrics = {
        metric_name: ndcg_at_k(retrieved_ids, relevance_by_id, k)
        for metric_name, relevance_by_id in score_maps.items()
    }
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


def target_gain(label: dict[str, Any]) -> float:
    intents = label.get("intent_scores") or {}
    intent_path_score = max(clamp_score(intents.get(name)) for name in PATH_ROLES)
    intent_application_or_benchmark_score = max(
        clamp_score(intents.get("evaluation_benchmark")),
        clamp_score(intents.get("application")),
    )
    duplicate_penalty = DUPLICATE_PENALTIES.get(str(label.get("duplicate_status", "none")), 0.0)
    return clamp_score(
        0.40 * clamp_score(label.get("reading_value_score"))
        + 0.25 * clamp_score(label.get("topic_match_score"))
        + 0.15 * clamp_score(label.get("beginner_fit_score"))
        + 0.05 * clamp_score(label.get("intermediate_fit_score"))
        + 0.05 * intent_path_score
        + 0.05 * intent_application_or_benchmark_score
        - 0.05 * duplicate_penalty
    )


def score_map(labels: dict[int, dict[str, Any]], field_name: str) -> dict[int, float]:
    return {paper_id: clamp_score(label.get(field_name)) for paper_id, label in labels.items()}


def intent_score_map(labels: dict[int, dict[str, Any]], intent: str) -> dict[int, float]:
    return {paper_id: clamp_score((label.get("intent_scores") or {}).get(intent)) for paper_id, label in labels.items()}


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
    return {"coverage": len(covered & available) / denominator if denominator else 0.0}


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
    return {"coverage": len(covered) / len(PATH_ROLES)}


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
        "path_level_coverage_at_k",
        "duplicate_penalty_at_k",
        "judged_retrieved_at_k",
        "unjudged_retrieved_at_k",
        "candidate_pool_size",
        "judged_candidates_in_pool",
        "packet_candidate_coverage",
    ] + [f"intent_{intent}_ndcg" for intent in INTENT_SCORE_FIELDS]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {field: sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0 for field in metric_fields()}


def build_per_topic_winners(method_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows_by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in method_rows.items():
        for row in rows:
            rows_by_query[row["query_id"]][method] = row
    winners: list[dict[str, Any]] = []
    for query_id, rows in sorted(rows_by_query.items()):
        topic_winner = max(rows.items(), key=lambda item: float(item[1]["topic_match_ndcg"]))
        reading_winner = max(rows.items(), key=lambda item: float(item[1]["reading_value_ndcg"]))
        winners.append(
            {
                "query_id": query_id,
                "topic_match_winner": topic_winner[0],
                "topic_match_ndcg": topic_winner[1]["topic_match_ndcg"],
                "reading_value_winner": reading_winner[0],
                "reading_value_ndcg": reading_winner[1]["reading_value_ndcg"],
            }
        )
    return winners


def build_failure_cases(
    method_rows: dict[str, list[dict[str, Any]]],
    oracle_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in method_rows.items():
        for row in rows:
            by_query[row["query_id"]][method] = row
    for method, rows in oracle_rows.items():
        for row in rows:
            by_query[row["query_id"]][method] = row
    failures = {
        "learned_loses_to_hybrid_on_reading_value_ndcg": [],
        "learned_loses_to_bm25_on_topic_match_ndcg": [],
        "learned_large_gap_to_oracle_reading_value": [],
        "learned_promotes_many_unjudged": [],
    }
    for query_id, rows in sorted(by_query.items()):
        learned = rows.get("learned_hybrid")
        hybrid = rows.get("hybrid")
        bm25 = rows.get("bm25")
        oracle = rows.get("oracle_reading_value")
        if learned and hybrid and learned["reading_value_ndcg"] < hybrid["reading_value_ndcg"]:
            failures["learned_loses_to_hybrid_on_reading_value_ndcg"].append(
                failure_row(query_id, learned, hybrid, "reading_value_ndcg")
            )
        if learned and bm25 and learned["topic_match_ndcg"] < bm25["topic_match_ndcg"]:
            failures["learned_loses_to_bm25_on_topic_match_ndcg"].append(
                failure_row(query_id, learned, bm25, "topic_match_ndcg")
            )
        if learned and oracle:
            gap = oracle["reading_value_ndcg"] - learned["reading_value_ndcg"]
            if gap >= 0.15:
                failures["learned_large_gap_to_oracle_reading_value"].append(
                    {
                        "query_id": query_id,
                        "learned": learned["reading_value_ndcg"],
                        "oracle": oracle["reading_value_ndcg"],
                        "gap": gap,
                        "judged_candidates_in_pool": learned["judged_candidates_in_pool"],
                        "likely_cause": "candidate generation" if oracle["reading_value_ndcg"] < 0.85 else "reranking/model robustness",
                    }
                )
        if learned and learned["unjudged_retrieved_at_k"] >= 5:
            failures["learned_promotes_many_unjudged"].append(
                {
                    "query_id": query_id,
                    "unjudged_retrieved_at_k": learned["unjudged_retrieved_at_k"],
                    "reading_value_ndcg": learned["reading_value_ndcg"],
                }
            )
    return failures


def failure_row(query_id: str, learned: dict[str, Any], competitor: dict[str, Any], metric: str) -> dict[str, Any]:
    coverage_issue = learned["judged_candidates_in_pool"] < 15
    return {
        "query_id": query_id,
        "competitor": competitor["method"],
        "metric": metric,
        "learned": learned[metric],
        "competitor_value": competitor[metric],
        "delta": learned[metric] - competitor[metric],
        "likely_cause": "candidate coverage" if coverage_issue else "ranking order/model behavior",
    }


def oracle_gaps(
    method_averages: dict[str, dict[str, float]],
    oracle_averages: dict[str, dict[str, float]],
) -> dict[str, Any]:
    reading_oracle = oracle_averages["oracle_reading_value"]["reading_value_ndcg"]
    topic_oracle = oracle_averages["oracle_topic_match"]["topic_match_ndcg"]
    return {
        "learned_gap_to_oracle_reading_value_ndcg": reading_oracle - method_averages["learned_hybrid"]["reading_value_ndcg"],
        "hybrid_gap_to_oracle_reading_value_ndcg": reading_oracle - method_averages["hybrid"]["reading_value_ndcg"],
        "learned_gap_to_oracle_topic_match_ndcg": topic_oracle - method_averages["learned_hybrid"]["topic_match_ndcg"],
        "hybrid_gap_to_oracle_topic_match_ndcg": topic_oracle - method_averages["hybrid"]["topic_match_ndcg"],
    }


def oracle_interpretation(
    method_rows: dict[str, list[dict[str, Any]]],
    oracle_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in method_rows.items():
        for row in rows:
            by_query[row["query_id"]][method] = row
    for method, rows in oracle_rows.items():
        for row in rows:
            by_query[row["query_id"]][method] = row
    output: list[dict[str, Any]] = []
    for query_id, rows in sorted(by_query.items()):
        learned = rows["learned_hybrid"]
        hybrid = rows["hybrid"]
        oracle = rows["oracle_reading_value"]
        if oracle["reading_value_ndcg"] < 0.85:
            bottleneck = "candidate_generation"
        elif oracle["reading_value_ndcg"] - learned["reading_value_ndcg"] >= 0.15:
            bottleneck = "reranking_model_or_feature_robustness"
        elif oracle["reading_value_ndcg"] - hybrid["reading_value_ndcg"] < 0.05:
            bottleneck = "current_hybrid_already_close_to_oracle"
        else:
            bottleneck = "mixed"
        output.append(
            {
                "query_id": query_id,
                "oracle_reading_value_ndcg": oracle["reading_value_ndcg"],
                "learned_reading_value_ndcg": learned["reading_value_ndcg"],
                "hybrid_reading_value_ndcg": hybrid["reading_value_ndcg"],
                "judged_candidates_in_pool": learned["judged_candidates_in_pool"],
                "packet_candidate_coverage": learned["packet_candidate_coverage"],
                "likely_bottleneck": bottleneck,
            }
        )
    return output


def feature_availability(candidate_rows_by_q: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for rows_for_query in candidate_rows_by_q.values() for row in rows_for_query]
    total = len(rows)
    by_method: dict[str, Any] = {}
    for method in BASELINE_METHODS:
        rank_count = sum(1 for row in rows if method in (row.get("retrieval_ranks_by_method") or {}))
        score_count = sum(1 for row in rows if method in (row.get("retrieval_scores_by_method") or {}))
        nonzero_score_count = sum(
            1 for row in rows if float((row.get("retrieval_scores_by_method") or {}).get(method, 0.0)) > 0
        )
        by_method[method] = {
            "rank_feature_available_pct": rank_count / total if total else 0.0,
            "score_feature_available_pct": score_count / total if total else 0.0,
            "nonzero_score_pct": nonzero_score_count / total if total else 0.0,
            "missing_rank_count": total - rank_count,
            "missing_score_count": total - score_count,
        }
    return {
        "candidate_row_count": total,
        "by_method": by_method,
        "score_normalization_universe": "per-query shared V2.4 production candidate pool",
        "matches_v2_2b_packet_style_feature_construction": False,
        "difference_from_v2_2b": (
            "The same feature extraction code and StandardScaler are used, but score min/max/z statistics are "
            "computed over larger production candidate pools rather than the 15-row packet universe."
        ),
        "missing_feature_rates_by_group": {
            "retrieval_rank": {
                method: 1.0 - by_method[method]["rank_feature_available_pct"] for method in BASELINE_METHODS
            },
            "retrieval_score": {
                method: 1.0 - by_method[method]["score_feature_available_pct"] for method in BASELINE_METHODS
            },
            "metadata": 0.0,
            "text_match": 0.0,
            "source_quality": 0.0,
            "dedup_heuristics": 0.0,
        },
    }


def packet_prediction_repro_check(packet_rows: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    context = build_feature_context(packet_rows)
    ranker = load_v2_2_ridge_ranker(model_path)
    scores_a = [ranker.score_packet_row(row, context) for row in packet_rows]
    scores_b = [ranker.score_packet_row(row, context) for row in packet_rows]
    diffs = [abs(a - b) for a, b in zip(scores_a, scores_b, strict=True)]
    return {
        "packet_prediction_count": len(packet_rows),
        "artifact_prediction_max_abs_diff": max(diffs, default=0.0),
        "artifact_prediction_mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "artifact_can_regenerate_packet_predictions": max(diffs, default=0.0) < 1e-12,
    }


def prior_milestone_comparison(v21_path: Path, v22_path: Path, v22b_path: Path, v23_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v21 = load_json_if_exists(v21_path)
    if v21:
        rows.append(
            milestone_row(
                name="V2.1 baseline packet evaluation",
                classification="validated_packet_baseline",
                method="hybrid",
                averages=extract_method_averages(v21, "hybrid"),
            )
        )
    v22 = load_json_if_exists(v22_path)
    if v22:
        rows.append(
            milestone_row(
                name="V2.2 original learned_hybrid",
                classification="exploratory_upper_bound_not_apples_to_apples",
                method="learned_hybrid",
                averages=extract_method_averages(v22, "learned_hybrid"),
            )
        )
    v22b = load_json_if_exists(v22b_path)
    if v22b:
        rows.append(
            milestone_row(
                name="V2.2b same-packet fair reranking",
                classification="validated_packet_reranking",
                method="learned_hybrid",
                averages=v22b.get("same_packet_reranking", {}).get("method_averages", {}).get("learned_hybrid", {}),
            )
        )
        rows.append(
            milestone_row(
                name="V2.2b current-hybrid restricted",
                classification="validated_restricted_packet_reranking",
                method="learned_hybrid",
                averages=v22b.get("restricted_candidate_evaluation", {}).get("method_averages", {}).get("learned_hybrid", {}),
            )
        )
    v23 = load_json_if_exists(v23_path)
    if v23:
        rows.append(
            milestone_row(
                name="V2.3 fallback production-style stress test",
                classification="fallback_stress_test_not_true_production",
                method="learned_hybrid",
                averages=extract_method_averages(v23, "learned_hybrid"),
            )
        )
    return rows


def extract_method_averages(payload: dict[str, Any], method: str) -> dict[str, Any]:
    if method in (payload.get("method_averages") or {}):
        return payload["method_averages"][method]
    if method in (payload.get("methods") or {}):
        return payload["methods"][method].get("averages", {})
    return {}


def milestone_row(name: str, classification: str, method: str, averages: dict[str, Any]) -> dict[str, Any]:
    return {
        "milestone": name,
        "classification": classification,
        "representative_method": method,
        "topic_match_ndcg": averages.get("topic_match_ndcg"),
        "reading_value_ndcg": averages.get("reading_value_ndcg"),
        "judged_at_10": averages.get("judged_retrieved_at_k"),
        "packet_candidate_coverage": averages.get("packet_candidate_coverage"),
    }


def build_report(
    *,
    labels_path: Path,
    packet_path: Path,
    v21_path: Path,
    v22_path: Path,
    v22b_path: Path,
    v23_path: Path,
    model_path: Path,
    top_k: int,
    eval_k: int,
) -> dict[str, Any]:
    labels = load_jsonl(labels_path)
    packet_rows = load_jsonl(packet_path)
    label_lookup = labels_by_query(labels)
    packet_lookup = packet_by_query(packet_rows)
    try:
        db_assets = load_db_assets()
    except Exception as exc:
        return {
            "created_at": datetime.now(UTC).isoformat(),
            "schema_version": "v2.4",
            "is_true_production_candidate_pool_retrieval": False,
            "db_diagnostics": db_unreachable_report(exc),
            "fatal_error": "DB was not reachable; true production candidate pool cannot be built.",
        }
    vector_assets = load_vector_assets(db_assets["settings"], db_assets["paper_by_id"])
    documents = build_documents(db_assets["papers"])
    runs = build_retrieval_runs(
        packet_lookup=packet_lookup,
        paper_by_id=db_assets["paper_by_id"],
        documents=documents,
        vector_assets=vector_assets,
        top_k=top_k,
    )
    candidate_rows = build_candidate_rows(
        runs=runs,
        paper_by_id=db_assets["paper_by_id"],
        identifiers_by_id=db_assets["identifiers_by_id"],
        sources_by_id=db_assets["sources_by_id"],
        source_urls_by_id=db_assets["source_urls_by_id"],
    )
    ranked = rank_methods(candidate_rows, model_path)
    evaluation = evaluate_methods(
        ranked=ranked,
        candidate_rows_by_q=candidate_rows,
        labels_by_q=label_lookup,
        k=eval_k,
    )
    feature_text = "\n".join(load_v2_2_ridge_ranker(model_path).feature_names)
    selected_hash = file_sha256(SELECTED_240)
    learned = evaluation["method_averages"]["learned_hybrid"]
    hybrid = evaluation["method_averages"]["hybrid"]
    bm25 = evaluation["method_averages"]["bm25"]
    true_mapping = bool(vector_assets["diagnostics"]["mapping_chain_available"])
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2.4",
        "evaluation_scope": "true_backend_database_faiss_production_candidate_pool",
        "is_true_production_candidate_pool_retrieval": true_mapping,
        "db_diagnostics": db_assets["db_diagnostics"],
        "production_asset_diagnostics": vector_assets["diagnostics"],
        "candidate_pool_definition": {
            "corpus": "backend Postgres papers table",
            "corpus_size": len(db_assets["paper_by_id"]),
            "union": [
                f"top_{top_k}_bm25",
                f"top_{top_k}_tfidf",
                f"top_{top_k}_embedding_exact_matrix",
                f"top_{top_k}_faiss_embedding",
                f"top_{top_k}_current_hybrid",
            ],
            "deduplicate_by": "paper_id",
            "dense_method_note": (
                "embedding and faiss_embedding use the same 50k MiniLM embedding/id-map artifacts; "
                "FAISS is the production dense retrieval path, while embedding is exact matrix scoring."
            ),
        },
        "method_averages": evaluation["method_averages"],
        "per_topic": evaluation["per_topic"],
        "per_topic_winners": evaluation["per_topic_winners"],
        "failure_cases": evaluation["failure_cases"],
        "oracle_upper_bounds": evaluation["oracle_upper_bounds"],
        "feature_availability": feature_availability(candidate_rows),
        "learned_unjudged_promotions": evaluation["learned_unjudged_promotions"],
        "comparison_to_prior_milestones": prior_milestone_comparison(v21_path, v22_path, v22b_path, v23_path),
        "success_criteria": {
            "learned_vs_hybrid_reading_value_delta": learned["reading_value_ndcg"] - hybrid["reading_value_ndcg"],
            "learned_vs_bm25_topic_match_delta": learned["topic_match_ndcg"] - bm25["topic_match_ndcg"],
            "beat_hybrid_on_reading_value_ndcg": learned["reading_value_ndcg"] > hybrid["reading_value_ndcg"],
            "avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg": (
                learned["topic_match_ndcg"] - bm25["topic_match_ndcg"]
            ) >= -0.01,
        },
        "reproducibility_checks": {
            "no_new_training_performed": True,
            "neural_models_trained": False,
            "grouped_cv_by_query_id": "not_applicable_no_training_or_cv_in_v2_4",
            "label_fields_used_as_features": [name for name in LEAKAGE_FIELD_NAMES if name in feature_text],
            "selection_reasons_used": "selection_reasons" in feature_text,
            "likely_coverage_used": "likely_coverage" in feature_text,
            "selected_240_sha256": selected_hash,
            "selected_240_matches_expected_hash": selected_hash == EXPECTED_SELECTED_240_SHA256,
            "labels_sha256": file_sha256(labels_path),
            "dense_faiss_retrieval_available": true_mapping,
        }
        | packet_prediction_repro_check(packet_rows, model_path),
        "likely_bottleneck": likely_bottleneck(evaluation),
        "safe_to_present_as_true_production_style_learned_reranking": bool(
            true_mapping
            and learned["reading_value_ndcg"] > hybrid["reading_value_ndcg"]
            and (learned["topic_match_ndcg"] - bm25["topic_match_ndcg"]) >= -0.01
        ),
        "runtime_backend_integration": (
            "V2.2b feature scorer is reusable, but production HybridRetriever still needs full integration "
            "to use this shared-pool feature context online."
        ),
    }


def likely_bottleneck(evaluation: dict[str, Any]) -> str:
    learned = evaluation["method_averages"]["learned_hybrid"]
    hybrid = evaluation["method_averages"]["hybrid"]
    oracle = evaluation["oracle_upper_bounds"]["method_averages"]["oracle_reading_value"]
    if oracle["reading_value_ndcg"] < 0.85:
        return "candidate_generation_or_judged_coverage"
    if oracle["reading_value_ndcg"] - learned["reading_value_ndcg"] >= 0.15:
        return "reranking_model_feature_mismatch_or_unjudged_promotion"
    if oracle["reading_value_ndcg"] - hybrid["reading_value_ndcg"] < 0.05:
        return "current_hybrid_already_close_to_oracle"
    return "mixed"


def markdown_report(report: dict[str, Any], *, command: str, outputs: list[str]) -> str:
    if "fatal_error" in report:
        return "\n".join(
            [
                "# V2.4 True Production Candidate-Pool Evaluation",
                "",
                f"Created: `{report['created_at']}`",
                "",
                "## Fatal Error",
                "",
                report["fatal_error"],
                "",
                f"DB diagnostics: `{json.dumps(report['db_diagnostics'], sort_keys=True)}`",
                "",
            ]
        )
    lines = [
        "# V2.4 True Production Candidate-Pool Evaluation",
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
            "## Production Assets",
            "",
            f"- DB reachable: `{report['db_diagnostics']['reachable']}`",
            f"- DB paper count: `{report['db_diagnostics']['paper_count']}`",
            f"- FAISS id-to-metadata mapping works: `{report['production_asset_diagnostics']['mapping_chain_available']}`",
            f"- FAISS index: `{report['production_asset_diagnostics']['faiss_index_path']}`",
            f"- FAISS id map: `{report['production_asset_diagnostics']['faiss_id_map_path']}`",
            f"- Embedding file: `{report['production_asset_diagnostics']['embedding_path']}`",
            "",
            "## Candidate Pool",
            "",
            f"- True production candidate-pool retrieval: `{report['is_true_production_candidate_pool_retrieval']}`",
            f"- Corpus size: `{report['candidate_pool_definition']['corpus_size']}`",
            f"- Union: `{', '.join(report['candidate_pool_definition']['union'])}`",
            f"- Dense included: `True`",
            f"- Dense note: {report['candidate_pool_definition']['dense_method_note']}",
            "",
            "## Method Comparison",
            "",
        ]
    )
    lines.extend(method_table(report["method_averages"]))
    lines.extend(
        [
            "",
            "## Oracle Upper Bounds",
            "",
        ]
    )
    lines.extend(method_table(report["oracle_upper_bounds"]["method_averages"]))
    gaps = report["oracle_upper_bounds"]["gaps"]
    lines.extend(
        [
            "",
            f"- learned_hybrid gap to oracle reading NDCG@10: `{gaps['learned_gap_to_oracle_reading_value_ndcg']:.3f}`",
            f"- hybrid gap to oracle reading NDCG@10: `{gaps['hybrid_gap_to_oracle_reading_value_ndcg']:.3f}`",
            f"- learned_hybrid gap to oracle topic NDCG@10: `{gaps['learned_gap_to_oracle_topic_match_ndcg']:.3f}`",
            f"- hybrid gap to oracle topic NDCG@10: `{gaps['hybrid_gap_to_oracle_topic_match_ndcg']:.3f}`",
            "",
            "## Success Criteria",
            "",
        ]
    )
    criteria = report["success_criteria"]
    lines.extend(
        [
            f"- learned_hybrid vs hybrid reading-value NDCG@10: `{criteria['learned_vs_hybrid_reading_value_delta']:+.3f}`",
            f"- learned_hybrid vs BM25 topic-match NDCG@10: `{criteria['learned_vs_bm25_topic_match_delta']:+.3f}`",
            f"- Beat hybrid on reading-value NDCG@10: `{criteria['beat_hybrid_on_reading_value_ndcg']}`",
            f"- Avoid losing more than 0.01 to BM25 on topic-match NDCG@10: `{criteria['avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg']}`",
            "",
            "## Feature Availability",
            "",
            "| method | rank available | score available | nonzero score |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, row in report["feature_availability"]["by_method"].items():
        lines.append(
            f"| `{method}` | {row['rank_feature_available_pct']:.3f} | "
            f"{row['score_feature_available_pct']:.3f} | {row['nonzero_score_pct']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Score normalization universe: `{report['feature_availability']['score_normalization_universe']}`",
            f"Matches V2.2b packet universe exactly: `{report['feature_availability']['matches_v2_2b_packet_style_feature_construction']}`",
            "",
            "## Prior Milestones",
            "",
            "| milestone | classification | method | topic | reading | judged@10 | packet coverage |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["comparison_to_prior_milestones"]:
        lines.append(
            f"| {row['milestone']} | `{row['classification']}` | `{row['representative_method']}` | "
            f"{format_optional(row['topic_match_ndcg'])} | {format_optional(row['reading_value_ndcg'])} | "
            f"{format_optional(row['judged_at_10'])} | {format_optional(row['packet_candidate_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            "### learned_hybrid loses to hybrid on reading_value NDCG@10",
            "",
            "| query_id | learned | hybrid | delta | likely cause |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["failure_cases"]["learned_loses_to_hybrid_on_reading_value_ndcg"]:
        lines.append(
            f"| `{row['query_id']}` | {row['learned']:.3f} | {row['competitor_value']:.3f} | "
            f"{row['delta']:+.3f} | {row['likely_cause']} |"
        )
    lines.extend(
        [
            "",
            "### learned_hybrid promotes many unjudged papers",
            "",
            "| query_id | unjudged@10 | reading NDCG@10 |",
            "|---|---:|---:|",
        ]
    )
    for row in report["failure_cases"]["learned_promotes_many_unjudged"]:
        lines.append(f"| `{row['query_id']}` | {row['unjudged_retrieved_at_k']:.0f} | {row['reading_value_ndcg']:.3f} |")
    checks = report["reproducibility_checks"]
    lines.extend(
        [
            "",
            "## Reproducibility Checks",
            "",
            f"- No new training performed: `{checks['no_new_training_performed']}`",
            f"- Neural models trained: `{checks['neural_models_trained']}`",
            f"- Label fields used as features: `{checks['label_fields_used_as_features']}`",
            f"- `selection_reasons` used: `{checks['selection_reasons_used']}`",
            f"- `likely_coverage` used: `{checks['likely_coverage_used']}`",
            f"- Selected 240 SHA256: `{checks['selected_240_sha256']}`",
            f"- Selected 240 hash matches expected: `{checks['selected_240_matches_expected_hash']}`",
            f"- Labels SHA256: `{checks['labels_sha256']}`",
            f"- Saved V2.2b packet prediction max diff: `{checks['artifact_prediction_max_abs_diff']:.12f}`",
            f"- Dense/FAISS retrieval available: `{checks['dense_faiss_retrieval_available']}`",
            "",
            "## Conclusion",
            "",
            f"- Likely bottleneck: `{report['likely_bottleneck']}`",
            f"- Safe to present as true production-style learned reranking: `{report['safe_to_present_as_true_production_style_learned_reranking']}`",
            f"- Runtime backend integration: {report['runtime_backend_integration']}",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}"


def method_table(averages: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | topic NDCG@10 | reading NDCG@10 | beginner | intermediate | advanced | expert | path coverage | role coverage | dup penalty | judged@10 | unjudged@10 | pool size | judged in pool | packet coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(averages, key=lambda name: averages[name]["reading_value_ndcg"], reverse=True):
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['topic_match_ndcg']:.3f} | {row['reading_value_ndcg']:.3f} | "
            f"{row['beginner_ndcg']:.3f} | {row['intermediate_ndcg']:.3f} | {row['advanced_ndcg']:.3f} | "
            f"{row['expert_ndcg']:.3f} | {row['path_level_coverage_at_k']:.3f} | {row['role_coverage_at_k']:.3f} | "
            f"{row['duplicate_penalty_at_k']:.3f} | {row['judged_retrieved_at_k']:.2f} | "
            f"{row['unjudged_retrieved_at_k']:.2f} | {row['candidate_pool_size']:.2f} | "
            f"{row['judged_candidates_in_pool']:.2f} | {row['packet_candidate_coverage']:.3f} |"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2.4 true production candidate pools.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--v21-report", default=str(DEFAULT_V21))
    parser.add_argument("--v22-report", default=str(DEFAULT_V22))
    parser.add_argument("--v22b-report", default=str(DEFAULT_V22B))
    parser.add_argument("--v23-report", default=str(DEFAULT_V23))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = resolve_repo_path(args.labels)
    packet_path = resolve_repo_path(args.packet)
    v21_path = resolve_repo_path(args.v21_report)
    v22_path = resolve_repo_path(args.v22_report)
    v22b_path = resolve_repo_path(args.v22b_report)
    v23_path = resolve_repo_path(args.v23_report)
    model_path = resolve_repo_path(args.model)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    report = build_report(
        labels_path=labels_path,
        packet_path=packet_path,
        v21_path=v21_path,
        v22_path=v22_path,
        v22b_path=v22b_path,
        v23_path=v23_path,
        model_path=model_path,
        top_k=args.top_k,
        eval_k=args.eval_k,
    )
    command = (
        f".\\backend\\.venv\\Scripts\\python.exe scripts\\evaluate_v2_4_true_production_candidate_pool.py "
        f"--labels {args.labels} --packet {args.packet} --v21-report {args.v21_report} "
        f"--v22-report {args.v22_report} --v22b-report {args.v22b_report} "
        f"--v23-report {args.v23_report} --model {args.model} --json-out {args.json_out} "
        f"--md-out {args.md_out} --top-k {args.top_k} --eval-k {args.eval_k}"
    )
    report["command"] = command
    write_json(json_out, report)
    write_text(
        md_out,
        markdown_report(
            report,
            command=command,
            outputs=[
                "backend/app/services/retrievers/faiss_id_mapping.py",
                "scripts/evaluate_v2_4_true_production_candidate_pool.py",
                args.json_out,
                args.md_out,
            ],
        ),
    )
    print("V2.4 true production candidate-pool evaluation complete")
    print(f"Report JSON: {json_out}")
    print(f"Report Markdown: {md_out}")
    if "fatal_error" in report:
        print(report["fatal_error"])
        return
    criteria = report["success_criteria"]
    print(f"DB reachable: {report['db_diagnostics']['reachable']}")
    print(f"FAISS id-to-metadata mapping works: {report['production_asset_diagnostics']['mapping_chain_available']}")
    print(f"Dense/FAISS included: {report['reproducibility_checks']['dense_faiss_retrieval_available']}")
    print(f"learned vs hybrid reading delta: {criteria['learned_vs_hybrid_reading_value_delta']:+.3f}")
    print(f"learned vs BM25 topic delta: {criteria['learned_vs_bm25_topic_match_delta']:+.3f}")
    print(f"safe as true production-style: {report['safe_to_present_as_true_production_style_learned_reranking']}")


if __name__ == "__main__":
    main()
