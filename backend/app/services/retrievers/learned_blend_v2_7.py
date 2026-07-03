from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.models.paper import Paper
from app.services.bm25 import BM25Retriever, tokenize as bm25_tokenize
from app.services.difficulty import score_difficulty
from app.services.quality_signals import compute_quality_signals, hybrid_score, normalize_scores
from app.services.retrievers.base import PaperDocument, ScoredDocument, papers_to_documents
from app.services.retrievers.embedding import EmbeddingRetriever, normalize_rows
from app.services.retrievers.faiss import FaissRetriever
from app.services.retrievers.tfidf import TfidfRetriever
from app.services.v2_2_learned_ranker import V22RidgeRanker, build_feature_context, load_v2_2_ridge_ranker


BASELINE_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
V27_METHOD_NAME = "learned_blend_v2_7"
V27_SELECTED_BLEND_WEIGHTS = {
    "old_v2_2b_score": 0.60,
    "v2_6_score": 0.20,
    "hybrid_score": 0.10,
    "embedding_score": 0.10,
}


@dataclass(frozen=True)
class LearnedBlendV27ScoredDocument(ScoredDocument):
    components: dict[str, Any]


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def top_ids_from_scores(scores: dict[int, float], k: int) -> list[dict[str, Any]]:
    ranked = sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)[:k]
    return [
        {"paper_id": int(paper_id), "score": float(score), "rank": rank}
        for rank, (paper_id, score) in enumerate(ranked, start=1)
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


class LearnedBlendV27Retriever:
    method_name = V27_METHOD_NAME

    def __init__(
        self,
        papers: list[Paper],
        *,
        embedding_index_path: Path | None = None,
        faiss_index_path: Path | None = None,
        faiss_id_map_path: Path | None = None,
        old_model_path: Path | None = None,
        v26_model_path: Path | None = None,
        blend_artifact_path: Path | None = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        exact_embedding_retriever: EmbeddingRetriever | None = None,
        faiss_retriever: FaissRetriever | None = None,
        old_ranker: V22RidgeRanker | None = None,
        v26_ranker: V22RidgeRanker | None = None,
        current_year: int = 2026,
    ) -> None:
        self.papers = papers
        self.paper_by_id = {paper.id: paper for paper in papers}
        self.documents = papers_to_documents(papers)
        self.bm25 = BM25Retriever(self.documents)
        self.tfidf = TfidfRetriever(self.documents)
        self.embedding = exact_embedding_retriever or self._load_embedding(
            embedding_index_path=embedding_index_path,
            embedding_model_name=embedding_model_name,
        )
        self.faiss = faiss_retriever or self._load_faiss(
            faiss_index_path=faiss_index_path,
            faiss_id_map_path=faiss_id_map_path,
            embedding_model_name=embedding_model_name,
        )
        self.old_ranker = old_ranker or self._load_ranker(old_model_path, "old V2.2b")
        self.v26_ranker = v26_ranker or self._load_ranker(v26_model_path, "V2.6")
        if blend_artifact_path is not None:
            self._validate_blend_artifact(blend_artifact_path)
        self.current_year = current_year
        self.last_components: dict[int, dict[str, Any]] = {}
        self.last_candidate_pool_ids: list[int] = []

    def fit(self, papers: list[PaperDocument]) -> None:
        self.documents = papers
        self.bm25.fit(papers)
        self.tfidf.fit(papers)

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        debug = self.score_query(query=query, top_k=50, exclude_ids=exclude_ids)
        ranked = sorted(debug["scores"].items(), key=lambda item: (item[1]["v2_7_score"], -item[0]), reverse=True)
        results: list[LearnedBlendV27ScoredDocument] = []
        self.last_components = {}
        self.last_candidate_pool_ids = list(debug["candidate_pool_ids"])
        for paper_id, values in ranked[:k]:
            components = dict(values)
            self.last_components[paper_id] = components
            results.append(
                LearnedBlendV27ScoredDocument(
                    document_id=paper_id,
                    score=float(values["v2_7_score"]),
                    components=components,
                )
            )
        return results

    def components_for(self, paper_id: int) -> dict[str, Any]:
        return self.last_components.get(paper_id, {})

    def score_query(
        self,
        *,
        query: str,
        top_k: int = 50,
        exclude_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        if not query.strip() or not self.papers:
            return {"query": query, "candidate_pool_ids": [], "rows": [], "scores": {}}
        exclude_ids = exclude_ids or set()
        runs = self._build_retrieval_run(query=query, top_k=top_k, exclude_ids=exclude_ids)
        rows = self._build_candidate_rows(query=query, run=runs)
        context = build_feature_context(rows)
        raw_baselines: dict[str, dict[int, float]] = {method: {} for method in BASELINE_METHODS}
        for row in rows:
            paper_id = int(row["paper_id"])
            for method in BASELINE_METHODS:
                raw_baselines[method][paper_id] = float(row["retrieval_scores_by_method"].get(method, 0.0))
        normalized = {method: minmax(values) for method, values in raw_baselines.items()}
        scores: dict[int, dict[str, float]] = {}
        for row in rows:
            paper_id = int(row["paper_id"])
            old_score = float(self.old_ranker.score_packet_row(row, context))
            v26_score = float(self.v26_ranker.score_packet_row(row, context))
            normalized_hybrid = normalized["hybrid"].get(paper_id, 0.0)
            normalized_embedding = normalized["embedding"].get(paper_id, 0.0)
            v27_score = (
                V27_SELECTED_BLEND_WEIGHTS["old_v2_2b_score"] * old_score
                + V27_SELECTED_BLEND_WEIGHTS["v2_6_score"] * v26_score
                + V27_SELECTED_BLEND_WEIGHTS["hybrid_score"] * normalized_hybrid
                + V27_SELECTED_BLEND_WEIGHTS["embedding_score"] * normalized_embedding
            )
            scores[paper_id] = {
                "old_v2_2b_score": old_score,
                "v2_6_score": v26_score,
                "hybrid_score": normalized_hybrid,
                "embedding_score": normalized_embedding,
                "hybrid_raw_score": raw_baselines["hybrid"].get(paper_id, 0.0),
                "embedding_raw_score": raw_baselines["embedding"].get(paper_id, 0.0),
                "v2_7_score": float(v27_score),
            }
        return {
            "query": query,
            "candidate_pool_ids": runs["candidate_pool_ids"],
            "generation_runs": runs["generation_runs"],
            "pool_scores": runs["pool_scores"],
            "rows": rows,
            "scores": scores,
        }

    def _build_retrieval_run(self, *, query: str, top_k: int, exclude_ids: set[int]) -> dict[str, Any]:
        bm25_all_scores = self._bm25_scores_for_query(query, exclude_ids=exclude_ids)
        bm25_top = top_ids_from_scores(bm25_all_scores, top_k)
        tfidf_all_scores = self._tfidf_scores_for_query(query, exclude_ids=exclude_ids)
        tfidf_top = top_ids_from_scores(tfidf_all_scores, top_k)
        dense_all_scores = self._dense_scores_for_query(query, exclude_ids=exclude_ids)
        embedding_top = top_ids_from_scores(dense_all_scores, top_k)
        faiss_top = self._faiss_top(query=query, top_k=top_k, exclude_ids=exclude_ids) or list(embedding_top)

        hybrid_search_k = min(len(self.documents), max(top_k * 8, 50))
        bm25_hybrid = top_ids_from_scores(bm25_all_scores, hybrid_search_k)
        tfidf_hybrid = top_ids_from_scores(tfidf_all_scores, hybrid_search_k)
        dense_hybrid = top_ids_from_scores(dense_all_scores, hybrid_search_k)
        hybrid_candidate_ids = {
            int(row["paper_id"])
            for row in bm25_hybrid + tfidf_hybrid + dense_hybrid
        }
        hybrid_scores = self._score_hybrid_candidates(
            query=query,
            candidate_ids=hybrid_candidate_ids,
            bm25_scores={int(row["paper_id"]): float(row["score"]) for row in bm25_hybrid},
            tfidf_scores={int(row["paper_id"]): float(row["score"]) for row in tfidf_hybrid},
            dense_scores={int(row["paper_id"]): float(row["score"]) for row in dense_hybrid},
        )
        hybrid_top = top_ids_from_scores(hybrid_scores, top_k)

        pool_ids = {
            int(row["paper_id"])
            for row in bm25_top + tfidf_top + embedding_top + faiss_top + hybrid_top
        }
        pool_bm25_scores = {paper_id: bm25_all_scores.get(paper_id, 0.0) for paper_id in pool_ids}
        pool_tfidf_scores = {paper_id: tfidf_all_scores.get(paper_id, 0.0) for paper_id in pool_ids}
        pool_dense_scores = {paper_id: dense_all_scores.get(paper_id, 0.0) for paper_id in pool_ids}
        pool_hybrid_scores = self._score_hybrid_candidates(
            query=query,
            candidate_ids=pool_ids,
            bm25_scores=pool_bm25_scores,
            tfidf_scores=pool_tfidf_scores,
            dense_scores=pool_dense_scores,
        )
        return {
            "query": query,
            "candidate_pool_ids": sorted(pool_ids),
            "generation_runs": {
                "bm25": bm25_top,
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

    def _build_candidate_rows(self, *, query: str, run: dict[str, Any]) -> list[dict[str, Any]]:
        candidate_ids = list(run["candidate_pool_ids"])
        rankings = {
            method: rank_from_scores(run["pool_scores"][method], candidate_ids)
            for method in BASELINE_METHODS
        }
        rank_by_method = {
            method: {paper_id: rank for rank, paper_id in enumerate(ranked_ids, start=1)}
            for method, ranked_ids in rankings.items()
        }
        rows: list[dict[str, Any]] = []
        for paper_id in candidate_ids:
            paper = self.paper_by_id[paper_id]
            arxiv_id = self._identifier_from_external_id(paper.external_id, prefix="arxiv:")
            openalex_id = self._identifier_from_external_id(paper.external_id, prefix="openalex:")
            rows.append(
                {
                    "query_id": "runtime",
                    "query": query,
                    "paper_id": paper_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "venue": paper.venue,
                    "authors": self._parse_authors(paper.authors),
                    "citation_count": paper.citation_count,
                    "sources_provenance": [paper.source] if paper.source else [],
                    "identifiers": {
                        "arxiv_id": arxiv_id,
                        "doi": paper.doi,
                        "openalex_id": openalex_id,
                    },
                    "source_url": paper.source_url or paper.url,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
                    "evidence_availability": {"level": "title_abstract", "fulltext_available": False},
                    "duplicate_title_cluster": {"size": 1},
                    "retrieval_ranks_by_method": {
                        method: int(rank_by_method[method][paper_id])
                        for method in BASELINE_METHODS
                    },
                    "retrieval_scores_by_method": {
                        method: float(run["pool_scores"][method].get(paper_id, 0.0))
                        for method in BASELINE_METHODS
                    },
                }
            )
        return rows

    def _bm25_scores_for_query(self, query: str, *, exclude_ids: set[int]) -> dict[int, float]:
        query_terms = bm25_tokenize(query)
        scores: dict[int, float] = {}
        for document in self.documents:
            doc_id = int(document.document_id)
            if doc_id in exclude_ids or doc_id not in self.bm25.term_frequencies:
                continue
            doc_length = self.bm25.doc_lengths[doc_id]
            term_frequency = self.bm25.term_frequencies[doc_id]
            score = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.bm25.k1 * (
                    1 - self.bm25.b + self.bm25.b * doc_length / max(self.bm25.avg_doc_length, 1e-9)
                )
                score += self.bm25._idf(term) * (frequency * (self.bm25.k1 + 1)) / denominator
            if score > 0:
                scores[doc_id] = float(score)
        return scores

    def _tfidf_scores_for_query(self, query: str, *, exclude_ids: set[int]) -> dict[int, float]:
        if self.tfidf.matrix is None:
            return {}
        query_vector = self.tfidf.vectorizer.transform([query])
        scores = (self.tfidf.matrix @ query_vector.T).toarray().ravel()
        return {
            int(document.document_id): float(scores[index])
            for index, document in enumerate(self.tfidf.documents)
            if int(document.document_id) not in exclude_ids and float(scores[index]) > 0
        }

    def _dense_scores_for_query(self, query: str, *, exclude_ids: set[int]) -> dict[int, float]:
        if self.embedding is None or self.embedding.embeddings is None or len(self.embedding.document_ids) == 0:
            return {}
        query_embedding = self.embedding._encoder().encode(
            query,
            batch_size=self.embedding.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_embedding = normalize_rows(query_embedding)[0]
        scores = self.embedding.embeddings @ query_embedding
        return {
            int(paper_id): float(scores[position])
            for position, paper_id in enumerate(self.embedding.document_ids.tolist())
            if int(paper_id) in self.paper_by_id and int(paper_id) not in exclude_ids
        }

    def _faiss_top(self, *, query: str, top_k: int, exclude_ids: set[int]) -> list[dict[str, Any]]:
        if self.faiss is None:
            return []
        return [
            {"paper_id": int(item.document_id), "score": float(item.score), "rank": rank}
            for rank, item in enumerate(self.faiss.search(query, k=top_k, exclude_ids=exclude_ids), start=1)
        ]

    def _score_hybrid_candidates(
        self,
        *,
        query: str,
        candidate_ids: set[int],
        bm25_scores: dict[int, float],
        tfidf_scores: dict[int, float],
        dense_scores: dict[int, float],
    ) -> dict[int, float]:
        bm25_norm = normalize_scores(bm25_scores)
        tfidf_norm = normalize_scores(tfidf_scores)
        dense_norm = normalize_scores(dense_scores)
        scored: dict[int, float] = {}
        for paper_id in candidate_ids:
            paper = self.paper_by_id.get(paper_id)
            if paper is None:
                continue
            difficulty = score_difficulty(paper, current_year=self.current_year)
            signals = compute_quality_signals(
                query=query,
                paper=paper,
                difficulty=difficulty,
                background_level="basic_ml",
                bm25_score=bm25_norm.get(paper_id, 0.0),
                tfidf_score=tfidf_norm.get(paper_id, 0.0),
                faiss_score=dense_norm.get(paper_id, 0.0),
                current_year=self.current_year,
            )
            scored[paper_id] = float(hybrid_score(signals))
        return scored

    def _load_embedding(
        self,
        *,
        embedding_index_path: Path | None,
        embedding_model_name: str,
    ) -> EmbeddingRetriever | None:
        if embedding_index_path is None or not embedding_index_path.exists():
            return None
        retriever = EmbeddingRetriever(model_name=embedding_model_name)
        retriever.load_index(embedding_index_path)
        return retriever

    def _load_faiss(
        self,
        *,
        faiss_index_path: Path | None,
        faiss_id_map_path: Path | None,
        embedding_model_name: str,
    ) -> FaissRetriever | None:
        if faiss_index_path is None or faiss_id_map_path is None:
            return None
        if not faiss_index_path.exists() or not faiss_id_map_path.exists():
            return None
        retriever = FaissRetriever(model_name=embedding_model_name)
        retriever.load_index(faiss_index_path, faiss_id_map_path)
        return retriever

    def _load_ranker(self, path: Path | None, label: str) -> V22RidgeRanker:
        if path is None or not path.exists():
            raise ValueError(f"{label} ranker artifact was not found: {path}")
        return load_v2_2_ridge_ranker(path)

    def _validate_blend_artifact(self, path: Path) -> None:
        if not path.exists():
            raise ValueError(f"V2.7 blend artifact was not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected_method = str(payload.get("selected_method") or "")
        expected_method = "blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding"
        if selected_method != expected_method:
            raise ValueError(f"Unexpected V2.7 selected method '{selected_method}'. Expected '{expected_method}'.")
        weights = (payload.get("fixed_blends") or {}).get(expected_method) or {}
        expected_weights = {"old": 0.6, "v26": 0.2, "hybrid": 0.1, "embedding": 0.1}
        for key, expected in expected_weights.items():
            if float(weights.get(key, -1.0)) != expected:
                raise ValueError(f"Unexpected V2.7 blend weight for '{key}': {weights.get(key)}.")

    @staticmethod
    def _identifier_from_external_id(value: str | None, *, prefix: str) -> str | None:
        if not value:
            return None
        if value.startswith(prefix):
            return value.removeprefix(prefix)
        return value if ":" not in value else None

    @staticmethod
    def _parse_authors(value: Any) -> list[str]:
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
