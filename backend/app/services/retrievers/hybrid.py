from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.models.paper import Paper
from app.services.bm25 import BM25Retriever
from app.services.difficulty import score_difficulty
from app.services.quality_signals import (
    QualitySignals,
    compute_quality_signals,
    hybrid_score,
    normalize_scores,
)
from app.services.retrievers.base import PaperDocument, ScoredDocument, papers_to_documents
from app.services.retrievers.faiss import FaissRetriever
from app.services.retrievers.tfidf import TfidfRetriever


@dataclass(frozen=True)
class HybridScoredDocument(ScoredDocument):
    components: dict[str, float]


class HybridRetriever:
    method_name = "hybrid"

    def __init__(
        self,
        papers: list[Paper],
        *,
        faiss_index_path: Path | None = None,
        faiss_id_map_path: Path | None = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        background_level: str = "basic_ml",
        current_year: int | None = None,
    ) -> None:
        self.papers = papers
        self.paper_by_id = {paper.id: paper for paper in papers}
        self.documents = papers_to_documents(papers)
        self.bm25 = BM25Retriever(self.documents)
        self.tfidf = TfidfRetriever(self.documents)
        self.faiss = self._load_faiss(
            faiss_index_path=faiss_index_path,
            faiss_id_map_path=faiss_id_map_path,
            embedding_model_name=embedding_model_name,
        )
        self.background_level = background_level
        self.current_year = current_year or datetime.now(UTC).year
        self.last_components: dict[int, dict[str, float]] = {}

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
        if not query.strip() or not self.papers:
            return []

        exclude_ids = exclude_ids or set()
        search_k = min(len(self.papers), max(k * 8, 50))
        bm25_raw = {item.document_id: item.score for item in self.bm25.search(query, k=search_k, exclude_ids=exclude_ids)}
        tfidf_raw = {
            item.document_id: item.score for item in self.tfidf.search(query, k=search_k, exclude_ids=exclude_ids)
        }
        faiss_raw: dict[int, float] = {}
        if self.faiss is not None:
            faiss_raw = {
                item.document_id: item.score
                for item in self.faiss.search(query, k=search_k, exclude_ids=exclude_ids)
            }

        bm25_scores = normalize_scores(bm25_raw)
        tfidf_scores = normalize_scores(tfidf_raw)
        faiss_scores = normalize_scores(faiss_raw)
        candidate_ids = set(bm25_scores) | set(tfidf_scores) | set(faiss_scores)
        scored: list[HybridScoredDocument] = []
        self.last_components = {}

        for paper_id in candidate_ids:
            paper = self.paper_by_id.get(paper_id)
            if paper is None:
                continue
            difficulty = score_difficulty(paper, current_year=self.current_year)
            signals = compute_quality_signals(
                query=query,
                paper=paper,
                difficulty=difficulty,
                background_level=self.background_level,
                bm25_score=bm25_scores.get(paper_id, 0.0),
                tfidf_score=tfidf_scores.get(paper_id, 0.0),
                faiss_score=faiss_scores.get(paper_id, 0.0),
                current_year=self.current_year,
            )
            score = hybrid_score(signals)
            components = signals.diagnostics() | {"hybrid_score": round(score, 6)}
            self.last_components[paper_id] = components
            if score <= 0:
                continue
            scored.append(HybridScoredDocument(document_id=paper_id, score=score, components=components))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def components_for(self, paper_id: int) -> dict[str, float]:
        return self.last_components.get(paper_id, {})

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
        try:
            retriever.load_index(faiss_index_path, faiss_id_map_path)
        except (RuntimeError, ValueError, OSError):
            return None
        return retriever
