import math

from app.services.bm25 import BM25Retriever
from app.services.retrievers.base import (
    PaperDocument,
    ScoredDocument,
    coerce_documents,
)


class CitationRecencyRetriever:
    method_name = "citation_recency"

    def __init__(
        self,
        documents: list[PaperDocument] | list[tuple[int, str]] | None = None,
        *,
        alpha: float = 0.1,
        beta: float = 0.2,
    ) -> None:
        self.documents: list[PaperDocument] = []
        self.alpha = alpha
        self.beta = beta
        self.text_retriever = BM25Retriever()
        self.year_min: int | None = None
        self.year_max: int | None = None
        if documents is not None:
            self.fit(documents)

    def fit(self, papers: list[PaperDocument] | list[tuple[int, str]]) -> None:
        self.documents = coerce_documents(papers)
        self.text_retriever.fit(self.documents)
        years = [document.year for document in self.documents if document.year is not None]
        self.year_min = min(years) if years else None
        self.year_max = max(years) if years else None

    def _recency_score(self, year: int | None) -> float:
        if year is None or self.year_min is None or self.year_max is None:
            return 0.0
        if self.year_max == self.year_min:
            return 1.0
        return (year - self.year_min) / (self.year_max - self.year_min)

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        if not query.strip() or not self.documents:
            return []

        exclude_ids = exclude_ids or set()
        text_scores = {
            result.document_id: result.score
            for result in self.text_retriever.search(query, k=len(self.documents), exclude_ids=exclude_ids)
        }
        results: list[ScoredDocument] = []

        for document in self.documents:
            if document.document_id in exclude_ids or document.document_id not in text_scores:
                continue
            score = (
                text_scores[document.document_id]
                + self.alpha * math.log1p(max(document.citation_count, 0))
                + self.beta * self._recency_score(document.year)
            )
            results.append(ScoredDocument(document_id=document.document_id, score=score))

        return sorted(results, key=lambda item: item.score, reverse=True)[:k]
