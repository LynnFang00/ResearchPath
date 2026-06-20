import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.retrievers.base import (
    PaperDocument,
    ScoredDocument,
    coerce_documents,
)


class TfidfRetriever:
    method_name = "tfidf"

    def __init__(self, documents: list[PaperDocument] | list[tuple[int, str]] | None = None) -> None:
        self.documents: list[PaperDocument] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = None
        if documents is not None:
            self.fit(documents)

    def fit(self, papers: list[PaperDocument] | list[tuple[int, str]]) -> None:
        self.documents = coerce_documents(papers)
        if not self.documents:
            self.matrix = None
            return
        self.matrix = self.vectorizer.fit_transform([document.text for document in self.documents])

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        if not query.strip() or self.matrix is None or not self.documents:
            return []

        exclude_ids = exclude_ids or set()
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        ranked_indices = np.argsort(scores)[::-1]
        results: list[ScoredDocument] = []

        for index in ranked_indices:
            document = self.documents[int(index)]
            score = float(scores[int(index)])
            if document.document_id in exclude_ids or score <= 0:
                continue
            results.append(ScoredDocument(document_id=document.document_id, score=score))
            if len(results) >= k:
                break

        return results
