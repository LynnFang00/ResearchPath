import math
import re
from collections import Counter

from app.services.retrievers.base import PaperDocument, ScoredDocument, coerce_documents


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    method_name = "bm25"

    def __init__(
        self,
        documents: list[PaperDocument] | list[tuple[int, str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[PaperDocument] = []
        self.doc_tokens: dict[int, list[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.avg_doc_length = 0.0
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.document_frequencies: Counter[str] = Counter()
        if documents is not None:
            self.fit(documents)

    def fit(self, documents: list[PaperDocument] | list[tuple[int, str]]) -> None:
        self.documents = coerce_documents(documents)
        self.doc_tokens = {
            document.document_id: tokenize(document.text) for document in self.documents
        }
        self.doc_lengths = {doc_id: len(tokens) for doc_id, tokens in self.doc_tokens.items()}
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_frequencies = {
            doc_id: Counter(tokens) for doc_id, tokens in self.doc_tokens.items()
        }
        self.document_frequencies = self._build_document_frequencies()

    def _build_document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in self.doc_tokens.values():
            frequencies.update(set(tokens))
        return frequencies

    def _idf(self, term: str) -> float:
        total_docs = len(self.documents)
        doc_freq = self.document_frequencies.get(term, 0)
        return math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, k: int = 10, exclude_ids: set[int] | None = None) -> list[ScoredDocument]:
        if not query.strip() or not self.documents:
            return []

        exclude_ids = exclude_ids or set()
        query_terms = tokenize(query)
        scores: list[ScoredDocument] = []

        for document in self.documents:
            doc_id = document.document_id
            if doc_id in exclude_ids:
                continue
            score = 0.0
            doc_length = self.doc_lengths[doc_id]
            term_frequency = self.term_frequencies[doc_id]

            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                score += self._idf(term) * (frequency * (self.k1 + 1)) / denominator

            if score > 0:
                scores.append(ScoredDocument(document_id=doc_id, score=score))

        return sorted(scores, key=lambda item: item.score, reverse=True)[:k]
