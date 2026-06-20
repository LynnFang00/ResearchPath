from pathlib import Path
from typing import Protocol

import numpy as np

from app.services.retrievers.base import (
    PaperDocument,
    ScoredDocument,
    coerce_documents,
)


class TextEncoder(Protocol):
    def encode(
        self,
        texts: list[str] | str,
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_sentence_transformer(model_name: str) -> TextEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised by environment, not unit tests.
        raise RuntimeError(
            "sentence-transformers is not installed. Install the ML extras with "
            "`python -m pip install -e .[ml]` from the backend directory."
        ) from exc
    return SentenceTransformer(model_name)


class EmbeddingRetriever:
    method_name = "embedding"

    def __init__(
        self,
        documents: list[PaperDocument] | list[tuple[int, str]] | None = None,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: TextEncoder | None = None,
        batch_size: int = 32,
    ) -> None:
        self.documents: list[PaperDocument] = []
        self.document_ids = np.array([], dtype=np.int64)
        self.embeddings: np.ndarray | None = None
        self.model_name = model_name
        self.encoder = encoder
        self.batch_size = batch_size
        if documents is not None:
            self.fit(documents)

    def fit(self, papers: list[PaperDocument] | list[tuple[int, str]]) -> None:
        self.documents = coerce_documents(papers)
        self.document_ids = np.array([document.document_id for document in self.documents], dtype=np.int64)
        if not self.documents:
            self.embeddings = None
            return

        encoder = self._encoder()
        raw_embeddings = encoder.encode(
            [document.text for document in self.documents],
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.embeddings = normalize_rows(raw_embeddings)

    def load_index(self, path: Path) -> None:
        payload = np.load(path, allow_pickle=False)
        self.document_ids = payload["document_ids"].astype(np.int64)
        self.embeddings = normalize_rows(payload["embeddings"])
        if "model_name" in payload.files:
            model_name = str(payload["model_name"])
            if model_name:
                self.model_name = model_name

    def save_index(self, path: Path) -> None:
        if self.embeddings is None:
            raise ValueError("Cannot save an embedding index before fitting or loading embeddings.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            document_ids=self.document_ids,
            embeddings=self.embeddings.astype(np.float32),
            model_name=np.array(self.model_name),
        )

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        if not query.strip() or self.embeddings is None or len(self.document_ids) == 0:
            return []

        exclude_ids = exclude_ids or set()
        query_embedding = self._encoder().encode(
            query,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_embedding = normalize_rows(query_embedding)[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]

        results: list[ScoredDocument] = []
        for index in ranked_indices:
            document_id = int(self.document_ids[int(index)])
            if document_id in exclude_ids:
                continue
            results.append(ScoredDocument(document_id=document_id, score=float(scores[int(index)])))
            if len(results) >= k:
                break
        return results

    def _encoder(self) -> TextEncoder:
        if self.encoder is None:
            self.encoder = load_sentence_transformer(self.model_name)
        return self.encoder
