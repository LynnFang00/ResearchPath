from pathlib import Path
from typing import Any

import numpy as np

from app.services.retrievers.base import ScoredDocument
from app.services.retrievers.embedding import TextEncoder, load_sentence_transformer, normalize_rows


def load_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on optional platform package.
        raise RuntimeError(
            "faiss is not installed. Install the FAISS extras with "
            "`python -m pip install -e .[faiss]` from the backend directory. "
            "On native Windows, use WSL or Docker if pip cannot find a FAISS wheel."
        ) from exc
    return faiss


class FaissRetriever:
    method_name = "faiss_embedding"

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: TextEncoder | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.encoder = encoder
        self.batch_size = batch_size
        self.document_ids = np.array([], dtype=np.int64)
        self.index = None
        self.dimension = 0

    def build_from_embeddings(self, document_ids: np.ndarray, embeddings: np.ndarray) -> None:
        embeddings = normalize_rows(embeddings).astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D matrix.")

        faiss = load_faiss()
        self.document_ids = np.asarray(document_ids, dtype=np.int64)
        self.dimension = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def load_index(self, index_path: Path, id_map_path: Path) -> None:
        faiss = load_faiss()
        payload = np.load(id_map_path, allow_pickle=False)
        self.document_ids = payload["document_ids"].astype(np.int64)
        if "model_name" in payload.files:
            model_name = str(payload["model_name"])
            if model_name:
                self.model_name = model_name
        self.index = faiss.read_index(str(index_path))
        self.dimension = int(self.index.d)

    def save_index(self, index_path: Path, id_map_path: Path) -> None:
        if self.index is None:
            raise ValueError("Cannot save a FAISS index before building or loading it.")
        faiss = load_faiss()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        id_map_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        np.savez_compressed(
            id_map_path,
            document_ids=self.document_ids,
            model_name=np.array(self.model_name),
        )

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        if not query.strip() or self.index is None or len(self.document_ids) == 0:
            return []

        exclude_ids = exclude_ids or set()
        query_embedding = self._encoder().encode(
            query,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_embedding = normalize_rows(query_embedding).astype(np.float32)
        search_k = min(int(self.index.ntotal), max(k + len(exclude_ids), k * 5))
        scores, indices = self.index.search(query_embedding, search_k)

        results: list[ScoredDocument] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            document_id = int(self.document_ids[int(index)])
            if document_id in exclude_ids:
                continue
            results.append(ScoredDocument(document_id=document_id, score=float(score)))
            if len(results) >= k:
                break
        return results

    def _encoder(self) -> TextEncoder:
        if self.encoder is None:
            self.encoder = load_sentence_transformer(self.model_name)
        return self.encoder
