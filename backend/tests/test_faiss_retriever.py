import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.services.retrievers.faiss import FaissRetriever  # noqa: E402
from scripts.build_faiss_index import default_id_map_path, load_embedding_payload  # noqa: E402


class FakeFaissIndex:
    def __init__(self, dimension):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self):
        return self.vectors.shape[0]

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query, k):
        scores = self.vectors @ query[0]
        ranked = np.argsort(scores)[::-1][:k]
        return scores[ranked].reshape(1, -1), ranked.astype(np.int64).reshape(1, -1)


class FakeEncoder:
    def encode(
        self,
        texts,
        *,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if "agent" in lowered else 0.0, 1.0 if "protein" in lowered else 0.0])
        return np.array(vectors, dtype=np.float32)


def test_faiss_retriever_searches_with_inner_product(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(IndexFlatIP=FakeFaissIndex))
    retriever = FaissRetriever(encoder=FakeEncoder())
    retriever.build_from_embeddings(
        np.array([1, 2], dtype=np.int64),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    results = retriever.search("agent retrieval", k=1)

    assert results[0].document_id == 1


def test_default_id_map_path():
    assert default_id_map_path(Path("index.faiss")) == Path("index.ids.npz")


def test_load_embedding_payload(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        path,
        document_ids=np.array([1, 2], dtype=np.int64),
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        model_name=np.array("fake-model"),
    )

    payload = load_embedding_payload(path)

    assert payload["model_name"] == "fake-model"
    assert payload["embeddings"].shape == (2, 2)
