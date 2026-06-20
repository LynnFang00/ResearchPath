import numpy as np

from app.services.retrievers.base import PaperDocument
from app.services.retrievers.embedding import EmbeddingRetriever, normalize_rows


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
            vectors.append(
                [
                    1.0 if "agent" in lowered else 0.0,
                    1.0 if "protein" in lowered else 0.0,
                    1.0 if "graph" in lowered else 0.0,
                ]
            )
        return np.array(vectors, dtype=np.float32)


def test_normalize_rows_handles_zero_vectors():
    matrix = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))

    assert np.allclose(matrix[0], [0.6, 0.8])
    assert np.allclose(matrix[1], [0.0, 0.0])


def test_embedding_retriever_searches_by_cosine_similarity():
    retriever = EmbeddingRetriever(
        [
            PaperDocument(document_id=1, text="AI agents for scientific discovery"),
            PaperDocument(document_id=2, text="protein folding benchmark"),
            PaperDocument(document_id=3, text="graph neural network survey"),
        ],
        encoder=FakeEncoder(),
    )

    results = retriever.search("agent systems", k=2)

    assert [result.document_id for result in results][:1] == [1]
    assert results[0].score > results[1].score


def test_embedding_retriever_saves_and_loads_index(tmp_path):
    index_path = tmp_path / "embeddings.npz"
    retriever = EmbeddingRetriever(
        [PaperDocument(document_id=10, text="protein design")],
        encoder=FakeEncoder(),
        model_name="fake-model",
    )
    retriever.save_index(index_path)

    loaded = EmbeddingRetriever(encoder=FakeEncoder())
    loaded.load_index(index_path)

    assert loaded.model_name == "fake-model"
    assert loaded.search("protein", k=1)[0].document_id == 10
