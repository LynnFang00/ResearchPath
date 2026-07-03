from pathlib import Path
import sys

import numpy as np
import pytest

from app.models.paper import Paper
from app.services.retrievers.base import PaperDocument
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.learned_blend_v2_7 import (
    LearnedBlendV27Retriever,
    V27_SELECTED_BLEND_WEIGHTS,
)
from app.services.v2_2_learned_ranker import feature_names, v2_2_ridge_ranker_from_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_v2_7_score_blends import minmax as offline_minmax  # noqa: E402


class KeywordEncoder:
    def encode(
        self,
        texts: list[str] | str,
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lowered = text.lower()
            graph = 1.0 if "graph" in lowered else 0.0
            transformer = 1.0 if "transformer" in lowered else 0.0
            retrieval = 1.0 if "retrieval" in lowered else 0.0
            vectors.append([graph + 0.2 * retrieval, transformer + 0.1 * retrieval])
        return np.asarray(vectors, dtype=np.float32)


def constant_ranker(score: float):
    names = feature_names()
    return v2_2_ridge_ranker_from_payload(
        {
            "feature_names": names,
            "weights": [0.0 for _ in names],
            "intercept": score,
            "standard_scaler_mean": [0.0 for _ in names],
            "standard_scaler_scale": [1.0 for _ in names],
            "target": {"name": "constant_test_score"},
            "metadata": {},
            "version": "test",
        }
    )


def test_learned_blend_v2_7_matches_offline_score_formula_components_and_order() -> None:
    papers = [
        Paper(
            id=1,
            title="Graph Attention Networks",
            abstract="Graph neural network attention methods.",
            authors="A",
            year=2018,
            categories="cs.LG",
            citation_count=500,
        ),
        Paper(
            id=2,
            title="Transformer Retrieval",
            abstract="Transformer retrieval model for search.",
            authors="B",
            year=2020,
            categories="cs.CL",
            citation_count=200,
        ),
        Paper(
            id=3,
            title="Graph Transformer Survey",
            abstract="Survey of graph transformer architectures.",
            authors="C",
            year=2023,
            categories="cs.LG",
            citation_count=80,
        ),
        Paper(
            id=4,
            title="Unrelated Medical Study",
            abstract="Clinical dataset analysis.",
            authors="D",
            year=2024,
            categories="q-bio",
            citation_count=5,
        ),
    ]
    documents = [
        PaperDocument(
            document_id=paper.id,
            text=paper.searchable_text,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            citation_count=paper.citation_count,
        )
        for paper in papers
    ]
    embedding = EmbeddingRetriever(documents, encoder=KeywordEncoder())
    retriever = LearnedBlendV27Retriever(
        papers,
        exact_embedding_retriever=embedding,
        old_ranker=constant_ranker(0.7),
        v26_ranker=constant_ranker(0.3),
    )

    debug = retriever.score_query(query="graph transformer", top_k=50)

    expected_pool_ids = sorted(
        {
            int(row["paper_id"])
            for rows in debug["generation_runs"].values()
            for row in rows
        }
    )
    assert debug["candidate_pool_ids"] == expected_pool_ids

    offline_hybrid = offline_minmax(debug["pool_scores"]["hybrid"])
    offline_embedding = offline_minmax(debug["pool_scores"]["embedding"])
    for paper_id, values in debug["scores"].items():
        assert values["old_v2_2b_score"] == pytest.approx(0.7)
        assert values["v2_6_score"] == pytest.approx(0.3)
        assert values["hybrid_score"] == pytest.approx(offline_hybrid[paper_id])
        assert values["embedding_score"] == pytest.approx(offline_embedding[paper_id])
        expected_score = (
            V27_SELECTED_BLEND_WEIGHTS["old_v2_2b_score"] * values["old_v2_2b_score"]
            + V27_SELECTED_BLEND_WEIGHTS["v2_6_score"] * values["v2_6_score"]
            + V27_SELECTED_BLEND_WEIGHTS["hybrid_score"] * offline_hybrid[paper_id]
            + V27_SELECTED_BLEND_WEIGHTS["embedding_score"] * offline_embedding[paper_id]
        )
        assert values["v2_7_score"] == pytest.approx(expected_score)

    expected_order = [
        paper_id
        for paper_id, _ in sorted(
            ((paper_id, values["v2_7_score"]) for paper_id, values in debug["scores"].items()),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
    ]
    results = retriever.search("graph transformer", k=4)
    assert [result.document_id for result in results] == expected_order[:4]


def test_learned_blend_v2_7_reports_missing_ranker_artifacts() -> None:
    with pytest.raises(ValueError, match="old V2.2b ranker artifact was not found"):
        LearnedBlendV27Retriever(
            [],
            old_model_path=Path("missing-old.json"),
            v26_ranker=constant_ranker(0.3),
        )


def test_learned_blend_v2_7_reports_missing_blend_artifact() -> None:
    with pytest.raises(ValueError, match="V2.7 blend artifact was not found"):
        LearnedBlendV27Retriever(
            [],
            old_ranker=constant_ranker(0.7),
            v26_ranker=constant_ranker(0.3),
            blend_artifact_path=Path("missing-v2-7.json"),
        )
