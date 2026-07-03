from pathlib import Path
import sys

import numpy as np
import pytest

from app.models.paper import Paper
from app.services.ltr_model_service import RAW_SCORE_FEATURES, V33LTRModel, load_v3_3_ltr_model
from app.services.recommendation_service import recommend_from_query
from app.services.retrievers.base import PaperDocument, ScoredDocument
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.v3_3_ltr import V33LTRRetriever
from app.services.v2_2_learned_ranker import feature_names, v2_2_ridge_ranker_from_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_v3_5_unjudged_audit_labels import (  # noqa: E402
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    DEFAULT_SELECTED_240,
    file_sha256,
)
from scripts.evaluate_v3_6_revealed_v3_5_labels import DEFAULT_V35_LABELS, EXPECTED_V35_LABELS_SHA256  # noqa: E402


MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
V32_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"


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


class SumEstimator:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return matrix.sum(axis=1)


class FakeRetriever:
    method_name = "fake"

    def search(self, query: str, k: int = 10, exclude_ids: set[int] | None = None) -> list[ScoredDocument]:
        return [ScoredDocument(document_id=2, score=0.9), ScoredDocument(document_id=1, score=0.1)]


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


def papers() -> list[Paper]:
    return [
        Paper(id=1, title="Graph Attention Networks", abstract="Graph neural network attention methods.", authors="A", year=2018, categories="cs.LG", citation_count=500),
        Paper(id=2, title="Transformer Retrieval", abstract="Transformer retrieval model for search.", authors="B", year=2020, categories="cs.CL", citation_count=200),
        Paper(id=3, title="Graph Transformer Survey", abstract="Survey of graph transformer architectures.", authors="C", year=2023, categories="cs.LG", citation_count=80),
        Paper(id=4, title="Unrelated Medical Study", abstract="Clinical dataset analysis.", authors="D", year=2024, categories="q-bio", citation_count=5),
    ]


def fake_v33_model(feature_columns: list[str] | None = None) -> V33LTRModel:
    real_metadata = load_v3_3_ltr_model(MODEL_DIR).metadata
    metadata = dict(real_metadata)
    metadata["feature_columns"] = feature_columns or list(real_metadata["feature_columns"])
    return V33LTRModel(
        model_name="random_forest_no_v27_compact",
        artifact_path=Path("fake.joblib"),
        metadata_path=Path("fake.metadata.json"),
        metadata=metadata,
        estimator=SumEstimator(),
    )


def build_runtime_retriever(v33_model: V33LTRModel | None = None) -> V33LTRRetriever:
    rows = papers()
    documents = [
        PaperDocument(
            document_id=paper.id,
            text=paper.searchable_text,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            citation_count=paper.citation_count,
        )
        for paper in rows
    ]
    embedding = EmbeddingRetriever(documents, encoder=KeywordEncoder())
    return V33LTRRetriever(
        rows,
        exact_embedding_retriever=embedding,
        old_ranker=constant_ranker(0.7),
        v26_ranker=constant_ranker(0.3),
        v33_model=v33_model or fake_v33_model(),
    )


def test_v3_3_runtime_model_artifact_loads_with_expected_feature_contract() -> None:
    model = load_v3_3_ltr_model(MODEL_DIR)

    assert model.model_name == "random_forest_no_v27_compact"
    assert model.metadata["uses_v2_7_feature"] is False
    assert model.metadata["raw_score_feature_intersection"] == []
    assert "v2_7_score" not in model.feature_columns
    assert RAW_SCORE_FEATURES.isdisjoint(model.feature_columns)
    assert hasattr(model.estimator, "predict")


def test_v3_3_runtime_feature_builder_matches_artifact_feature_order() -> None:
    model = load_v3_3_ltr_model(MODEL_DIR)
    retriever = build_runtime_retriever(fake_v33_model(model.feature_columns))

    debug = retriever.score_query(query="graph transformer", top_k=50)
    feature_row = debug["feature_rows"][0]

    assert debug["model"]["feature_columns"] == model.feature_columns
    assert list(debug["scores"][int(feature_row["paper_id"])]["v3_3_feature_values"]) == model.feature_columns
    assert debug["scores"][1]["v3_3_feature_values"]["paper_age_years"] == 8.0
    assert debug["scores"][1]["v3_3_feature_values"]["citation_count_log1p"] > 0.0
    assert "v2_7_score" not in model.feature_columns
    assert RAW_SCORE_FEATURES.isdisjoint(model.feature_columns)


def test_v3_3_runtime_scoring_returns_finite_sorted_scores() -> None:
    retriever = build_runtime_retriever()

    results = retriever.search("graph transformer", k=4)
    scores = [result.score for result in results]

    assert len(results) == 4
    assert all(np.isfinite(score) for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert all(result.components["v3_3_model_name"] == "random_forest_no_v27_compact" for result in results)


def test_v3_3_runtime_missing_artifact_reports_clear_error() -> None:
    with pytest.raises(ValueError, match="V3.3 LTR artifact index was not found"):
        load_v3_3_ltr_model(REPO_ROOT / "data" / "processed" / "models" / "missing_v3_3_ltr")


def test_recommend_from_query_default_still_uses_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_papers = papers()
    paper_by_id = {paper.id: paper for paper in loaded_papers}

    def fake_load_corpus(db):
        return (1, 4, "static"), loaded_papers, paper_by_id, []

    observed = {}

    def fake_get_cached_retriever(**kwargs):
        observed["method"] = kwargs["method"]
        return FakeRetriever()

    monkeypatch.setattr("app.services.recommendation_service._load_corpus", fake_load_corpus)
    monkeypatch.setattr("app.services.recommendation_service.get_cached_retriever", fake_get_cached_retriever)

    results = recommend_from_query(db=object(), query="graph")

    assert observed["method"] == "bm25"
    assert [result.method for result in results] == ["bm25", "bm25"]


def test_v3_7_protected_hashes_are_unchanged() -> None:
    assert file_sha256(V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
