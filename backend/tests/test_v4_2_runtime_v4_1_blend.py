from pathlib import Path
import sys

import numpy as np
import pytest

from app.models.paper import Paper
from app.services.ltr_model_service import (
    V33LTRModel,
    V41BlendModel,
    V41WeightedLTRModel,
    load_v3_3_ltr_model,
    load_v4_1_blend_model,
)
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, normalize_method, recommend_from_query
from app.services.retrievers.base import PaperDocument, ScoredDocument
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.v4_1_blend import V41BlendRetriever
from app.services.v2_2_learned_ranker import feature_names, v2_2_ridge_ranker_from_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_v3_5_unjudged_audit_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    file_sha256,
)
from scripts.evaluate_v3_6_revealed_v3_5_labels import DEFAULT_V35_LABELS, EXPECTED_V35_LABELS_SHA256  # noqa: E402


V33_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
V41_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"
BLEND_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"
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
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return matrix.sum(axis=1) + self.offset


class FakeRetriever:
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


def fake_blend_model() -> V41BlendModel:
    real_v33 = load_v3_3_ltr_model(V33_MODEL_DIR)
    v33_metadata = dict(real_v33.metadata)
    v41_metadata = dict(real_v33.metadata)
    v41_metadata.update(
        {
            "score_column": "v4_1_downweight_v3_9_active_learning_score",
            "uses_v2_7_feature": False,
        }
    )
    v33 = V33LTRModel(
        model_name="random_forest_no_v27_compact",
        artifact_path=Path("fake_v33.joblib"),
        metadata_path=Path("fake_v33.metadata.json"),
        metadata=v33_metadata,
        estimator=SumEstimator(0.0),
    )
    v41 = V41WeightedLTRModel(
        model_name="downweight_v3_9_active_learning",
        artifact_path=Path("fake_v41.joblib"),
        metadata_path=Path("fake_v41.metadata.json"),
        metadata=v41_metadata,
        estimator=SumEstimator(4.0),
    )
    return V41BlendModel(
        v33_model=v33,
        v41_model=v41,
        config_path=BLEND_CONFIG,
        config={
            "method": "blend_0_75_v33_0_25_v41",
            "created_at": "test",
            "spec": {"weights": {"v3_3_random_forest_no_v27_compact": 0.75, "__v41__": 0.25}},
        },
    )


def build_runtime_retriever() -> V41BlendRetriever:
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
    return V41BlendRetriever(
        rows,
        exact_embedding_retriever=embedding,
        old_ranker=constant_ranker(0.7),
        v26_ranker=constant_ranker(0.3),
        blend_model=fake_blend_model(),
    )


def test_v4_1_blend_method_is_registered_without_changing_default() -> None:
    assert "v4_1_blend" in SUPPORTED_RETRIEVAL_METHODS
    assert normalize_method("v4_1_blend") == "v4_1_blend"


def test_v4_1_blend_artifacts_load_with_matching_feature_order() -> None:
    model = load_v4_1_blend_model(
        v33_model_dir=V33_MODEL_DIR,
        v41_model_dir=V41_MODEL_DIR,
        blend_config_path=BLEND_CONFIG,
    )
    assert model.blend_name == "blend_0_75_v33_0_25_v41"
    assert model.v33_weight == 0.75
    assert model.v41_weight == 0.25
    assert model.v33_model.feature_columns == model.v41_model.feature_columns
    assert "v2_7_score" not in model.v41_model.feature_columns


def test_v4_1_blend_missing_config_reports_clear_error() -> None:
    with pytest.raises(ValueError, match="V4.1 blend config was not found"):
        load_v4_1_blend_model(
            v33_model_dir=V33_MODEL_DIR,
            v41_model_dir=V41_MODEL_DIR,
            blend_config_path=REPO_ROOT / "data" / "processed" / "models" / "missing_v4_1_blend.json",
        )


def test_v4_1_blend_formula_is_deterministic_and_sorted() -> None:
    retriever = build_runtime_retriever()
    first = retriever.search("graph transformer", k=4)
    second = retriever.search("graph transformer", k=4)
    assert [row.document_id for row in first] == [row.document_id for row in second]
    assert [row.score for row in first] == sorted([row.score for row in first], reverse=True)
    assert all(np.isfinite(row.score) for row in first)
    for result in first:
        components = result.components
        expected = 0.75 * components["v3_3_score"] + 0.25 * components["v4_1_weighted_score"]
        assert components["v4_1_blend_score"] == pytest.approx(expected)
        assert components["v4_1_blend_method"] == "v4_1_blend"


def test_recommend_from_query_default_and_v4_1_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_papers = papers()
    paper_by_id = {paper.id: paper for paper in loaded_papers}

    def fake_load_corpus(db):
        return (1, 4, "static"), loaded_papers, paper_by_id, []

    observed = []

    def fake_get_cached_retriever(**kwargs):
        observed.append(kwargs["method"])
        return FakeRetriever()

    monkeypatch.setattr("app.services.recommendation_service._load_corpus", fake_load_corpus)
    monkeypatch.setattr("app.services.recommendation_service.get_cached_retriever", fake_get_cached_retriever)

    default_results = recommend_from_query(db=object(), query="graph")
    opt_in_results = recommend_from_query(db=object(), query="graph", method="v4_1_blend")

    assert observed == ["bm25", "v4_1_blend"]
    assert [result.method for result in default_results] == ["bm25", "bm25"]
    assert [result.method for result in opt_in_results] == ["v4_1_blend", "v4_1_blend"]


def test_v4_2_protected_hashes_are_unchanged() -> None:
    assert file_sha256(V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
