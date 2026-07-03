import inspect
import json
from pathlib import Path
import sys

import pytest

from app.services.ltr_model_service import (
    V49_METHOD_NAME,
    V49GuardedTextBlendModel,
    load_v4_9_guarded_text_blend_model,
)
from app.services.recommendation_service import (
    SUPPORTED_RETRIEVAL_METHODS,
    build_retriever,
    normalize_method,
    recommend_from_query,
)
from app.services.retrievers.base import ScoredDocument


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.validate_v3_5_unjudged_audit_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    file_sha256,
)
from scripts.evaluate_v3_6_revealed_v3_5_labels import DEFAULT_V35_LABELS, EXPECTED_V35_LABELS_SHA256  # noqa: E402


V21_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
V25_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
V32_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
V39_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl"
V33_MODEL_DIR = ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
V41_MODEL_DIR = ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"
V43_MODEL_DIR = ROOT / "data" / "processed" / "models" / "v4_3_text_reranker"
V41_BLEND_CONFIG = ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"
V49_CONFIG = ROOT / "data" / "processed" / "models" / "v4_9_guarded_text_blend_candidate.json"
REPORT = ROOT / "data" / "eval" / "results" / "v5_0_v4_9_guarded_text_runtime_smoke_report.json"


class FakeRetriever:
    def search(self, query: str, k: int = 10, exclude_ids: set[int] | None = None) -> list[ScoredDocument]:
        return [ScoredDocument(document_id=2, score=0.9), ScoredDocument(document_id=1, score=0.1)]


def test_v5_0_v4_9_method_is_opt_in_and_default_remains_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    assert V49_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS
    assert normalize_method("V4_9_Guarded_Text_Blend") == V49_METHOD_NAME
    assert inspect.signature(recommend_from_query).parameters["method"].default == "bm25"

    loaded_papers = []
    paper_by_id = {}
    observed = []

    def fake_load_corpus(db):
        return (0, 0, "static"), loaded_papers, paper_by_id, []

    def fake_get_cached_retriever(**kwargs):
        observed.append(kwargs["method"])
        return FakeRetriever()

    monkeypatch.setattr("app.services.recommendation_service._load_corpus", fake_load_corpus)
    monkeypatch.setattr("app.services.recommendation_service.get_cached_retriever", fake_get_cached_retriever)

    recommend_from_query(db=object(), query="contrastive learning")
    recommend_from_query(db=object(), query="contrastive learning", method=V49_METHOD_NAME)

    assert observed == ["bm25", V49_METHOD_NAME]


def test_v5_0_build_retriever_requires_full_metadata_for_v4_9() -> None:
    with pytest.raises(ValueError, match="V4.9 guarded text blend retrieval requires full paper metadata"):
        build_retriever(V49_METHOD_NAME, [])


def test_v5_0_candidate_config_and_models_load() -> None:
    model = load_v4_9_guarded_text_blend_model(
        v33_model_dir=V33_MODEL_DIR,
        v41_model_dir=V41_MODEL_DIR,
        v43_model_dir=V43_MODEL_DIR,
        v41_blend_config_path=V41_BLEND_CONFIG,
        v49_candidate_config_path=V49_CONFIG,
    )

    assert model.config["schema_version"] == "v4.9_guarded_text_blend_candidate_config"
    assert model.blend_name == "v4_6_guardrail_text_when_rankers_not_confident"
    assert model.text_model.score_column == "v4_3_text_plus_ranker_features_ridge_score"
    assert model.text_model.metadata["excludes_v2_7_feature"] is True
    assert model.text_model.metadata["excludes_raw_score_features"] is True


def test_v5_0_formula_parity_with_offline_reference() -> None:
    v33 = {1: 0.1, 2: 0.4, 3: 0.9, 4: 0.9}
    v41 = {1: 0.2, 2: 0.3, 3: 0.5, 4: 0.95}
    text = {1: 0.9, 2: 0.2, 3: 0.8, 4: 0.1}
    scored = V49GuardedTextBlendModel.score_component_maps(
        v33_scores=v33,
        v41_scores=v41,
        text_scores=text,
    )
    normalized_v33 = V49GuardedTextBlendModel.normalize_scores(v33)
    normalized_v41 = V49GuardedTextBlendModel.normalize_scores(v41)
    normalized_text = V49GuardedTextBlendModel.normalize_scores(text)

    for paper_id, row in scored.items():
        text_component = 0.10 * normalized_text[paper_id] if max(normalized_v33[paper_id], normalized_v41[paper_id]) < 0.85 else 0.0
        expected = 0.675 * normalized_v33[paper_id] + 0.225 * normalized_v41[paper_id] + text_component
        assert row["v4_9_guarded_text_blend_score"] == pytest.approx(expected)


def test_v5_0_no_forbidden_runtime_features_required() -> None:
    config = json.loads(V49_CONFIG.read_text(encoding="utf-8"))
    features = set(config["formula"]["features_used"])
    assert features == {
        "normalized_v3_3_score",
        "normalized_v4_1_score",
        "normalized_v4_3_text_score",
        "max_ranker_confidence",
    }
    forbidden = {
        "true_relevance_label",
        "reading_value_score",
        "topic_match_score",
        "hard_negative_label",
        "section_label",
        "difficulty_label",
        "per_topic_known_hurt_status",
        "per_topic_known_improved_status",
    }
    assert features.isdisjoint(forbidden)


def test_v5_0_frontend_defaults_unchanged_and_no_frontend_v4_9_option() -> None:
    search_page = (ROOT / "frontend" / "src" / "pages" / "SearchPage.tsx").read_text(encoding="utf-8")
    client = (ROOT / "frontend" / "src" / "api" / "client.ts").read_text(encoding="utf-8")

    assert "useState('embedding')" in search_page
    assert "method = 'hybrid'" in client
    assert V49_METHOD_NAME not in search_page
    assert V49_METHOD_NAME not in client


def test_v5_0_smoke_report_schema() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v5.0_v4_9_guarded_text_runtime_smoke_report"
    assert report["runtime_method_added"] is True
    assert report["runtime_method_name"] == V49_METHOD_NAME
    assert report["default_remains_bm25"] is True
    assert report["formula_parity"]["max_delta"] <= 1e-12
    assert report["offline_live_candidate_set_jaccard"]["jaccard"] == 1.0
    assert report["runtime_integration_status"] == "opt_in_only_not_default"
    assert report["frontend_defaults_changed_by_this_script"] is False


def test_v5_0_protected_hashes_are_unchanged() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == report["protected_hashes"]["v3_9_labels_sha256"]
