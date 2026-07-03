import inspect
import json
from pathlib import Path
import sys

import pytest

from app.services.ltr_model_service import (
    V64_METHOD_NAME,
    V64SafeFusionModel,
    load_v6_4_safe_fusion_model,
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
V64_CONFIG = ROOT / "data" / "processed" / "models" / "v6_4_safe_fusion_candidate.json"
REPORT = ROOT / "data" / "eval" / "results" / "v6_5_safe_fusion_runtime_smoke_report.json"


class FakeRetriever:
    def search(self, query: str, k: int = 10, exclude_ids: set[int] | None = None) -> list[ScoredDocument]:
        return [ScoredDocument(document_id=2, score=0.9), ScoredDocument(document_id=1, score=0.1)]


def reference_scores(v49: dict[int, float], ridge: dict[int, float]) -> dict[int, float]:
    paper_ids = sorted(set(v49) | set(ridge))
    v49_ranked = sorted(paper_ids, key=lambda paper_id: (v49.get(paper_id, 0.0), -paper_id), reverse=True)
    ridge_ranked = sorted(paper_ids, key=lambda paper_id: (ridge.get(paper_id, 0.0), -paper_id), reverse=True)
    v49_top = v49_ranked[:10]
    entrants = [paper_id for paper_id in ridge_ranked[:10] if paper_id not in set(v49_top)][:2]
    tenth = min(v49[paper_id] for paper_id in v49_top) if v49_top else 0.0
    cap = tenth + 0.004 * 3
    output = {}
    for paper_id in paper_ids:
        v49_score = float(v49.get(paper_id, 0.0))
        ridge_score = float(ridge.get(paper_id, v49_score))
        if paper_id in entrants:
            output[paper_id] = max(v49_score, min(ridge_score, cap))
        else:
            output[paper_id] = v49_score + max(-0.005, min(0.005, ridge_score - v49_score))
    return output


def test_v6_5_method_is_opt_in_and_default_remains_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    assert V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS
    assert normalize_method("V6_4_SAFE_FUSION") == V64_METHOD_NAME
    assert inspect.signature(recommend_from_query).parameters["method"].default == "bm25"

    observed = []

    def fake_load_corpus(db):
        return (0, 0, "static"), [], {}, []

    def fake_get_cached_retriever(**kwargs):
        observed.append(kwargs["method"])
        return FakeRetriever()

    monkeypatch.setattr("app.services.recommendation_service._load_corpus", fake_load_corpus)
    monkeypatch.setattr("app.services.recommendation_service.get_cached_retriever", fake_get_cached_retriever)

    recommend_from_query(db=object(), query="multimodal learning")
    recommend_from_query(db=object(), query="multimodal learning", method=V64_METHOD_NAME)

    assert observed == ["bm25", V64_METHOD_NAME]


def test_v6_5_build_retriever_requires_full_metadata_for_v6_4() -> None:
    with pytest.raises(ValueError, match="V6.4 safe fusion retrieval requires full paper metadata"):
        build_retriever(V64_METHOD_NAME, [])


def test_v6_5_candidate_config_loads() -> None:
    model = load_v6_4_safe_fusion_model(V64_CONFIG)

    assert model.config["schema_version"] == "v6.4_safe_fusion_candidate"
    assert model.config["method"] == "v6_4_top10_swap_limited_2"
    assert model.config["acceptance_passed"] is True
    assert model.config["forbidden_features_used"] == []


def test_v6_5_formula_parity_with_offline_reference() -> None:
    v49 = {
        1: 0.90,
        2: 0.82,
        3: 0.75,
        4: 0.68,
        5: 0.60,
        6: 0.54,
        7: 0.48,
        8: 0.40,
        9: 0.33,
        10: 0.26,
        11: 0.20,
        12: 0.12,
    }
    ridge = {
        1: 0.88,
        2: 0.81,
        3: 0.77,
        4: 0.66,
        5: 0.61,
        6: 0.52,
        7: 0.47,
        8: 0.39,
        9: 0.34,
        10: 0.25,
        11: 0.72,
        12: 0.63,
    }
    scored = V64SafeFusionModel.score_component_maps(v49_scores=v49, ridge_scores=ridge)
    expected = reference_scores(v49, ridge)

    for paper_id, row in scored.items():
        assert row["v6_4_safe_fusion_score"] == pytest.approx(expected[paper_id])


def test_v6_5_missing_ridge_fallback_preserves_v4_9_scores() -> None:
    v49 = {1: 0.9, 2: 0.6, 3: 0.3}
    scored = V64SafeFusionModel.score_component_maps(v49_scores=v49, ridge_scores=dict(v49))

    for paper_id, row in scored.items():
        assert row["v6_4_safe_fusion_score"] == pytest.approx(v49[paper_id])


def test_v6_5_no_forbidden_runtime_features_required() -> None:
    config = json.loads(V64_CONFIG.read_text(encoding="utf-8"))
    features = set(config["inference_safe_features"])
    forbidden = {
        "true_relevance_label",
        "relevance_bucket",
        "reading_value_score",
        "topic_match_score",
        "hard_negative",
        "hard_negative_label",
        "section_label",
        "difficulty_label",
        "generic_survey",
        "eval_outcomes",
        "known_hurt_topic",
        "known_improved_topic",
    }
    assert config["forbidden_features_used"] == []
    assert features.isdisjoint(forbidden)


def test_v6_5_frontend_defaults_unchanged_and_no_frontend_v6_4_option() -> None:
    search_page = (ROOT / "frontend" / "src" / "pages" / "SearchPage.tsx").read_text(encoding="utf-8")
    client = (ROOT / "frontend" / "src" / "api" / "client.ts").read_text(encoding="utf-8")

    assert "useState('embedding')" in search_page
    assert "method = 'hybrid'" in client
    assert V64_METHOD_NAME not in search_page
    assert V64_METHOD_NAME not in client


def test_v6_5_smoke_report_schema() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v6.5_safe_fusion_runtime_smoke_report"
    assert report["runtime_method_added"] is True
    assert report["runtime_method_name"] == V64_METHOD_NAME
    assert report["default_remains_bm25"] is True
    assert report["formula_parity"]["max_delta"] <= 1e-12
    assert report["offline_live_candidate_set_jaccard"]["jaccard"] == 1.0
    assert report["forbidden_runtime_features_required"] == []
    assert report["runtime_integration_status"] == "opt_in_only_not_default"
    assert report["frontend_defaults_changed_by_this_script"] is False


def test_v6_5_protected_hashes_are_unchanged() -> None:
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
