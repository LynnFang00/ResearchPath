import inspect
import json
from pathlib import Path
import sys

import pytest

from app.services.ltr_model_service import V64_METHOD_NAME, V64SafeFusionModel
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, normalize_method, recommend_from_query
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


PARITY_REPORT = ROOT / "data" / "eval" / "results" / "v6_5_1_safe_fusion_runtime_parity_report.json"
SERIALIZATION_REPORT = ROOT / "data" / "eval" / "results" / "v6_5_1_safe_fusion_scorer_serialization_report.json"
V651_CANDIDATE = ROOT / "data" / "processed" / "models" / "v6_5_1_safe_fusion_runtime_candidate.json"
V21_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
V25_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
V32_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
V39_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl"


class FakeRetriever:
    def search(self, query: str, k: int = 10, exclude_ids: set[int] | None = None) -> list[ScoredDocument]:
        return [ScoredDocument(document_id=1, score=1.0)]


def test_v6_5_1_reports_schema_and_no_complete_candidate_created() -> None:
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
    serialization = json.loads(SERIALIZATION_REPORT.read_text(encoding="utf-8"))

    assert parity["schema_version"] == "v6.5.1_safe_fusion_runtime_parity_report"
    assert serialization["schema_version"] == "v6.5.1_safe_fusion_scorer_serialization_report"
    assert parity["complete_v6_5_1_candidate_config_created"] is False
    assert serialization["complete_candidate_config_created"] is False
    assert not V651_CANDIDATE.exists()


def test_v6_5_1_exact_ridge_reconstruction_is_reported_blocked() -> None:
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
    serialization = json.loads(SERIALIZATION_REPORT.read_text(encoding="utf-8"))

    assert parity["exact_ridge_scorer_reconstruction_possible"] is False
    assert parity["runtime_computes_v6_3_ridge_internally"] is False
    assert serialization["ridge_reconstruction"]["exact_reconstruction_possible"] is False
    assert "ridge_intercept" in serialization["ridge_reconstruction"]["missing_required_fields"]
    assert "standard_scaler_mean" in serialization["ridge_reconstruction"]["missing_required_fields"]
    assert "fold_specific_or_production_coefficient_vector" in serialization["ridge_reconstruction"]["missing_required_fields"]


def test_v6_5_1_safe_formula_still_has_parity_when_ridge_is_supplied() -> None:
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))

    assert parity["safe_formula_parity_max_delta_when_ridge_supplied"] <= 1e-12
    assert parity["v6_4_safe_formula_parity_with_supplied_ridge"]["max_delta"] <= 1e-12
    assert parity["offline_live_candidate_set_jaccard"]["jaccard"] == 1.0


def test_v6_5_1_runtime_fallback_preserves_v4_9_when_ridge_features_are_missing() -> None:
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
    v49 = {1: 0.9, 2: 0.7, 3: 0.2}
    scored = V64SafeFusionModel.score_component_maps(v49_scores=v49, ridge_scores=dict(v49))

    assert parity["fallback_behavior"]["preserves_v4_9_order_when_ridge_missing"] is True
    assert parity["fallback_behavior"]["max_delta_from_v4_9_when_ridge_missing"] <= 1e-12
    for paper_id, details in scored.items():
        assert details["v6_4_safe_fusion_score"] == pytest.approx(v49[paper_id])


def test_v6_5_1_no_forbidden_runtime_features_required() -> None:
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
    assert parity["forbidden_runtime_features_required"] == []
    assert parity["acceptance"]["criteria"]["forbidden_runtime_features_required_empty"] is True


def test_v6_5_1_default_and_opt_in_method_status(monkeypatch: pytest.MonkeyPatch) -> None:
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

    recommend_from_query(db=object(), query="safe fusion")
    recommend_from_query(db=object(), query="safe fusion", method=V64_METHOD_NAME)

    assert observed == ["bm25", V64_METHOD_NAME]


def test_v6_5_1_frontend_defaults_unchanged() -> None:
    search_page = (ROOT / "frontend" / "src" / "pages" / "SearchPage.tsx").read_text(encoding="utf-8")
    client = (ROOT / "frontend" / "src" / "api" / "client.ts").read_text(encoding="utf-8")

    assert "useState('embedding')" in search_page
    assert "method = 'hybrid'" in client
    assert V64_METHOD_NAME not in search_page
    assert V64_METHOD_NAME not in client


def test_v6_5_1_protected_hashes_are_unchanged() -> None:
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in parity["protected_hashes"].items() if key.endswith("_hash_unchanged")}

    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == parity["protected_hashes"]["v3_9_labels_sha256"]
