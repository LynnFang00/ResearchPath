import inspect
import json
from pathlib import Path
import sys

import pytest

from app.services.ltr_model_service import (
    V64_METHOD_NAME,
    V64SafeFusionModel,
    load_v6_6_safe_fusion_ridge_scorer,
)
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, normalize_method, recommend_from_query


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


SCORER = ROOT / "data" / "processed" / "models" / "v6_6_safe_fusion_ridge_scorer.json"
CANDIDATE = ROOT / "data" / "processed" / "models" / "v6_6_safe_fusion_candidate.json"
REPRO_REPORT = ROOT / "data" / "eval" / "results" / "v6_6_safe_fusion_scorer_reproduction_report.json"
PARITY_REPORT = ROOT / "data" / "eval" / "results" / "v6_6_safe_fusion_runtime_parity_report.json"
V21_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
V25_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
V32_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
V39_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl"


def synthetic_components() -> dict[int, dict[str, float]]:
    return {
        1: {"bm25_score": 0.2, "v3_3_ltr_score": 0.9, "v4_1_blend_score": 0.88, "v4_9_guarded_text_blend_score": 0.90, "v6_1_neural_score": 0.4, "v6_2_multitask_neural_score": 0.7, "v4_9_max_ranker_confidence": 0.95},
        2: {"bm25_score": 0.8, "v3_3_ltr_score": 0.5, "v4_1_blend_score": 0.52, "v4_9_guarded_text_blend_score": 0.62, "v6_1_neural_score": 0.8, "v6_2_multitask_neural_score": 0.9, "v4_9_max_ranker_confidence": 0.62},
        3: {"bm25_score": 0.4, "v3_3_ltr_score": 0.3, "v4_1_blend_score": 0.35, "v4_9_guarded_text_blend_score": 0.40, "v6_1_neural_score": 0.9, "v6_2_multitask_neural_score": 0.2, "v4_9_max_ranker_confidence": 0.40},
    }


def test_v6_6_scorer_config_schema_and_parameters_present() -> None:
    config = json.loads(SCORER.read_text(encoding="utf-8"))

    assert config["schema_version"] == "v6.6_safe_fusion_ridge_scorer"
    assert config["model_type"] == "ridge"
    assert config["feature_names"]
    assert config["feature_order"] == config["feature_names"]
    assert len(config["coefficients"]) == len(config["feature_names"])
    assert isinstance(config["intercept"], float)
    assert len(config["scaler"]["mean"]) == len(config["feature_names"])
    assert len(config["scaler"]["scale"]) == len(config["feature_names"])
    assert len(config["scaler"]["feature_order"]) == len(config["feature_names"])


def test_v6_6_forbidden_inference_features_are_excluded() -> None:
    config = json.loads(SCORER.read_text(encoding="utf-8"))
    forbidden = {
        "relevance_label",
        "relevance_bucket",
        "reading_value_score",
        "topic_match_score",
        "hard_negative",
        "section_label",
        "difficulty_label",
        "generic_survey",
        "eval_outcomes",
        "known_hurt_topic",
    }

    assert config["forbidden_features_used"] == []
    assert set(config["feature_names"]).isdisjoint(forbidden)


def test_v6_6_candidate_config_created_only_when_acceptance_passes() -> None:
    report = json.loads(REPRO_REPORT.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    assert report["acceptance"]["passes"] is True
    assert report["candidate_config_created"] is True
    assert candidate["schema_version"] == "v6.6_safe_fusion_candidate"
    assert candidate["acceptance_passed"] is True
    assert candidate["runtime_method"] == V64_METHOD_NAME
    assert candidate["ridge_scorer_config_path"].endswith("v6_6_safe_fusion_ridge_scorer.json")


def test_v6_6_runtime_computes_ridge_internally_when_features_are_present() -> None:
    scorer = load_v6_6_safe_fusion_ridge_scorer(SCORER)
    ridge_rows = scorer.score_component_maps(synthetic_components())

    assert set(ridge_rows) == {1, 2, 3}
    for row in ridge_rows.values():
        assert isinstance(row["v6_6_ridge_fusion_score"], float)
        assert row["v6_6_ridge_feature_values"]


def test_v6_6_safe_fusion_uses_internal_ridge_scores_and_fallback_preserves_v4_9() -> None:
    scorer = load_v6_6_safe_fusion_ridge_scorer(SCORER)
    components = synthetic_components()
    ridge_rows = scorer.score_component_maps(components)
    v49 = {paper_id: values["v4_9_guarded_text_blend_score"] for paper_id, values in components.items()}
    ridge = {paper_id: values["v6_6_ridge_fusion_score"] for paper_id, values in ridge_rows.items()}
    safe = V64SafeFusionModel.score_component_maps(v49_scores=v49, ridge_scores=ridge)

    assert set(safe) == set(components)
    fallback = V64SafeFusionModel.score_component_maps(v49_scores=v49, ridge_scores=dict(v49))
    for paper_id, row in fallback.items():
        assert row["v6_4_safe_fusion_score"] == pytest.approx(v49[paper_id])


def test_v6_6_runtime_parity_report() -> None:
    report = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))

    assert report["schema_version"] == "v6.6_safe_fusion_runtime_parity_report"
    assert report["ridge_score_computed_internally"] is True
    assert report["formula_parity_max_delta"] <= 1e-12
    assert report["offline_live_candidate_set_jaccard"]["jaccard"] == 1.0
    assert report["fallback_behavior"]["preserves_v4_9_order"] is True
    assert report["forbidden_runtime_features_required"] == []
    assert report["acceptance"]["passes"] is True


def test_v6_6_default_remains_bm25_and_method_is_opt_in() -> None:
    assert V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS
    assert normalize_method("V6_4_SAFE_FUSION") == V64_METHOD_NAME
    assert inspect.signature(recommend_from_query).parameters["method"].default == "bm25"


def test_v6_6_frontend_defaults_unchanged() -> None:
    search_page = (ROOT / "frontend" / "src" / "pages" / "SearchPage.tsx").read_text(encoding="utf-8")
    client = (ROOT / "frontend" / "src" / "api" / "client.ts").read_text(encoding="utf-8")

    assert "useState('embedding')" in search_page
    assert "method = 'hybrid'" in client


def test_v6_6_protected_hashes_are_unchanged() -> None:
    report = json.loads(REPRO_REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}

    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == report["protected_hashes"]["v3_9_labels_sha256"]
