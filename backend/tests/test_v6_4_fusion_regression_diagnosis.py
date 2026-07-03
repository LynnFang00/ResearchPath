import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_v6_4_fusion_regressions import (  # noqa: E402
    FORBIDDEN_INFERENCE_FEATURES,
    REFINED_METHODS,
    SAFE_FEATURES,
    method_passes,
    refined_feature_leakage,
)
from scripts.validate_v3_9_semantic_expansion_labels import (  # noqa: E402
    DEFAULT_LABELS as V39_LABELS,
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    EXPECTED_V35_LABELS_SHA256,
    file_sha256,
)


DIAG = ROOT / "data" / "eval" / "results" / "v6_4_fusion_regression_diagnosis_report.json"
EXPERIMENTS = ROOT / "data" / "eval" / "results" / "v6_4_safe_fusion_experiments.json"
EXAMPLES = ROOT / "data" / "eval" / "results" / "v6_4_fusion_regression_examples.jsonl"
CANDIDATE = ROOT / "data" / "processed" / "models" / "v6_4_safe_fusion_candidate.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v6_4_report_schema_and_input_diagnostics() -> None:
    report = load_json(DIAG)
    assert report["schema_version"] == "v6.4_fusion_regression_diagnosis_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_model_artifacts_overwritten"] is False
    assert report["input_diagnostics"]["v6_3_prediction_rows"] == 2480
    assert report["input_diagnostics"]["v6_1_8_prediction_rows"] == 2480
    assert report["input_diagnostics"]["v6_1_7_feature_matrix_rows"] == 2480


def test_v6_4_severe_regression_examples_are_exported() -> None:
    report = load_json(DIAG)
    examples = load_jsonl(EXAMPLES)
    assert report["v6_3_ridge_severe_regression_count"] == 3
    assert report["v6_3_ridge_non_weak_severe_regression_count"] == 2
    assert len(report["severe_regression_groups"]) == 3
    assert examples
    assert {row["schema_version"] for row in examples} == {"v6.4_fusion_regression_example"}
    topics = {row["query_id"] for row in examples}
    assert {"v2_graph_recommendation", "v2_causal_representation_learning", "v2_diffusion_image_generation"} <= topics
    assert report["cause_distribution"]["neural_overrides_v4_9_too_much"] == 3
    assert report["cause_distribution"]["foundational_good_paper_demoted"] == 3


def test_v6_4_refined_features_exclude_forbidden_fields() -> None:
    leakage = refined_feature_leakage()
    assert leakage["uses_only_inference_safe_features"] is True
    assert leakage["forbidden_features_used"] == []
    assert set(SAFE_FEATURES).isdisjoint(FORBIDDEN_INFERENCE_FEATURES)
    report = load_json(DIAG)
    assert report["refined_feature_leakage"] == leakage


def test_v6_4_safe_fusion_experiment_metrics_and_gate() -> None:
    diag = load_json(DIAG)
    experiments = load_json(EXPERIMENTS)
    assert experiments["schema_version"] == "v6.4_safe_fusion_experiments"
    assert experiments["best_safe_method"] == "v6_4_top10_swap_limited_2"
    assert diag["best_safe_method"] == "v6_4_top10_swap_limited_2"
    assert set(REFINED_METHODS) <= set(experiments["methods"])

    base = experiments["methods"]["v4_9_guarded_text_blend"]["test_metrics"]
    best = experiments["methods"]["v6_4_top10_swap_limited_2"]
    best_metrics = best["test_metrics"]
    assert best_metrics["reading_ndcg_at_10"]["mean"] == pytest.approx(0.7430484350944331)
    assert best_metrics["topic_ndcg_at_10"]["mean"] == pytest.approx(0.8355085323660981)
    assert best_metrics["hard_neg_at_10"]["mean"] == pytest.approx(0.08750000000000001)
    assert best_metrics["reading_ndcg_at_10"]["mean"] >= base["reading_ndcg_at_10"]["mean"]
    assert best_metrics["topic_ndcg_at_10"]["mean"] >= base["topic_ndcg_at_10"]["mean"]
    assert best_metrics["hard_neg_at_10"]["mean"] <= base["hard_neg_at_10"]["mean"]
    assert best["severe_regression_count"] == 0
    assert best["non_weak_severe_regression_count"] == 0
    assert experiments["acceptance"]["v6_4_top10_swap_limited_2"]["passes"] is True
    assert experiments["acceptance"]["v6_3_ridge_fusion"]["passes"] is False
    assert method_passes(
        {"test": {method: row["test_metrics"] for method, row in experiments["methods"].items()}},
        {
            method: {
                "severe_regression_count": row["severe_regression_count"],
                "non_weak_severe_regression_count": row["non_weak_severe_regression_count"],
                "judged_at_10_decreases": row["judged_at_10_decreases"],
            }
            for method, row in experiments["methods"].items()
        },
        diag["protected_hashes"],
        "v6_4_top10_swap_limited_2",
    )


def test_v6_4_candidate_config_created_only_after_acceptance() -> None:
    diag = load_json(DIAG)
    candidate = load_json(CANDIDATE)
    assert diag["candidate_config_created"] is True
    assert candidate["schema_version"] == "v6.4_safe_fusion_candidate"
    assert candidate["method"] == "v6_4_top10_swap_limited_2"
    assert candidate["acceptance_passed"] is True
    assert candidate["runtime_integration_status"] == "offline_only_not_integrated"
    assert candidate["production_candidate_created"] is False
    assert candidate["forbidden_features_used"] == []


def test_v6_4_protected_hashes_are_unchanged() -> None:
    report = load_json(DIAG)
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == report["protected_hashes"]["v3_9_labels_sha256"]


def test_v6_4_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_4_safe_fusion.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_4_safe_fusion.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_4_safe_fusion.ts").exists()
