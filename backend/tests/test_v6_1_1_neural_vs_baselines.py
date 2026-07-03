import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v6_1_1_neural_vs_baselines import METHODS, NEURAL_METHOD, method_metrics  # noqa: E402
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


REPORT = ROOT / "data" / "eval" / "results" / "v6_1_1_neural_vs_baselines_report.json"
VAL_PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_1_neural_vs_baselines_predictions_val.jsonl"
TEST_PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_1_neural_vs_baselines_predictions_test.jsonl"
FAILURES = ROOT / "data" / "eval" / "results" / "v6_1_1_neural_validation_failure_examples.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_1_1_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.1_neural_vs_baselines_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False
    assert report["model_retrained_by_this_script"] is False
    assert report["methods"] == METHODS
    assert report["all_baselines_comparable"] is False


def test_v6_1_1_method_coverage_is_reported() -> None:
    report = load_report()
    assert report["comparison"]["val"][NEURAL_METHOD]["coverage"]["coverage"] == 1.0
    assert report["comparison"]["test"][NEURAL_METHOD]["coverage"]["coverage"] == 1.0
    assert report["comparison"]["val"]["bm25"]["coverage"]["available_rows"] == 310
    assert report["comparison"]["test"]["bm25"]["coverage"]["available_rows"] == 306
    for method in ("v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"):
        assert report["comparison"]["val"][method]["coverage"]["available_rows"] == 0
        assert report["comparison"]["val"][method]["metrics"] is None
        assert report["baseline_recovery"][method].startswith("Unavailable")


def test_v6_1_1_metrics_computation_smoke() -> None:
    rows = [
        {"query_id": "q", "v6_1_neural_score": 0.9, "reading_value_score": 1.0, "topic_match_score": 1.0, "relevance_bucket": 2, "hard_negative": False},
        {"query_id": "q", "v6_1_neural_score": 0.2, "reading_value_score": 0.2, "topic_match_score": 0.2, "relevance_bucket": 0, "hard_negative": True},
    ]
    metrics = method_metrics(rows, NEURAL_METHOD)
    assert metrics["coverage"]["available_rows"] == 2
    assert metrics["metrics"]["reading_ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["metrics"]["hard_neg_at_10"] == pytest.approx(0.1)


def test_v6_1_1_validation_diagnosis_fields_exist() -> None:
    report = load_report()
    diagnosis = report["validation_diagnosis"]
    assert diagnosis["val_topics"] == ["v2_ai_for_scientific_discovery", "v2_multimodal_learning"]
    assert diagnosis["test_topics"] == ["v2_retrieval_augmented_generation", "v2_robot_learning"]
    assert diagnosis["validation_harder_assessment"]["validation_appears_harder_than_test"] is True
    assert diagnosis["validation_harder_assessment"]["likely_high_variance"] is True
    assert diagnosis["failure_summary"]
    assert diagnosis["top_validation_failure_examples"]
    assert "v2_ai_for_scientific_discovery" in diagnosis["val_topic_profile"]
    assert "bm25" in diagnosis["per_topic_baseline_metrics_val"]


def test_v6_1_1_predictions_and_failures_schema() -> None:
    val_rows = load_jsonl(VAL_PREDICTIONS)
    test_rows = load_jsonl(TEST_PREDICTIONS)
    failures = load_jsonl(FAILURES)
    required = {
        "schema_version",
        "split",
        "query_id",
        "paper_id",
        "title",
        "scores",
        "score_available",
        "reading_value_score",
        "topic_match_score",
        "relevance_bucket",
        "hard_negative",
    }
    assert len(val_rows) == 332
    assert len(test_rows) == 328
    assert all(required <= set(row) for row in val_rows[:10] + test_rows[:10])
    assert all(row["schema_version"] == "v6.1.1_neural_vs_baselines_prediction" for row in val_rows + test_rows)
    assert all(row["score_available"][NEURAL_METHOD] is True for row in val_rows + test_rows)
    assert failures
    assert all(row["schema_version"] == "v6.1.1_neural_validation_failure_example" for row in failures)


def test_v6_1_1_v4_9_win_loss_is_marked_unavailable() -> None:
    report = load_report()
    assert report["topic_win_loss_tie_vs_v4_9"]["val"]["available"] is False
    assert report["topic_win_loss_tie_vs_v4_9"]["test"]["available"] is False
    assert report["topic_win_loss_tie_vs_v4_9"]["val"]["severe_regression_count_reading_delta_lte_minus_0_05"] is None


def test_v6_1_1_model_selection_recommendation() -> None:
    report = load_report()
    assert report["model_selection_diagnosis"]["recommendation"] == "improve_cross_validation_first"
    assert report["leave_one_topic_out"]["available"] is False


def test_v6_1_1_protected_hashes_are_unchanged() -> None:
    report = load_report()
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == report["protected_hashes"]["v3_9_labels_sha256"]


def test_v6_1_1_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_1_1_neural_vs_baselines.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "neural_baseline_comparison.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_1_neural_vs_baselines.ts").exists()
