import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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


REPORT = ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_baselines_report.json"
BLEND_REPORT = ROOT / "data" / "eval" / "results" / "v6_1_8_neural_blend_probe_report.json"
PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_predictions.jsonl"
FAILURES = ROOT / "data" / "eval" / "results" / "v6_1_8_neural_vs_v4_9_failure_examples.jsonl"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v6_1_8_report_schema_and_offline_flags() -> None:
    report = load_json(REPORT)
    assert report["schema_version"] == "v6.1.8_exact_neural_vs_learned_baselines_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["models_trained_by_this_script"] is False
    assert report["existing_model_artifacts_overwritten"] is False
    assert report["production_candidate_created"] is False
    assert report["recommendation"] == "neural_is_not_competitive_keep_as_dl_baseline"


def test_v6_1_8_exact_join_coverage_has_no_silent_drops() -> None:
    report = load_json(REPORT)
    join = report["join_diagnostics"]
    assert join["neural_rows"] == 2480
    assert join["learned_rows"] == 2480
    assert join["feature_matrix_rows"] == 2480
    assert join["feature_matrix_unique_keys"] == 2480
    assert join["feature_matrix_duplicate_keys"] == 0
    assert join["joined_rows"] == 2480
    assert join["missing_learned_rows"] == 0
    assert join["missing_feature_matrix_rows"] == 0
    assert join["rows_silently_dropped"] == 0


def test_v6_1_8_predictions_schema_and_scores() -> None:
    rows = load_jsonl(PREDICTIONS)
    assert len(rows) == 2480
    required_methods = {
        "neural_cross_encoder_v6_1",
        "bm25",
        "v3_3_ltr",
        "v4_1_blend",
        "v4_9_guarded_text_blend",
        "probe_v4_9_90_neural_10",
        "probe_v4_9_80_neural_20",
        "probe_v4_9_neural_hard_negative_guardrail",
        "probe_v4_9_neural_when_v49_confidence_low",
    }
    for row in rows[:100]:
        assert row["schema_version"] == "v6.1.8_exact_neural_vs_learned_prediction"
        assert {"fold_id", "split", "query_id", "paper_id", "title", "scores"} <= set(row)
        assert required_methods <= set(row["scores"])
        assert all(math.isfinite(float(row["scores"][method])) for method in required_methods)


def test_v6_1_8_test_metrics_match_expected_exact_comparison() -> None:
    report = load_json(REPORT)
    test = report["aggregate"]["test"]
    assert test["neural_cross_encoder_v6_1"]["reading_ndcg_at_10"]["mean"] == pytest.approx(0.7426644255504069)
    assert test["neural_cross_encoder_v6_1"]["topic_ndcg_at_10"]["mean"] == pytest.approx(0.7569937561221735)
    assert test["neural_cross_encoder_v6_1"]["hard_neg_at_10"]["mean"] == pytest.approx(0.125)
    assert test["v4_9_guarded_text_blend"]["reading_ndcg_at_10"]["mean"] == pytest.approx(0.7241826722124772)
    assert test["v4_9_guarded_text_blend"]["topic_ndcg_at_10"]["mean"] == pytest.approx(0.8273612122808424)
    assert test["v4_9_guarded_text_blend"]["hard_neg_at_10"]["mean"] == pytest.approx(0.1)
    assert test["v3_3_ltr"]["topic_ndcg_at_10"]["mean"] > test["neural_cross_encoder_v6_1"]["topic_ndcg_at_10"]["mean"]


def test_v6_1_8_acceptance_diagnostics_show_neural_instability() -> None:
    diag = load_json(REPORT)["acceptance_diagnostics"]["neural_cross_encoder_v6_1"]
    assert diag["reading_delta_vs_v4_9_test_mean"]["mean"] == pytest.approx(0.01848175333792959)
    assert diag["topic_delta_vs_v4_9_test_mean"]["mean"] == pytest.approx(-0.07036745615866895)
    assert diag["hard_neg_delta_vs_v4_9_test_mean"]["mean"] == pytest.approx(0.024999999999999994)
    assert diag["severe_regression_count"] == 10
    assert diag["non_weak_severe_regression_count"] == 5
    assert diag["judged_at_10_decreases"] is False


def test_v6_1_8_blend_probe_report_identifies_safe_probe_separately() -> None:
    blend = load_json(BLEND_REPORT)
    assert blend["schema_version"] == "v6.1.8_neural_blend_probe_report"
    assert blend["offline_only"] is True
    assert blend["runtime_integration_status"] == "offline_only_not_integrated"
    assert blend["production_candidate_created"] is False
    assert blend["highest_reading_probe"]["method"] == "probe_v4_9_80_neural_20"
    assert blend["highest_reading_probe"]["severe_regression_count"] == 4
    assert blend["safest_probe"]["method"] == "probe_v4_9_neural_when_v49_confidence_low"
    assert blend["safest_probe"]["severe_regression_count"] == 0
    assert blend["safest_probe"]["non_weak_severe_regression_count"] == 0
    assert blend["safest_probe"]["reading_delta_vs_v4_9_test_mean"] == pytest.approx(0.0003116466981666066)
    assert blend["best_probe"] == blend["safest_probe"]


def test_v6_1_8_failure_examples_schema() -> None:
    rows = load_jsonl(FAILURES)
    assert rows
    required = {
        "schema_version",
        "failure_type",
        "fold_id",
        "split",
        "query_id",
        "paper_id",
        "title",
        "rank",
        "neural_score",
        "v4_9_score",
        "relevance_bucket",
        "hard_negative",
    }
    for row in rows[:50]:
        assert row["schema_version"] == "v6.1.8_neural_vs_v4_9_failure_example"
        assert required <= set(row)
        assert math.isfinite(float(row["neural_score"]))
        assert math.isfinite(float(row["v4_9_score"]))


def test_v6_1_8_protected_hashes_are_unchanged() -> None:
    report = load_json(REPORT)
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == report["protected_hashes"]["v3_9_labels_sha256"]


def test_v6_1_8_does_not_create_runtime_frontend_or_model_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_8_neural_blend.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_8_neural_blend.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_8_neural_blend_candidate.json").exists()
