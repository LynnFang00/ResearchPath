import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_v6_2_pairwise_neural_reranker import (  # noqa: E402
    FORBIDDEN_INFERENCE_INPUTS,
    HARD_NEGATIVE_PAIR_WEIGHT,
    PAIRWISE_METHOD,
    MULTITASK_METHOD,
    assert_inference_safe,
    build_pair_candidates,
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


EXAMPLES = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
PAIRWISE = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_pairwise_train.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_2_pairwise_neural_reranker_report.json"
BLEND = ROOT / "data" / "eval" / "results" / "v6_2_neural_blend_probe_report.json"
PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_2_neural_predictions.jsonl"
FAILURES = ROOT / "data" / "eval" / "results" / "v6_2_neural_vs_v4_9_failure_examples.jsonl"
CONFIG = ROOT / "data" / "processed" / "models" / "v6_2_neural_reranker_config.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v6_2_pairwise_data_loading_and_hard_negative_weights() -> None:
    examples = load_jsonl(EXAMPLES)
    uploaded_pairs = load_jsonl(PAIRWISE)
    assert len(examples) == 2480
    assert len(uploaded_pairs) == 7200
    train_rows = [row for row in examples if row["v6_0_2_grouped_split"] == "train"]
    pairs = build_pair_candidates(train_rows, uploaded_pairs)
    assert pairs
    assert all(pair["preferred_paper_id"] != pair["rejected_paper_id"] for pair in pairs)
    assert any(pair["hard_negative_pair"] for pair in pairs)
    assert all(pair["weight"] == HARD_NEGATIVE_PAIR_WEIGHT for pair in pairs if pair["hard_negative_pair"])
    assert all(pair["weight"] == 1.0 for pair in pairs if not pair["hard_negative_pair"])


def test_v6_2_model_input_excludes_forbidden_fields() -> None:
    leakage = assert_inference_safe()
    assert leakage["input_is_inference_safe"] is True
    assert leakage["input_fields"] == ["abstract", "query", "title"]
    assert leakage["forbidden_fields_used"] == []
    assert "hard_negative" in FORBIDDEN_INFERENCE_INPUTS
    assert leakage["hard_negative_used_as_inference_input"] is False
    assert leakage["hard_negative_used_as_training_target_or_weight_only"] is True


def test_v6_2_report_schema_and_offline_flags() -> None:
    report = load_json(REPORT)
    assert report["schema_version"] == "v6.2_pairwise_neural_reranker_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_model_artifacts_overwritten"] is False
    assert report["production_candidate_created"] is False
    assert report["large_checkpoints_created"] is False
    assert report["variants_trained"] == [PAIRWISE_METHOD, MULTITASK_METHOD]
    assert report["fold_training"]["pairwise"]["trained_successfully"] is True
    assert report["fold_training"]["multitask"]["trained_successfully"] is True
    assert report["recommendation"] == "research_baseline_only_do_not_integrate"


def test_v6_2_predictions_schema_and_metric_values() -> None:
    rows = load_jsonl(PREDICTIONS)
    assert len(rows) == 2480
    for row in rows[:100]:
        assert row["schema_version"] == "v6.2_neural_prediction"
        assert {PAIRWISE_METHOD, MULTITASK_METHOD, "neural_cross_encoder_v6_1", "v4_9_guarded_text_blend"} <= set(row["scores"])
        assert math.isfinite(float(row["scores"][PAIRWISE_METHOD]))
        assert math.isfinite(float(row["scores"][MULTITASK_METHOD]))

    report = load_json(REPORT)
    test = report["aggregate"]["test"]
    assert test[PAIRWISE_METHOD]["reading_ndcg_at_10"]["mean"] == pytest.approx(0.6292473402791575)
    assert test[MULTITASK_METHOD]["reading_ndcg_at_10"]["mean"] == pytest.approx(0.7097874966243601)
    assert test[MULTITASK_METHOD]["topic_ndcg_at_10"]["mean"] == pytest.approx(0.7426144000104442)
    assert test[MULTITASK_METHOD]["hard_neg_at_10"]["mean"] == pytest.approx(0.1125)
    assert test["v4_9_guarded_text_blend"]["topic_ndcg_at_10"]["mean"] > test[MULTITASK_METHOD]["topic_ndcg_at_10"]["mean"]


def test_v6_2_acceptance_diagnostics_vs_v4_9_and_v6_1() -> None:
    report = load_json(REPORT)
    multi = report["acceptance_diagnostics"][MULTITASK_METHOD]
    pairwise = report["acceptance_diagnostics"][PAIRWISE_METHOD]
    v61 = report["acceptance_diagnostics"]["neural_cross_encoder_v6_1"]
    assert multi["severe_regression_count"] == 7
    assert multi["non_weak_severe_regression_count"] == 5
    assert pairwise["severe_regression_count"] == 8
    assert report["v6_1_severe_regression_reference_count"] == 10
    assert report["v6_2_reduces_v6_1_severe_regressions"] is True
    assert report["v6_2_reduces_hard_negative_promotions"] is True
    assert multi["hard_neg_delta_vs_v4_9_test_mean"]["mean"] == pytest.approx(0.0125)
    assert v61["hard_neg_delta_vs_v4_9_test_mean"]["mean"] == pytest.approx(0.024999999999999994)


def test_v6_2_blend_probes_are_offline_only() -> None:
    blend = load_json(BLEND)
    assert blend["schema_version"] == "v6.2_neural_blend_probe_report"
    assert blend["offline_only"] is True
    assert blend["runtime_integration_status"] == "offline_only_not_integrated"
    assert blend["production_candidate_created"] is False
    assert blend["best_probe"]["method"] == "probe_v4_9_v6_2_multitask_neural_when_v49_confidence_low"
    assert blend["best_probe"]["severe_regression_count"] == 0
    assert blend["best_probe"]["non_weak_severe_regression_count"] == 0
    assert blend["best_probe"]["reading_delta_vs_v4_9_test_mean"] == pytest.approx(0.0003116466981666066)


def test_v6_2_config_is_offline_and_no_checkpoint_saved() -> None:
    config = load_json(CONFIG)
    assert config["schema_version"] == "v6.2_neural_reranker_config"
    assert config["runtime_integration_status"] == "offline_only_not_integrated"
    assert config["production_candidate_created"] is False
    assert config["checkpoint_saved"] is False
    assert config["weights_committed"] is False
    assert config["input_fields"] == ["abstract", "query", "title"]
    assert config["forbidden_inference_fields_used"] == []


def test_v6_2_failure_examples_schema() -> None:
    rows = load_jsonl(FAILURES)
    assert rows
    required = {
        "schema_version",
        "diagnosed_variant",
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
        assert row["schema_version"] == "v6.2_neural_vs_v4_9_failure_example"
        assert required <= set(row)


def test_v6_2_protected_hashes_are_unchanged() -> None:
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


def test_v6_2_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_2_pairwise_neural_reranker.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_2_pairwise_neural.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_2_pairwise_neural.ts").exists()
