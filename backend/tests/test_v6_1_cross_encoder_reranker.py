import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_v6_1_cross_encoder_reranker import (  # noqa: E402
    FORBIDDEN_INPUT_FIELDS,
    assert_no_forbidden_input_fields,
    construct_model_input,
    ranking_metrics,
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


TRAIN = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_train.jsonl"
VAL = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_val.jsonl"
TEST = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_test.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_reranker_report.json"
VAL_PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_predictions_val.jsonl"
TEST_PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_predictions_test.jsonl"
FAILURES = ROOT / "data" / "eval" / "results" / "v6_1_cross_encoder_failure_examples.jsonl"
CONFIG = ROOT / "data" / "processed" / "models" / "v6_1_cross_encoder_reranker_config.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_1_dataset_loading_counts() -> None:
    assert len(load_jsonl(TRAIN)) == 1820
    assert len(load_jsonl(VAL)) == 332
    assert len(load_jsonl(TEST)) == 328
    assert sum(1 for row in load_jsonl(TRAIN) if row["label_source"] == "v6_0_2") == 20
    assert sum(1 for row in load_jsonl(VAL) if row["label_source"] == "v6_0_2") == 32
    assert sum(1 for row in load_jsonl(TEST) if row["label_source"] == "v6_0_2") == 28


def test_v6_1_input_construction_excludes_forbidden_features() -> None:
    row = load_jsonl(TRAIN)[0]
    model_input = construct_model_input(row)
    leakage = assert_no_forbidden_input_fields({"query", "title", "abstract"})
    assert "[SEP]" in model_input
    assert row["query"] in model_input
    assert row["title"] in model_input
    assert leakage["input_is_inference_safe"] is True
    assert leakage["forbidden_fields_used"] == []
    assert "reading_value_score" in FORBIDDEN_INPUT_FIELDS
    assert "baseline_rankings" in FORBIDDEN_INPUT_FIELDS


def test_v6_1_metrics_computation_smoke() -> None:
    rows = [
        {"query_id": "q", "score": 0.9, "reading_value_score": 1.0, "topic_match_score": 1.0, "relevance_bucket": 2, "hard_negative": False},
        {"query_id": "q", "score": 0.3, "reading_value_score": 0.4, "topic_match_score": 0.5, "relevance_bucket": 1, "hard_negative": False},
        {"query_id": "q", "score": 0.1, "reading_value_score": 0.0, "topic_match_score": 0.0, "relevance_bucket": 0, "hard_negative": True},
    ]
    metrics = ranking_metrics(rows, "score")
    assert metrics["aggregate"]["reading_ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["aggregate"]["topic_ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["aggregate"]["mrr_at_10"] == pytest.approx(1.0)
    assert metrics["aggregate"]["hard_neg_at_10"] == pytest.approx(0.1)


def test_v6_1_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1_cross_encoder_reranker_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_model_artifacts_overwritten"] is False
    assert report["model_type"] in {"cross_encoder_transformer", "fallback_neural_mlp"}
    assert report["dataset"] == {
        "train_rows": 1820,
        "val_rows": 332,
        "test_rows": 328,
        "train_v6_0_2_rows": 20,
        "val_v6_0_2_rows": 32,
        "test_v6_0_2_rows": 28,
    }
    assert report["leakage_check"]["input_is_inference_safe"] is True


def test_v6_1_prediction_file_schema_is_deterministic() -> None:
    val_rows = load_jsonl(VAL_PREDICTIONS)
    test_rows = load_jsonl(TEST_PREDICTIONS)
    required = {
        "schema_version",
        "split",
        "query_id",
        "query",
        "paper_id",
        "title",
        "neural_score",
        "reading_value_score",
        "topic_match_score",
        "relevance_bucket",
        "hard_negative",
        "label_source",
        "baseline_rankings",
    }
    assert len(val_rows) == 332
    assert len(test_rows) == 328
    assert all(required <= set(row) for row in val_rows[:10] + test_rows[:10])
    assert all(row["schema_version"] == "v6.1_cross_encoder_prediction" for row in val_rows + test_rows)
    assert all(0.0 <= row["neural_score"] <= 1.0 for row in val_rows + test_rows)
    assert [row["paper_id"] for row in val_rows[:5]] == [6, 26, 31, 36, 37]


def test_v6_1_report_metrics_and_baseline_availability() -> None:
    report = load_report()
    assert report["best_validation_epoch"] >= 1
    assert report["validation"]["ranking"]["aggregate"]["judged_at_10"] == 10.0
    assert report["test"]["ranking"]["aggregate"]["judged_at_10"] == 10.0
    assert report["baseline_comparison"]["val"]["bm25"]["available_rows"] == 310
    assert report["baseline_comparison"]["test"]["bm25"]["available_rows"] == 306
    assert report["baseline_comparison"]["val"]["v3_3_ltr"]["metrics"] is None
    assert report["baseline_comparison"]["test"]["v4_9_guarded_text_blend"]["metrics"] is None
    assert report["pairwise_data_audit"]["pairwise_example_count"] == 7200
    assert report["pairwise_data_audit"]["used_in_v6_1_training"] is False
    assert report["pairwise_data_audit"]["v6_0_2_labels_represented_pair_count"] == 266


def test_v6_1_config_and_failure_examples() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    failures = load_jsonl(FAILURES)
    assert config["schema_version"] == "v6.1_cross_encoder_reranker_config"
    assert config["runtime_integration_status"] == "offline_only_not_integrated"
    assert config["checkpoint_saved"] is False
    assert config["weights_committed"] is False
    assert config["forbidden_fields_used"] == []
    assert failures
    assert all(row["schema_version"] == "v6.1_cross_encoder_failure_example" for row in failures[:10])


def test_v6_1_protected_hashes_are_unchanged() -> None:
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


def test_v6_1_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_1_cross_encoder_reranker.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "cross_encoder_reranker_service.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_cross_encoder_reranker.ts").exists()
