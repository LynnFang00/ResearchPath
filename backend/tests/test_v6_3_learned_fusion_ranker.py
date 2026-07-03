import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_v6_3_learned_fusion_ranker import (  # noqa: E402
    FEATURE_NAMES,
    FORBIDDEN_INFERENCE_FEATURES,
    FUSION_METHODS,
    build_scored_rows,
    candidate_passes,
    leakage_check,
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


V618 = ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_predictions.jsonl"
V62 = ROOT / "data" / "eval" / "results" / "v6_2_neural_predictions.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_ranker_report.json"
PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_predictions.jsonl"
FAILURES = ROOT / "data" / "eval" / "results" / "v6_3_fusion_vs_v4_9_failure_examples.jsonl"
IMPORTANCE = ROOT / "data" / "eval" / "results" / "v6_3_fusion_feature_importance.json"
CANDIDATE = ROOT / "data" / "processed" / "models" / "v6_3_learned_fusion_candidate.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v6_3_feature_construction_excludes_forbidden_fields() -> None:
    check = leakage_check()
    assert check["uses_only_inference_safe_features"] is True
    assert check["forbidden_features_used"] == []
    assert set(FEATURE_NAMES).isdisjoint(FORBIDDEN_INFERENCE_FEATURES)
    rows = build_scored_rows(V618, V62)
    assert len(rows) == 2480
    for row in rows[:50]:
        assert set(row["fusion_features"]) == set(FEATURE_NAMES)
        assert len(row["feature_vector"]) == len(FEATURE_NAMES)
        assert all(math.isfinite(float(value)) for value in row["feature_vector"])


def test_v6_3_grouped_folds_prevent_query_leakage() -> None:
    report = load_json(REPORT)
    assert report["query_id_leakage_detected"] is False
    for fold in report["folds"]:
        train = set(fold["train_topics"])
        eval_topics = set(fold["val_topics"]) | set(fold["test_topics"])
        assert train.isdisjoint(eval_topics)
        assert fold["query_id_leakage"] is False
        assert fold["train_rows"] > 0
        assert fold["val_rows"] > 0
        assert fold["test_rows"] > 0


def test_v6_3_report_schema_and_offline_flags() -> None:
    report = load_json(REPORT)
    assert report["schema_version"] == "v6.3_learned_fusion_ranker_report"
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
    assert report["fusion_methods"] == FUSION_METHODS
    assert report["best_fusion_method"] == "v6_3_ridge_fusion"
    assert report["candidate_config_created"] is False
    assert report["recommendation"] == "research_only_no_candidate"


def test_v6_3_predictions_and_metrics() -> None:
    rows = load_jsonl(PREDICTIONS)
    assert len(rows) == 2480
    for row in rows[:100]:
        assert row["schema_version"] == "v6.3_learned_fusion_prediction"
        assert set(FUSION_METHODS) <= set(row["scores"])
        assert set(row["fusion_features"]) == set(FEATURE_NAMES)
        assert all(math.isfinite(float(row["scores"][method])) for method in FUSION_METHODS)

    report = load_json(REPORT)
    test = report["aggregate"]["test"]
    assert test["v6_3_ridge_fusion"]["reading_ndcg_at_10"]["mean"] == pytest.approx(0.8045949525774437)
    assert test["v6_3_ridge_fusion"]["topic_ndcg_at_10"]["mean"] == pytest.approx(0.836165366039902)
    assert test["v6_3_ridge_fusion"]["hard_neg_at_10"]["mean"] == pytest.approx(0.05)
    assert test["v6_3_ridge_fusion"]["reading_ndcg_at_10"]["mean"] > test["v4_9_guarded_text_blend"]["reading_ndcg_at_10"]["mean"]
    assert test["v6_3_ridge_fusion"]["topic_ndcg_at_10"]["mean"] > test["v4_9_guarded_text_blend"]["topic_ndcg_at_10"]["mean"]
    assert test["v6_3_ridge_fusion"]["hard_neg_at_10"]["mean"] < test["v4_9_guarded_text_blend"]["hard_neg_at_10"]["mean"]


def test_v6_3_acceptance_and_candidate_gate() -> None:
    report = load_json(REPORT)
    ridge = report["acceptance_diagnostics"]["v6_3_ridge_fusion"]
    guarded = report["acceptance_diagnostics"]["v6_3_guarded_fusion"]
    assert ridge["severe_regression_count"] == 3
    assert ridge["non_weak_severe_regression_count"] == 2
    assert guarded["severe_regression_count"] == 0
    assert candidate_passes(report, "v6_3_ridge_fusion") is False
    assert report["candidate_config_created"] is False
    assert not CANDIDATE.exists()


def test_v6_3_feature_importance_report() -> None:
    importance = load_json(IMPORTANCE)
    assert importance["schema_version"] == "v6.3_fusion_feature_importance"
    assert importance["feature_names"] == FEATURE_NAMES
    assert "ridge_coefficients" in importance
    assert "random_forest_feature_importance" in importance
    assert importance["top_random_forest_features"]
    assert importance["top_ridge_abs_coefficients"]
    assert importance["top_random_forest_features"][0]["feature"] == "v4_9_guarded_text_blend_score"
    assert importance["top_ridge_abs_coefficients"][0]["feature"] == "v4_9_guarded_text_blend_score"


def test_v6_3_failure_examples_schema() -> None:
    rows = load_jsonl(FAILURES)
    assert rows
    required = {
        "schema_version",
        "failure_type",
        "method",
        "fold_id",
        "split",
        "query_id",
        "paper_id",
        "title",
        "fusion_score",
        "v4_9_score",
        "relevance_bucket",
        "hard_negative",
    }
    for row in rows[:50]:
        assert row["schema_version"] == "v6.3_fusion_vs_v4_9_failure_example"
        assert required <= set(row)


def test_v6_3_protected_hashes_are_unchanged() -> None:
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


def test_v6_3_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_3_learned_fusion_ranker.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_3_learned_fusion.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_3_learned_fusion.ts").exists()
