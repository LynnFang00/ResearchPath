import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v6_1_3_exact_learned_baselines import (  # noqa: E402
    LEARNED_METHODS,
    METHODS,
    NEURAL_METHOD,
    evaluate_method,
    exact_feature_status,
    load_jsonl,
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


REPORT = ROOT / "data" / "eval" / "results" / "v6_1_3_exact_learned_baseline_report.json"
PARITY = ROOT / "data" / "eval" / "results" / "v6_1_3_exact_scoring_parity_report.json"
PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_3_exact_baseline_predictions.jsonl"
FAILURES = ROOT / "data" / "eval" / "results" / "v6_1_3_neural_vs_v4_9_failure_examples.jsonl"


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_1_3_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.3_exact_learned_baseline_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["models_retrained_by_this_script"] is False
    assert report["existing_model_artifacts_overwritten"] is False
    assert report["production_candidate_created"] is False
    assert report["dataset"] == {"fold_count": 4, "row_count": 2480, "topic_count": 16}
    assert report["methods"] == METHODS


def test_v6_1_3_method_coverage_and_explicit_scoring_failures() -> None:
    report = load_report()
    assert report["aggregate"]["test"][NEURAL_METHOD]["reading_ndcg_at_10"]["count"] == 4
    assert report["aggregate"]["test"]["bm25"]["reading_ndcg_at_10"]["count"] == 4
    for method in LEARNED_METHODS:
        status = report["scoring_status"][method]
        assert status["artifact_loaded"] is True
        assert status["exact_scoring_succeeded"] is False
        assert status["scorable_rows"] == 0
        assert status["unscorable_rows"] == 2480
        assert status["unavailable_reason"]
        assert status["missing_feature_counts"]["tfidf_score"] == 2480
        assert status["missing_feature_counts"]["old_v2_2b_score"] == 2480
        assert status["sample_unscorable_rows"]
        assert report["aggregate"]["test"][method]["reading_ndcg_at_10"]["count"] == 0
        assert report["fold_reports"][0]["test"][method]["coverage"]["coverage"] == 0.0
        assert report["fold_reports"][0]["test"][method]["coverage"]["unavailable_reason"] == "No rows have exact scores for this method."


def test_v6_1_3_predictions_and_failure_files_are_explicit() -> None:
    predictions = load_jsonl(PREDICTIONS)
    failures = load_jsonl(FAILURES)
    assert len(predictions) == 2480
    assert all(row["schema_version"] == "v6.1.3_exact_baseline_prediction" for row in predictions[:50])
    assert all(row["score_available"][NEURAL_METHOD] is True for row in predictions)
    assert sum(1 for row in predictions if row["score_available"]["bm25"]) == 2422
    for method in LEARNED_METHODS:
        assert sum(1 for row in predictions if row["score_available"][method]) == 0
        assert all(row["scores"][method] is None for row in predictions[:100])
    assert failures == [
        {
            "failure_type": "v4_9_exact_scores_unavailable",
            "reason": "Cannot create neural-vs-V4.9 row-level failure examples until exact V4.9 runtime feature inputs are available.",
            "schema_version": "v6.1.3_neural_vs_v4_9_failure_example",
        }
    ]


def test_v6_1_3_v4_9_parity_and_forbidden_feature_audit() -> None:
    report = load_report()
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    assert report["parity"] == parity
    assert parity["schema_version"] == "v6.1.3_exact_scoring_parity_report"
    assert parity["offline_only"] is True
    assert parity["runtime_scoring_invoked"] is False
    assert parity["config_matches_runtime_constants"] is True
    assert parity["forbidden_features_required"] == []
    assert parity["uses_only_inference_time_features"] is True
    assert parity["v4_9_formula"]["features_used"] == [
        "normalized_v3_3_score",
        "normalized_v4_1_score",
        "normalized_v4_3_text_score",
        "max_ranker_confidence",
    ]


def test_v6_1_3_metrics_computation_smoke() -> None:
    rows = [
        {
            "query_id": "q",
            f"{NEURAL_METHOD}_score_for_eval": 0.9,
            "reading_value_score": 1.0,
            "topic_match_score": 1.0,
            "relevance_bucket": 2,
            "hard_negative": False,
        },
        {
            "query_id": "q",
            f"{NEURAL_METHOD}_score_for_eval": 0.1,
            "reading_value_score": 0.0,
            "topic_match_score": 0.0,
            "relevance_bucket": 0,
            "hard_negative": True,
        },
    ]
    metrics = evaluate_method(rows, NEURAL_METHOD)
    assert metrics["coverage"]["coverage"] == 1.0
    assert metrics["metrics"]["reading_ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["metrics"]["hard_neg_at_10"] == pytest.approx(0.1)


def test_v6_1_3_feature_status_reports_missing_inputs() -> None:
    rows = [{"query_id": "q", "paper_id": 1, "baseline_rankings": {"bm25": {"score": 0.5, "rank": 2}}, "year": 2024}]
    scorable, status = exact_feature_status(rows, ["bm25_score", "bm25_rank", "tfidf_score", "old_v2_2b_score"])
    assert scorable == []
    assert status["scorable_rows"] == 0
    assert status["unscorable_rows"] == 1
    assert status["missing_feature_counts"] == {"old_v2_2b_score": 1, "tfidf_score": 1}
    assert status["sample_unscorable_rows"][0]["missing_features"] == ["tfidf_score", "old_v2_2b_score"]


def test_v6_1_3_recommendation_is_to_fix_scoring_infrastructure_first() -> None:
    report = load_report()
    assert report["recommendation"]["recommendation"] == "fix_exact_scoring_infrastructure_first"
    assert report["diagnostic_questions"]["does_neural_beat_bm25_consistently"] is False
    assert report["diagnostic_questions"]["does_neural_beat_any_learned_baseline"] is None
    assert report["diagnostic_questions"]["does_neural_reduce_hard_negatives_vs_v4_9"] is None
    assert report["severe_regressions_vs_v4_9"]["available"] is False


def test_v6_1_3_protected_hashes_are_unchanged() -> None:
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


def test_v6_1_3_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_1_3_exact_learned_baselines.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_3_neural_ranker.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_3_exact_learned_baselines.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_3_neural_candidate.json").exists()
