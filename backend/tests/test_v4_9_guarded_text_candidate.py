import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.validate_v4_9_guarded_text_candidate import acceptance_check  # noqa: E402
from scripts.validate_v4_6_guarded_text_blend import guardrail_feature_audit  # noqa: E402

REPORT = ROOT / "data" / "eval" / "results" / "v4_9_guarded_text_candidate_validation_report.json"
AUDIT = ROOT / "data" / "eval" / "results" / "v4_9_guarded_text_leakage_audit.json"
FAILURES = ROOT / "data" / "eval" / "results" / "v4_9_guarded_text_failure_examples.jsonl"
CONFIG = ROOT / "data" / "processed" / "models" / "v4_9_guarded_text_blend_candidate.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v4_9_report_schema_and_offline_flags() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v4.9_guarded_text_candidate_validation_report"
    assert report["guardrail_method"] == "v4_6_guardrail_text_when_rankers_not_confident"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False


def test_v4_9_v4_8_override_behavior() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    override = report["v4_8_override_behavior"]
    assert override["base_label_count"] == 2400
    assert override["v4_8_label_count"] == 25
    assert override["effective_label_count"] == 2400
    assert override["overlap_with_base_labels"] == 25
    assert override["v4_8_duplicate_query_paper_rows"] == []
    assert override["combined_duplicate_query_paper_rows"] == []
    assert report["v4_8_preflight"]["all_query_ids_contrastive"] is True
    assert report["v4_8_preflight"]["all_v4_8_rows_in_v4_7_candidates"] is True


def test_v4_9_leakage_audit_and_forbidden_feature_detection() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["schema_version"] == "v4.6_guardrail_leakage_audit"
    assert audit["uses_only_inference_time_features"] is True
    assert audit["forbidden_features_used"] == []
    assert audit["unknown_features_used"] == []

    dirty = guardrail_feature_audit({"hard_negative_label"})
    assert dirty["uses_only_inference_time_features"] is False
    assert "hard_negative_label" in dirty["forbidden_features_used"]


def test_v4_9_candidate_config_only_written_when_acceptance_passes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["candidate_config_created"] == report["acceptance"]["passes"]
    assert CONFIG.exists() == report["acceptance"]["passes"]
    if CONFIG.exists():
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        assert config["schema_version"] == "v4.9_guarded_text_blend_candidate_config"
        assert config["warning"].startswith("Offline-only candidate")
        assert config["label_table"]["effective_judged_rows"] == 2400


def test_v4_9_acceptance_rejects_failed_override_contract() -> None:
    evaluation = {
        "method_averages": {
            "v4_6_guardrail_text_when_rankers_not_confident": {
                "reading_value_ndcg": 1.0,
                "topic_match_ndcg": 1.0,
                "hard_negative_rate_at_10": 0.0,
                "judged_at_10": 10.0,
            },
            "blend_0_75_v33_0_25_v41": {
                "reading_value_ndcg": 0.9,
                "topic_match_ndcg": 0.9,
                "hard_negative_rate_at_10": 0.1,
                "judged_at_10": 10.0,
            },
        },
        "per_topic": {
            "v4_6_guardrail_text_when_rankers_not_confident": [
                {
                    "query_id": "v2_contrastive_learning",
                    "reading_value_ndcg": 0.9,
                    "topic_match_ndcg": 0.9,
                    "mrr_at_10": 1.0,
                    "positive_recall_at_10": 1.0,
                    "hard_negative_rate_at_10": 0.0,
                    "judged_at_10": 10.0,
                    "unjudged_at_10": 0.0,
                },
                {
                    "query_id": "v2_multimodal_learning",
                    "reading_value_ndcg": 0.9,
                    "topic_match_ndcg": 0.9,
                    "mrr_at_10": 1.0,
                    "positive_recall_at_10": 1.0,
                    "hard_negative_rate_at_10": 0.0,
                    "judged_at_10": 10.0,
                    "unjudged_at_10": 0.0,
                },
            ],
            "blend_0_75_v33_0_25_v41": [
                {
                    "query_id": "v2_contrastive_learning",
                    "reading_value_ndcg": 0.9,
                    "topic_match_ndcg": 0.9,
                    "mrr_at_10": 1.0,
                    "positive_recall_at_10": 1.0,
                    "hard_negative_rate_at_10": 0.1,
                    "judged_at_10": 10.0,
                    "unjudged_at_10": 0.0,
                },
                {
                    "query_id": "v2_multimodal_learning",
                    "reading_value_ndcg": 0.9,
                    "topic_match_ndcg": 0.9,
                    "mrr_at_10": 1.0,
                    "positive_recall_at_10": 1.0,
                    "hard_negative_rate_at_10": 0.0,
                    "judged_at_10": 10.0,
                    "unjudged_at_10": 0.0,
                },
            ],
        },
    }
    result = acceptance_check(
        evaluation=evaluation,
        leakage_audit={"uses_only_inference_time_features": True},
        protected={"v2_1_labels_hash_unchanged": True},
        override_summary={
            "base_label_count": 2400,
            "v4_8_label_count": 25,
            "effective_label_count": 2425,
            "overlap_with_base_labels": 0,
            "v4_8_duplicate_query_paper_rows": [],
            "combined_duplicate_query_paper_rows": [],
        },
    )
    assert result["passes"] is False
    assert result["criteria"]["v4_8_overrides_resolve_without_duplicates"] is False


def test_v4_9_protected_hashes_no_runtime_config_and_failure_examples() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert isinstance(load_jsonl(FAILURES), list)
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v4_9_guarded_text_blend.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v4_9_text_blend.py").exists()
