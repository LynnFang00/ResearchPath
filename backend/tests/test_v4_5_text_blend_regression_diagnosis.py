import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "data" / "eval" / "results" / "v4_5_text_blend_regression_diagnosis_report.json"
GUARDRAILS = ROOT / "data" / "eval" / "results" / "v4_5_guardrail_experiments.json"
FAILURES = ROOT / "data" / "eval" / "results" / "v4_5_multimodal_learning_failure_examples.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v4_5_report_schema_and_read_only_flags() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v4.5_text_blend_regression_diagnosis_report"
    assert report["target_regression_topic"] == "v2_multimodal_learning"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False


def test_v4_5_top10_and_failure_examples() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    top10 = report["top10_by_topic_and_ranker"]["v2_multimodal_learning"]
    expected = {
        "v3_3_random_forest_no_v27_compact",
        "v4_1_downweight_v3_9_active_learning",
        "blend_0_75_v33_0_25_v41",
        "text_weight_0_05",
        "text_weight_0_10",
        "text_weight_0_15",
    }
    assert set(top10) == expected
    assert all(len(rows) == 10 for rows in top10.values())
    failures = load_jsonl(FAILURES)
    assert failures
    assert {row["query_id"] for row in failures} == {"v2_multimodal_learning"}


def test_v4_5_guardrails_are_deterministic_and_offline() -> None:
    first = json.loads(GUARDRAILS.read_text(encoding="utf-8"))
    second = json.loads(GUARDRAILS.read_text(encoding="utf-8"))
    assert first["schema_version"] == "v4.5_guardrail_experiments_report"
    assert first["experiments"] == second["experiments"]
    assert first["runtime_integration_status"] == "offline_only_not_integrated"
    assert first["runtime_defaults_changed_by_this_script"] is False
    assert first["frontend_defaults_changed_by_this_script"] is False
    assert first["safest_method"]["method"]


def test_v4_5_protected_hashes_and_no_runtime_config() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert not (ROOT / "data" / "processed" / "models" / "v4_5_text_blend_candidate.json").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v4_5_text_blend.py").exists()
