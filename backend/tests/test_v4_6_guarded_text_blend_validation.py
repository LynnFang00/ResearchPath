import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.validate_v4_6_guarded_text_blend import guardrail_feature_audit  # noqa: E402

REPORT = ROOT / "data" / "eval" / "results" / "v4_6_guarded_text_blend_validation_report.json"
AUDIT = ROOT / "data" / "eval" / "results" / "v4_6_guardrail_leakage_audit.json"
FAILURES = ROOT / "data" / "eval" / "results" / "v4_6_guardrail_failure_examples.jsonl"
CONFIG = ROOT / "data" / "processed" / "models" / "v4_6_guarded_text_blend_candidate.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v4_6_report_schema_and_offline_flags() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v4.6_guarded_text_blend_validation_report"
    assert report["guardrail_method"] == "v4_6_guardrail_text_when_rankers_not_confident"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False


def test_v4_6_guardrail_formula_determinism_and_leakage_detection() -> None:
    clean = guardrail_feature_audit()
    repeat = guardrail_feature_audit()
    dirty = guardrail_feature_audit({"reading_value_score"})
    assert clean["formula"] == repeat["formula"]
    assert clean["uses_only_inference_time_features"] is True
    assert dirty["uses_only_inference_time_features"] is False
    assert "reading_value_score" in dirty["forbidden_features_used"]
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["uses_only_inference_time_features"] is True


def test_v4_6_candidate_config_only_written_when_acceptance_passes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["candidate_config_created"] == report["acceptance"]["passes"]
    assert CONFIG.exists() == report["candidate_config_created"]
    if CONFIG.exists():
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        assert config["schema_version"] == "v4.6_guarded_text_blend_candidate_config"
        assert config["warning"].startswith("Offline-only")


def test_v4_6_no_runtime_config_and_protected_hashes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v4_6_guarded_text_blend.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v4_6_text_blend.py").exists()
    assert isinstance(load_jsonl(FAILURES), list)
