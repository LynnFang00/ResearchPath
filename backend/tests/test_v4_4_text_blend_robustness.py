import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "data" / "eval" / "results" / "v4_4_text_blend_robustness_report.json"
FAILURES = ROOT / "data" / "eval" / "results" / "v4_4_text_blend_failure_examples.jsonl"
CONFIG = ROOT / "data" / "processed" / "models" / "v4_4_text_blend_candidate.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v4_4_robustness_report_created() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v4.4_text_blend_robustness_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["selected_blend"]["method"]
    assert isinstance(report["acceptance"]["passes"], bool)
    assert report["per_topic_wins_losses_vs_v4_1_blend"]


def test_v4_4_config_creation_matches_acceptance() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["candidate_config_created"] == report["acceptance"]["passes"]
    assert CONFIG.exists() == report["candidate_config_created"]
    if CONFIG.exists():
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        assert config["schema_version"] == "v4.4_text_blend_candidate_config"
        assert config["warning"].startswith("Offline-only")


def test_v4_4_failure_examples_jsonl_and_hashes() -> None:
    examples = load_jsonl(FAILURES)
    assert isinstance(examples, list)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
