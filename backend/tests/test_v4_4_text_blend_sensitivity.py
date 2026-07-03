import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "data" / "eval" / "results" / "v4_4_text_blend_sensitivity_report.json"


def test_v4_4_sensitivity_report_created_and_weights() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v4.4_text_blend_sensitivity_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert {row["text_weight"] for row in report["sensitivity"]} == {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}
    assert report["selection"]["best"]["method"]


def test_v4_4_sensitivity_metrics_and_hashes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for row in report["sensitivity"]:
        assert 0.0 <= row["reading_value_ndcg"] <= 1.0
        assert 0.0 <= row["topic_match_ndcg"] <= 1.0
        assert row["judged_at_10"] >= 0.0
        assert row["unjudged_at_10"] >= 0.0
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
