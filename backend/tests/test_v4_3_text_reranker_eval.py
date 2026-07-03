import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_production_pool_report.json"


def test_v4_3_production_report_contains_required_methods() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    averages = report["evaluation"]["method_averages"]
    assert "v3_3_random_forest_no_v27_compact" in averages
    assert "blend_0_75_v33_0_25_v41" in averages
    assert report["comparisons"]["best_v43_method"] in averages
    assert "blend_0_70_v33_0_20_v41_0_10_v43_text" in averages
    assert "blend_0_70_v41_blend_0_30_v43_text" in averages


def test_v4_3_production_report_is_offline_only() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["comparisons"]["recommend_runtime_integration"] is False


def test_v4_3_production_report_protected_hashes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
