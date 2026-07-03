import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_0_production_pool_2400_report.json"


def test_v4_0_production_pool_report_exists_and_has_expected_methods() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    averages = report["evaluation"]["method_averages"]
    v4_method = report["comparison"]["best_v4_no_v27_method"]

    assert report["dataset"]["label_count"] == 2400
    assert report["dataset"]["duplicate_query_paper_rows"] == []
    assert "v2_7" in averages
    assert "v2_6" in averages
    assert "v3_3_random_forest_no_v27_compact" in averages
    assert v4_method in averages
    assert report["runtime_integration_status"] == "offline_only_not_integrated"


def test_v4_0_production_pool_reports_comparison_fields() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    comparison = report["comparison"]

    assert "v4_vs_v3_3_reading_delta" in comparison
    assert "v4_vs_v2_7_reading_delta" in comparison
    assert "v4_vs_v3_3_hard_negative_delta" in comparison
    assert comparison["best_v4_no_v27_method"].startswith("v4_0_")
    assert report["corpus_expanded_by_this_script"] is False
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
