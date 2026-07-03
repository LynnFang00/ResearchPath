import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_v4_0_vs_v3_3_error_analysis.json"
EXAMPLES = REPO_ROOT / "data" / "eval" / "results" / "v4_1_v4_0_vs_v3_3_disagreement_examples.jsonl"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v4_1_error_analysis_report_exists_and_is_safe() -> None:
    report = load_json(REPORT)
    assert report["schema_version"] == "v4.1_v4_0_vs_v3_3_error_analysis"
    assert report["dataset"]["label_count"] == 2400
    assert report["labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["methods"]["v3_3"] in report["evaluation"]["method_averages"]
    assert report["methods"]["v4_0"] in report["evaluation"]["method_averages"]
    assert all(value is True for key, value in report["protected_hashes"].items() if key.endswith("_unchanged"))


def test_v4_1_disagreement_examples_jsonl() -> None:
    assert EXAMPLES.exists()
    rows = [json.loads(line) for line in EXAMPLES.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert {"kind", "query_id", "paper_id", "v3_3_rank", "v4_0_rank", "feature_values"} <= set(rows[0])
