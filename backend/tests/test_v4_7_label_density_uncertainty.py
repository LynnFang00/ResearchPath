import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "data" / "eval" / "results" / "v4_7_label_density_uncertainty_report.json"
BOUNDARY = ROOT / "data" / "eval" / "results" / "v4_7_contrastive_learning_boundary_cases.jsonl"
CANDIDATES = ROOT / "data" / "eval" / "results" / "v4_7_targeted_labeling_candidates.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v4_7_report_schema_and_read_only_flags() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v4.7_label_density_uncertainty_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert "v2_contrastive_learning" in report["label_density_by_topic"]
    assert "v2_multimodal_learning" in report["label_density_by_topic"]


def test_v4_7_boundary_and_candidates_have_required_fields() -> None:
    boundary = load_jsonl(BOUNDARY)
    candidates = load_jsonl(CANDIDATES)
    assert boundary
    assert candidates
    assert {row["query_id"] for row in boundary} == {"v2_contrastive_learning"}
    assert all(row.get("reason_selected") for row in candidates)
    assert all(row["query_id"] == "v2_contrastive_learning" for row in candidates)


def test_v4_7_protected_hashes_and_no_runtime_config() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v4_7_guarded_text_blend.py").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v4_7_guarded_text_blend_candidate.json").exists()
