import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v4_8_targeted_contrastive import (  # noqa: E402
    CONTRASTIVE_TOPIC,
    combine_with_v48_overrides,
    normalize_v48_label,
)

LABELS = ROOT / "data" / "eval" / "manual_labels_v4_8_targeted_contrastive.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v4_8_targeted_contrastive_reeval_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v4_8_label_file_shape() -> None:
    rows = load_jsonl(LABELS)
    keys = [(row["query_id"], int(row["paper_id"])) for row in rows]
    assert len(rows) == 25
    assert len(keys) == len(set(keys))
    assert {row["query_id"] for row in rows} == {CONTRASTIVE_TOPIC}
    assert {int(row["relevance_label"]["bucket"]) for row in rows} <= {0, 1, 2}
    assert all(0.0 <= float(row["relevance_label"]["reading_value_score"]) <= 1.0 for row in rows)
    assert all(0.0 <= float(row["relevance_label"]["topic_match_score"]) <= 1.0 for row in rows)


def test_v4_8_override_semantics_do_not_duplicate_base_rows() -> None:
    v48_row = {
        "query_id": CONTRASTIVE_TOPIC,
        "paper_id": 123,
        "title": "Boundary case",
        "relevance_label": {"bucket": 1, "reading_value_score": 0.4, "topic_match_score": 0.6},
        "section_label": "application",
        "difficulty_label": "intermediate+advanced",
        "hard_negative": False,
        "label_notes": "targeted replacement",
    }
    base = [
        {"query_id": CONTRASTIVE_TOPIC, "paper_id": 123, "reading_value_score": 1.0, "topic_match_score": 1.0},
        {"query_id": "v2_other", "paper_id": 456, "reading_value_score": 1.0, "topic_match_score": 1.0},
    ]

    combined, summary = combine_with_v48_overrides(base, [v48_row])

    assert summary["base_label_count"] == 2
    assert summary["v4_8_label_count"] == 1
    assert summary["effective_label_count"] == 2
    assert summary["overlap_with_base_labels"] == 1
    assert summary["combined_duplicate_query_paper_rows"] == []
    replacement = next(row for row in combined if row["query_id"] == CONTRASTIVE_TOPIC)
    assert replacement["judged_source"] == "v4_8"
    assert replacement["reading_value_score"] == 0.4
    assert replacement["intermediate_fit_score"] == 1.0
    assert replacement["advanced_fit_score"] == 1.0


def test_v4_8_normalization_marks_hard_negatives() -> None:
    normalized = normalize_v48_label(
        {
            "query_id": CONTRASTIVE_TOPIC,
            "paper_id": 999,
            "title": "Contrastive divergence",
            "relevance_label": {"bucket": 0, "reading_value_score": 0.1, "topic_match_score": 0.25},
            "section_label": "background",
            "difficulty_label": "expert",
            "hard_negative": True,
            "label_notes": "different contrastive tradition",
        }
    )
    assert normalized["hard_negative"] is True
    assert normalized["positive"] is False
    assert normalized["primary_role"] == "negative"
    assert normalized["expert_fit_score"] == 1.0


def test_v4_8_report_schema_and_offline_flags() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v4.8_targeted_contrastive_reeval_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["preflight"]["row_count"] == 25
    assert report["label_combination"]["base_label_count"] == 2400
    assert report["label_combination"]["effective_label_count"] == 2400
    assert report["label_combination"]["overlap_with_base_labels"] == 25


def test_v4_8_protected_hashes_and_no_runtime_config() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v4_8_targeted_contrastive.py").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v4_8_targeted_contrastive_candidate.json").exists()
