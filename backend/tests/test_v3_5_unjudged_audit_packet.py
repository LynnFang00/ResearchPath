from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v3_5_unjudged_top10_audit_packet import (  # noqa: E402
    DEFAULT_CANDIDATES_OUT,
    DEFAULT_REPORT_JSON_OUT,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    SELECTED_240,
    file_sha256,
    load_json,
    load_jsonl,
)
from scripts.validate_v3_5_unjudged_audit_labels import validate_labels  # noqa: E402


def test_v3_5_packet_has_unique_labelable_candidates_if_present() -> None:
    if not DEFAULT_CANDIDATES_OUT.exists():
        return

    rows = load_jsonl(DEFAULT_CANDIDATES_OUT)
    keys = [(row["query_id"], int(row["paper_id"])) for row in rows]

    assert len(rows) >= 27
    assert len(rows) <= 160
    assert len(keys) == len(set(keys))
    assert all(row.get("title") and len(row["title"].strip()) >= 5 for row in rows)
    assert all(row.get("abstract") and len(row["abstract"].split()) >= 30 for row in rows)
    assert all(row.get("diagnostic_reasons") for row in rows)
    assert sum("v3_3_high_scoring_unjudged_top10" in row["diagnostic_reasons"] for row in rows) == 27


def test_v3_5_packet_report_matches_candidates_if_present() -> None:
    if not DEFAULT_CANDIDATES_OUT.exists() or not DEFAULT_REPORT_JSON_OUT.exists():
        return

    rows = load_jsonl(DEFAULT_CANDIDATES_OUT)
    report = load_json(DEFAULT_REPORT_JSON_OUT)

    assert report["candidate_count"] == len(rows)
    assert report["labelable_candidates"] is True
    assert report["all_27_v3_3_high_scoring_unjudged_top10_included"] is True
    assert report["duplicate_query_paper_rows"] == []


def test_v3_5_label_validation_passes_for_current_progress_if_present() -> None:
    candidates_path = DEFAULT_CANDIDATES_OUT
    labels_path = REPO_ROOT / "data" / "eval" / "manual_labels_v3_5_unjudged_audit.jsonl"
    if not candidates_path.exists() or not labels_path.exists():
        return

    report = validate_labels(
        labels=load_jsonl(labels_path),
        candidates=load_jsonl(candidates_path),
        v21_labels=load_jsonl(DEFAULT_V21_LABELS),
        v25_labels=load_jsonl(DEFAULT_V25_LABELS),
        v32_labels=load_jsonl(DEFAULT_V32_LABELS),
        v21_labels_path=DEFAULT_V21_LABELS,
        v25_labels_path=DEFAULT_V25_LABELS,
        v32_labels_path=DEFAULT_V32_LABELS,
        selected_240_path=SELECTED_240,
    )

    assert report["schema_valid"] is True
    assert report["is_valid"] is True
    assert 0 <= report["labeled_candidate_count"] <= report["candidate_count"]


def test_v3_5_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256
