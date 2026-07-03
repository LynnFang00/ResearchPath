import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.create_v6_1_6b_canonical_postgres_snapshot import acceptance  # noqa: E402
from scripts.validate_v3_9_semantic_expansion_labels import (  # noqa: E402
    DEFAULT_LABELS as V39_LABELS,
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    EXPECTED_V35_LABELS_SHA256,
    file_sha256,
)


REPORT = ROOT / "data" / "eval" / "results" / "v6_1_6b_canonical_postgres_snapshot_report.json"
COVERAGE = ROOT / "data" / "eval" / "results" / "v6_1_6b_label_coverage_against_canonical.jsonl"
AMBIGUOUS = ROOT / "data" / "eval" / "results" / "v6_1_6b_ambiguous_title_year_groups.jsonl"
SNAPSHOT = ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_paper_metadata.jsonl"
MANIFEST = ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_corpus_manifest.json"


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v6_1_6b_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.6b_canonical_postgres_snapshot_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["models_trained_by_this_script"] is False
    assert report["existing_model_artifacts_overwritten"] is False
    assert report["feature_artifacts_regenerated"] is False
    assert report["external_api_calls_made"] is False
    assert report["fuzzy_remapping_performed"] is False
    assert report["paper_id_mapping_guessed"] is False


def test_v6_1_6b_snapshot_created_from_explicit_paper_id_identity() -> None:
    report = load_report()
    assert report["postgresql_available"] is True
    assert report["acceptance_passed"] is True
    assert report["acceptance_failure_reasons"] == []
    assert report["canonical_snapshot_created"] is True
    assert report["row_count"] == 50424
    assert report["duplicate_paper_id_count"] == 0
    assert report["corpus_hash"] and len(report["corpus_hash"]) == 64
    assert SNAPSHOT.exists()
    assert MANIFEST.exists()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "v6.1.6b_canonical_corpus_manifest"
    assert manifest["row_count"] == report["row_count"]
    assert manifest["corpus_hash"] == report["corpus_hash"]
    assert manifest["id_policy"]["primary_identity_key"] == "explicit ResearchPath paper_id from papers.id"
    assert manifest["id_policy"]["normalized_title_year_role"] == "diagnostic_only_warning_not_blocker"


def test_v6_1_6b_snapshot_rows_have_required_identity_fields() -> None:
    rows = load_jsonl(SNAPSHOT)
    assert len(rows) == 50424
    first = rows[0]
    required = {
        "schema_version",
        "paper_id",
        "title",
        "abstract",
        "year",
        "venue",
        "source",
        "external_ids",
        "source_table",
        "identity_key",
        "normalized_title",
        "normalized_title_year_group_size",
        "identity_validation_status",
    }
    assert required <= set(first)
    assert first["schema_version"] == "v6.1.6b_canonical_paper_metadata"
    assert first["identity_key"] == "paper_id"
    assert first["paper_id"] == 1
    assert first["title"] == "Attention Is All You Need"
    assert len({row["paper_id"] for row in rows}) == len(rows)


def test_v6_1_6b_v6_identity_validation_and_ambiguous_warning_policy() -> None:
    report = load_report()
    validation = report["v6_example_identity_validation"]
    assert validation["total_v6_unique_paper_ids"] == 2165
    assert validation["resolved_by_explicit_paper_id"] == 2165
    assert validation["missing_count"] == 0
    assert validation["true_title_year_mismatch_count"] == 0
    assert validation["ambiguous_normalized_title_year_groups_are_blockers"] is False
    assert report["ambiguous_normalized_title_year_group_count"] == 14
    assert report["ambiguous_normalized_title_year_groups_are_warnings"] is True

    ambiguous = load_jsonl(AMBIGUOUS)
    assert len(ambiguous) == 14
    assert all(row["identity_policy"] == "warning_only_explicit_paper_id_is_primary" for row in ambiguous)
    assert all(row["group_size"] > 1 for row in ambiguous)


def test_v6_1_6b_raw_row_numbers_and_fuzzy_matching_are_not_used() -> None:
    report = load_report()
    assert report["id_policy"]["raw_jsonl_row_numbers_used_as_paper_id"] is False
    assert report["id_policy"]["fuzzy_title_matching_used"] is False
    assert report["id_policy"]["paper_id_mapping_guessed"] is False
    alignment = report["raw_row_number_alignment"]
    assert alignment["raw_jsonl_row_numbers_used_as_paper_id"] is False
    assert alignment["raw_jsonl_row_number_safe_as_paper_id"] is False
    assert alignment["conclusion"] == "unsafe"


def test_v6_1_6b_label_coverage_is_reported_and_resolved() -> None:
    rows = load_jsonl(COVERAGE)
    sources = {row["source"] for row in rows}
    assert sources == {"v2_1", "v2_5", "v3_2", "v3_5", "v3_9", "v4_8_override", "v6_0_2_neural_examples"}
    for row in rows:
        assert row["schema_version"] == "v6.1.6b_label_coverage_against_canonical"
        assert row["missing_paper_id_count"] == 0
        assert row["true_title_year_mismatch_count"] == 0
        assert row["resolved_by_explicit_paper_id"] == row["rows_checked"]
        assert row["identity_policy"] == "explicit_paper_id_primary_no_fuzzy_remapping"
    v602 = next(row for row in rows if row["source"] == "v6_0_2_neural_examples")
    assert v602["rows_checked"] == 2480
    assert v602["unique_paper_ids"] == 2165
    assert v602["ambiguous_normalized_title_year_warning_row_count"] == 2


def test_v6_1_6b_acceptance_fails_without_explicit_identity_proof() -> None:
    protected = {
        "selected_240_hash_unchanged": True,
        "v2_1_labels_hash_unchanged": True,
        "v2_5_labels_hash_unchanged": True,
        "v3_2_labels_hash_unchanged": True,
        "v3_5_labels_hash_unchanged": True,
        "v3_9_labels_hash_unchanged": True,
    }
    ok, reasons = acceptance(
        rows=[{"id": 1}],
        connection_error=None,
        duplicate_paper_id_count=0,
        validation={
            "total_v6_unique_paper_ids": 2,
            "resolved_by_explicit_paper_id": 1,
            "true_title_year_mismatch_count": 0,
        },
        protected=protected,
    )
    assert ok is False
    assert any("row count" in reason for reason in reasons)
    assert any("did not resolve 100%" in reason for reason in reasons)


def test_v6_1_6b_protected_hashes_are_unchanged() -> None:
    report = load_report()
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == report["protected_hashes"]["v3_9_labels_sha256"]


def test_v6_1_6b_does_not_create_runtime_frontend_or_model_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_6b_canonical_postgres_snapshot.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_6b_canonical_postgres_snapshot.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_6b_canonical_postgres_snapshot.json").exists()
