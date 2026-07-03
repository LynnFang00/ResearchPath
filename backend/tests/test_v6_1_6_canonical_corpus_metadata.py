import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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


REPORT = ROOT / "data" / "eval" / "results" / "v6_1_6_canonical_corpus_metadata_report.json"
CANDIDATES = ROOT / "data" / "eval" / "results" / "v6_1_6_canonical_source_candidates.jsonl"
COVERAGE = ROOT / "data" / "eval" / "results" / "v6_1_6_label_coverage_against_canonical.jsonl"
CANONICAL = ROOT / "data" / "processed" / "corpus" / "v6_1_6_canonical_paper_metadata.jsonl"
MANIFEST = ROOT / "data" / "processed" / "corpus" / "v6_1_6_canonical_corpus_manifest.json"


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v6_1_6_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.6_canonical_corpus_metadata_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["models_retrained_by_this_script"] is False
    assert report["existing_model_artifacts_overwritten"] is False
    assert report["production_candidate_created"] is False
    assert report["external_api_calls_made"] is False
    assert report["fuzzy_remapping_performed"] is False


def test_v6_1_6_postgresql_full_corpus_is_seen_but_not_accepted() -> None:
    report = load_report()
    postgres = report["postgresql"]
    assert postgres["available"] is True
    assert postgres["full_corpus_available"] is True
    assert postgres["paper_count"] == 50424
    assert {"id", "title", "abstract", "year", "venue", "source"} <= set(postgres["schema"])

    validation = postgres["v6_identity_validation"]
    assert validation["v6_unique_paper_ids"] == 2165
    assert validation["resolved_by_paper_id"] == 2165
    assert validation["missing_by_paper_id"] == 0
    assert validation["accepted_identity_matches"] == 2165
    assert validation["mismatch_count"] == 0
    assert validation["ambiguous_normalized_title_year_count"] > 0

    assert report["safe_canonical_metadata_source_found"] is False
    assert report["canonical_metadata_snapshot_created"] is False
    assert report["feature_regeneration_can_proceed_next"] is False
    assert report["decision_recommendation"] == "no_safe_metadata_source_found_rebuild_from_ingestion_required"
    assert not CANONICAL.exists()
    assert not MANIFEST.exists()


def test_v6_1_6_source_candidates_record_blockers() -> None:
    rows = load_jsonl(CANDIDATES)
    assert len(rows) >= 4
    by_name = {row["name"]: row for row in rows}
    assert {"backend_default_postgresql", "embeddings_50k", "faiss_id_map_50k"} <= set(by_name)
    assert "corpus_provenance_validation_v2_50k_report" in by_name
    assert all(row["schema_version"] == "v6.1.6_canonical_source_candidate" for row in rows)
    assert all(row["can_safely_become_canonical"] is False for row in rows)

    postgres = by_name["backend_default_postgresql"]
    assert postgres["row_count"] == 50424
    assert "ambiguous normalized title+year rows are present" in postgres["canonical_rejection_reasons"]
    assert "V6 title/year mismatches are present" not in postgres["canonical_rejection_reasons"]
    assert by_name["embeddings_50k"]["has_explicit_paper_id"] is True
    assert by_name["embeddings_50k"]["has_title"] is False
    assert by_name["faiss_id_map_50k"]["has_explicit_paper_id"] is True


def test_v6_1_6_raw_row_numbers_are_not_used_as_paper_ids() -> None:
    report = load_report()
    alignment = report["raw_row_number_alignment"]
    assert alignment["raw_jsonl_row_numbers_used_as_paper_id"] is False
    assert alignment["raw_jsonl_row_number_safe_as_paper_id"] is False
    assert len(alignment["mismatches"]) >= 4
    assert any(row["paper_id"] == 1 and row["v6_title"] == "Attention Is All You Need" for row in alignment["mismatches"])


def test_v6_1_6_label_coverage_is_blocked_without_canonical_snapshot() -> None:
    rows = load_jsonl(COVERAGE)
    sources = {row["source"] for row in rows}
    assert sources >= {"v2_1", "v2_5", "v3_2", "v3_5", "v3_9", "v4_8_override", "v6_0_2_neural_examples"}
    for row in rows:
        assert row["schema_version"] == "v6.1.6_label_coverage_against_canonical"
        assert row["canonical_available"] is False
        assert row["status"] == "not_evaluated_no_canonical_source"
        assert row["rows_resolved"] == 0
        assert row["rows_missing"] == row["total_rows"]
    v602 = next(row for row in rows if row["source"] == "v6_0_2_neural_examples")
    assert v602["total_rows"] == 2480


def test_v6_1_6_protected_hashes_are_unchanged() -> None:
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


def test_v6_1_6_does_not_create_runtime_frontend_or_model_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_6_canonical_corpus_metadata.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_1_6_canonical_corpus_metadata.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_6_canonical_corpus_metadata.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_6_canonical_corpus_candidate.json").exists()
