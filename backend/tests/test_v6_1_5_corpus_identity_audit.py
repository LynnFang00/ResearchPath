import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_v6_1_5_corpus_identity_and_ids import (  # noqa: E402
    normalize_title,
)
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


REPORT = ROOT / "data" / "eval" / "results" / "v6_1_5_corpus_identity_audit_report.json"
ID_REPORT = ROOT / "data" / "eval" / "results" / "v6_1_5_paper_id_resolution_report.jsonl"
FEATURE_REPORT = ROOT / "data" / "eval" / "results" / "v6_1_5_feature_artifact_provenance_report.json"


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def load_id_rows() -> list[dict]:
    return [json.loads(line) for line in ID_REPORT.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v6_1_5_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.5_corpus_identity_audit_report"
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
    assert report["automatic_fuzzy_remapping_performed"] is False
    assert report["judged_label_rows_after_v6_0_2"] == 2480


def test_v6_1_5_corpus_sizes_and_db_config_are_reported() -> None:
    report = load_report()
    runtime = report["runtime_db_configuration"]
    assert runtime["backend_config_database_url_default"].startswith("postgresql+psycopg://")
    assert runtime["backend_default_uses_sqlite"] is False
    assert runtime["configured_sqlite_status"]["row_or_paper_count"] == 6
    assert runtime["six_paper_db_assessment"] == "dev_or_smoke_fixture_not_full_runtime_corpus"

    sources = {row["name"]: row for row in report["corpus_sources"]}
    assert sources["embeddings_50k"]["row_or_paper_count"] == 50424
    assert sources["faiss_id_map_50k"]["row_or_paper_count"] == 50424
    assert sources["raw_arxiv_50k_incremental"]["row_or_paper_count"] == 35000
    assert sources["v6_0_2_neural_examples"]["row_or_paper_count"] == 2480
    assert sources["embeddings_50k"]["safe_join_to_labels"] is False
    assert sources["raw_arxiv_50k_incremental"]["contains_researchpath_paper_id"] is False


def test_v6_1_5_raw_jsonl_row_number_is_explicitly_unsafe() -> None:
    report = load_report()
    alignment = report["raw_row_number_alignment"]
    assert alignment["raw_jsonl_row_number_safe_as_paper_id"] is False
    assert alignment["conclusion"] == "unsafe"
    assert len(alignment["mismatches"]) >= 4
    assert any(row["paper_id"] == 1 and row["v6_title"] == "Attention Is All You Need" for row in alignment["mismatches"])


def test_v6_1_5_paper_id_resolution_is_explicit_not_fuzzy() -> None:
    rows = load_id_rows()
    assert {row["source"] for row in rows} >= {
        "v2_1",
        "v2_5",
        "v3_2",
        "v3_5",
        "v3_9",
        "v4_8_override",
        "v6_0_2_neural_examples",
    }
    for row in rows:
        assert row["schema_version"] == "v6.1.5_paper_id_resolution_row"
        assert row["automatic_mapping_safe"] is False
        assert row["fuzzy_remapping_performed"] is False
        assert row["full_50k_source_has_valid_paper_id_field"] is False
        assert row["resolving_in_full_50k_source_by_paper_id"] is None
        assert "No full 50K metadata source" in row["blocker"]
    v602 = next(row for row in rows if row["source"] == "v6_0_2_neural_examples")
    assert v602["total_rows"] == 2480
    assert v602["resolving_in_runtime_sqlite_by_id"] == 7
    assert v602["resolving_in_embedding_or_faiss_id_map"] == 2480
    assert v602["exact_title_year_matches_in_raw_metadata"] > 0


def test_v6_1_5_model_artifact_provenance_is_reported() -> None:
    report = load_report()
    provenance = report["model_artifact_provenance"]
    assert set(provenance) == {"v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"}
    for method, row in provenance.items():
        assert row["exists"] is True
        assert row["can_safely_apply_to_v6_0_2_rows"] is False
        assert row["records_corpus_hash_or_source"] is False
        assert row["records_label_version"] is True
        assert row["cannot_apply_reasons"]
    assert "bm25_score" in provenance["v3_3_ltr"]["expected_feature_order"]
    assert "v2_6_score" in provenance["v4_1_blend"]["expected_feature_order"]
    assert provenance["v4_9_guarded_text_blend"]["expected_feature_order"] == [
        "normalized_v3_3_score",
        "normalized_v4_1_score",
        "normalized_v4_3_text_score",
        "max_ranker_confidence",
    ]


def test_v6_1_5_feature_artifact_provenance_is_reported() -> None:
    feature_report = json.loads(FEATURE_REPORT.read_text(encoding="utf-8"))
    assert feature_report["schema_version"] == "v6.1.5_feature_artifact_provenance_report"
    groups = feature_report["feature_groups"]
    assert set(groups) == {
        "tfidf_bm25",
        "dense_faiss",
        "hybrid",
        "v2_2b",
        "v2_6",
        "citation_quality",
        "full_text_availability",
    }
    assert groups["dense_faiss"]["exists"] is True
    assert groups["dense_faiss"]["coverage_over_v6_0_2_paper_ids"] == 2165
    assert groups["tfidf_bm25"]["coverage_over_v6_0_2_paper_ids"] == 2422
    assert groups["hybrid"]["safe_join_possible"] is False
    assert groups["v2_2b"]["feature_count"] == 78
    assert groups["v2_6"]["feature_count"] == 78
    assert all(row["safe_join_possible"] is False for row in groups.values())


def test_v6_1_5_recommendations_and_smallest_safe_fix() -> None:
    report = load_report()
    assert report["exact_learned_ranker_comparison_possible"] is False
    assert "need_to_rebuild_full_corpus_db_from_processed_source" in report["decision_recommendations"]
    assert "need_to_create_paper_id_mapping_table" in report["decision_recommendations"]
    assert "need_to_regenerate_feature_artifacts_keyed_by_paper_id" in report["decision_recommendations"]
    assert "Build a read-only full-corpus metadata/feature snapshot" in report["smallest_safe_next_fix"]


def test_v6_1_5_protected_hashes_are_unchanged() -> None:
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


def test_v6_1_5_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_5_corpus_identity.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_1_5_corpus_identity.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_5_corpus_identity.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_5_candidate.json").exists()


def test_v6_1_5_normalization_helper_is_deterministic_and_not_fuzzy() -> None:
    assert normalize_title("  Attention\nIs  All You Need ") == "attention is all you need"
