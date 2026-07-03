import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.materialize_v6_1_4_learned_ranker_features import (  # noqa: E402
    FORBIDDEN_FEATURES,
    feature_hash,
    load_jsonl,
    materialize_available_features,
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


FEATURE_MATRIX = ROOT / "data" / "eval" / "neural_reranker" / "v6_1_4_learned_ranker_feature_matrix.jsonl"
PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_4_learned_baseline_predictions.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_1_4_learned_ranker_feature_materialization_report.json"
PARITY = ROOT / "data" / "eval" / "results" / "v6_1_4_feature_parity_report.json"


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_1_4_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.4_learned_ranker_feature_materialization_report"
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
    assert report["dataset"] == {"fold_count": 4, "row_count": 2480, "topic_count": 16}


def test_v6_1_4_feature_snapshot_schema_and_hashes() -> None:
    rows = load_jsonl(FEATURE_MATRIX)
    assert len(rows) == 2480
    required = {
        "schema_version",
        "query_id",
        "query",
        "paper_id",
        "title",
        "fold_id",
        "split",
        "feature_values",
        "missing_feature_names",
        "missing_features_by_method",
        "feature_source",
        "feature_hash",
        "forbidden_features_used",
        "exact_runtime_feature_complete",
    }
    assert all(required <= set(row) for row in rows[:25])
    assert all(row["schema_version"] == "v6.1.4_learned_ranker_feature_row" for row in rows[:50])
    first = rows[0]
    assert first["feature_hash"] == feature_hash(first["feature_values"], first["missing_feature_names"])
    assert first["forbidden_features_used"] == []
    assert first["exact_runtime_feature_complete"] is False


def test_v6_1_4_no_forbidden_label_or_eval_fields_are_used_as_features() -> None:
    rows = load_jsonl(FEATURE_MATRIX)
    for row in rows[:200]:
        assert set(row["feature_values"]).isdisjoint(FORBIDDEN_FEATURES)
        assert set(row["feature_source"]).isdisjoint(FORBIDDEN_FEATURES)
        assert row["forbidden_features_used"] == []


def test_v6_1_4_missing_features_are_explicit_not_silent() -> None:
    report = load_report()
    assert report["rows_silently_dropped"] == 0
    for method in ("v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"):
        coverage = report["learned_baseline_coverage"][method]
        assert coverage["rows_scored"] == 0
        assert coverage["rows_missing"] == 2480
        assert coverage["exact_scoring_available"] is False
        assert coverage["rows_silently_dropped"] == 0
        assert coverage["missing_feature_counts"]["tfidf_score"] == 2480
        assert coverage["missing_feature_counts"]["old_v2_2b_score"] == 2480
        assert coverage["missing_feature_counts"]["v2_6_score"] == 2480
    assert report["blockers"]["runtime_corpus_database"] == "Runtime SQLite corpus has 6 rows, not the current ~50K paper corpus."
    assert report["blockers"]["raw_corpus_alignment"] == "Raw JSONL row number does not align with ResearchPath paper_id."


def test_v6_1_4_predictions_report_learned_scores_unavailable() -> None:
    predictions = load_jsonl(PREDICTIONS)
    assert len(predictions) == 2480
    assert all(row["schema_version"] == "v6.1.4_learned_baseline_prediction" for row in predictions[:50])
    for row in predictions[:200]:
        assert row["v3_3_score"] is None
        assert row["v4_1_score"] is None
        assert row["v4_9_score"] is None
        assert row["missing_score_reasons"]["v3_3_ltr"].startswith("Missing exact runtime features:")
        assert row["missing_score_reasons"]["v4_1_blend"].startswith("Missing exact runtime features:")
        assert row["missing_score_reasons"]["v4_9_guarded_text_blend"].startswith("Missing exact runtime features:")


def test_v6_1_4_parity_report_is_explicitly_not_comparable() -> None:
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    assert parity["schema_version"] == "v6.1.4_feature_parity_report"
    assert parity["offline_only"] is True
    assert parity["runtime_scoring_invoked"] is False
    assert parity["parity_comparable"] is False
    assert parity["max_score_delta"] == {
        "v3_3_ltr": None,
        "v4_1_blend": None,
        "v4_9_guarded_text_blend": None,
    }
    assert parity["forbidden_features_required"] == []
    assert parity["forbidden_features_used"] == []
    assert parity["runtime_corpus_database"]["paper_count"] == 6
    assert parity["raw_corpus_alignment"]["usable_for_exact_runtime_features"] is False


def test_v6_1_4_materialization_is_deterministic_for_available_features() -> None:
    row = {
        "baseline_rankings": {"bm25": {"score": 3.5, "rank": 7}},
        "year": 2020,
        "citation_count": 10,
        "full_text_available": False,
    }
    values1, sources1 = materialize_available_features(row)
    values2, sources2 = materialize_available_features(row)
    assert values1 == values2
    assert sources1 == sources2
    assert values1["bm25_reciprocal_rank"] == 1 / 7
    assert values1["paper_age_years"] == 6
    assert feature_hash(values1, ["tfidf_score"]) == feature_hash(values2, ["tfidf_score"])


def test_v6_1_4_protected_hashes_are_unchanged() -> None:
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


def test_v6_1_4_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_1_4_learned_ranker_features.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_4_feature_materialization.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_4_learned_ranker_features.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_4_neural_candidate.json").exists()
