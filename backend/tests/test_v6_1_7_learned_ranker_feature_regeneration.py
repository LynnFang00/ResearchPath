import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.materialize_v6_1_4_learned_ranker_features import FORBIDDEN_FEATURES, feature_hash  # noqa: E402
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


CANONICAL = ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_paper_metadata.jsonl"
MANIFEST = ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_corpus_manifest.json"
EXAMPLES = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
MATRIX = ROOT / "data" / "eval" / "neural_reranker" / "v6_1_7_learned_ranker_feature_matrix.jsonl"
PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_7_learned_baseline_predictions.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_1_7_learned_ranker_feature_regeneration_report.json"
COVERAGE = ROOT / "data" / "eval" / "results" / "v6_1_7_feature_coverage_report.json"
PARITY = ROOT / "data" / "eval" / "results" / "v6_1_7_scoring_parity_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_1_7_canonical_snapshot_loads_and_v6_ids_resolve() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["row_count"] == 50424
    assert manifest["corpus_hash"] == "83B0FF757E74B2F4D9C429BC6655038B9AA2282CA682DEFE2F9825D749A3E296"

    canonical_ids = {int(row["paper_id"]) for row in load_jsonl(CANONICAL)}
    example_ids = {int(row["paper_id"]) for row in load_jsonl(EXAMPLES)}
    assert len(example_ids) == 2165
    assert example_ids <= canonical_ids


def test_v6_1_7_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.7_learned_ranker_feature_regeneration_report"
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
    assert report["external_api_calls_made"] is False
    assert report["feature_regeneration_succeeded"] is True
    assert report["exact_neural_vs_v4_9_comparison_possible"] is True
    assert report["rows_silently_dropped"] == 0


def test_v6_1_7_feature_matrix_schema_and_no_forbidden_features() -> None:
    rows = load_jsonl(MATRIX)
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
    for row in rows[:100]:
        assert required <= set(row)
        assert row["schema_version"] == "v6.1.7_learned_ranker_feature_row"
        assert row["forbidden_features_used"] == []
        assert set(row["feature_values"]).isdisjoint(FORBIDDEN_FEATURES)
        assert set(row["feature_source"]).isdisjoint(FORBIDDEN_FEATURES)
        assert row["missing_feature_names"] == []
        assert row["exact_runtime_feature_complete"] is True
        assert row["feature_hash"] == feature_hash(row["feature_values"], row["missing_feature_names"])


def test_v6_1_7_learned_baseline_coverage_and_predictions() -> None:
    report = load_report()
    for method in ("v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"):
        coverage = report["learned_baseline_coverage"][method]
        assert coverage["rows_scored"] == 2480
        assert coverage["rows_missing"] == 0
        assert coverage["coverage"] == 1.0
        assert coverage["exact_scoring_available"] is True
        assert coverage["missing_feature_counts"] == {}
        assert coverage["rows_silently_dropped"] == 0

    predictions = load_jsonl(PREDICTIONS)
    assert len(predictions) == 2480
    first = predictions[0]
    assert first["schema_version"] == "v6.1.7_learned_baseline_prediction"
    assert all(first["missing_score_reasons"][method] is None for method in ("v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"))
    for key in ("v3_3_score", "v4_1_score", "v4_9_score"):
        assert isinstance(first[key], float)
        assert math.isfinite(first[key])
    assert first["v4_9_details"]["v4_9_text_enabled"] in {True, False}


def test_v6_1_7_feature_coverage_report_by_group() -> None:
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    assert coverage["schema_version"] == "v6.1.7_feature_coverage_report"
    groups = coverage["by_group"]
    expected = {
        "bm25",
        "tfidf",
        "dense_embedding",
        "faiss_embedding",
        "hybrid",
        "v2_2b",
        "v2_6",
        "citation_quality",
        "metadata",
        "full_text",
    }
    assert expected <= set(groups)
    assert all(groups[name]["available"] is True for name in expected)
    assert coverage["learned_baseline_coverage"]["v4_9_guarded_text_blend"]["rows_scored"] == 2480


def test_v6_1_7_scoring_is_deterministic_by_hash_sample() -> None:
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    assert parity["schema_version"] == "v6.1.7_scoring_parity_report"
    assert parity["v4_9_formula_validated_against_candidate_config"] is True
    assert parity["forbidden_features_required"] == []
    assert parity["forbidden_features_used"] == []
    sample = parity["deterministic_prediction_hash_sample"]
    assert len(sample) == 25
    assert sample == json.loads(PARITY.read_text(encoding="utf-8"))["deterministic_prediction_hash_sample"]


def test_v6_1_7_protected_hashes_are_unchanged() -> None:
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


def test_v6_1_7_does_not_create_runtime_frontend_or_model_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_7_learned_ranker_features.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_7_learned_ranker_features.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_7_learned_ranker_features.json").exists()
