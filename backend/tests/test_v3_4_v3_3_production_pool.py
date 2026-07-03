from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V33_MODEL_DIR,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    SELECTED_240,
    file_sha256,
    load_json,
    load_jsonl,
    load_v33_model,
)


def test_v3_4_v3_3_model_artifact_loads() -> None:
    payload = load_v33_model(DEFAULT_V33_MODEL_DIR, "random_forest_no_v27_compact")

    assert payload["metadata"]["model_name"] == "random_forest_no_v27_compact"
    assert payload["metadata"]["uses_v2_7_feature"] is False
    assert payload["metadata"]["raw_score_feature_intersection"] == []
    assert hasattr(payload["estimator"], "predict")


def test_v3_4_report_candidate_pool_dedup_if_present() -> None:
    report_path = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_production_pool_report.json"
    if not report_path.exists():
        return

    report = load_json(report_path)

    assert report["candidate_pool_definition"]["duplicate_candidate_rows"] == []
    assert len(report["candidate_pool_definition"]["candidate_pool_sizes_by_topic"]) == 16
    assert max(report["candidate_pool_definition"]["candidate_pool_sizes_by_topic"].values()) <= 250


def test_v3_4_error_examples_have_unique_query_paper_if_present() -> None:
    examples_path = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_error_examples.jsonl"
    if not examples_path.exists():
        return

    rows = load_jsonl(examples_path)
    keys = [(row["query_id"], int(row["paper_id"])) for row in rows]

    assert len(keys) == len(set(keys))
    assert all("reasons" in row and row["reasons"] for row in rows)
    assert all("observations" in row and row["observations"] for row in rows)


def test_v3_4_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert DEFAULT_V32_LABELS.exists()
