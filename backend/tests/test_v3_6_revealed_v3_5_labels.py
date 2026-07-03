from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_V21_PACKET,
    DEFAULT_V33_MODEL_DIR,
    load_json,
    load_jsonl,
    load_v33_model,
    production_candidate_rows,
)
from scripts.evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_JSON_OUT,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_CANDIDATES,
    DEFAULT_V35_LABELS,
    DEFAULT_SELECTED_240,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    EXPECTED_V35_LABELS_SHA256,
    duplicate_keys,
    file_sha256,
    load_labels_with_sources,
    v35_label_distribution,
)


def test_v3_6_combined_judgment_table_has_831_unique_rows() -> None:
    combined, labels_by_source = load_labels_with_sources(
        v21_labels_path=DEFAULT_V21_LABELS,
        v25_labels_path=DEFAULT_V25_LABELS,
        v32_labels_path=DEFAULT_V32_LABELS,
        v35_labels_path=DEFAULT_V35_LABELS,
    )

    assert len(combined) == 831
    assert {source: len(rows) for source, rows in labels_by_source.items()} == {
        "v2_1": 240,
        "v2_5": 96,
        "v3_2": 464,
        "v3_5": 31,
    }
    assert duplicate_keys(combined) == []


def test_v3_6_v3_5_labels_are_included_in_distribution() -> None:
    _, labels_by_source = load_labels_with_sources(
        v21_labels_path=DEFAULT_V21_LABELS,
        v25_labels_path=DEFAULT_V25_LABELS,
        v32_labels_path=DEFAULT_V32_LABELS,
        v35_labels_path=DEFAULT_V35_LABELS,
    )
    distribution = v35_label_distribution(labels_by_source["v3_5"], load_jsonl(DEFAULT_V35_CANDIDATES))

    assert distribution["label_count"] == 31
    assert distribution["count_by_bucket"] == {"positive": 28, "partial": 0, "negative": 3}
    assert distribution["v3_3_high_scoring_unjudged_top10_count"] == 27
    assert distribution["v3_3_high_scoring_unjudged_top10_by_bucket"] == {"positive": 24, "partial": 0, "negative": 3}


def test_v3_6_candidate_pool_dedup_behavior_is_unchanged() -> None:
    rows_by_query, _diagnostics = production_candidate_rows(packet_rows=load_jsonl(DEFAULT_V21_PACKET), top_k=50)

    assert len(rows_by_query) == 16
    for rows in rows_by_query.values():
        paper_ids = [int(row["paper_id"]) for row in rows]
        assert len(paper_ids) == len(set(paper_ids))


def test_v3_6_frozen_v3_3_artifact_loads() -> None:
    payload = load_v33_model(DEFAULT_V33_MODEL_DIR, "random_forest_no_v27_compact")

    assert payload["metadata"]["model_name"] == "random_forest_no_v27_compact"
    assert payload["metadata"]["uses_v2_7_feature"] is False
    assert hasattr(payload["estimator"], "predict")


def test_v3_6_report_if_present_uses_831_judgments() -> None:
    if not DEFAULT_JSON_OUT.exists():
        return

    report = load_json(DEFAULT_JSON_OUT)

    assert report["dataset"]["label_count"] == 831
    assert report["dataset"]["duplicate_query_paper_rows"] == []
    assert report["v3_5_label_distribution"]["label_count"] == 31


def test_v3_6_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
