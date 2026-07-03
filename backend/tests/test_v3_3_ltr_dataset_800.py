from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v3_3_ltr_dataset_800 import (  # noqa: E402
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    SELECTED_240,
    build_topic_splits,
    duplicate_keys,
    file_sha256,
    validate_splits,
)


def test_v3_3_duplicate_key_detection() -> None:
    rows = [
        {"query_id": "q1", "paper_id": 1},
        {"query_id": "q1", "paper_id": 1},
        {"query_id": "q2", "paper_id": 1},
    ]

    assert duplicate_keys(rows) == [{"query_id": "q1", "paper_id": 1, "count": 2}]


def test_v3_3_topic_splits_have_no_leakage() -> None:
    topics = [f"topic_{index:02d}" for index in range(16)]

    validation = validate_splits(build_topic_splits(topics))

    assert validation["train_dev_test_no_topic_leakage"] is True
    assert validation["leave_topic_out_no_topic_leakage"] is True
    assert validation["train_topic_count"] == 10
    assert validation["dev_topic_count"] == 3
    assert validation["test_topic_count"] == 3
    assert validation["leave_topic_out_fold_count"] == 16


def test_v3_3_built_dataset_has_expected_shape_if_present() -> None:
    dataset_path = REPO_ROOT / "data" / "eval" / "training" / "v3_3_ltr_dataset_800.jsonl"
    if not dataset_path.exists():
        return
    import json
    from collections import Counter

    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(row["query_id"] for row in rows)

    assert len(rows) == 800
    assert len(counts) == 16
    assert set(counts.values()) == {50}
    assert duplicate_keys(rows) == []
    assert Counter(row["judged_source"] for row in rows) == {"v2_1": 240, "v2_5": 96, "v3_2": 464}


def test_v3_3_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256
