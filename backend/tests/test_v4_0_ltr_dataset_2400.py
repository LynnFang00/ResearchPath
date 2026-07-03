from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v4_0_ltr_dataset_2400 import DEFAULT_DATASET_JSONL, load_jsonl, row_key  # noqa: E402
from scripts.validate_v4_0_ltr_dataset_2400 import EXPECTED_SOURCE_COUNTS, validate_dataset  # noqa: E402
from scripts.validate_v3_9_semantic_expansion_labels import (  # noqa: E402
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


def test_v4_0_dataset_has_expected_rows_sources_and_topics() -> None:
    rows = load_jsonl(DEFAULT_DATASET_JSONL)

    assert len(rows) == 2400
    assert len({row_key(row) for row in rows}) == 2400
    assert len({row["query_id"] for row in rows}) == 16

    counts = {}
    for row in rows:
        counts[row["judged_source"]] = counts.get(row["judged_source"], 0) + 1
    assert counts == EXPECTED_SOURCE_COUNTS


def test_v4_0_dataset_validator_passes() -> None:
    report = validate_dataset(
        dataset_path=REPO_ROOT / "data" / "processed" / "training" / "v4_0_ltr_2400" / "ltr_dataset_2400.jsonl",
        csv_path=REPO_ROOT / "data" / "processed" / "training" / "v4_0_ltr_2400" / "ltr_dataset_2400.csv",
        splits_path=REPO_ROOT / "data" / "processed" / "training" / "v4_0_ltr_2400" / "splits_2400.json",
        v21_labels_path=DEFAULT_V21_LABELS,
        v25_labels_path=DEFAULT_V25_LABELS,
        v32_labels_path=DEFAULT_V32_LABELS,
        v35_labels_path=DEFAULT_V35_LABELS,
        selected_240_path=DEFAULT_SELECTED_240,
    )

    assert report["valid"] is True
    assert report["score_scale_violations"] == []


def test_v4_0_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
