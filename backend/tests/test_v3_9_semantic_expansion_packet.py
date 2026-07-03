from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v3_9_semantic_label_expansion_packet import (  # noqa: E402
    DEFAULT_CANDIDATES_OUT,
    DEFAULT_LABEL_OUT,
    DEFAULT_REPORT_JSON,
    MIN_NEW_LABELS,
    TARGET_TOTAL_LABELS,
    TOPICS,
    abstract_is_valid,
    load_jsonl,
    title_is_valid,
)
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
    validate_labels,
)


def test_v3_9_packet_reaches_minimum_label_target() -> None:
    candidates = load_jsonl(DEFAULT_CANDIDATES_OUT)
    report = __import__("json").loads(DEFAULT_REPORT_JSON.read_text(encoding="utf-8"))

    assert len(candidates) >= MIN_NEW_LABELS
    assert report["candidate_count"] == len(candidates)
    assert report["projected_total_after_labeling"] >= TARGET_TOTAL_LABELS
    assert report["target_reached"] is True


def test_v3_9_packet_has_no_duplicates_or_existing_label_overlap() -> None:
    candidates = load_jsonl(DEFAULT_CANDIDATES_OUT)
    existing = (
        load_jsonl(DEFAULT_V21_LABELS)
        + load_jsonl(DEFAULT_V25_LABELS)
        + load_jsonl(DEFAULT_V32_LABELS)
        + load_jsonl(DEFAULT_V35_LABELS)
    )
    candidate_keys = [(str(row["query_id"]), int(row["paper_id"])) for row in candidates]
    existing_keys = {(str(row["query_id"]), int(row["paper_id"])) for row in existing}

    assert len(candidate_keys) == len(set(candidate_keys))
    assert not (set(candidate_keys) & existing_keys)


def test_v3_9_packet_candidates_are_labelable_and_cover_all_topics() -> None:
    candidates = load_jsonl(DEFAULT_CANDIDATES_OUT)
    topics = {str(row["query_id"]) for row in candidates}

    assert topics == set(TOPICS)
    assert all(title_is_valid(row.get("title")) for row in candidates)
    assert all(abstract_is_valid(row.get("abstract")) for row in candidates)


def test_v3_9_per_topic_quotas_are_reasonably_balanced() -> None:
    candidates = load_jsonl(DEFAULT_CANDIDATES_OUT)
    counts = {topic: 0 for topic in TOPICS}
    for row in candidates:
        counts[str(row["query_id"])] += 1

    assert min(counts.values()) >= 95
    assert max(counts.values()) <= 100
    assert max(counts.values()) - min(counts.values()) <= 5


def test_v3_9_label_file_validates_current_progress() -> None:
    labels = load_jsonl(DEFAULT_LABEL_OUT, missing_ok=True)
    candidates = load_jsonl(DEFAULT_CANDIDATES_OUT)
    existing = (
        load_jsonl(DEFAULT_V21_LABELS)
        + load_jsonl(DEFAULT_V25_LABELS)
        + load_jsonl(DEFAULT_V32_LABELS)
        + load_jsonl(DEFAULT_V35_LABELS)
    )

    report = validate_labels(
        labels=labels,
        candidates=candidates,
        existing_labels=existing,
        v21_labels_path=DEFAULT_V21_LABELS,
        v25_labels_path=DEFAULT_V25_LABELS,
        v32_labels_path=DEFAULT_V32_LABELS,
        v35_labels_path=DEFAULT_V35_LABELS,
        selected_240_path=DEFAULT_SELECTED_240,
    )

    assert report["is_valid"] is True
    assert report["schema_valid"] is True
    assert report["label_file_rows"] == len(labels)
    assert not report["missing_required_fields"]
    assert not report["invalid_values"]
    assert not report["duplicate_query_paper_rows"]
    assert not report["labels_not_in_candidates"]
    assert not report["overlap_existing_labels"]
    assert report["projected_total_judged_rows"] >= TARGET_TOTAL_LABELS


def test_v3_9_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
