from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_v3_0_ltr_errors import (  # noqa: E402
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    SELECTED_240,
    file_sha256,
    select_active_learning_candidates,
    validate_packet,
)


def _candidate(topic: str, paper_id: int, ridge: float, v27: float, v26: float, hybrid: float) -> dict:
    return {
        "query_id": topic,
        "query": topic.replace("_", " "),
        "topic": topic.replace("_", " "),
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "Synthetic candidate.",
        "year": 2025,
        "authors": [],
        "venue": "Synthetic",
        "citation_count": 0,
        "identifiers": {},
        "source_url": None,
        "pdf_url": None,
        "sources_provenance": [],
        "bm25_score": 0.0,
        "tfidf_score": 0.0,
        "embedding_score": 0.0,
        "faiss_embedding_score": 0.0,
        "hybrid_score": hybrid,
        "old_v2_2b_score": 0.0,
        "v2_6_score": v26,
        "v2_7_score": v27,
        "v3_0_ridge_no_v27_score": ridge,
        "bm25_rank": 999999,
        "tfidf_rank": 999999,
        "embedding_rank": 999999,
        "faiss_embedding_rank": 999999,
        "hybrid_rank": paper_id,
    }


def test_v3_1_packet_validation_rejects_duplicates_and_labeled_rows() -> None:
    packet = [
        {"query_id": "q1", "paper_id": 1, "diagnostic_only": False},
        {"query_id": "q1", "paper_id": 1, "diagnostic_only": False},
        {"query_id": "q2", "paper_id": 2, "diagnostic_only": False},
    ]

    validation = validate_packet(packet, labeled_keys={("q2", 2)}, expected_topics={"q1", "q2", "q3"})

    assert validation["no_duplicate_query_paper_rows"] is False
    assert validation["no_already_labeled_non_diagnostic_rows"] is False
    assert validation["all_expected_topics_represented"] is False
    assert validation["missing_topics"] == ["q3"]


def test_v3_1_active_learning_selection_exports_unique_unlabeled_topics() -> None:
    candidates = []
    for topic_index in range(2):
        topic = f"topic_{topic_index}"
        for offset in range(6):
            candidates.append(
                _candidate(
                    topic,
                    topic_index * 100 + offset,
                    ridge=1.0 - offset * 0.05,
                    v27=0.1 + offset * 0.1,
                    v26=0.2 + offset * 0.05,
                    hybrid=0.15 + offset * 0.02,
                )
            )
    v30_report = {
        "splits": {"train_dev_test": {"test_topics": ["topic_1"]}},
        "failure_summary": {
            "loto_topics_losing_reading_ndcg_vs_v2_7": [{"topic": "topic_1", "reading_delta": -0.2}],
            "loto_topics_losing_reading_ndcg_vs_v2_6": [],
        },
    }

    packet, summary = select_active_learning_candidates(
        candidates,
        v30_report=v30_report,
        per_topic_min=2,
        per_topic_max=3,
    )

    assert summary["exported_count"] == 6
    assert summary["per_topic_counts"] == {"topic_0": 3, "topic_1": 3}
    assert len({(row["query_id"], row["paper_id"]) for row in packet}) == len(packet)
    assert all(row["already_labeled"] is False for row in packet)


def test_v3_1_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256
