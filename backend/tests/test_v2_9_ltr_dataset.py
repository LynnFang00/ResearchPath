from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v2_9_ltr_dataset import build_topic_splits, validate_no_duplicate_keys, validate_splits  # noqa: E402
from scripts.evaluate_v2_9_score_columns import evaluate_score_columns  # noqa: E402


def test_v2_9_duplicate_key_detection() -> None:
    rows = [
        {"query_id": "q1", "paper_id": 1},
        {"query_id": "q1", "paper_id": 1},
        {"query_id": "q1", "paper_id": 2},
    ]

    duplicates = validate_no_duplicate_keys(rows)

    assert duplicates == [{"query_id": "q1", "paper_id": 1, "count": 2}]


def test_v2_9_topic_splits_have_no_leakage() -> None:
    topics = [f"topic_{index:02d}" for index in range(16)]

    validation = validate_splits(build_topic_splits(topics))

    assert validation["train_dev_test_no_topic_leakage"] is True
    assert validation["leave_topic_out_no_topic_leakage"] is True
    assert validation["leave_topic_out_fold_count"] == 16


def test_v2_9_score_column_evaluator_orders_by_score_then_paper_id() -> None:
    rows = [
        {
            "query_id": "q1",
            "paper_id": 2,
            "judged_source": "v2_1",
            "score": 0.9,
            "reading_value_score": 1.0,
            "topic_match_score": 1.0,
            "positive": True,
            "hard_negative": False,
            "duplicate_status": "none",
        },
        {
            "query_id": "q1",
            "paper_id": 1,
            "judged_source": "v2_1",
            "score": 0.9,
            "reading_value_score": 0.5,
            "topic_match_score": 0.5,
            "positive": True,
            "hard_negative": False,
            "duplicate_status": "none",
        },
        {
            "query_id": "q1",
            "paper_id": 3,
            "judged_source": "v2_1",
            "score": 0.1,
            "reading_value_score": 0.0,
            "topic_match_score": 0.0,
            "positive": False,
            "hard_negative": True,
            "duplicate_status": "none",
        },
    ]

    report = evaluate_score_columns(rows, score_columns=["score"], k=2)

    assert report["v2_1"]["per_topic"]["score"][0]["retrieved_ids_at_k"] == [1, 2]
    assert report["v2_1"]["method_averages"]["score"]["positive_recall_at_10"] == 1.0
