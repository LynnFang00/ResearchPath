from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_v3_0_ltr_models import (  # noqa: E402
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    ModelSpec,
    SELECTED_240,
    file_sha256,
    train_models_for_topics,
    validate_no_duplicate_query_paper,
    validate_topic_splits,
    evaluate_rows_with_scores,
)


def _row(query_id: str, paper_id: int, score: float, relevance: float, *, positive: bool) -> dict:
    return {
        "query_id": query_id,
        "paper_id": paper_id,
        "judged_source": "v2_1",
        "bm25_score": score,
        "tfidf_score": score,
        "embedding_score": score,
        "faiss_embedding_score": score,
        "hybrid_score": score,
        "old_v2_2b_score": score,
        "v2_6_score": score,
        "v2_7_score": score,
        "reading_value_score": relevance,
        "topic_match_score": relevance,
        "relevance_label": relevance,
        "positive": positive,
        "hard_negative": not positive,
        "hidden_positive": False,
        "duplicate_status": "none",
    }


def test_v3_0_duplicate_key_detection() -> None:
    duplicates = validate_no_duplicate_query_paper(
        [
            {"query_id": "q1", "paper_id": 1},
            {"query_id": "q1", "paper_id": 1},
            {"query_id": "q2", "paper_id": 1},
        ]
    )

    assert duplicates == [{"query_id": "q1", "paper_id": 1, "count": 2}]


def test_v3_0_topic_split_validation_detects_no_leakage() -> None:
    splits = {
        "train_dev_test": {
            "train_topics": ["q1", "q2"],
            "dev_topics": ["q3"],
            "test_topics": ["q4"],
        },
        "leave_topic_out": [
            {"fold_id": "loto_01", "train_topics": ["q2", "q3", "q4"], "dev_topics": [], "test_topics": ["q1"]},
            {"fold_id": "loto_02", "train_topics": ["q1", "q3", "q4"], "dev_topics": [], "test_topics": ["q2"]},
        ],
    }

    validation = validate_topic_splits(splits)

    assert validation["train_dev_test_no_topic_leakage"] is True
    assert validation["leave_topic_out_no_topic_leakage"] is True


def test_v3_0_training_uses_only_train_topics() -> None:
    rows = [
        _row("q1", 1, 0.9, 1.0, positive=True),
        _row("q1", 2, 0.1, 0.0, positive=False),
        _row("q2", 3, 0.8, 1.0, positive=True),
        _row("q2", 4, 0.2, 0.0, positive=False),
    ]
    spec = ModelSpec(
        name="ridge_test",
        model_type="pointwise_ridge_regression",
        feature_columns=["bm25_score", "hybrid_score"],
        target="relevance_label",
        hyperparameters={"alpha": 1.0, "random_state": 17},
    )

    trained = train_models_for_topics(rows, {"q1"}, [spec])

    assert trained[0].training_topics == ["q1"]
    assert trained[0].training_row_count == 2


def test_v3_0_evaluator_scores_baselines_and_trained_models() -> None:
    rows = [
        _row("q1", 1, 0.9, 1.0, positive=True),
        _row("q1", 2, 0.1, 0.0, positive=False),
        _row("q2", 3, 0.8, 1.0, positive=True),
        _row("q2", 4, 0.2, 0.0, positive=False),
    ]
    spec = ModelSpec(
        name="ridge_test",
        model_type="pointwise_ridge_regression",
        feature_columns=["bm25_score", "hybrid_score"],
        target="relevance_label",
        hyperparameters={"alpha": 1.0, "random_state": 17},
    )
    trained = train_models_for_topics(rows, {"q1"}, [spec])

    report = evaluate_rows_with_scores(rows, trained, k=2)

    assert "bm25_score" in report["combined"]["method_averages"]
    assert "v3_0_ridge_test_score" in report["combined"]["method_averages"]
    assert report["combined"]["per_topic"]["v3_0_ridge_test_score"][0]["retrieved_ids_at_k"]


def test_v3_0_protected_hashes_are_unchanged() -> None:
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256
