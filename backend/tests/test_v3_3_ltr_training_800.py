from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_v3_3_ltr_models_800 import (  # noqa: E402
    BASELINE_COLUMNS,
    FEATURE_VARIANTS,
    RAW_SCORE_FEATURES,
    ModelSpec,
    build_model_specs,
    evaluate_rows_with_scores,
    model_score_column,
    train_models_for_topics,
)


def _row(query_id: str, paper_id: int, score: float, relevance: float, *, positive: bool) -> dict:
    row = {
        "query_id": query_id,
        "paper_id": paper_id,
        "judged_source": "v3_2",
        "reading_value_score": relevance,
        "topic_match_score": relevance,
        "relevance_label": relevance,
        "positive": positive,
        "hard_negative": not positive,
        "hidden_positive": False,
        "duplicate_status": "none",
        "year": 2024,
        "citation_count": 10,
        "full_text_available": False,
        "old_v2_2b_score": score,
        "v2_6_score": score,
        "v2_7_score": score,
    }
    for method in ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]:
        row[f"{method}_score"] = score
        row[f"{method}_raw_score"] = score
        row[f"{method}_rank"] = 1 if score >= 0.5 else 2
        row[f"{method}_reciprocal_rank"] = 1.0 if score >= 0.5 else 0.5
    return row


def test_v3_3_feature_variants_drop_raw_score_features() -> None:
    for columns in FEATURE_VARIANTS.values():
        assert not (set(columns) & RAW_SCORE_FEATURES)
    assert "v2_7_score" not in FEATURE_VARIANTS["no_v27_compact"]
    assert "v2_7_score" in FEATURE_VARIANTS["with_v27_compact"]


def test_v3_3_model_specs_include_requested_variants() -> None:
    names = {spec.name for spec in build_model_specs()}

    assert "ridge_rank_features_only" in names
    assert "ridge_normalized_scores_only" in names
    assert "ridge_compact_combined" in names
    assert "ridge_no_v27_compact" in names
    assert "ridge_with_v27_compact" in names
    assert "pairwise_logistic_no_v27_compact" in names


def test_v3_3_training_uses_only_train_topics() -> None:
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


def test_v3_3_evaluator_scores_baselines_and_trained_models() -> None:
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

    assert set(BASELINE_COLUMNS) <= set(report["combined"]["method_averages"])
    assert model_score_column("ridge_test") in report["combined"]["method_averages"]
    assert report["combined"]["per_topic"][model_score_column("ridge_test")][0]["retrieved_ids_at_k"]
