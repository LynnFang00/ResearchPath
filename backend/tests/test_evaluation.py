from app.ml.evaluation import evaluate_ranking


def test_evaluate_ranking_metrics() -> None:
    metrics = evaluate_ranking(
        retrieved_ids=[10, 20, 30],
        relevant_ids={20, 40},
        k=3,
        latency_ms=12.5,
    )

    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 1 / 3
    assert metrics.mrr == 0.5
    assert metrics.latency_ms == 12.5
    assert 0 < metrics.ndcg_at_k < 1


def test_evaluate_ranking_empty_relevance_edge_case() -> None:
    metrics = evaluate_ranking(
        retrieved_ids=[1, 2, 3],
        relevant_ids=set(),
        k=5,
    )

    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.ndcg_at_k == 0.0
    assert metrics.mrr == 0.0
