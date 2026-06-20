import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    latency_ms: float | None = None


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in relevant_ids) / k


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    dcg = 0.0
    for index, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def mean_reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for index, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / index
    return 0.0


def evaluate_ranking(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
    latency_ms: float | None = None,
) -> RankingMetrics:
    return RankingMetrics(
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
        latency_ms=latency_ms,
    )


def time_call(fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000
