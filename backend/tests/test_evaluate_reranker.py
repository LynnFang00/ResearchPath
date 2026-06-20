from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_reranker import rerank_candidate_ids, score_query_candidates  # noqa: E402


class FakePaper:
    def __init__(self, paper_id: int, text: str):
        self.id = paper_id
        self.searchable_text = text


def test_rerank_candidate_ids_orders_by_score_descending_stably():
    reranked = rerank_candidate_ids([10, 20, 30], [0.2, 0.9, 0.9], k=3)

    assert reranked == [20, 30, 10]


def test_score_query_candidates_builds_query_candidate_pairs():
    paper_by_id = {
        1: FakePaper(1, "first paper"),
        2: FakePaper(2, "second paper"),
    }

    scores = score_query_candidates(
        query="query",
        candidate_ids=[1, 2],
        paper_by_id=paper_by_id,
        score_pairs=lambda pairs: [float(len(candidate)) for _, candidate in pairs],
    )

    assert scores == [11.0, 12.0]
