import sys
from pathlib import Path

from app.models.paper import Paper


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_retrieval import evaluate_retrievers  # noqa: E402


def test_evaluation_runner_output_format(tmp_path) -> None:
    papers = [
        Paper(id=1, title="AI Agents", abstract="Agents use tools.", citation_count=10, year=2024),
        Paper(id=2, title="Graph Systems", abstract="Graphs and databases.", citation_count=5, year=2020),
    ]
    examples = [
        {
            "query_id": "agents",
            "query": "AI agents tools",
            "relevant_paper_ids": [1],
        }
    ]

    report = evaluate_retrievers(examples=examples, papers=papers, output_dir=tmp_path)

    assert report["paper_count"] == 2
    assert report["query_count"] == 1
    assert set(report["methods"]) == {"bm25", "tfidf", "citation_recency"}
    assert "recall_at_5" in report["methods"]["bm25"]["averages"]
    assert Path(report["output_path"]).exists()
