from app.services.bm25 import BM25Retriever


def test_bm25_ranks_keyword_match_first() -> None:
    retriever = BM25Retriever(
        [
            (1, "Graph neural networks for molecule discovery"),
            (2, "A beginner introduction to AI agents and tool use"),
            (3, "Sorting algorithms and data structures"),
        ]
    )

    results = retriever.search("AI agents", k=2)

    assert results[0].document_id == 2
    assert results[0].score > 0
