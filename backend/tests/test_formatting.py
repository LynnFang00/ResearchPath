from app.models.paper import Paper
from app.services.formatting import paper_to_recommendation


def test_recommendation_response_format() -> None:
    paper = Paper(
        id=7,
        title="Learning to Rank Papers",
        abstract="This paper studies ranking models for research paper recommendation." * 10,
        authors="Ada Lovelace; Alan Turing",
        year=2026,
        venue="ResearchPath Test",
        categories="cs.IR",
        citation_count=3,
        source_url="https://example.com/paper",
    )

    response = paper_to_recommendation(
        paper=paper,
        score=1.23456789,
        method="bm25",
        explanation="matched important keywords",
    )

    assert response.paper_id == 7
    assert response.authors == ["Ada Lovelace", "Alan Turing"]
    assert response.score == 1.234568
    assert response.method == "bm25"
    assert response.explanation == "matched important keywords"
    assert response.abstract_snippet.endswith("...")
