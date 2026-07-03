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
    assert response.paper_url == "https://example.com/paper"
    assert response.source_url == "https://example.com/paper"
    assert response.abstract_snippet.endswith("...")


def test_recommendation_response_includes_arxiv_pdf_and_doi_links() -> None:
    paper = Paper(
        id=8,
        title="Attention Is All You Need",
        abstract="Transformer architecture paper.",
        authors="A; B",
        year=2017,
        venue="NeurIPS",
        categories="cs.CL",
        citation_count=1000,
        source_url="https://arxiv.org/abs/1706.03762",
        doi="10.48550/arXiv.1706.03762",
    )

    response = paper_to_recommendation(
        paper=paper,
        score=2.0,
        method="bm25",
        explanation="matched important keywords",
    )

    assert response.paper_url == "https://arxiv.org/abs/1706.03762"
    assert response.pdf_url == "https://arxiv.org/pdf/1706.03762"
    assert response.doi_url == "https://doi.org/10.48550/arXiv.1706.03762"
