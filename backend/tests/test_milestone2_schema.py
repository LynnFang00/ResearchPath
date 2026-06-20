from pydantic import ValidationError

from app.models.citation_edge import CitationEdge
from app.models.paper import Paper
from app.schemas.paper import CitationEdgeCreate, PaperCreate


def test_paper_schema_keeps_old_records_compatible() -> None:
    paper = PaperCreate(
        title="A Baseline Paper",
        abstract="This is still enough to parse older records.",
    )

    assert paper.external_id is None
    assert paper.references_count == 0
    assert paper.influential_citation_count == 0
    assert paper.references == []


def test_paper_model_has_milestone2_fields() -> None:
    paper = Paper(
        title="Paper",
        abstract="Abstract text",
        external_id="arxiv:1234.5678",
        source="arxiv",
        doi="10.1234/example",
        url="https://example.com",
        references_count=2,
        influential_citation_count=1,
        abstract_word_count=2,
    )

    assert paper.external_id == "arxiv:1234.5678"
    assert paper.url == "https://example.com"
    assert paper.abstract_word_count == 2


def test_citation_edge_schema_validates_relationship_type() -> None:
    edge = CitationEdgeCreate(
        source_paper_id=1,
        target_paper_id=2,
        relationship_type="co_citation",
        source="test",
    )

    assert edge.relationship_type == "co_citation"

    try:
        CitationEdgeCreate(source_paper_id=1, target_paper_id=2, relationship_type="unknown")
    except ValidationError as exc:
        assert "Unsupported relationship_type" in str(exc)
    else:
        raise AssertionError("Expected invalid relationship_type to fail.")


def test_citation_edge_model_fields() -> None:
    edge = CitationEdge(
        source_paper_id=1,
        target_paper_id=2,
        relationship_type="reference",
        source="sample",
    )

    assert edge.source_paper_id == 1
    assert edge.target_paper_id == 2
    assert edge.relationship_type == "reference"
