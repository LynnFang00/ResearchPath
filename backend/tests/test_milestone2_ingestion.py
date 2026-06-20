import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.citation_edge import CitationEdge
from app.models.paper import Paper
from app.services.ingestion import ingest_papers_from_file, parse_citation_edge_record


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_parse_citation_edge_record() -> None:
    edge = parse_citation_edge_record(
        {
            "source_paper_id": 1,
            "target_paper_id": 2,
            "relationship_type": "bibliographic_coupling",
            "source": "unit-test",
        },
        row_number=1,
    )

    assert edge.relationship_type == "bibliographic_coupling"


def test_ingestion_deduplicates_and_creates_reference_edges(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    records = [
        {
            "external_id": "paper:a",
            "title": "First Paper",
            "abstract": "Graph retrieval for papers.",
            "authors": ["A"],
            "year": 2022,
            "references": [],
        },
        {
            "external_id": "paper:b",
            "title": "Second Paper",
            "abstract": "This paper references graph retrieval.",
            "authors": ["B"],
            "year": 2023,
            "references": ["paper:a"],
        },
        {
            "external_id": "paper:b",
            "title": "Duplicate Second Paper",
            "abstract": "Duplicate record.",
            "authors": ["B"],
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    db = _session()

    result = ingest_papers_from_file(path, db, write_manifest=False)

    assert result["inserted"] == 2
    assert result["skipped"] == 1
    assert result["citation_edges_inserted"] == 1
    assert any("duplicate paper" in warning for warning in result["warnings"])
    assert db.scalar(select(Paper).where(Paper.external_id == "paper:a")) is not None
    edge = db.scalar(select(CitationEdge))
    assert edge is not None
    assert edge.relationship_type == "reference"
