import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.paper import Paper
from app.models.paper_identifier import PaperIdentifier
from app.models.paper_source import PaperSource
from app.services.deduplication import is_near_duplicate_title, normalize_title
from app.services.ingestion import _extract_arxiv_id, ingest_papers_from_file


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_title_normalization_ignores_case_and_punctuation() -> None:
    assert normalize_title("A Survey: Large-Language Models!") == normalize_title(
        "a survey large language models"
    )


def test_near_duplicate_title_detection_handles_survey_variants() -> None:
    assert is_near_duplicate_title(
        "A Survey of Large Language Model Agents",
        "Survey on Large Language Model Agents",
    )


def test_old_style_arxiv_doi_extracts_arxiv_identifier() -> None:
    assert _extract_arxiv_id("10.48550/arxiv.cmp-lg/9806001") == "cmp-lg/9806001"


def test_ingestion_merges_duplicate_metadata(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    records = [
        {
            "external_id": "arxiv:1",
            "title": "A Survey of Large Language Model Agents",
            "abstract": "Short abstract.",
            "authors": ["A"],
            "year": 2024,
        },
        {
            "doi": "10.1234/agents",
            "title": "A survey: large language model agents",
            "abstract": "Longer abstract with more complete metadata about LLM agents and tool use.",
            "authors": ["A", "B"],
            "venue": "Journal of AI",
            "citation_count": 10,
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    db = _session()

    result = ingest_papers_from_file(path, db, write_manifest=False)

    paper = db.scalar(select(Paper))
    assert result["inserted"] == 1
    assert result["skipped"] == 1
    assert paper is not None
    assert paper.doi == "10.1234/agents"
    assert paper.venue == "Journal of AI"
    assert "B" in paper.authors
    assert paper.citation_count == 10


def test_openalex_enrichment_merges_by_arxiv_identifier_with_exact_dedupe(tmp_path) -> None:
    arxiv_path = tmp_path / "arxiv.jsonl"
    openalex_path = tmp_path / "openalex.jsonl"
    arxiv_records = [
        {
            "external_id": "arxiv:cmp-lg/9806001",
            "source": "arxiv",
            "title": "Learning Correlations between Linguistic Indicators and Semantic Constraints",
            "abstract": "Original arXiv abstract.",
            "authors": ["A"],
            "year": 1998,
            "identifiers": {"arxiv": "cmp-lg/9806001"},
        },
    ]
    openalex_records = [
        {
            "external_id": "openalex:W4294349317",
            "source": "openalex",
            "title": "Learning Correlations between Linguistic Indicators and Semantic Constraints",
            "abstract": "Longer OpenAlex abstract with more complete metadata.",
            "authors": ["A", "B"],
            "year": 1998,
            "doi": "10.48550/arxiv.cmp-lg/9806001",
            "identifiers": {
                "arxiv": "cmp-lg/9806001",
                "openalex": "W4294349317",
                "doi": "10.48550/arxiv.cmp-lg/9806001",
            },
        },
    ]
    arxiv_path.write_text("\n".join(json.dumps(record) for record in arxiv_records), encoding="utf-8")
    openalex_path.write_text("\n".join(json.dumps(record) for record in openalex_records), encoding="utf-8")
    db = _session()

    first_result = ingest_papers_from_file(arxiv_path, db, write_manifest=False, allow_near_title_dedupe=False)
    second_result = ingest_papers_from_file(openalex_path, db, write_manifest=False, allow_near_title_dedupe=False)

    papers = db.scalars(select(Paper)).all()
    identifiers = db.scalars(select(PaperIdentifier)).all()
    sources = db.scalars(select(PaperSource)).all()
    assert first_result["inserted"] == 1
    assert second_result["inserted"] == 0
    assert second_result["skipped"] == 1
    assert len(papers) == 1
    assert {(item.source, item.identifier) for item in identifiers} == {
        ("arxiv", "cmp-lg/9806001"),
        ("doi", "10.48550/arxiv.cmp-lg/9806001"),
        ("openalex", "W4294349317"),
    }
    assert {item.source for item in sources} == {"arxiv", "openalex"}
