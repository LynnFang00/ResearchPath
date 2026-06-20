import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import Base
from app.models.paper import Paper
from app.services.deduplication import is_near_duplicate_title, normalize_title
from app.services.ingestion import ingest_papers_from_file


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
