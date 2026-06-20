import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fetch_openalex_papers import (  # noqa: E402
    build_filter,
    external_work_id,
    normalize_doi,
    openalex_work_to_paper_record,
    reconstruct_abstract,
)


def test_reconstruct_abstract_from_openalex_inverted_index() -> None:
    abstract = reconstruct_abstract({"Paper": [0], "ranking": [2], "recommendation": [1]})

    assert abstract == "Paper recommendation ranking"


def test_openalex_id_and_doi_normalization() -> None:
    assert external_work_id("https://openalex.org/W123") == "openalex:W123"
    assert normalize_doi("https://doi.org/10.1145/example") == "10.1145/example"
    assert normalize_doi("doi:10.1000/ABC") == "10.1000/abc"


def test_openalex_work_to_researchpath_record() -> None:
    work = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1234/test",
        "display_name": "Learning to Rank Papers",
        "abstract_inverted_index": {"Learning": [0], "rank": [2], "to": [1], "papers": [3]},
        "authorships": [
            {"raw_author_name": "Ada Lovelace"},
            {"author": {"display_name": "Alan Turing"}},
        ],
        "publication_year": 2026,
        "primary_location": {
            "landing_page_url": "https://example.com/paper",
            "source": {"display_name": "ResearchPath Conf"},
        },
        "cited_by_count": 42,
        "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
        "topics": [{"display_name": "Information Retrieval"}],
        "keywords": [{"display_name": "Recommender Systems"}],
        "updated_date": "2026-06-19",
    }

    record = openalex_work_to_paper_record(work)

    assert record["external_id"] == "openalex:W123"
    assert record["doi"] == "10.1234/test"
    assert record["title"] == "Learning to Rank Papers"
    assert record["abstract"] == "Learning to rank papers"
    assert record["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert record["venue"] == "ResearchPath Conf"
    assert record["citation_count"] == 42
    assert record["references"] == ["openalex:W1", "openalex:W2"]
    assert record["categories"] == ["Information Retrieval", "Recommender Systems"]
    assert record["updated_at"] == "2026-06-19T00:00:00+00:00"


def test_build_filter() -> None:
    filter_value = build_filter(
        from_year=2020,
        to_year=2026,
        work_type="article",
        min_citations=10,
        topic_ids=["T1", "T2"],
        extra_filter="open_access.is_oa:true",
    )

    assert "type:article" in filter_value
    assert "publication_year:2020-2026" in filter_value
    assert "cited_by_count:>9" in filter_value
    assert "topics.id:T1|T2" in filter_value
    assert "open_access.is_oa:true" in filter_value
