import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.expand_openalex_references import (  # noqa: E402
    collect_missing_reference_counts,
    expand_reference_records,
    select_reference_ids,
)


def test_collect_missing_reference_counts() -> None:
    records = [
        {"external_id": "openalex:W1", "references": ["openalex:W2", "openalex:W3"]},
        {"external_id": "openalex:W2", "references": ["openalex:W3"]},
    ]

    counts = collect_missing_reference_counts(records)

    assert counts["openalex:W3"] == 2
    assert "openalex:W2" not in counts


def test_select_reference_ids_respects_frequency_and_limit() -> None:
    counts = collect_missing_reference_counts(
        [
            {"external_id": "openalex:W1", "references": ["openalex:W9", "openalex:W8"]},
            {"external_id": "openalex:W2", "references": ["openalex:W9", "openalex:W7"]},
        ]
    )

    selected = select_reference_ids(counts, max_references=1, min_frequency=2)

    assert selected == ["openalex:W9"]


def test_expand_reference_records_with_fake_fetcher() -> None:
    def fake_fetcher(reference_id: str):
        return {
            "id": reference_id.replace("openalex:", "https://openalex.org/"),
            "display_name": "Expanded Paper",
            "abstract_inverted_index": {"Expanded": [0], "abstract": [1]},
            "authorships": [{"raw_author_name": "A Researcher"}],
            "publication_year": 2024,
            "primary_location": {"source": {"display_name": "Test Venue"}},
            "cited_by_count": 3,
            "referenced_works": [],
            "topics": [],
            "keywords": [],
        }

    records, warnings = expand_reference_records(["openalex:W123"], fake_fetcher)

    assert warnings == []
    assert records[0]["external_id"] == "openalex:W123"
    assert records[0]["title"] == "Expanded Paper"
