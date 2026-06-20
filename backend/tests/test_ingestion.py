import pytest

from app.services.ingestion import parse_paper_record


def test_parse_paper_record_accepts_valid_record() -> None:
    record = {
        "title": "Attention Is All You Need",
        "abstract": "A transformer architecture for sequence modeling.",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "venue": "NeurIPS",
        "categories": ["cs.CL", "cs.LG"],
        "citation_count": 100000,
        "source_url": "https://arxiv.org/abs/1706.03762",
    }

    paper = parse_paper_record(record, row_number=1)

    assert paper.title == "Attention Is All You Need"
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.categories == ["cs.CL", "cs.LG"]


def test_parse_paper_record_reports_row_number() -> None:
    with pytest.raises(ValueError, match="Row 4"):
        parse_paper_record({"title": "", "abstract": "Missing title"}, row_number=4)
