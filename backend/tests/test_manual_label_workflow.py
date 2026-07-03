import json
from pathlib import Path
from types import SimpleNamespace
import sys

from app.models.paper import Paper


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.export_labeling_candidates import collect_labeling_candidates, load_query_set  # noqa: E402
from scripts.export_fulltext_label_packets import build_packets  # noqa: E402
from scripts.fetch_fulltext_for_labeling import (  # noqa: E402
    extract_arxiv_id,
    manifest_row_for,
    source_from_arxiv_feed,
    source_from_openalex_work,
)
from scripts.validate_manual_labels import load_jsonl, validate_labels  # noqa: E402


def test_load_query_set_requires_all_expected_sections(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "transformer architecture",
                    "target_audience": "beginner",
                    "expected_sections": ["background", "foundational", "core_methods", "recent_frontier"],
                }
            ]
        ),
        encoding="utf-8",
    )

    queries = load_query_set(path)

    assert queries[0]["query_id"] == "q1"
    assert queries[0]["query"] == "transformer architecture"


def test_collect_labeling_candidates_deduplicates_query_paper_by_method() -> None:
    paper = Paper(
        id=1,
        title="Attention Is All You Need",
        abstract="A transformer architecture paper.",
        authors="A; B",
        year=2017,
        venue="NeurIPS",
        doi="10.0000/test",
        external_id="openalex:1",
        citation_count=1000,
    )
    queries = [
        {
            "query_id": "q1",
            "query": "transformer architecture",
            "target_audience": "beginner",
            "expected_sections": ["background", "foundational", "core_methods", "recent_frontier"],
        }
    ]

    rows = collect_labeling_candidates(
        queries=queries,
        methods=["bm25", "hybrid"],
        top_k=10,
        retrieve=lambda query, method, k: [SimpleNamespace(paper_id=1)],
        get_paper=lambda paper_id: paper if paper_id == 1 else None,
    )

    assert len(rows) == 1
    assert rows[0]["paper_id"] == 1
    assert rows[0]["authors"] == ["A", "B"]
    assert rows[0]["source_methods"] == ["bm25", "hybrid"]
    assert rows[0]["retrieval_ranks_by_method"] == {"bm25": 1, "hybrid": 1}
    assert {"query_id", "paper_id", "abstract_snippet", "doi", "external_id"} <= set(rows[0])


def test_validate_manual_labels_reports_distribution_and_pool_mismatch(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    label_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "transformer architecture",
                "paper_id": 1,
                "title": "Attention Is All You Need",
                "relevance": 3,
                "best_section": "foundational",
                "section_fit": 3,
                "difficulty": "intermediate",
                "difficulty_fit_for_beginner": 2,
                "too_narrow": False,
                "too_advanced": False,
                "duplicate": False,
                "would_recommend": True,
                "notes": "Central transformer paper.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_labels(load_jsonl(label_path), pool_keys={("q1", 2)})

    assert report["label_count"] == 1
    assert report["labels_per_query"] == {"q1": 1}
    assert report["relevance_distribution"] == {3: 1}
    assert report["section_distribution"] == {"foundational": 1}
    assert report["difficulty_distribution"] == {"intermediate": 1}
    assert report["would_recommend_count"] == 1
    assert report["labels_not_in_pool"] == [{"query_id": "q1", "paper_id": 1, "line": 1}]
    assert report["is_valid"] is False


def test_validate_manual_labels_detects_missing_invalid_and_duplicate_rows(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    row = {
        "query_id": "q1",
        "query": "transformer architecture",
        "paper_id": 1,
        "title": "Bad Label",
        "relevance": 4,
        "best_section": "wrong",
        "section_fit": 0,
        "difficulty": "expert",
        "difficulty_fit_for_beginner": 0,
        "too_narrow": "false",
        "too_advanced": False,
        "duplicate": False,
        "would_recommend": False,
    }
    label_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    report = validate_labels(load_jsonl(label_path))

    assert report["missing_required_fields"] == [
        {"line": 1, "missing": ["notes"]},
        {"line": 2, "missing": ["notes"]},
    ]
    assert report["duplicate_query_paper_rows"] == [{"query_id": "q1", "paper_id": 1, "first_line": 1, "line": 2}]
    assert any(item["field"] == "relevance" for item in report["invalid_values"])
    assert any(item["field"] == "too_narrow" for item in report["invalid_values"])
    assert report["is_valid"] is False


def test_fulltext_manifest_schema_for_unavailable_text(tmp_path: Path) -> None:
    row = {
        "paper_id": 10,
        "title": "A Paper",
        "doi": "10.0000/example",
        "external_id": "openalex:1",
    }

    manifest = manifest_row_for(
        row,
        source=None,
        pdf_path=tmp_path / "pdfs" / "10.pdf",
        text_path=tmp_path / "text" / "10.txt",
        status="no_open_source",
    )

    assert manifest == {
        "paper_id": 10,
        "title": "A Paper",
        "doi": "10.0000/example",
        "external_id": "openalex:1",
        "full_text_available": False,
        "source_type": None,
        "source_url": None,
        "lookup_attempts": [],
        "pdf_path": None,
        "text_path": None,
        "status": "no_open_source",
        "error": "",
        "text_char_count": 0,
    }
    assert extract_arxiv_id("10.48550/arXiv.2010.11929") == "2010.11929"


def test_openalex_source_uses_only_open_access_direct_pdf() -> None:
    source = source_from_openalex_work(
        {
            "open_access": {"is_oa": True},
            "best_oa_location": {
                "is_oa": True,
                "pdf_url": "https://openaccess.thecvf.com/content/CVPR2021/papers/example.pdf",
            },
        }
    )
    blocked = source_from_openalex_work(
        {
            "open_access": {"is_oa": True},
            "best_oa_location": {
                "is_oa": True,
                "landing_page_url": "https://publisher.example.com/article/123",
            },
        }
    )

    assert source is not None
    assert source.source_type == "openalex_pdf"
    assert blocked is None


def test_arxiv_title_feed_returns_matching_pdf_only() -> None:
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2010.11929v2</id>
        <title>An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</title>
        <link href="https://arxiv.org/pdf/2010.11929" rel="related" title="pdf"/>
      </entry>
    </feed>
    """

    source = source_from_arxiv_feed(feed, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")

    assert source is not None
    assert source.source_type == "arxiv_title"
    assert source.source_url == "https://arxiv.org/pdf/2010.11929"


def test_fulltext_label_packet_schema_without_full_text() -> None:
    pool_rows = [
        {
            "query_id": "q1",
            "query": "transformer architecture",
            "target_audience": "beginner",
            "paper_id": 1,
            "title": "Attention",
            "abstract": "This paper describes transformer attention and architecture concepts clearly enough for abstract-only labeling.",
            "year": 2017,
            "venue": "NeurIPS",
            "citation_count": 100,
            "source_methods": ["bm25"],
            "retrieval_ranks_by_method": {"bm25": 1},
        }
    ]

    packets = build_packets(pool_rows=pool_rows, manifest_rows=[], per_query=10)

    assert packets[0]["full_text_available"] is False
    assert packets[0]["evidence_level"] == "abstract_only"
    assert packets[0]["selection_policy"] == "full_text_balanced_v1"
    assert packets[0]["text_path"] is None
    assert packets[0]["intro_excerpt"] == ""


def test_fulltext_label_packet_schema_with_full_text(tmp_path: Path) -> None:
    text_path = tmp_path / "1.txt"
    text_path.write_text(
        "Introduction This paper introduces the topic. Method We propose a model. Conclusion It works.",
        encoding="utf-8",
    )
    pool_rows = [
        {
            "query_id": "q1",
            "query": "transformer architecture",
            "target_audience": "beginner",
            "paper_id": 1,
            "title": "Attention",
            "abstract": "Transformer paper.",
            "year": 2017,
            "venue": "NeurIPS",
            "citation_count": 100,
            "source_methods": ["bm25"],
            "retrieval_ranks_by_method": {"bm25": 1},
        }
    ]
    manifest_rows = [
        {
            "paper_id": 1,
            "full_text_available": True,
            "text_path": str(text_path),
        }
    ]

    packets = build_packets(pool_rows=pool_rows, manifest_rows=manifest_rows, per_query=10)

    assert packets[0]["full_text_available"] is True
    assert packets[0]["evidence_level"] == "full_text_skim"
    assert "Introduction" in packets[0]["intro_excerpt"]
    assert "Method" in packets[0]["method_or_contribution_excerpt"]


def test_fulltext_packet_selection_keeps_top_ten_when_coverage_is_good() -> None:
    pool_rows = [
        {
            "query_id": "q1",
            "query": "transformer architecture",
            "target_audience": "beginner",
            "paper_id": paper_id,
            "title": f"Paper {paper_id}",
            "abstract": "This abstract is clear enough to support abstract-only labeling when needed.",
            "source_methods": ["bm25"],
            "retrieval_ranks_by_method": {"bm25": paper_id},
        }
        for paper_id in range(1, 13)
    ]
    manifest_rows = [{"paper_id": paper_id, "full_text_available": paper_id <= 7} for paper_id in range(1, 13)]

    packets = build_packets(pool_rows=pool_rows, manifest_rows=manifest_rows, per_query=10)

    assert [packet["paper_id"] for packet in packets] == list(range(1, 11))
    assert {packet["selection_reason"] for packet in packets} == {"top_k_has_enough_full_text"}


def test_fulltext_packet_selection_expands_when_coverage_is_low() -> None:
    pool_rows = [
        {
            "query_id": "q1",
            "query": "transformer architecture",
            "target_audience": "beginner",
            "paper_id": paper_id,
            "title": f"Paper {paper_id}",
            "abstract": "Too short." if paper_id in {4, 5} else "This is a clear abstract with enough detail for abstract-only labeling decisions.",
            "source_methods": ["bm25", "hybrid"] if paper_id in {1, 12, 13} else ["bm25"],
            "retrieval_ranks_by_method": {"bm25": paper_id, "hybrid": 1} if paper_id in {1, 12, 13} else {"bm25": paper_id},
        }
        for paper_id in range(1, 16)
    ]
    manifest_rows = [{"paper_id": paper_id, "full_text_available": paper_id in {2, 8, 11, 12, 13, 14, 15}} for paper_id in range(1, 16)]

    packets = build_packets(pool_rows=pool_rows, manifest_rows=manifest_rows, per_query=10, expand_to=15)
    packet_ids = [packet["paper_id"] for packet in packets]

    assert len(packets) == 10
    assert 1 in packet_ids
    assert 12 in packet_ids
    assert 4 not in packet_ids
    assert 5 not in packet_ids
    assert {packet["selection_reason"] for packet in packets} == {"expanded_for_low_full_text_coverage"}


def test_validate_manual_labels_fulltext_fields() -> None:
    label = {
        "query_id": "q1",
        "query": "transformer architecture",
        "paper_id": 1,
        "title": "Attention",
        "relevance": 3,
        "best_section": "foundational",
        "section_fit": 3,
        "difficulty": "intermediate",
        "difficulty_fit_for_beginner": 2,
        "too_narrow": False,
        "too_advanced": False,
        "duplicate": False,
        "would_recommend": True,
        "notes": "Good.",
        "evidence_level": "full_text_skim",
        "full_text_available": True,
        "label_confidence": "high",
    }

    valid = validate_labels([label], require_fulltext_fields=True)
    invalid = validate_labels([{**label, "evidence_level": "skimmed", "label_confidence": "certain"}])

    assert valid["is_valid"] is True
    assert valid["missing_required_fields"] == []
    assert any(item["field"] == "evidence_level" for item in invalid["invalid_values"])
    assert any(item["field"] == "label_confidence" for item in invalid["invalid_values"])
