from app.models.paper import Paper
from app.services.difficulty import score_difficulty
from app.services.paper_types import classify_paper_types
from app.services.personalization import score_personalization
from app.services.reading_path import PathCandidate, plan_reading_path


def _paper(paper_id: int, title: str, abstract: str, *, year: int = 2024, citations: int = 10) -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract=abstract,
        authors="A",
        year=year,
        categories="cs.LG",
        citation_count=citations,
        references_count=20,
    )


def test_paper_type_classification() -> None:
    paper = _paper(
        1,
        "A Survey and Benchmark of Transformer Architectures",
        "A tutorial survey with benchmark results for transformer architecture methods.",
        year=2024,
        citations=1500,
    )

    tags = classify_paper_types(paper, current_year=2026)

    assert "survey" in tags
    assert "tutorial" in tags
    assert "benchmark" in tags
    assert "foundation" in tags
    assert "frontier" in tags


def test_personalization_score_changes_for_saved_and_skipped_papers() -> None:
    saved = _paper(1, "Transformer Agents", "Tool using transformer agents.")
    candidate = _paper(2, "Transformer Agent Planning", "Tool using transformer agents for planning.")
    skipped = _paper(3, "Unrelated Medical Dataset", "Clinical dataset study.")
    paper_by_id = {paper.id: paper for paper in [saved, candidate, skipped]}

    boosted = score_personalization(
        paper=candidate,
        paper_by_id=paper_by_id,
        profile={"saved_paper_ids": [1], "skipped_paper_ids": [], "too_hard_paper_ids": [], "too_easy_paper_ids": []},
        difficulty=score_difficulty(candidate, current_year=2026),
        background_level="basic_ml",
    )
    penalized = score_personalization(
        paper=candidate,
        paper_by_id=paper_by_id,
        profile={"saved_paper_ids": [], "skipped_paper_ids": [2], "too_hard_paper_ids": [], "too_easy_paper_ids": []},
        difficulty=score_difficulty(candidate, current_year=2026),
        background_level="basic_ml",
    )

    assert boosted.score > 0
    assert penalized.score < 0


def test_personalization_uses_preferred_topics() -> None:
    candidate = _paper(2, "Transformer Agent Planning", "Tool using transformer agents for planning.")

    result = score_personalization(
        paper=candidate,
        paper_by_id={candidate.id: candidate},
        profile={
            "saved_paper_ids": [],
            "skipped_paper_ids": [],
            "too_hard_paper_ids": [],
            "too_easy_paper_ids": [],
            "preferred_topics": ["transformer"],
        },
        difficulty=score_difficulty(candidate, current_year=2026),
        background_level="basic_ml",
    )

    assert result.topic_similarity == 1.0
    assert result.score > 0


def test_reading_path_response_includes_product_explanations() -> None:
    candidate = PathCandidate(
        paper=_paper(
            1,
            "A Survey of Large Language Model Agents",
            "A survey and tutorial overview for beginner researchers.",
            year=2024,
            citations=200,
        ),
        score=1.0,
        method="hybrid",
        explanation="combined signals",
        retrieval_components={"bm25_score": 1.0, "personalization_score": 0.0},
    )

    path = plan_reading_path(
        query="large language model agents",
        candidates=[candidate],
        method="hybrid",
        papers_per_section=1,
        current_year=2026,
    )

    paper = path.sections["background"][0]
    assert paper.why_recommended
    assert paper.why_this_section
    assert paper.confidence_label in {"high", "medium", "low"}
    assert "survey" in paper.paper_type_tags
    assert paper.explanation_signals
