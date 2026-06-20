import pytest

from app.models.paper import Paper
from app.services.reading_path import PathCandidate, normalize_background_level, plan_reading_path


def make_candidate(
    paper_id: int,
    title: str,
    *,
    year: int,
    citations: int,
    abstract: str = "This paper studies machine learning systems.",
    score: float = 1.0,
) -> PathCandidate:
    return PathCandidate(
        paper=Paper(
            id=paper_id,
            title=title,
            abstract=abstract,
            authors="Ada Lovelace",
            year=year,
            categories="cs.LG",
            citation_count=citations,
            references_count=20,
        ),
        score=score,
        method="bm25",
        explanation="matched query terms",
    )


def test_reading_path_groups_candidates_into_learning_sections():
    candidates = [
        make_candidate(
            1,
            "A Survey of AI Agents",
            year=2021,
            citations=900,
            abstract="A survey and tutorial introduction to AI agents.",
        ),
        make_candidate(2, "Foundational Methods for Agent Learning", year=2016, citations=1200),
        make_candidate(3, "Planning Algorithms for Tool-Using Agents", year=2022, citations=80),
        make_candidate(
            4,
            "Frontier Transformer Agents for Scientific Discovery",
            year=2026,
            citations=10,
            abstract="Transformer attention and contrastive representation learning for scientific agents.",
        ),
    ]

    path = plan_reading_path(
        query="AI agents for scientific discovery",
        candidates=candidates,
        method="bm25",
        papers_per_section=1,
        current_year=2026,
    )

    assert path.sections["background"][0].paper_id == 1
    assert path.sections["foundational"][0].paper_id == 2
    assert path.sections["core_methods"][0].paper_id == 3
    assert path.sections["recent_frontier"][0].paper_id == 4
    assert path.sections["recent_frontier"][0].difficulty_label in {"intermediate", "advanced"}


def test_normalize_background_level_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported background level"):
        normalize_background_level("expert")
