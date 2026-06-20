from app.models.paper import Paper
from app.services.reading_path import PathCandidate, duplicate_penalties_for_candidates, plan_reading_path


def _candidate(paper_id: int, title: str, *, score: float = 1.0, year: int = 2024) -> PathCandidate:
    return PathCandidate(
        paper=Paper(
            id=paper_id,
            title=title,
            abstract="This paper introduces a method and architecture for machine learning systems.",
            authors="Ada Lovelace",
            year=year,
            categories="cs.LG",
            citation_count=20,
            references_count=20,
        ),
        score=score,
        method="bm25",
        explanation="matched query terms",
    )


def test_reading_path_includes_quality_diagnostics() -> None:
    path = plan_reading_path(
        query="LLM agents",
        candidates=[
            _candidate(
                1,
                "A Survey of Large Language Model Agents",
                score=1.0,
                year=2024,
            )
        ],
        method="bm25",
        papers_per_section=1,
        current_year=2026,
    )

    paper = path.sections["background"][0]
    assert paper.relevance_score is not None
    assert paper.citation_score is not None
    assert paper.recency_score is not None
    assert paper.section_score is not None
    assert paper.duplicate_penalty == 0.0
    assert paper.final_path_score == paper.section_score


def test_near_duplicate_candidate_gets_penalty() -> None:
    candidates = [
        _candidate(1, "A Survey of Large Language Model Agents", score=1.0),
        _candidate(2, "A survey: large language model agents", score=0.8),
    ]

    penalties = duplicate_penalties_for_candidates(candidates)

    assert penalties[1] > 0


def test_reading_path_does_not_force_fill_weak_sections() -> None:
    path = plan_reading_path(
        query="transformer",
        candidates=[
            PathCandidate(
                paper=Paper(
                    id=1,
                    title="Unrelated Clinical Dataset Note",
                    abstract="A narrow clinical case study unrelated to transformer learning paths.",
                    authors="A",
                    year=2026,
                    categories="medicine",
                    citation_count=0,
                    references_count=5,
                ),
                score=0.01,
                method="bm25",
                explanation="weak match",
            )
        ],
        method="bm25",
        papers_per_section=2,
        current_year=2026,
    )

    assert len(path.sections["background"]) < 2
    assert path.section_status["background"].section_complete is False
    assert path.section_status["background"].fill_reason == "not_enough_high_quality_candidates"
