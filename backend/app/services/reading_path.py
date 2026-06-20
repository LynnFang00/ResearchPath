from dataclasses import dataclass, field
from datetime import UTC, datetime
import math

from app.models.paper import Paper
from app.schemas.paper import PathPaper, PathSectionStatus, ReadingPathResponse
from app.services.difficulty import DifficultyResult, score_difficulty
from app.services.formatting import make_snippet, split_semicolon_field
from app.services.deduplication import completeness_score, duplicate_match
from app.services.quality_signals import (
    SECTION_NAMES,
    compute_quality_signals,
    normalize_scores,
    section_minimum_score,
    section_reason,
    section_score as score_for_section,
)


PATH_SECTIONS = SECTION_NAMES
BEGINNER_LEVELS = {"basic_ml", "beginner", "intermediate", "advanced"}


@dataclass(frozen=True)
class PathCandidate:
    paper: Paper
    score: float
    method: str
    explanation: str
    retrieval_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedPaper:
    candidate: PathCandidate
    section: str
    section_score: float
    difficulty: DifficultyResult
    reason: str
    diagnostics: dict[str, float]


def normalize_background_level(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in BEGINNER_LEVELS:
        allowed = ", ".join(sorted(BEGINNER_LEVELS))
        raise ValueError(f"Unsupported background level '{value}'. Use one of: {allowed}.")
    return normalized


def plan_reading_path(
    *,
    query: str,
    candidates: list[PathCandidate],
    method: str,
    background_level: str = "basic_ml",
    papers_per_section: int = 4,
    current_year: int | None = None,
) -> ReadingPathResponse:
    background_level = normalize_background_level(background_level)
    current_year = current_year or datetime.now(UTC).year
    planned_by_section: dict[str, list[PlannedPaper]] = {section: [] for section in PATH_SECTIONS}
    duplicate_penalties = duplicate_penalties_for_candidates(candidates)
    normalized_candidate_scores = normalize_scores(
        {index: candidate.score for index, candidate in enumerate(candidates)}
    )

    for index, candidate in enumerate(candidates):
        difficulty = score_difficulty(candidate.paper, current_year=current_year)
        signals = _signals_for_candidate(
            query=query,
            candidate=candidate,
            difficulty=difficulty,
            background_level=background_level,
            current_year=current_year,
            duplicate_penalty=duplicate_penalties.get(index, 0.0),
            normalized_score=normalized_candidate_scores.get(index, 0.0),
        )
        for section in PATH_SECTIONS:
            section_score = score_for_section(section, signals, query=query)
            diagnostics = signals.diagnostics() | {
                "section_score": round(section_score, 6),
                "final_path_score": round(section_score, 6),
            }
            planned_by_section[section].append(
                PlannedPaper(
                    candidate=candidate,
                    section=section,
                    section_score=section_score,
                    difficulty=difficulty,
                    reason=section_reason(section),
                    diagnostics=diagnostics,
                )
            )

    selected_ids: set[int] = set()
    sections: dict[str, list[PathPaper]] = {}
    section_status: dict[str, PathSectionStatus] = {}
    for section in PATH_SECTIONS:
        ranked = sorted(planned_by_section[section], key=lambda item: item.section_score, reverse=True)
        section_papers: list[PathPaper] = []
        for planned in ranked:
            paper = planned.candidate.paper
            if paper.id in selected_ids:
                continue
            if section == "recent_frontier" and planned.diagnostics["recency_score"] < 0.35:
                continue
            if planned.section_score < section_minimum_score(section):
                continue
            section_papers.append(planned_to_response(planned))
            selected_ids.add(paper.id)
            if len(section_papers) >= papers_per_section:
                break
        sections[section] = section_papers
        section_complete = len(section_papers) >= papers_per_section
        section_status[section] = PathSectionStatus(
            section_complete=section_complete,
            fill_reason=None if section_complete else "not_enough_high_quality_candidates",
        )

    return ReadingPathResponse(
        query=query,
        method=method,
        background_level=background_level,
        sections=sections,
        section_status=section_status,
    )


def _signals_for_candidate(
    *,
    query: str,
    candidate: PathCandidate,
    difficulty: DifficultyResult,
    background_level: str,
    current_year: int,
    duplicate_penalty: float,
    normalized_score: float,
):
    components = candidate.retrieval_components
    if not components and candidate.score < 0.05:
        normalized_score = min(normalized_score, max(candidate.score, 0.0))
    bm25_score = components.get("bm25_score", 0.0)
    tfidf_score = components.get("tfidf_score", 0.0)
    faiss_score = components.get("faiss_score", 0.0)
    if not components:
        if candidate.method == "tfidf":
            tfidf_score = normalized_score
        elif candidate.method in {"faiss_embedding", "embedding"}:
            faiss_score = normalized_score
        else:
            bm25_score = normalized_score

    return compute_quality_signals(
        query=query,
        paper=candidate.paper,
        difficulty=difficulty,
        background_level=background_level,
        bm25_score=bm25_score,
        tfidf_score=tfidf_score,
        faiss_score=faiss_score,
        duplicate_penalty=duplicate_penalty,
        current_year=current_year,
    )


def duplicate_penalties_for_candidates(candidates: list[PathCandidate]) -> dict[int, float]:
    penalties: dict[int, float] = {}
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            match = duplicate_match(left.paper, right.paper)
            if match is None:
                continue
            left_quality = _duplicate_survivor_quality(left)
            right_quality = _duplicate_survivor_quality(right)
            penalized_index = right_index if left_quality >= right_quality else left_index
            penalty = 0.75 if match.kind in {"external_id", "doi", "title"} else 0.35
            penalties[penalized_index] = max(penalties.get(penalized_index, 0.0), penalty)
    return penalties


def _duplicate_survivor_quality(candidate: PathCandidate) -> float:
    year = candidate.paper.year or 9999
    older_bonus = max(0.0, (2030 - year) / 100.0)
    citation_bonus = min(math.log1p(candidate.paper.citation_count or 0) / 8.0, 1.5)
    return candidate.score + citation_bonus + older_bonus + 0.03 * completeness_score(candidate.paper)


def planned_to_response(planned: PlannedPaper, *, override_section: str | None = None) -> PathPaper:
    paper = planned.candidate.paper
    section = override_section or planned.section
    return PathPaper(
        paper_id=paper.id,
        title=paper.title,
        abstract_snippet=make_snippet(paper.abstract),
        year=paper.year,
        authors=split_semicolon_field(paper.authors),
        score=round(float(planned.candidate.score), 6),
        method=planned.candidate.method,
        explanation=planned.candidate.explanation,
        difficulty_label=planned.difficulty.label,
        difficulty_score=planned.difficulty.score,
        difficulty_explanation=planned.difficulty.explanation,
        path_section=section,
        path_reason=planned.reason if override_section is None else f"used to fill the {section} section from available candidates",
        relevance_score=planned.diagnostics["relevance_score"],
        bm25_score=planned.diagnostics["bm25_score"],
        tfidf_score=planned.diagnostics["tfidf_score"],
        faiss_score=planned.diagnostics["faiss_score"],
        citation_score=planned.diagnostics["citation_score"],
        influence_score=planned.diagnostics["influence_score"],
        recency_score=planned.diagnostics["recency_score"],
        difficulty_fit_score=planned.diagnostics["difficulty_fit_score"],
        background_signal=planned.diagnostics["background_signal"],
        method_signal=planned.diagnostics["method_signal"],
        narrow_application_score=planned.diagnostics["narrow_application_score"],
        too_narrow_penalty=planned.diagnostics["too_narrow_penalty"],
        section_score=planned.diagnostics["section_score"],
        duplicate_penalty=planned.diagnostics["duplicate_penalty"],
        final_path_score=planned.diagnostics["final_path_score"],
    )
