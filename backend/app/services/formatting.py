from app.models.paper import Paper
from app.schemas.paper import RecommendationResponse


def split_semicolon_field(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def make_snippet(text: str, max_chars: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def paper_to_recommendation(
    paper: Paper,
    score: float,
    method: str,
    explanation: str,
) -> RecommendationResponse:
    return RecommendationResponse(
        paper_id=paper.id,
        title=paper.title,
        abstract_snippet=make_snippet(paper.abstract),
        year=paper.year,
        authors=split_semicolon_field(paper.authors),
        score=round(float(score), 6),
        method=method,
        explanation=explanation,
    )
