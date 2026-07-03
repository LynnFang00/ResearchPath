from datetime import UTC, datetime

from app.models.paper import Paper


def classify_paper_types(paper: Paper, *, current_year: int | None = None) -> list[str]:
    current_year = current_year or datetime.now(UTC).year
    text = f"{paper.title} {paper.abstract} {paper.venue or ''}".lower()
    tags: list[str] = []

    if any(term in text for term in ("survey", "review", "systematic review")):
        tags.append("survey")
    if any(term in text for term in ("tutorial", "primer", "introduction to", "guide")):
        tags.append("tutorial")
    if any(term in text for term in ("book", "handbook", "monograph")):
        tags.append("book")
    if (paper.citation_count or 0) >= 1000 or any(term in text for term in ("introduced", "seminal", "foundational")):
        tags.append("foundation")
    if any(term in text for term in ("method", "architecture", "algorithm", "model", "framework")):
        tags.append("method")
    if any(term in text for term in ("benchmark", "dataset", "evaluation suite", "leaderboard")):
        tags.append("benchmark")
    if any(term in text for term in ("application", "case study", "clinical", "medical", "domain-specific")):
        tags.append("application")
    if any(term in text for term in ("position", "perspective", "opinion", "viewpoint")):
        tags.append("position/opinion")
    if paper.year is not None and paper.year >= current_year - 3:
        tags.append("frontier")

    return tags or ["research"]
