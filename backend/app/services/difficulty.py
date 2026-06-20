from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re

from app.models.paper import Paper


TECHNICAL_TERMS = {
    "attention",
    "backpropagation",
    "bayesian",
    "contrastive",
    "diffusion",
    "embedding",
    "fine-tuning",
    "graph neural",
    "in-context",
    "latent",
    "optimization",
    "policy gradient",
    "reinforcement",
    "representation learning",
    "transformer",
    "variational",
}

BEGINNER_SIGNALS = {
    "a primer",
    "beginner",
    "introduction",
    "introductory",
    "overview",
    "review",
    "survey",
    "tutorial",
}


@dataclass(frozen=True)
class DifficultyResult:
    label: str
    score: float
    explanation: str


def word_count_for_paper(paper: Paper) -> int:
    if paper.abstract_word_count:
        return paper.abstract_word_count
    return len(re.findall(r"\b\w+\b", paper.abstract or ""))


def technical_term_density(text: str) -> float:
    normalized = text.lower()
    words = re.findall(r"\b\w+\b", normalized)
    if not words:
        return 0.0
    hits = sum(1 for term in TECHNICAL_TERMS if term in normalized)
    return hits / max(len(words), 1)


def score_difficulty(
    paper: Paper,
    *,
    current_year: int | None = None,
) -> DifficultyResult:
    current_year = current_year or datetime.now(UTC).year
    text = f"{paper.title} {paper.abstract}".lower()
    word_count = word_count_for_paper(paper)
    density = technical_term_density(text)
    score = 0.35
    reasons: list[str] = []

    if any(signal in text for signal in BEGINNER_SIGNALS):
        score -= 0.24
        reasons.append("survey/tutorial framing")

    if word_count >= 350:
        score += 0.18
        reasons.append("long abstract")
    elif word_count >= 220:
        score += 0.10
        reasons.append("moderate abstract length")
    elif word_count <= 90:
        score -= 0.08
        reasons.append("short abstract")

    if density >= 0.025:
        score += 0.24
        reasons.append("dense technical terminology")
    elif density >= 0.012:
        score += 0.12
        reasons.append("some technical terminology")

    if (paper.references_count or 0) >= 80:
        score += 0.10
        reasons.append("large reference list")

    citation_count = paper.citation_count or 0
    if paper.year is not None and paper.year >= current_year - 2 and citation_count < 50:
        score += 0.12
        reasons.append("recent frontier paper")

    if citation_count >= 1000:
        score -= 0.08
        reasons.append("widely cited foundation")
    elif citation_count >= 200:
        score -= 0.04
        reasons.append("well-established paper")

    score = max(0.0, min(1.0, score))
    if score < 0.34:
        label = "beginner"
    elif score < 0.67:
        label = "intermediate"
    else:
        label = "advanced"

    if not reasons:
        reasons.append("balanced metadata signals")
    return DifficultyResult(
        label=label,
        score=round(score, 3),
        explanation=", ".join(reasons),
    )


def citation_strength(paper: Paper) -> float:
    return math.log1p(max(paper.citation_count, 0))
