from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re

from app.models.paper import Paper
from app.services.difficulty import DifficultyResult, citation_strength, score_difficulty


SECTION_NAMES = ("background", "foundational", "core_methods", "recent_frontier")

BACKGROUND_TERMS = {
    "book",
    "guide",
    "handbook",
    "introduction",
    "overview",
    "primer",
    "review",
    "survey",
    "tutorial",
}
METHOD_TERMS = {
    "algorithm",
    "architecture",
    "benchmark",
    "framework",
    "method",
    "model",
    "network",
    "optimization",
    "system",
    "technique",
    "training",
}
NARROW_APPLICATION_TERMS = {
    "case study",
    "clinical",
    "dataset",
    "domain-specific",
    "drug",
    "education",
    "industrial",
    "materials",
    "medical",
    "metal-organic",
    "molecular",
    "news",
    "protein",
    "single-cell",
    "usmle",
}
BROAD_QUERY_TERMS = {
    "agent",
    "agents",
    "deep learning",
    "diffusion model",
    "diffusion models",
    "graph neural network",
    "graph neural networks",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "recommendation systems",
    "reinforcement learning",
    "transformer",
    "transformers",
}
DOMAIN_QUERY_TERMS = NARROW_APPLICATION_TERMS | {
    "biology",
    "chemistry",
    "healthcare",
    "medicine",
    "science",
    "scientific discovery",
}


@dataclass(frozen=True)
class QualitySignals:
    relevance_score: float
    bm25_score: float
    tfidf_score: float
    faiss_score: float
    citation_score: float
    influence_score: float
    recency_score: float
    difficulty_score: float
    difficulty_fit_score: float
    background_signal: float
    method_signal: float
    narrow_application_score: float
    duplicate_penalty: float
    too_narrow_penalty: float

    def diagnostics(self) -> dict[str, float]:
        return {
            "relevance_score": round(self.relevance_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "tfidf_score": round(self.tfidf_score, 6),
            "faiss_score": round(self.faiss_score, 6),
            "citation_score": round(self.citation_score, 6),
            "influence_score": round(self.influence_score, 6),
            "recency_score": round(self.recency_score, 6),
            "difficulty_score": round(self.difficulty_score, 6),
            "difficulty_fit_score": round(self.difficulty_fit_score, 6),
            "background_signal": round(self.background_signal, 6),
            "method_signal": round(self.method_signal, 6),
            "narrow_application_score": round(self.narrow_application_score, 6),
            "duplicate_penalty": round(self.duplicate_penalty, 6),
            "too_narrow_penalty": round(self.too_narrow_penalty, 6),
        }


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    positive = {doc_id: max(score, 0.0) for doc_id, score in scores.items() if score > 0}
    if not positive:
        return {doc_id: 0.0 for doc_id in scores}
    max_score = max(positive.values())
    if max_score <= 0:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: max(score, 0.0) / max_score for doc_id, score in scores.items()}


def compute_quality_signals(
    *,
    query: str,
    paper: Paper,
    difficulty: DifficultyResult,
    background_level: str,
    bm25_score: float = 0.0,
    tfidf_score: float = 0.0,
    faiss_score: float = 0.0,
    duplicate_penalty: float = 0.0,
    current_year: int | None = None,
) -> QualitySignals:
    current_year = current_year or datetime.now(UTC).year
    text = f"{paper.title} {paper.abstract} {paper.venue or ''} {paper.categories}".lower()
    relevance_score = (0.48 * bm25_score) + (0.22 * tfidf_score) + (0.30 * faiss_score)
    if relevance_score == 0:
        relevance_score = max(bm25_score, tfidf_score, faiss_score, 0.0)
    narrow_application_score = keyword_score(text, NARROW_APPLICATION_TERMS)
    return QualitySignals(
        relevance_score=relevance_score,
        bm25_score=bm25_score,
        tfidf_score=tfidf_score,
        faiss_score=faiss_score,
        citation_score=normalized_citation_score(paper),
        influence_score=normalized_influence_score(paper),
        recency_score=recency_score(paper, current_year=current_year),
        difficulty_score=difficulty.score,
        difficulty_fit_score=difficulty_fit_score(difficulty, background_level),
        background_signal=keyword_score(text, BACKGROUND_TERMS),
        method_signal=keyword_score(text, METHOD_TERMS),
        narrow_application_score=narrow_application_score,
        duplicate_penalty=duplicate_penalty,
        too_narrow_penalty=too_narrow_penalty(query=query, paper=paper, narrow_score=narrow_application_score),
    )


def hybrid_score(signals: QualitySignals) -> float:
    return (
        0.48 * signals.relevance_score
        + 0.12 * signals.citation_score
        + 0.08 * signals.influence_score
        + 0.08 * signals.recency_score
        + 0.08 * signals.difficulty_fit_score
        + 0.07 * signals.background_signal
        + 0.07 * signals.method_signal
        - signals.duplicate_penalty
        - signals.too_narrow_penalty
    )


def section_score(section: str, signals: QualitySignals, *, query: str) -> float:
    if section == "background":
        return (
            0.40 * signals.relevance_score
            + 0.34 * signals.background_signal
            + 0.18 * signals.difficulty_fit_score
            + 0.06 * signals.citation_score
            - signals.duplicate_penalty
            - 1.2 * signals.too_narrow_penalty
        )
    if section == "foundational":
        return (
            0.28 * signals.relevance_score
            + 0.42 * signals.citation_score
            + 0.18 * signals.influence_score
            + 0.18 * (1.0 - signals.recency_score)
            - 0.35 * signals.narrow_application_score * generic_query_factor(query)
            - signals.duplicate_penalty
        )
    if section == "core_methods":
        return (
            0.42 * signals.relevance_score
            + 0.28 * signals.method_signal
            + 0.12 * signals.difficulty_fit_score
            + 0.10 * signals.citation_score
            - 0.60 * signals.too_narrow_penalty
            - signals.duplicate_penalty
        )
    if section == "recent_frontier":
        return (
            0.44 * signals.relevance_score
            + 0.30 * signals.recency_score
            + 0.10 * signals.method_signal
            + 0.08 * signals.citation_score
            - 0.18 * signals.background_signal
            - signals.too_narrow_penalty
            - signals.duplicate_penalty
        )
    raise ValueError(f"Unsupported reading path section: {section}")


def section_minimum_score(section: str) -> float:
    thresholds = {
        "background": 0.28,
        "foundational": 0.24,
        "core_methods": 0.26,
        "recent_frontier": 0.30,
    }
    return thresholds[section]


def section_reason(section: str) -> str:
    reasons = {
        "background": "strong beginner, survey, tutorial, book, or broad-overview fit",
        "foundational": "central older paper with strong citation or influence signal",
        "core_methods": "method, architecture, algorithm, benchmark, or technical contribution fit",
        "recent_frontier": "recent relevant non-survey work with enough topical fit",
    }
    return reasons[section]


def normalized_citation_score(paper: Paper) -> float:
    return min(citation_strength(paper) / 8.0, 1.0)


def normalized_influence_score(paper: Paper) -> float:
    influence = paper.influential_citation_count or 0
    return min(math.log1p(max(influence, 0)) / 6.0, 1.0)


def recency_score(paper: Paper, *, current_year: int) -> float:
    if paper.year is None:
        return 0.0
    age = max(current_year - paper.year, 0)
    if age <= 1:
        return 1.0
    if age <= 3:
        return 0.82
    if age <= 5:
        return 0.35
    if age <= 8:
        return 0.18
    return 0.05


def difficulty_fit_score(difficulty: DifficultyResult, background_level: str) -> float:
    if background_level in {"advanced"}:
        return 1.0
    if background_level == "intermediate":
        return 1.0 - max(difficulty.score - 0.75, 0.0)
    if difficulty.label == "beginner":
        return 1.0
    if difficulty.label == "intermediate":
        return 0.72
    return 0.28


def keyword_score(text: str, terms: set[str]) -> float:
    if not text:
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return min(hits / 2.0, 1.0)


def generic_query_factor(query: str) -> float:
    normalized = normalize_query(query)
    if normalized in BROAD_QUERY_TERMS:
        return 1.0
    if any(term == normalized for term in BROAD_QUERY_TERMS):
        return 1.0
    tokens = normalized.split()
    if len(tokens) <= 3 and any(term in normalized for term in BROAD_QUERY_TERMS):
        return 0.85
    return 0.0


def query_requests_domain(query: str) -> bool:
    normalized = normalize_query(query)
    return any(term in normalized for term in DOMAIN_QUERY_TERMS)


def too_narrow_penalty(*, query: str, paper: Paper, narrow_score: float) -> float:
    if narrow_score <= 0 or query_requests_domain(query):
        return 0.0
    if generic_query_factor(query) <= 0:
        return 0.0
    centrality = max(normalized_citation_score(paper), normalized_influence_score(paper))
    if centrality >= 0.78:
        return 0.0
    return 0.22 * narrow_score * generic_query_factor(query)


def normalize_query(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", query.lower()))
