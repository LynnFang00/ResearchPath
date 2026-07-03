from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.models.paper import Paper
from app.services.difficulty import DifficultyResult


@dataclass(frozen=True)
class PersonalizationResult:
    score: float
    reason: str
    saved_similarity: float
    skipped_similarity: float
    too_hard_similarity: float
    topic_similarity: float

    def diagnostics(self) -> dict[str, float]:
        return {
            "personalization_score": round(self.score, 6),
            "saved_similarity": round(self.saved_similarity, 6),
            "skipped_similarity": round(self.skipped_similarity, 6),
            "too_hard_similarity": round(self.too_hard_similarity, 6),
            "topic_similarity": round(self.topic_similarity, 6),
        }


def score_personalization(
    *,
    paper: Paper,
    paper_by_id: dict[int, Paper],
    profile: dict[str, Any] | None,
    difficulty: DifficultyResult,
    background_level: str,
) -> PersonalizationResult:
    if not profile:
        return PersonalizationResult(0.0, "no profile signals yet", 0.0, 0.0, 0.0, 0.0)

    saved_ids = set(profile.get("saved_paper_ids", []))
    skipped_ids = set(profile.get("skipped_paper_ids", []))
    too_easy_ids = set(profile.get("too_easy_paper_ids", []))
    too_hard_ids = set(profile.get("too_hard_paper_ids", []))
    preferred_topics = [str(topic) for topic in profile.get("preferred_topics", [])]
    avoid_topics = [str(topic) for topic in profile.get("avoid_topics", [])]
    research_goal = str(profile.get("research_goal", "learn_topic") or "learn_topic")
    paper_taste = str(profile.get("paper_taste", "balanced") or "balanced")
    saved_similarity = max_similarity(paper, [paper_by_id[item] for item in saved_ids if item in paper_by_id])
    skipped_similarity = max_similarity(paper, [paper_by_id[item] for item in skipped_ids if item in paper_by_id])
    too_hard_similarity = max_similarity(paper, [paper_by_id[item] for item in too_hard_ids if item in paper_by_id])
    topic_similarity = topic_match_score(paper, preferred_topics)
    avoid_similarity = topic_match_score(paper, avoid_topics)

    score = 0.0
    reasons: list[str] = []
    if paper.id in saved_ids:
        score += 0.10
        reasons.append("paper was saved")
    if saved_similarity > 0:
        score += 0.12 * saved_similarity
        reasons.append("similar to saved papers")
    if topic_similarity > 0:
        score += 0.10 * topic_similarity
        reasons.append("matches preferred topics")
    if avoid_similarity > 0:
        score -= 0.16 * avoid_similarity
        reasons.append("matches avoided topics")
    if paper.id in skipped_ids:
        score -= 0.30
        reasons.append("paper was marked not useful")
    if skipped_similarity > 0:
        score -= 0.12 * skipped_similarity
        reasons.append("similar to skipped papers")
    if background_level in {"basic_ml", "beginner"} and too_hard_similarity > 0:
        score -= 0.14 * too_hard_similarity
        reasons.append("similar to papers marked too hard")

    difficulty_shift = len(too_easy_ids) - len(too_hard_ids)
    if difficulty_shift >= 2 and difficulty.label in {"intermediate", "advanced"}:
        score += 0.04
        reasons.append("profile is ready for harder papers")
    elif difficulty_shift <= -2 and difficulty.label == "beginner":
        score += 0.04
        reasons.append("profile prefers easier papers")
    elif difficulty_shift <= -2 and difficulty.label == "advanced":
        score -= 0.08
        reasons.append("profile has marked harder papers as too hard")

    taste_score, taste_reason = score_paper_taste(paper, paper_taste, research_goal)
    if taste_score:
        score += taste_score
        reasons.append(taste_reason)

    return PersonalizationResult(
        score=max(min(score, 0.25), -0.45),
        reason=", ".join(reasons) if reasons else "no matching profile signal",
        saved_similarity=saved_similarity,
        skipped_similarity=skipped_similarity,
        too_hard_similarity=too_hard_similarity,
        topic_similarity=topic_similarity,
    )


def max_similarity(paper: Paper, others: list[Paper]) -> float:
    if not others:
        return 0.0
    paper_tokens = tokens_for(paper)
    if not paper_tokens:
        return 0.0
    scores = []
    for other in others:
        other_tokens = tokens_for(other)
        if not other_tokens:
            continue
        scores.append(len(paper_tokens & other_tokens) / len(paper_tokens | other_tokens))
    return max(scores, default=0.0)


def tokens_for(paper: Paper) -> set[str]:
    text = f"{paper.title} {paper.abstract}".lower()
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2}


def topic_match_score(paper: Paper, topics: list[str]) -> float:
    if not topics:
        return 0.0
    paper_tokens = tokens_for(paper)
    if not paper_tokens:
        return 0.0
    topic_scores: list[float] = []
    for topic in topics:
        topic_tokens = {token for token in re.findall(r"[a-z0-9]+", topic.lower()) if len(token) > 2}
        if not topic_tokens:
            continue
        topic_scores.append(len(topic_tokens & paper_tokens) / len(topic_tokens))
    return max(topic_scores, default=0.0)


def score_paper_taste(paper: Paper, paper_taste: str, research_goal: str) -> tuple[float, str]:
    text = f"{paper.title} {paper.abstract}".lower()
    citation_count = int(paper.citation_count or 0)
    year = int(paper.year or 0)
    survey_like = any(term in text for term in ("survey", "review", "tutorial", "overview"))
    method_like = any(term in text for term in ("method", "algorithm", "architecture", "benchmark", "framework"))
    recent_like = year >= 2022

    if paper_taste == "surveys_first" and survey_like:
        return 0.07, "profile prefers surveys first"
    if paper_taste == "foundational_first" and citation_count >= 500:
        return 0.06, "profile prefers foundational papers"
    if paper_taste == "recent_first" and recent_like:
        return 0.06, "profile prefers recent work"
    if paper_taste == "implementation_first" and method_like:
        return 0.05, "profile prefers implementation-oriented papers"
    if research_goal == "literature_review" and survey_like:
        return 0.05, "research goal favors survey papers"
    if research_goal in {"start_project", "find_baseline", "implement_method"} and method_like:
        return 0.04, "research goal favors method papers"
    return 0.0, ""
