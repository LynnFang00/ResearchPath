from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Protocol


NEAR_DUPLICATE_TITLE_THRESHOLD = 0.92


class PaperLike(Protocol):
    title: str
    abstract: str
    authors: object
    year: int | None
    venue: str | None
    categories: object
    citation_count: int
    source_url: str | None
    external_id: str | None
    source: str | None
    doi: str | None
    url: str | None
    references_count: int
    influential_citation_count: int
    abstract_word_count: int


@dataclass(frozen=True)
class DuplicateMatch:
    kind: str
    key: str
    similarity: float = 1.0


def normalize_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def normalize_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def title_similarity(left: str, right: str) -> float:
    left_normalized = normalize_title(left)
    right_normalized = normalize_title(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    sequence_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    return max(sequence_score, token_jaccard(left, right))


def is_near_duplicate_title(left: str, right: str, *, threshold: float = NEAR_DUPLICATE_TITLE_THRESHOLD) -> bool:
    left_normalized = normalize_title(left)
    right_normalized = normalize_title(right)
    if len(left_normalized) < 20 or len(right_normalized) < 20:
        return left_normalized == right_normalized
    return title_similarity(left, right) >= threshold


def lookup_keys_for_record(record: PaperLike) -> list[str]:
    keys: list[str] = []
    external_id = normalize_identifier(record.external_id)
    doi = normalize_identifier(record.doi)
    if external_id:
        keys.append(f"external_id:{external_id}")
    if doi:
        keys.append(f"doi:{doi}")
    title_key = normalize_title(record.title)
    if title_key:
        keys.append(f"title:{title_key}")
    return keys


def completeness_score(record: PaperLike) -> float:
    score = 0.0
    for field_name in (
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "categories",
        "citation_count",
        "source_url",
        "external_id",
        "source",
        "doi",
        "url",
        "references_count",
        "influential_citation_count",
        "abstract_word_count",
    ):
        value = getattr(record, field_name, None)
        if value in (None, "", [], 0):
            continue
        score += 1.0
    score += min(len(record.abstract or "") / 1000, 1.0)
    score += min(max(record.citation_count or 0, 0) / 1000, 1.0)
    return score


def duplicate_match(left: PaperLike, right: PaperLike) -> DuplicateMatch | None:
    left_external_id = normalize_identifier(left.external_id)
    right_external_id = normalize_identifier(right.external_id)
    if left_external_id and right_external_id and left_external_id == right_external_id:
        return DuplicateMatch(kind="external_id", key=left_external_id)

    left_doi = normalize_identifier(left.doi)
    right_doi = normalize_identifier(right.doi)
    if left_doi and right_doi and left_doi == right_doi:
        return DuplicateMatch(kind="doi", key=left_doi)

    left_title = normalize_title(left.title)
    right_title = normalize_title(right.title)
    if left_title and left_title == right_title:
        return DuplicateMatch(kind="title", key=left_title)

    similarity = title_similarity(left.title, right.title)
    if is_near_duplicate_title(left.title, right.title):
        return DuplicateMatch(kind="near_title", key=left_title or right_title, similarity=similarity)
    return None
