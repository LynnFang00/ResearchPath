from datetime import datetime

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


VALID_RELATIONSHIP_TYPES = {
    "citation",
    "reference",
    "co_citation",
    "bibliographic_coupling",
}


class PaperBase(BaseModel):
    title: str = Field(..., min_length=1)
    abstract: str = Field(..., min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1800, le=2100)
    venue: str | None = None
    categories: list[str] = Field(default_factory=list)
    citation_count: int = Field(default=0, ge=0)
    source_url: str | None = None
    external_id: str | None = None
    source: str | None = None
    doi: str | None = None
    url: str | None = None
    references_count: int = Field(default=0, ge=0)
    influential_citation_count: int = Field(default=0, ge=0)
    abstract_word_count: int = Field(default=0, ge=0)
    updated_at: datetime | None = None
    references: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    @field_validator("authors", "categories", "references", "citations", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(";") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("Expected a list or semicolon-separated string.")

    @field_validator("external_id", "source", "doi", "url", "source_url", mode="before")
    @classmethod
    def blank_string_to_none(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return str(value)

    @field_validator("abstract_word_count", mode="before")
    @classmethod
    def coerce_optional_count(cls, value: object) -> int:
        if value in (None, ""):
            return 0
        return int(value)

    model_config = ConfigDict(extra="ignore")


class PaperCreate(PaperBase):
    pass


class PaperRead(PaperBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CitationEdgeCreate(BaseModel):
    source_paper_id: int = Field(..., ge=1)
    target_paper_id: int = Field(..., ge=1)
    relationship_type: str = "citation"
    source: str | None = None

    @field_validator("relationship_type")
    @classmethod
    def validate_relationship_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in VALID_RELATIONSHIP_TYPES:
            allowed = ", ".join(sorted(VALID_RELATIONSHIP_TYPES))
            raise ValueError(f"Unsupported relationship_type '{value}'. Use one of: {allowed}.")
        return normalized


class CitationEdgeRead(CitationEdgeCreate):
    id: int

    model_config = ConfigDict(from_attributes=True)


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to a local JSONL or CSV file.")


class IngestResponse(BaseModel):
    inserted: int
    skipped: int
    citation_edges_inserted: int = 0
    paper_count: int | None = None
    citation_edge_count: int | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    manifest_path: str | None = None


class RecommendationResponse(BaseModel):
    paper_id: int
    title: str
    abstract_snippet: str
    year: int | None
    authors: list[str]
    venue: str | None = None
    paper_url: str | None = None
    pdf_url: str | None = None
    doi_url: str | None = None
    source_url: str | None = None
    doi: str | None = None
    score: float
    method: str
    explanation: str


class PathPaper(RecommendationResponse):
    difficulty_label: str
    difficulty_score: float
    difficulty_explanation: str
    path_section: str
    path_reason: str
    relevance_score: float | None = None
    bm25_score: float | None = None
    tfidf_score: float | None = None
    faiss_score: float | None = None
    citation_score: float | None = None
    influence_score: float | None = None
    recency_score: float | None = None
    difficulty_fit_score: float | None = None
    background_signal: float | None = None
    method_signal: float | None = None
    narrow_application_score: float | None = None
    section_score: float | None = None
    duplicate_penalty: float | None = None
    too_narrow_penalty: float | None = None
    final_path_score: float | None = None
    why_recommended: str | None = None
    why_this_section: str | None = None
    confidence_label: str | None = None
    read_before: list[str] = Field(default_factory=list)
    read_after: list[str] = Field(default_factory=list)
    explanation_signals: list[str] = Field(default_factory=list)
    paper_type_tags: list[str] = Field(default_factory=list)
    personalization_score: float | None = None
    personalization_reason: str | None = None
    saved_similarity: float | None = None
    skipped_similarity: float | None = None
    too_hard_similarity: float | None = None
    topic_similarity: float | None = None
    learned_ranker_score: float | None = None
    learned_ranker_adjustment: float | None = None
    learned_ranker_version: str | None = None


class PathSectionStatus(BaseModel):
    section_complete: bool
    fill_reason: str | None = None


class ReadingPathResponse(BaseModel):
    query: str
    method: str
    background_level: str
    sections: dict[str, list[PathPaper]]
    section_status: dict[str, PathSectionStatus] = Field(default_factory=dict)


class DatasetStatusResponse(BaseModel):
    dataset_name: str
    source: str
    paper_count: int
    citation_edge_count: int
    last_updated_timestamp: str | None
    model_index_version: str
    embedding_model_name: str
    faiss_index_path: str
    manifest_path: str | None = None


FEEDBACK_ACTIONS = {
    "save",
    "already_read",
    "too_easy",
    "too_hard",
    "not_relevant",
    "more_like_this",
    "less_like_this",
}


class FeedbackCreate(BaseModel):
    paper_id: int = Field(..., ge=1)
    query: str = ""
    section: str | None = None
    action: str
    method: str = ""
    background_level: str = "basic_ml"
    tags: list[str] = Field(default_factory=list)

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in FEEDBACK_ACTIONS:
            allowed = ", ".join(sorted(FEEDBACK_ACTIONS))
            raise ValueError(f"Unsupported feedback action '{value}'. Use one of: {allowed}.")
        return normalized

    @field_validator("tags")
    @classmethod
    def normalize_feedback_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in value:
            clean = str(tag).strip().lower()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized


class FeedbackRead(FeedbackCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeedbackSummaryResponse(BaseModel):
    total_events: int
    action_counts: dict[str, int]
    saved_paper_ids: list[int]
    skipped_paper_ids: list[int]
    too_easy_paper_ids: list[int]
    too_hard_paper_ids: list[int]


class ProfileRead(BaseModel):
    background_level: str
    saved_paper_ids: list[int] = Field(default_factory=list)
    skipped_paper_ids: list[int] = Field(default_factory=list)
    too_easy_paper_ids: list[int] = Field(default_factory=list)
    too_hard_paper_ids: list[int] = Field(default_factory=list)
    preferred_topics: list[str] = Field(default_factory=list)
    avoid_topics: list[str] = Field(default_factory=list)
    current_status: str = "exploring"
    research_goal: str = "learn_topic"
    paper_taste: str = "balanced"
    updated_at: datetime | None = None


class ProfileUpdate(BaseModel):
    background_level: str | None = None
    saved_paper_ids: list[int] | None = None
    skipped_paper_ids: list[int] | None = None
    too_easy_paper_ids: list[int] | None = None
    too_hard_paper_ids: list[int] | None = None
    preferred_topics: list[str] | None = None
    avoid_topics: list[str] | None = None
    current_status: str | None = None
    research_goal: str | None = None
    paper_taste: str | None = None


class LibraryItemUpsert(BaseModel):
    paper_id: int = Field(..., ge=1)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in value:
            clean = str(tag).strip().lower()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized


class LibraryItemResponse(BaseModel):
    id: int
    paper_id: int
    tags: list[str]
    notes: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    paper: RecommendationResponse


class LibraryResponse(BaseModel):
    items: list[LibraryItemResponse]
    tags: list[str]
