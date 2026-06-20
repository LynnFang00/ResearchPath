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
