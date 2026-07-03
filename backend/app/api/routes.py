from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.paper import Paper
from app.schemas.paper import (
    DatasetStatusResponse,
    FeedbackCreate,
    FeedbackRead,
    FeedbackSummaryResponse,
    IngestRequest,
    IngestResponse,
    LibraryItemResponse,
    LibraryItemUpsert,
    LibraryResponse,
    ProfileRead,
    ProfileUpdate,
    ReadingPathResponse,
    RecommendationResponse,
)
from app.services.dataset_manifest import get_dataset_status
from app.services.feedback import create_feedback_event, feedback_summary
from app.services.ingestion import ingest_papers_from_file
from app.services.library import delete_library_item, list_library_items, upsert_library_item
from app.services.profile import get_or_create_profile, profile_to_read, update_profile
from app.services.recommendation_service import (
    SUPPORTED_RETRIEVAL_METHODS,
    recommend_from_paper,
    recommend_from_query,
    recommend_reading_path,
)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/dataset/status", response_model=DatasetStatusResponse)
def dataset_status(db: Session = Depends(get_db)) -> DatasetStatusResponse:
    return DatasetStatusResponse(**get_dataset_status(db=db))


@router.post("/feedback", response_model=FeedbackRead)
def create_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)) -> FeedbackRead:
    paper = db.get(Paper, payload.paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper with id={payload.paper_id} was not found.")
    return FeedbackRead.model_validate(create_feedback_event(db, payload))


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def get_feedback_summary(db: Session = Depends(get_db)) -> FeedbackSummaryResponse:
    return feedback_summary(db)


@router.get("/profile", response_model=ProfileRead)
def get_profile(db: Session = Depends(get_db)) -> ProfileRead:
    return profile_to_read(get_or_create_profile(db))


@router.patch("/profile", response_model=ProfileRead)
def patch_profile(payload: ProfileUpdate, db: Session = Depends(get_db)) -> ProfileRead:
    return update_profile(db, payload)


@router.get("/library", response_model=LibraryResponse)
def get_library(tag: str | None = Query(None, description="Optional tag filter."), db: Session = Depends(get_db)) -> LibraryResponse:
    return list_library_items(db, tag=tag)


@router.post("/library/items", response_model=LibraryItemResponse)
def save_library_item(payload: LibraryItemUpsert, db: Session = Depends(get_db)) -> LibraryItemResponse:
    try:
        return upsert_library_item(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/library/items/{paper_id}")
def remove_library_item(paper_id: int, db: Session = Depends(get_db)) -> dict[str, str]:
    delete_library_item(db, paper_id)
    return {"status": "deleted"}


@router.post("/papers/ingest", response_model=IngestResponse)
def ingest_papers(payload: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:
    try:
        result = ingest_papers_from_file(Path(payload.file_path), db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(**result)


@router.get("/papers/search", response_model=list[RecommendationResponse])
def search_papers(
    query: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=50),
    method: str = Query("bm25", description="Retrieval method to use."),
    db: Session = Depends(get_db),
) -> list[RecommendationResponse]:
    try:
        return recommend_from_query(db=db, query=query, k=k, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/recommend/query", response_model=list[RecommendationResponse])
def recommend_query(
    query: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=50),
    method: str = Query("bm25", description="Retrieval method to use."),
    db: Session = Depends(get_db),
) -> list[RecommendationResponse]:
    try:
        return recommend_from_query(db=db, query=query, k=k, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/recommend/paper/{paper_id}", response_model=list[RecommendationResponse])
def recommend_paper(
    paper_id: int,
    k: int = Query(10, ge=1, le=50),
    method: str = Query("bm25", description="Retrieval method to use."),
    db: Session = Depends(get_db),
) -> list[RecommendationResponse]:
    paper = db.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper with id={paper_id} was not found.")
    try:
        return recommend_from_paper(db=db, paper=paper, k=k, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/recommend/methods")
def recommend_methods() -> dict[str, list[str]]:
    return {"methods": sorted(SUPPORTED_RETRIEVAL_METHODS)}


@router.get("/path/query", response_model=ReadingPathResponse)
def reading_path_query(
    query: str = Query(..., min_length=1),
    k: int = Query(4, ge=1, le=10, description="Papers per reading-path section."),
    method: str = Query("bm25", description="First-stage retrieval method to use."),
    background_level: str = Query("basic_ml", description="User background level."),
    candidate_k: int = Query(40, ge=10, le=200, description="Candidate pool before path grouping."),
    db: Session = Depends(get_db),
) -> ReadingPathResponse:
    try:
        return recommend_reading_path(
            db=db,
            query=query,
            k=k,
            method=method,
            background_level=background_level,
            candidate_k=candidate_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
