from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.paper import Paper
from app.schemas.paper import (
    DatasetStatusResponse,
    IngestRequest,
    IngestResponse,
    ReadingPathResponse,
    RecommendationResponse,
)
from app.services.dataset_manifest import get_dataset_status
from app.services.ingestion import ingest_papers_from_file
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
