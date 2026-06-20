from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.paper import Paper
from app.schemas.paper import ReadingPathResponse, RecommendationResponse
from app.services.bm25 import BM25Retriever
from app.services.formatting import paper_to_recommendation
from app.services.reading_path import PathCandidate, plan_reading_path
from app.services.retrievers import PaperDocument, Retriever, papers_to_documents
from app.services.retrievers.citation_recency import CitationRecencyRetriever
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.faiss import FaissRetriever
from app.services.retrievers.hybrid import HybridRetriever
from app.services.retrievers.tfidf import TfidfRetriever


REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_RETRIEVAL_METHODS = {
    "bm25",
    "tfidf",
    "citation_recency",
    "embedding",
    "faiss_embedding",
    "hybrid",
}


def _load_papers(db: Session) -> list[Paper]:
    return list(db.scalars(select(Paper).order_by(Paper.id)).all())


def _build_bm25(papers: list[Paper]) -> BM25Retriever:
    return BM25Retriever(papers_to_documents(papers))


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_method(method: str) -> str:
    normalized_method = method.strip().lower()
    if normalized_method not in SUPPORTED_RETRIEVAL_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_RETRIEVAL_METHODS))
        raise ValueError(f"Unsupported recommendation method '{method}'. Use one of: {allowed}.")
    return normalized_method


def build_retriever(method: str, documents: list[PaperDocument]) -> Retriever:
    normalized_method = normalize_method(method)

    if normalized_method == "bm25":
        return BM25Retriever(documents)
    if normalized_method == "tfidf":
        return TfidfRetriever(documents)
    if normalized_method == "citation_recency":
        return CitationRecencyRetriever(documents)
    if normalized_method == "hybrid":
        raise ValueError("Hybrid retrieval requires full paper metadata. Use build_hybrid_retriever.")

    settings = get_settings()
    if normalized_method == "embedding":
        index_path = _resolve_repo_path(settings.embedding_index_path)
        if not index_path.exists():
            raise ValueError(f"Embedding index was not found: {index_path}")
        retriever = EmbeddingRetriever(model_name=settings.embedding_model_name)
        retriever.load_index(index_path)
        return retriever

    faiss_index_path = _resolve_repo_path(settings.faiss_index_path)
    faiss_id_map_path = _resolve_repo_path(settings.faiss_id_map_path)
    if not faiss_index_path.exists():
        raise ValueError(f"FAISS index was not found: {faiss_index_path}")
    if not faiss_id_map_path.exists():
        raise ValueError(f"FAISS id map was not found: {faiss_id_map_path}")
    retriever = FaissRetriever(model_name=settings.embedding_model_name)
    retriever.load_index(faiss_index_path, faiss_id_map_path)
    return retriever


def build_hybrid_retriever(
    papers: list[Paper],
    *,
    background_level: str = "basic_ml",
) -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        papers,
        faiss_index_path=_resolve_repo_path(settings.faiss_index_path),
        faiss_id_map_path=_resolve_repo_path(settings.faiss_id_map_path),
        embedding_model_name=settings.embedding_model_name,
        background_level=background_level,
    )


def _explanation_for_method(method: str) -> str:
    if method == "bm25":
        return "matched important keywords in the title and abstract"
    if method == "tfidf":
        return "matched important TF-IDF terms in the title and abstract"
    if method == "citation_recency":
        return "matched query text with a boost for citation count and recency"
    if method == "embedding":
        return "high semantic similarity from frozen transformer embeddings"
    if method == "faiss_embedding":
        return "high semantic similarity from FAISS vector search"
    if method == "hybrid":
        return "combined lexical, semantic, citation, recency, difficulty, and reading-path quality signals"
    return "ranked by the configured retrieval method"


def recommend_from_query(
    db: Session,
    query: str,
    k: int = 10,
    method: str = "bm25",
) -> list[RecommendationResponse]:
    method = normalize_method(method)
    papers = _load_papers(db)
    paper_by_id = {paper.id: paper for paper in papers}
    retriever = (
        build_hybrid_retriever(papers)
        if method == "hybrid"
        else build_retriever(method, papers_to_documents(papers))
    )
    results = retriever.search(query=query, k=k)
    explanation = _explanation_for_method(method)

    return [
        paper_to_recommendation(
            paper=paper_by_id[result.document_id],
            score=result.score,
            method=method,
            explanation=explanation,
        )
        for result in results
        if result.document_id in paper_by_id
    ]


def recommend_from_paper(
    db: Session,
    paper: Paper,
    k: int = 10,
    method: str = "bm25",
) -> list[RecommendationResponse]:
    method = normalize_method(method)
    papers = _load_papers(db)
    paper_by_id = {candidate.id: candidate for candidate in papers}
    retriever = (
        build_hybrid_retriever(papers)
        if method == "hybrid"
        else build_retriever(method, papers_to_documents(papers))
    )
    results = retriever.search(query=paper.searchable_text, k=k, exclude_ids={paper.id})
    explanation = _explanation_for_method(method)

    return [
        paper_to_recommendation(
            paper=paper_by_id[result.document_id],
            score=result.score,
            method=method,
            explanation=explanation,
        )
        for result in results
        if result.document_id in paper_by_id
    ]


def recommend_reading_path(
    db: Session,
    query: str,
    k: int = 4,
    method: str = "bm25",
    background_level: str = "basic_ml",
    candidate_k: int = 40,
) -> ReadingPathResponse:
    method = normalize_method(method)
    papers = _load_papers(db)
    paper_by_id = {paper.id: paper for paper in papers}
    retriever = (
        build_hybrid_retriever(papers, background_level=background_level)
        if method == "hybrid"
        else build_retriever(method, papers_to_documents(papers))
    )
    search_k = max(candidate_k, k * 8)
    results = retriever.search(query=query, k=search_k)
    explanation = _explanation_for_method(method)
    candidates = [
        PathCandidate(
            paper=paper_by_id[result.document_id],
            score=result.score,
            method=method,
            explanation=explanation,
            retrieval_components=(
                retriever.components_for(result.document_id)
                if method == "hybrid" and isinstance(retriever, HybridRetriever)
                else {}
            ),
        )
        for result in results
        if result.document_id in paper_by_id
    ]
    return plan_reading_path(
        query=query,
        candidates=candidates,
        method=method,
        background_level=background_level,
        papers_per_section=k,
    )
