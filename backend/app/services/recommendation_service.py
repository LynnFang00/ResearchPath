from pathlib import Path
import json

from sqlalchemy import func, select
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
from app.services.retrievers.learned_blend_v2_7 import LearnedBlendV27Retriever
from app.services.retrievers.v3_3_ltr import V33LTRRetriever
from app.services.retrievers.v4_1_blend import V41BlendRetriever
from app.services.retrievers.v4_9_guarded_text_blend import V49GuardedTextBlendRetriever
from app.services.retrievers.v6_4_safe_fusion import V64SafeFusionRetriever
from app.services.retrievers.tfidf import TfidfRetriever
from app.services.profile import profile_as_dict


REPO_ROOT = Path(__file__).resolve().parents[3]
_CORPUS_CACHE: dict[str, object] = {}
_RETRIEVER_CACHE: dict[tuple[object, ...], Retriever] = {}
SUPPORTED_RETRIEVAL_METHODS = {
    "bm25",
    "tfidf",
    "citation_recency",
    "embedding",
    "faiss_embedding",
    "hybrid",
    "learned_hybrid",
    "learned_blend_v2_7",
    "v3_3_ltr",
    "v4_1_blend",
    "v4_9_guarded_text_blend",
    "v6_4_safe_fusion",
}


def _corpus_signature(db: Session) -> tuple[int, int, str]:
    count, max_id, max_updated_at = db.execute(
        select(func.count(Paper.id), func.coalesce(func.max(Paper.id), 0), func.max(Paper.updated_at))
    ).one()
    return int(count or 0), int(max_id or 0), str(max_updated_at or "")


def _load_papers(db: Session) -> list[Paper]:
    return list(db.scalars(select(Paper).order_by(Paper.id)).all())


def _load_corpus(db: Session) -> tuple[tuple[int, int, str], list[Paper], dict[int, Paper], list[PaperDocument]]:
    signature = _corpus_signature(db)
    if _CORPUS_CACHE.get("signature") != signature:
        papers = _load_papers(db)
        _CORPUS_CACHE.clear()
        _RETRIEVER_CACHE.clear()
        _CORPUS_CACHE.update(
            {
                "signature": signature,
                "papers": papers,
                "paper_by_id": {paper.id: paper for paper in papers},
                "documents": papers_to_documents(papers),
            }
        )
    return (
        signature,
        _CORPUS_CACHE["papers"],  # type: ignore[return-value]
        _CORPUS_CACHE["paper_by_id"],  # type: ignore[return-value]
        _CORPUS_CACHE["documents"],  # type: ignore[return-value]
    )


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
    if normalized_method == "learned_blend_v2_7":
        raise ValueError("V2.7 learned blend retrieval requires full paper metadata. Use build_learned_blend_v2_7_retriever.")
    if normalized_method == "v3_3_ltr":
        raise ValueError("V3.3 LTR retrieval requires full paper metadata. Use build_v3_3_ltr_retriever.")
    if normalized_method == "v4_1_blend":
        raise ValueError("V4.1 blend retrieval requires full paper metadata. Use build_v4_1_blend_retriever.")
    if normalized_method == "v4_9_guarded_text_blend":
        raise ValueError("V4.9 guarded text blend retrieval requires full paper metadata. Use build_v4_9_guarded_text_blend_retriever.")
    if normalized_method == "v6_4_safe_fusion":
        raise ValueError("V6.4 safe fusion retrieval requires full paper metadata. Use build_v6_4_safe_fusion_retriever.")

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
    profile: dict | None = None,
    use_learned_ranker: bool = False,
) -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        papers,
        faiss_index_path=_resolve_repo_path(settings.faiss_index_path),
        faiss_id_map_path=_resolve_repo_path(settings.faiss_id_map_path),
        embedding_model_name=settings.embedding_model_name,
        background_level=background_level,
        profile=profile,
        learned_ranker_path=_resolve_repo_path(settings.learned_ranker_path) if use_learned_ranker else None,
    )


def build_learned_blend_v2_7_retriever(papers: list[Paper]) -> LearnedBlendV27Retriever:
    settings = get_settings()
    return LearnedBlendV27Retriever(
        papers,
        embedding_index_path=_resolve_repo_path(settings.embedding_index_path),
        faiss_index_path=_resolve_repo_path(settings.faiss_index_path),
        faiss_id_map_path=_resolve_repo_path(settings.faiss_id_map_path),
        old_model_path=_resolve_repo_path(settings.learned_ranker_v2_2b_path),
        v26_model_path=_resolve_repo_path(settings.learned_ranker_v2_6_path),
        blend_artifact_path=_resolve_repo_path(settings.learned_blend_v2_7_path),
        embedding_model_name=settings.embedding_model_name,
    )


def build_v3_3_ltr_retriever(papers: list[Paper]) -> V33LTRRetriever:
    settings = get_settings()
    return V33LTRRetriever(
        papers,
        embedding_index_path=_resolve_repo_path(settings.embedding_index_path),
        faiss_index_path=_resolve_repo_path(settings.faiss_index_path),
        faiss_id_map_path=_resolve_repo_path(settings.faiss_id_map_path),
        old_model_path=_resolve_repo_path(settings.learned_ranker_v2_2b_path),
        v26_model_path=_resolve_repo_path(settings.learned_ranker_v2_6_path),
        v33_model_dir=_resolve_repo_path(settings.learned_ranker_v3_3_dir),
        embedding_model_name=settings.embedding_model_name,
    )


def build_v4_1_blend_retriever(papers: list[Paper]) -> V41BlendRetriever:
    settings = get_settings()
    return V41BlendRetriever(
        papers,
        embedding_index_path=_resolve_repo_path(settings.embedding_index_path),
        faiss_index_path=_resolve_repo_path(settings.faiss_index_path),
        faiss_id_map_path=_resolve_repo_path(settings.faiss_id_map_path),
        old_model_path=_resolve_repo_path(settings.learned_ranker_v2_2b_path),
        v26_model_path=_resolve_repo_path(settings.learned_ranker_v2_6_path),
        v33_model_dir=_resolve_repo_path(settings.learned_ranker_v3_3_dir),
        v41_model_dir=_resolve_repo_path(settings.learned_ranker_v4_1_dir),
        blend_config_path=_resolve_repo_path(settings.learned_ranker_v4_1_blend_config_path),
        embedding_model_name=settings.embedding_model_name,
    )


def build_v4_9_guarded_text_blend_retriever(papers: list[Paper]) -> V49GuardedTextBlendRetriever:
    settings = get_settings()
    return V49GuardedTextBlendRetriever(
        papers,
        embedding_index_path=_resolve_repo_path(settings.embedding_index_path),
        faiss_index_path=_resolve_repo_path(settings.faiss_index_path),
        faiss_id_map_path=_resolve_repo_path(settings.faiss_id_map_path),
        old_model_path=_resolve_repo_path(settings.learned_ranker_v2_2b_path),
        v26_model_path=_resolve_repo_path(settings.learned_ranker_v2_6_path),
        v33_model_dir=_resolve_repo_path(settings.learned_ranker_v3_3_dir),
        v41_model_dir=_resolve_repo_path(settings.learned_ranker_v4_1_dir),
        v43_model_dir=_resolve_repo_path(settings.learned_ranker_v4_3_text_dir),
        v41_blend_config_path=_resolve_repo_path(settings.learned_ranker_v4_1_blend_config_path),
        v49_candidate_config_path=_resolve_repo_path(settings.learned_ranker_v4_9_guarded_text_config_path),
        embedding_model_name=settings.embedding_model_name,
    )


def build_v6_4_safe_fusion_retriever(papers: list[Paper]) -> V64SafeFusionRetriever:
    settings = get_settings()
    return V64SafeFusionRetriever(
        papers,
        embedding_index_path=_resolve_repo_path(settings.embedding_index_path),
        faiss_index_path=_resolve_repo_path(settings.faiss_index_path),
        faiss_id_map_path=_resolve_repo_path(settings.faiss_id_map_path),
        old_model_path=_resolve_repo_path(settings.learned_ranker_v2_2b_path),
        v26_model_path=_resolve_repo_path(settings.learned_ranker_v2_6_path),
        v33_model_dir=_resolve_repo_path(settings.learned_ranker_v3_3_dir),
        v41_model_dir=_resolve_repo_path(settings.learned_ranker_v4_1_dir),
        v43_model_dir=_resolve_repo_path(settings.learned_ranker_v4_3_text_dir),
        v41_blend_config_path=_resolve_repo_path(settings.learned_ranker_v4_1_blend_config_path),
        v49_candidate_config_path=_resolve_repo_path(settings.learned_ranker_v4_9_guarded_text_config_path),
        v64_candidate_config_path=_resolve_repo_path(settings.learned_ranker_v6_4_safe_fusion_config_path),
        v66_ridge_scorer_config_path=_resolve_repo_path(settings.learned_ranker_v6_6_safe_fusion_ridge_scorer_path),
        embedding_model_name=settings.embedding_model_name,
    )


def _profile_cache_key(profile: dict | None) -> str:
    if not profile:
        return "{}"
    return json.dumps(profile, sort_keys=True, default=str)


def get_cached_retriever(
    *,
    method: str,
    papers: list[Paper],
    documents: list[PaperDocument],
    corpus_signature: tuple[int, int, str],
    profile: dict | None = None,
) -> Retriever:
    settings = get_settings()
    key = (
        method,
        corpus_signature,
        settings.embedding_model_name,
        settings.embedding_index_path,
        settings.faiss_index_path,
        settings.faiss_id_map_path,
        settings.learned_ranker_path if method == "learned_hybrid" else None,
        settings.learned_ranker_v2_2b_path if method == "learned_blend_v2_7" else None,
        settings.learned_ranker_v2_6_path if method == "learned_blend_v2_7" else None,
        settings.learned_blend_v2_7_path if method == "learned_blend_v2_7" else None,
        settings.learned_ranker_v2_2b_path if method == "v3_3_ltr" else None,
        settings.learned_ranker_v2_6_path if method == "v3_3_ltr" else None,
        settings.learned_ranker_v3_3_dir if method == "v3_3_ltr" else None,
        settings.learned_ranker_v2_2b_path if method == "v4_1_blend" else None,
        settings.learned_ranker_v2_6_path if method == "v4_1_blend" else None,
        settings.learned_ranker_v3_3_dir if method == "v4_1_blend" else None,
        settings.learned_ranker_v4_1_dir if method == "v4_1_blend" else None,
        settings.learned_ranker_v4_1_blend_config_path if method == "v4_1_blend" else None,
        settings.learned_ranker_v2_2b_path if method in {"v4_9_guarded_text_blend", "v6_4_safe_fusion"} else None,
        settings.learned_ranker_v2_6_path if method in {"v4_9_guarded_text_blend", "v6_4_safe_fusion"} else None,
        settings.learned_ranker_v3_3_dir if method in {"v4_9_guarded_text_blend", "v6_4_safe_fusion"} else None,
        settings.learned_ranker_v4_1_dir if method in {"v4_9_guarded_text_blend", "v6_4_safe_fusion"} else None,
        settings.learned_ranker_v4_3_text_dir if method in {"v4_9_guarded_text_blend", "v6_4_safe_fusion"} else None,
        settings.learned_ranker_v4_1_blend_config_path if method in {"v4_9_guarded_text_blend", "v6_4_safe_fusion"} else None,
        settings.learned_ranker_v4_9_guarded_text_config_path if method in {"v4_9_guarded_text_blend", "v6_4_safe_fusion"} else None,
        settings.learned_ranker_v6_4_safe_fusion_config_path if method == "v6_4_safe_fusion" else None,
        settings.learned_ranker_v6_6_safe_fusion_ridge_scorer_path if method == "v6_4_safe_fusion" else None,
        _profile_cache_key(profile) if method in {"hybrid", "learned_hybrid"} else "{}",
    )
    if key not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[key] = (
            build_hybrid_retriever(papers, profile=profile, use_learned_ranker=method == "learned_hybrid")
            if method in {"hybrid", "learned_hybrid"}
            else build_learned_blend_v2_7_retriever(papers)
            if method == "learned_blend_v2_7"
            else build_v3_3_ltr_retriever(papers)
            if method == "v3_3_ltr"
            else build_v4_1_blend_retriever(papers)
            if method == "v4_1_blend"
            else build_v4_9_guarded_text_blend_retriever(papers)
            if method == "v4_9_guarded_text_blend"
            else build_v6_4_safe_fusion_retriever(papers)
            if method == "v6_4_safe_fusion"
            else build_retriever(method, documents)
        )
    return _RETRIEVER_CACHE[key]


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
    if method == "learned_hybrid":
        return "combined hybrid retrieval with a lightweight learned reranking adjustment from manual labels"
    if method == "learned_blend_v2_7":
        return "opt-in V2.7 blend of the original learned ranker, production-aware learned ranker, hybrid score, and dense embedding score"
    if method == "v3_3_ltr":
        return "opt-in V3.3 frozen RandomForest LTR reranking over the shared production-style candidate pool"
    if method == "v4_1_blend":
        return "opt-in V4.1 offline calibrated blend of V3.3 relevance ranking with a V4.1 hard-negative guardrail signal"
    if method == "v4_9_guarded_text_blend":
        return "opt-in V4.9 guarded text blend of V3.3, V4.1, and gated V4.3 text scores"
    if method == "v6_4_safe_fusion":
        return "opt-in V6.4 safe fusion that applies a top-10 swap-limited neural fusion adjustment to V4.9"
    return "ranked by the configured retrieval method"


def recommend_from_query(
    db: Session,
    query: str,
    k: int = 10,
    method: str = "bm25",
) -> list[RecommendationResponse]:
    method = normalize_method(method)
    signature, papers, paper_by_id, documents = _load_corpus(db)
    profile = profile_as_dict(db) if method in {"hybrid", "learned_hybrid"} else None
    retriever = get_cached_retriever(
        method=method,
        papers=papers,
        documents=documents,
        corpus_signature=signature,
        profile=profile,
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
    signature, papers, paper_by_id, documents = _load_corpus(db)
    profile = profile_as_dict(db) if method in {"hybrid", "learned_hybrid"} else None
    retriever = get_cached_retriever(
        method=method,
        papers=papers,
        documents=documents,
        corpus_signature=signature,
        profile=profile,
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
        build_hybrid_retriever(
            papers,
            background_level=background_level,
            profile=profile_as_dict(db),
            use_learned_ranker=method == "learned_hybrid",
        )
        if method in {"hybrid", "learned_hybrid"}
        else build_learned_blend_v2_7_retriever(papers)
        if method == "learned_blend_v2_7"
        else build_v3_3_ltr_retriever(papers)
        if method == "v3_3_ltr"
        else build_v4_1_blend_retriever(papers)
        if method == "v4_1_blend"
        else build_v4_9_guarded_text_blend_retriever(papers)
        if method == "v4_9_guarded_text_blend"
        else build_v6_4_safe_fusion_retriever(papers)
        if method == "v6_4_safe_fusion"
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
                if method in {"hybrid", "learned_hybrid"} and isinstance(retriever, HybridRetriever)
                else retriever.components_for(result.document_id)
                if method == "learned_blend_v2_7" and isinstance(retriever, LearnedBlendV27Retriever)
                else retriever.components_for(result.document_id)
                if method == "v3_3_ltr" and isinstance(retriever, V33LTRRetriever)
                else retriever.components_for(result.document_id)
                if method == "v4_1_blend" and isinstance(retriever, V41BlendRetriever)
                else retriever.components_for(result.document_id)
                if method == "v4_9_guarded_text_blend" and isinstance(retriever, V49GuardedTextBlendRetriever)
                else retriever.components_for(result.document_id)
                if method == "v6_4_safe_fusion" and isinstance(retriever, V64SafeFusionRetriever)
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
