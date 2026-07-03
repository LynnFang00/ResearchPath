from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.paper import Paper
from app.services.ltr_model_service import (
    V49_METHOD_NAME,
    V49GuardedTextBlendModel,
    V64_METHOD_NAME,
    V64SafeFusionModel,
    V66SafeFusionRidgeScorer,
    load_v6_4_safe_fusion_model,
    load_v6_6_safe_fusion_ridge_scorer,
)
from app.services.retrievers.base import ScoredDocument
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.faiss import FaissRetriever
from app.services.retrievers.v4_9_guarded_text_blend import V49GuardedTextBlendRetriever
from app.services.v2_2_learned_ranker import V22RidgeRanker


@dataclass(frozen=True)
class V64SafeFusionScoredDocument(ScoredDocument):
    components: dict[str, Any]


class V64SafeFusionRetriever(V49GuardedTextBlendRetriever):
    method_name = V64_METHOD_NAME

    def __init__(
        self,
        papers: list[Paper],
        *,
        v33_model_dir: Path | None = None,
        v41_model_dir: Path | None = None,
        v43_model_dir: Path | None = None,
        v41_blend_config_path: Path | None = None,
        v49_candidate_config_path: Path | None = None,
        v64_candidate_config_path: Path | None = None,
        v66_ridge_scorer_config_path: Path | None = None,
        guarded_model: V49GuardedTextBlendModel | None = None,
        safe_fusion_model: V64SafeFusionModel | None = None,
        ridge_scorer: V66SafeFusionRidgeScorer | None = None,
        embedding_index_path: Path | None = None,
        faiss_index_path: Path | None = None,
        faiss_id_map_path: Path | None = None,
        old_model_path: Path | None = None,
        v26_model_path: Path | None = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        exact_embedding_retriever: EmbeddingRetriever | None = None,
        faiss_retriever: FaissRetriever | None = None,
        old_ranker: V22RidgeRanker | None = None,
        v26_ranker: V22RidgeRanker | None = None,
        current_year: int = 2026,
    ) -> None:
        if safe_fusion_model is None:
            if v64_candidate_config_path is None:
                raise ValueError("V6.4 safe fusion candidate config path is required.")
            safe_fusion_model = load_v6_4_safe_fusion_model(v64_candidate_config_path)
        if ridge_scorer is None and v66_ridge_scorer_config_path is not None and v66_ridge_scorer_config_path.exists():
            ridge_scorer = load_v6_6_safe_fusion_ridge_scorer(v66_ridge_scorer_config_path)
        super().__init__(
            papers,
            v33_model_dir=v33_model_dir,
            v41_model_dir=v41_model_dir,
            v43_model_dir=v43_model_dir,
            v41_blend_config_path=v41_blend_config_path,
            v49_candidate_config_path=v49_candidate_config_path,
            guarded_model=guarded_model,
            embedding_index_path=embedding_index_path,
            faiss_index_path=faiss_index_path,
            faiss_id_map_path=faiss_id_map_path,
            old_model_path=old_model_path,
            v26_model_path=v26_model_path,
            embedding_model_name=embedding_model_name,
            exact_embedding_retriever=exact_embedding_retriever,
            faiss_retriever=faiss_retriever,
            old_ranker=old_ranker,
            v26_ranker=v26_ranker,
            current_year=current_year,
        )
        self.safe_fusion_model = safe_fusion_model
        self.ridge_scorer = ridge_scorer

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        debug = self.score_query(query=query, top_k=50, exclude_ids=exclude_ids)
        ranked = sorted(
            debug["scores"].items(),
            key=lambda item: (item[1]["v6_4_safe_fusion_score"], -item[0]),
            reverse=True,
        )
        results: list[V64SafeFusionScoredDocument] = []
        self.last_components = {}
        self.last_candidate_pool_ids = list(debug["candidate_pool_ids"])
        for paper_id, values in ranked[:k]:
            components = dict(values)
            self.last_components[paper_id] = components
            results.append(
                V64SafeFusionScoredDocument(
                    document_id=paper_id,
                    score=float(values["v6_4_safe_fusion_score"]),
                    components=components,
                )
            )
        return results

    def score_query(
        self,
        *,
        query: str,
        top_k: int = 50,
        exclude_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        debug = super().score_query(query=query, top_k=top_k, exclude_ids=exclude_ids)
        if not debug["scores"]:
            debug["model"] = self._model_diagnostics()
            return debug

        v49_scores = {
            paper_id: float(values["v4_9_guarded_text_blend_score"])
            for paper_id, values in debug["scores"].items()
        }
        internal_ridge_details = self.ridge_scorer.score_component_maps(debug["scores"]) if self.ridge_scorer else {}
        ridge_scores: dict[int, float] = {}
        ridge_sources: dict[int, str] = {}
        for paper_id, values in debug["scores"].items():
            if paper_id in internal_ridge_details:
                ridge_scores[paper_id] = float(internal_ridge_details[paper_id]["v6_6_ridge_fusion_score"])
                ridge_sources[paper_id] = "internal_v6_6_ridge_scorer"
            elif "v6_3_ridge_fusion_score" in values:
                ridge_scores[paper_id] = float(values["v6_3_ridge_fusion_score"])
                ridge_sources[paper_id] = "provided_component"
            else:
                ridge_scores[paper_id] = float(values["v4_9_guarded_text_blend_score"])
                ridge_sources[paper_id] = "fallback_to_v4_9_missing_serialized_v6_3_ridge"

        safe_scores = self.safe_fusion_model.score_component_maps(
            v49_scores=v49_scores,
            ridge_scores=ridge_scores,
        )
        for paper_id, details in safe_scores.items():
            debug["scores"][paper_id].update(
                {
                    **details,
                    **internal_ridge_details.get(paper_id, {}),
                    "v4_9_method": V49_METHOD_NAME,
                    "v6_4_blend_name": self.safe_fusion_model.blend_name,
                    "v6_4_method": self.method_name,
                    "v6_4_model_version": self.safe_fusion_model.model_version,
                    "v6_4_candidate_config_path": str(self.safe_fusion_model.config_path),
                    "v6_4_ridge_score_source": ridge_sources[paper_id],
                    "learned_ranker_score": float(details["v6_4_safe_fusion_score"]),
                    "learned_ranker_version": self.safe_fusion_model.model_version,
                }
            )
        debug["model"] = self._model_diagnostics()
        return debug

    def _model_diagnostics(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "method_label": "offline accepted V6.4 safe fusion",
            "description": "V6.4 opt-in top-10 swap-limited safe fusion over V4.9 plus supplied V6.3 ridge scores when available",
            "blend_name": self.safe_fusion_model.blend_name,
            "candidate_config_path": str(self.safe_fusion_model.config_path),
            "formula": self.safe_fusion_model.config.get("formula"),
            "v4_9_guarded_text_model_version": self.guarded_model.model_version,
            "v6_6_ridge_scorer_config_path": str(self.ridge_scorer.config_path) if self.ridge_scorer else None,
            "v6_6_ridge_scorer_loaded": self.ridge_scorer is not None,
            "uses_only_inference_time_features": True,
            "forbidden_runtime_features": [],
            "ridge_score_runtime_behavior": "computes internal V6.6 ridge score when all required inference-safe features are present; otherwise uses supplied v6_3 score or preserves V4.9 ordering",
        }
