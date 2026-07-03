from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.paper import Paper
from app.services.ltr_model_service import (
    V49_METHOD_NAME,
    V49GuardedTextBlendModel,
    load_v4_9_guarded_text_blend_model,
)
from app.services.retrievers.base import ScoredDocument
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.faiss import FaissRetriever
from app.services.retrievers.v4_1_blend import V41BlendRetriever
from app.services.v2_2_learned_ranker import V22RidgeRanker


@dataclass(frozen=True)
class V49GuardedTextBlendScoredDocument(ScoredDocument):
    components: dict[str, Any]


class V49GuardedTextBlendRetriever(V41BlendRetriever):
    method_name = V49_METHOD_NAME

    def __init__(
        self,
        papers: list[Paper],
        *,
        v33_model_dir: Path | None = None,
        v41_model_dir: Path | None = None,
        v43_model_dir: Path | None = None,
        v41_blend_config_path: Path | None = None,
        v49_candidate_config_path: Path | None = None,
        guarded_model: V49GuardedTextBlendModel | None = None,
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
        if guarded_model is None:
            if v33_model_dir is None:
                raise ValueError("V3.3 LTR model directory is required for V4.9 guarded text blend.")
            if v41_model_dir is None:
                raise ValueError("V4.1 weighted LTR model directory is required for V4.9 guarded text blend.")
            if v43_model_dir is None:
                raise ValueError("V4.3 text reranker model directory is required.")
            if v41_blend_config_path is None:
                raise ValueError("V4.1 blend config path is required for V4.9 guarded text blend.")
            if v49_candidate_config_path is None:
                raise ValueError("V4.9 guarded text candidate config path is required.")
            guarded_model = load_v4_9_guarded_text_blend_model(
                v33_model_dir=v33_model_dir,
                v41_model_dir=v41_model_dir,
                v43_model_dir=v43_model_dir,
                v41_blend_config_path=v41_blend_config_path,
                v49_candidate_config_path=v49_candidate_config_path,
            )
        super().__init__(
            papers,
            blend_model=guarded_model.v41_blend_model,
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
        self.guarded_model = guarded_model

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        debug = self.score_query(query=query, top_k=50, exclude_ids=exclude_ids)
        ranked = sorted(
            debug["scores"].items(),
            key=lambda item: (item[1]["v4_9_guarded_text_blend_score"], -item[0]),
            reverse=True,
        )
        results: list[V49GuardedTextBlendScoredDocument] = []
        self.last_components = {}
        self.last_candidate_pool_ids = list(debug["candidate_pool_ids"])
        for paper_id, values in ranked[:k]:
            components = dict(values)
            self.last_components[paper_id] = components
            results.append(
                V49GuardedTextBlendScoredDocument(
                    document_id=paper_id,
                    score=float(values["v4_9_guarded_text_blend_score"]),
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
        if not debug["rows"]:
            debug["model"] = self._model_diagnostics()
            return debug
        row_by_id = {int(row["paper_id"]): row for row in debug["rows"]}
        feature_rows = list(debug["feature_rows"])
        texts = [self._text_input(row_by_id[int(row["paper_id"])]) for row in feature_rows]
        text_predictions = self.guarded_model.text_model.score_text_rows(texts, feature_rows)
        text_scores = {
            int(row["paper_id"]): float(score)
            for row, score in zip(feature_rows, text_predictions, strict=True)
        }
        v33_scores = {
            paper_id: float(values["v3_3_score"])
            for paper_id, values in debug["scores"].items()
        }
        v41_scores = {
            paper_id: float(values["v4_1_weighted_score"])
            for paper_id, values in debug["scores"].items()
        }
        guarded_scores = self.guarded_model.score_component_maps(
            v33_scores=v33_scores,
            v41_scores=v41_scores,
            text_scores=text_scores,
        )
        for paper_id, details in guarded_scores.items():
            debug["scores"][paper_id].update(
                {
                    **details,
                    "v4_3_text_score": float(text_scores.get(paper_id, 0.0)),
                    "v4_3_text_model_name": self.guarded_model.text_model.model_name,
                    "v4_3_text_model_version": self.guarded_model.text_model.model_version,
                    "v4_9_blend_name": self.guarded_model.blend_name,
                    "v4_9_method": self.method_name,
                    "v4_9_model_version": self.guarded_model.model_version,
                    "v4_9_candidate_config_path": str(self.guarded_model.config_path),
                    "learned_ranker_score": float(details["v4_9_guarded_text_blend_score"]),
                    "learned_ranker_version": self.guarded_model.model_version,
                }
            )
        debug["model"] = self._model_diagnostics()
        return debug

    @staticmethod
    def _text_input(row: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Query: {row.get('query') or row.get('query_id') or ''}",
                f"Title: {row.get('title') or ''}",
                f"Abstract: {row.get('abstract') or ''}",
            ]
        )

    def _model_diagnostics(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "method_label": "offline accepted guarded text blend",
            "description": "V4.9 opt-in guarded blend of V3.3, V4.1 weighted, and gated V4.3 text scores",
            "blend_name": self.guarded_model.blend_name,
            "candidate_config_path": str(self.guarded_model.config_path),
            "formula": self.guarded_model.config.get("formula"),
            "v4_1_blend_model_version": self.blend_model.model_version,
            "v4_3_text_model_name": self.guarded_model.text_model.model_name,
            "v4_3_text_model_version": self.guarded_model.text_model.model_version,
            "uses_only_inference_time_features": True,
            "forbidden_runtime_features": [],
        }
