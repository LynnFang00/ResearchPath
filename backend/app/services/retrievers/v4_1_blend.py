from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.paper import Paper
from app.services.ltr_model_service import (
    V41_BLEND_METHOD_NAME,
    V41BlendModel,
    load_v4_1_blend_model,
    numeric_feature_value,
)
from app.services.retrievers.base import ScoredDocument
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.faiss import FaissRetriever
from app.services.retrievers.v3_3_ltr import V33LTRRetriever
from app.services.v2_2_learned_ranker import V22RidgeRanker


@dataclass(frozen=True)
class V41BlendScoredDocument(ScoredDocument):
    components: dict[str, Any]


class V41BlendRetriever(V33LTRRetriever):
    method_name = V41_BLEND_METHOD_NAME

    def __init__(
        self,
        papers: list[Paper],
        *,
        v33_model_dir: Path | None = None,
        v41_model_dir: Path | None = None,
        blend_config_path: Path | None = None,
        blend_model: V41BlendModel | None = None,
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
        if blend_model is None:
            if v33_model_dir is None:
                raise ValueError("V3.3 LTR model directory is required for V4.1 blend.")
            if v41_model_dir is None:
                raise ValueError("V4.1 weighted LTR model directory is required.")
            if blend_config_path is None:
                raise ValueError("V4.1 blend config path is required.")
            blend_model = load_v4_1_blend_model(
                v33_model_dir=v33_model_dir,
                v41_model_dir=v41_model_dir,
                blend_config_path=blend_config_path,
            )
        super().__init__(
            papers,
            v33_model=blend_model.v33_model,
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
        self.blend_model = blend_model

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        debug = self.score_query(query=query, top_k=50, exclude_ids=exclude_ids)
        ranked = sorted(debug["scores"].items(), key=lambda item: (item[1]["v4_1_blend_score"], -item[0]), reverse=True)
        results: list[V41BlendScoredDocument] = []
        self.last_components = {}
        self.last_candidate_pool_ids = list(debug["candidate_pool_ids"])
        for paper_id, values in ranked[:k]:
            components = dict(values)
            self.last_components[paper_id] = components
            results.append(
                V41BlendScoredDocument(
                    document_id=paper_id,
                    score=float(values["v4_1_blend_score"]),
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
        feature_rows = debug["feature_rows"]
        blend_scores = self.blend_model.score_feature_rows(feature_rows)
        for feature_row, scores in zip(feature_rows, blend_scores, strict=True):
            paper_id = int(feature_row["paper_id"])
            feature_values = {
                feature: float(numeric_feature_value(feature_row, feature))
                for feature in self.blend_model.v41_model.feature_columns
            }
            formula_score = self.blend_model.v33_weight * scores["v3_3_score"] + self.blend_model.v41_weight * scores["v4_1_weighted_score"]
            debug["scores"][paper_id].update(
                {
                    **feature_values,
                    "v3_3_score": scores["v3_3_score"],
                    "v4_1_weighted_score": scores["v4_1_weighted_score"],
                    "v4_1_blend_score": scores["v4_1_blend_score"],
                    "v4_1_blend_formula_score": float(formula_score),
                    "v4_1_blend_name": self.blend_model.blend_name,
                    "v4_1_blend_method": self.method_name,
                    "v4_1_blend_model_version": self.blend_model.model_version,
                    "v4_1_weighted_model_name": self.blend_model.v41_model.model_name,
                    "v4_1_weighted_model_version": self.blend_model.v41_model.model_version,
                    "v4_1_blend_feature_values": feature_values,
                    "learned_ranker_score": scores["v4_1_blend_score"],
                    "learned_ranker_version": self.blend_model.model_version,
                }
            )
        debug["model"] = self._model_diagnostics()
        return debug

    def _model_diagnostics(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "method_label": "offline calibrated blend",
            "description": "V3.3 relevance ranker plus V4.1 weighted guardrail signal",
            "blend_name": self.blend_model.blend_name,
            "blend_config_path": str(self.blend_model.config_path),
            "v3_3_weight": self.blend_model.v33_weight,
            "v4_1_weight": self.blend_model.v41_weight,
            "v3_3_model_name": self.blend_model.v33_model.model_name,
            "v3_3_model_version": self.blend_model.v33_model.model_version,
            "v4_1_weighted_model_name": self.blend_model.v41_model.model_name,
            "v4_1_weighted_model_version": self.blend_model.v41_model.model_version,
            "feature_columns": self.blend_model.v41_model.feature_columns,
            "uses_v2_7_feature": self.blend_model.v41_model.metadata.get("uses_v2_7_feature"),
            "raw_score_feature_intersection": self.blend_model.v41_model.metadata.get("raw_score_feature_intersection", []),
        }
