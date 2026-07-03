from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.paper import Paper
from app.services.ltr_model_service import V33_METHOD_NAME, V33LTRModel, load_v3_3_ltr_model, numeric_feature_value
from app.services.retrievers.base import ScoredDocument
from app.services.retrievers.embedding import EmbeddingRetriever
from app.services.retrievers.faiss import FaissRetriever
from app.services.retrievers.learned_blend_v2_7 import BASELINE_METHODS, LearnedBlendV27Retriever, minmax
from app.services.v2_2_learned_ranker import V22RidgeRanker


@dataclass(frozen=True)
class V33LTRScoredDocument(ScoredDocument):
    components: dict[str, Any]


class V33LTRRetriever(LearnedBlendV27Retriever):
    method_name = V33_METHOD_NAME

    def __init__(
        self,
        papers: list[Paper],
        *,
        v33_model_dir: Path | None = None,
        v33_model: V33LTRModel | None = None,
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
        super().__init__(
            papers,
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
        if v33_model is None:
            if v33_model_dir is None:
                raise ValueError("V3.3 LTR model directory is required.")
            v33_model = load_v3_3_ltr_model(v33_model_dir)
        self.v33_model = v33_model

    def search(
        self,
        query: str,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> list[ScoredDocument]:
        debug = self.score_query(query=query, top_k=50, exclude_ids=exclude_ids)
        ranked = sorted(debug["scores"].items(), key=lambda item: (item[1]["v3_3_ltr_score"], -item[0]), reverse=True)
        results: list[V33LTRScoredDocument] = []
        self.last_components = {}
        self.last_candidate_pool_ids = list(debug["candidate_pool_ids"])
        for paper_id, values in ranked[:k]:
            components = dict(values)
            self.last_components[paper_id] = components
            results.append(
                V33LTRScoredDocument(
                    document_id=paper_id,
                    score=float(values["v3_3_ltr_score"]),
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

        feature_rows = self.build_feature_rows(debug)
        predictions = self.v33_model.score_feature_rows(feature_rows)
        for feature_row, score in zip(feature_rows, predictions, strict=True):
            paper_id = int(feature_row["paper_id"])
            feature_values = {
                feature: float(numeric_feature_value(feature_row, feature))
                for feature in self.v33_model.feature_columns
            }
            debug["scores"][paper_id].update(
                {
                    **feature_values,
                    "v3_3_ltr_score": float(score),
                    "v3_3_model_name": self.v33_model.model_name,
                    "v3_3_model_version": self.v33_model.model_version,
                    "v3_3_score_column": self.v33_model.score_column,
                    "v3_3_feature_values": feature_values,
                    "learned_ranker_score": float(score),
                    "learned_ranker_version": self.v33_model.model_version,
                }
            )
        debug["feature_rows"] = feature_rows
        debug["model"] = self._model_diagnostics()
        return debug

    def build_feature_rows(self, debug: dict[str, Any]) -> list[dict[str, Any]]:
        candidate_ids = [int(row["paper_id"]) for row in debug["rows"]]
        normalized_scores = {
            method: minmax({paper_id: float(debug["pool_scores"][method].get(paper_id, 0.0)) for paper_id in candidate_ids})
            for method in BASELINE_METHODS
        }
        row_by_id = {int(row["paper_id"]): row for row in debug["rows"]}
        output: list[dict[str, Any]] = []
        for paper_id in candidate_ids:
            row = row_by_id[paper_id]
            ranks = row.get("retrieval_ranks_by_method") or {}
            v27_values = debug["scores"][paper_id]
            feature_row: dict[str, Any] = {
                "paper_id": paper_id,
                "year": row.get("year"),
                "citation_count": row.get("citation_count") or 0,
                "full_text_available": bool((row.get("evidence_availability") or {}).get("fulltext_available")),
                "old_v2_2b_score": float(v27_values.get("old_v2_2b_score", 0.0)),
                "v2_6_score": float(v27_values.get("v2_6_score", 0.0)),
            }
            for method in BASELINE_METHODS:
                rank = int(ranks.get(method, 0) or 0)
                feature_row[f"{method}_score"] = float(normalized_scores[method].get(paper_id, 0.0))
                feature_row[f"{method}_rank"] = float(rank)
                feature_row[f"{method}_reciprocal_rank"] = 1.0 / rank if rank > 0 else 0.0
            output.append(feature_row)
        return output

    def _model_diagnostics(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "model_name": self.v33_model.model_name,
            "model_version": self.v33_model.model_version,
            "score_column": self.v33_model.score_column,
            "feature_columns": self.v33_model.feature_columns,
            "uses_v2_7_feature": self.v33_model.metadata.get("uses_v2_7_feature"),
            "raw_score_feature_intersection": self.v33_model.metadata.get("raw_score_feature_intersection", []),
        }
