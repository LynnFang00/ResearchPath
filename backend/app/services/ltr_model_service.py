from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse


V33_MODEL_NAME = "random_forest_no_v27_compact"
V33_METHOD_NAME = "v3_3_ltr"
V41_WEIGHTED_MODEL_NAME = "downweight_v3_9_active_learning"
V41_BLEND_METHOD_NAME = "v4_1_blend"
V41_BLEND_NAME = "blend_0_75_v33_0_25_v41"
V43_TEXT_MODEL_NAME = "text_plus_ranker_features_ridge"
V43_TEXT_SCORE_COLUMN = "v4_3_text_plus_ranker_features_ridge_score"
V49_METHOD_NAME = "v4_9_guarded_text_blend"
V49_BLEND_NAME = "v4_6_guardrail_text_when_rankers_not_confident"
V49_V33_WEIGHT = 0.675
V49_V41_WEIGHT = 0.225
V49_TEXT_WEIGHT = 0.10
V49_CONFIDENCE_THRESHOLD = 0.85
V64_METHOD_NAME = "v6_4_safe_fusion"
V64_BLEND_NAME = "v6_4_top10_swap_limited_2"
V64_MAX_SWAPS = 2
V64_SWAP_SCORE_STEP = 0.004
V64_NON_ENTRANT_CAP = 0.005
V66_RIDGE_SCORER_SCHEMA = "v6.6_safe_fusion_ridge_scorer"
RAW_SCORE_FEATURES = {
    "bm25_raw_score",
    "tfidf_raw_score",
    "embedding_raw_score",
    "faiss_embedding_raw_score",
    "hybrid_raw_score",
}


@dataclass(frozen=True)
class V33LTRModel:
    model_name: str
    artifact_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    estimator: Any

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["feature_columns"])

    @property
    def score_column(self) -> str:
        return str(self.metadata.get("score_column") or "v3_3_random_forest_no_v27_compact_score")

    @property
    def model_version(self) -> str:
        created_at = str(self.metadata.get("created_at") or "")
        return f"{self.model_name}:{created_at}" if created_at else self.model_name

    def score_feature_rows(self, rows: list[dict[str, Any]]) -> list[float]:
        matrix = np.array(
            [[numeric_feature_value(row, feature) for feature in self.feature_columns] for row in rows],
            dtype=float,
        )
        predictions = self.estimator.predict(matrix)
        return [float(value) for value in predictions]


@dataclass(frozen=True)
class V41WeightedLTRModel:
    model_name: str
    artifact_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    estimator: Any

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["feature_columns"])

    @property
    def score_column(self) -> str:
        return str(self.metadata.get("score_column") or "v4_1_downweight_v3_9_active_learning_score")

    @property
    def model_version(self) -> str:
        created_at = str(self.metadata.get("created_at") or "")
        return f"{self.model_name}:{created_at}" if created_at else self.model_name

    def score_feature_rows(self, rows: list[dict[str, Any]]) -> list[float]:
        matrix = np.array(
            [[numeric_feature_value(row, feature) for feature in self.feature_columns] for row in rows],
            dtype=float,
        )
        predictions = self.estimator.predict(matrix)
        return [float(value) for value in predictions]


@dataclass(frozen=True)
class V41BlendModel:
    v33_model: V33LTRModel
    v41_model: V41WeightedLTRModel
    config_path: Path
    config: dict[str, Any]

    @property
    def v33_weight(self) -> float:
        return float(self.config["spec"]["weights"]["v3_3_random_forest_no_v27_compact"])

    @property
    def v41_weight(self) -> float:
        return float(self.config["spec"]["weights"]["__v41__"])

    @property
    def blend_name(self) -> str:
        return str(self.config["method"])

    @property
    def model_version(self) -> str:
        created_at = str(self.config.get("created_at") or "")
        return f"{self.blend_name}:{created_at}" if created_at else self.blend_name

    def score_feature_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        v33_scores = self.v33_model.score_feature_rows(rows)
        v41_scores = self.v41_model.score_feature_rows(rows)
        outputs: list[dict[str, float]] = []
        for v33_score, v41_score in zip(v33_scores, v41_scores, strict=True):
            blend_score = self.v33_weight * v33_score + self.v41_weight * v41_score
            values = {
                "v3_3_score": float(v33_score),
                "v4_1_weighted_score": float(v41_score),
                "v4_1_blend_score": float(blend_score),
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"V4.1 blend produced non-finite scores: {values}")
            outputs.append(values)
        return outputs


@dataclass(frozen=True)
class V43TextRerankerModel:
    model_name: str
    artifact_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    payload: dict[str, Any]

    @property
    def score_column(self) -> str:
        return str(self.metadata.get("score_column") or V43_TEXT_SCORE_COLUMN)

    @property
    def numeric_ranker_features(self) -> list[str]:
        return list(self.metadata.get("numeric_ranker_features") or [])

    @property
    def uses_ranker_features(self) -> bool:
        return bool(self.metadata.get("uses_ranker_features"))

    @property
    def model_version(self) -> str:
        created_at = str(self.metadata.get("created_at") or "")
        return f"{self.model_name}:{created_at}" if created_at else self.model_name

    def score_text_rows(self, texts: list[str], feature_rows: list[dict[str, Any]]) -> list[float]:
        x_text = self.payload["vectorizer"].transform(texts)
        if self.uses_ranker_features:
            matrix = np.array(
                [
                    [numeric_feature_value(row, feature) for feature in self.numeric_ranker_features]
                    for row in feature_rows
                ],
                dtype=float,
            )
            x_numeric = sparse.csr_matrix(self.payload["scaler"].transform(matrix))
            x_values = sparse.hstack([x_text, x_numeric], format="csr")
        else:
            x_values = x_text
        estimator = self.payload["estimator"]
        if str(self.metadata.get("model_type")) == "tfidf_logistic_positive":
            predictions = estimator.predict_proba(x_values)[:, 1]
        else:
            predictions = estimator.predict(x_values)
        return [float(value) for value in predictions]


@dataclass(frozen=True)
class V49GuardedTextBlendModel:
    v41_blend_model: V41BlendModel
    text_model: V43TextRerankerModel
    config_path: Path
    config: dict[str, Any]

    @property
    def blend_name(self) -> str:
        return str(self.config["blend_name"])

    @property
    def model_version(self) -> str:
        created_at = str(self.config.get("created_at") or "")
        return f"{V49_METHOD_NAME}:{created_at}" if created_at else V49_METHOD_NAME

    @staticmethod
    def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        values = list(scores.values())
        if not values:
            return {}
        low = min(values)
        high = max(values)
        if abs(high - low) < 1e-12:
            return {paper_id: 0.5 for paper_id in scores}
        return {paper_id: (score - low) / (high - low) for paper_id, score in scores.items()}

    @staticmethod
    def guarded_score(
        *,
        normalized_v33_score: float,
        normalized_v41_score: float,
        normalized_text_score: float,
    ) -> dict[str, float | bool]:
        max_ranker_confidence = max(normalized_v33_score, normalized_v41_score)
        text_enabled = max_ranker_confidence < V49_CONFIDENCE_THRESHOLD
        text_component = V49_TEXT_WEIGHT * normalized_text_score if text_enabled else 0.0
        score = V49_V33_WEIGHT * normalized_v33_score + V49_V41_WEIGHT * normalized_v41_score + text_component
        return {
            "v4_9_guarded_text_blend_score": float(score),
            "v4_9_text_component": float(text_component),
            "v4_9_text_enabled": bool(text_enabled),
            "v4_9_max_ranker_confidence": float(max_ranker_confidence),
        }

    @staticmethod
    def score_component_maps(
        *,
        v33_scores: dict[int, float],
        v41_scores: dict[int, float],
        text_scores: dict[int, float],
    ) -> dict[int, dict[str, float | bool]]:
        normalized_v33 = V49GuardedTextBlendModel.normalize_scores(v33_scores)
        normalized_v41 = V49GuardedTextBlendModel.normalize_scores(v41_scores)
        normalized_text = V49GuardedTextBlendModel.normalize_scores(text_scores)
        output: dict[int, dict[str, float | bool]] = {}
        for paper_id in sorted(set(normalized_v33) | set(normalized_v41) | set(normalized_text)):
            details = V49GuardedTextBlendModel.guarded_score(
                normalized_v33_score=float(normalized_v33.get(paper_id, 0.0)),
                normalized_v41_score=float(normalized_v41.get(paper_id, 0.0)),
                normalized_text_score=float(normalized_text.get(paper_id, 0.0)),
            )
            details.update(
                {
                    "normalized_v3_3_score": float(normalized_v33.get(paper_id, 0.0)),
                    "normalized_v4_1_score": float(normalized_v41.get(paper_id, 0.0)),
                    "normalized_v4_3_text_score": float(normalized_text.get(paper_id, 0.0)),
                }
            )
            output[paper_id] = details
        return output


@dataclass(frozen=True)
class V64SafeFusionModel:
    config_path: Path
    config: dict[str, Any]

    @property
    def blend_name(self) -> str:
        return str(self.config["method"])

    @property
    def model_version(self) -> str:
        created_at = str(self.config.get("created_at") or "")
        return f"{V64_METHOD_NAME}:{created_at}" if created_at else V64_METHOD_NAME

    @staticmethod
    def clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def score_component_maps(
        *,
        v49_scores: dict[int, float],
        ridge_scores: dict[int, float],
        max_swaps: int = V64_MAX_SWAPS,
    ) -> dict[int, dict[str, float | bool | int | None]]:
        paper_ids = sorted(set(v49_scores) | set(ridge_scores))
        if not paper_ids:
            return {}

        def ranked_ids(scores: dict[int, float]) -> list[int]:
            return sorted(paper_ids, key=lambda paper_id: (float(scores.get(paper_id, 0.0)), -paper_id), reverse=True)

        v49_ranked = ranked_ids(v49_scores)
        ridge_ranked = ranked_ids(ridge_scores)
        v49_top = v49_ranked[:10]
        ridge_top = ridge_ranked[:10]
        v49_top_ids = set(v49_top)
        entrants = [paper_id for paper_id in ridge_top if paper_id not in v49_top_ids][:max_swaps]
        v49_tenth = min(float(v49_scores.get(paper_id, 0.0)) for paper_id in v49_top) if v49_top else 0.0
        entrant_cap = v49_tenth + V64_SWAP_SCORE_STEP * (max_swaps + 1)
        v49_ranks = {paper_id: rank for rank, paper_id in enumerate(v49_ranked, start=1)}
        ridge_ranks = {paper_id: rank for rank, paper_id in enumerate(ridge_ranked, start=1)}

        output: dict[int, dict[str, float | bool | int | None]] = {}
        for paper_id in paper_ids:
            v49_score = float(v49_scores.get(paper_id, 0.0))
            ridge_score = float(ridge_scores.get(paper_id, v49_score))
            if paper_id in entrants:
                score = max(v49_score, min(ridge_score, entrant_cap))
                clipped_delta = score - v49_score
                entrant = True
            else:
                clipped_delta = V64SafeFusionModel.clip(
                    ridge_score - v49_score,
                    -V64_NON_ENTRANT_CAP,
                    V64_NON_ENTRANT_CAP,
                )
                score = v49_score + clipped_delta
                entrant = False
            output[paper_id] = {
                "v6_4_safe_fusion_score": float(score),
                "v6_4_v49_score": float(v49_score),
                "v6_4_ridge_fusion_score": float(ridge_score),
                "v6_4_score_delta_from_v4_9": float(score - v49_score),
                "v6_4_clipped_ridge_delta": float(clipped_delta),
                "v6_4_top10_swap_entrant": entrant,
                "v6_4_v49_rank": v49_ranks.get(paper_id),
                "v6_4_ridge_rank": ridge_ranks.get(paper_id),
                "v6_4_rank_gap_ridge_minus_v49": (
                    None
                    if v49_ranks.get(paper_id) is None or ridge_ranks.get(paper_id) is None
                    else int(ridge_ranks[paper_id] - v49_ranks[paper_id])
                ),
                "v6_4_v49_tenth_score": float(v49_tenth),
                "v6_4_entrant_cap": float(entrant_cap),
            }
        return output


@dataclass(frozen=True)
class V66SafeFusionRidgeScorer:
    config_path: Path
    config: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return list(self.config["feature_names"])

    @property
    def coefficients(self) -> list[float]:
        return [float(value) for value in self.config["coefficients"]]

    @property
    def intercept(self) -> float:
        return float(self.config["intercept"])

    @property
    def scaler_mean(self) -> list[float]:
        return [float(value) for value in self.config["scaler"]["mean"]]

    @property
    def scaler_scale(self) -> list[float]:
        return [float(value) if abs(float(value)) > 1e-12 else 1.0 for value in self.config["scaler"]["scale"]]

    @property
    def model_version(self) -> str:
        created_at = str(self.config.get("created_at") or "")
        return f"{V66_RIDGE_SCORER_SCHEMA}:{created_at}" if created_at else V66_RIDGE_SCORER_SCHEMA

    @staticmethod
    def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        return V49GuardedTextBlendModel.normalize_scores(scores)

    @staticmethod
    def rank_map(scores: dict[int, float]) -> dict[int, int]:
        ranked = sorted(scores, key=lambda paper_id: (float(scores.get(paper_id, 0.0)), -paper_id), reverse=True)
        return {paper_id: rank for rank, paper_id in enumerate(ranked, start=1)}

    @staticmethod
    def margin_to_10th(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        ranked_scores = sorted((float(score) for score in scores.values()), reverse=True)
        tenth = ranked_scores[min(9, len(ranked_scores) - 1)]
        return {paper_id: float(score) - tenth for paper_id, score in scores.items()}

    @staticmethod
    def feature_maps_from_components(component_maps: dict[int, dict[str, Any]]) -> dict[int, dict[str, float]]:
        required_methods = {
            "bm25": ("bm25_score", "bm25_raw_score"),
            "v3_3_ltr": ("v3_3_ltr_score", "v3_3_score"),
            "v4_1_blend": ("v4_1_blend_score",),
            "v4_9_guarded_text_blend": ("v4_9_guarded_text_blend_score",),
            "v6_1_neural": ("v6_1_neural_score", "neural_cross_encoder_v6_1_score"),
            "v6_2_multitask_neural": ("v6_2_multitask_neural_score",),
        }
        raw_scores: dict[str, dict[int, float]] = {method: {} for method in required_methods}
        for paper_id, values in component_maps.items():
            for method, aliases in required_methods.items():
                for alias in aliases:
                    if values.get(alias) is not None:
                        try:
                            raw_scores[method][paper_id] = float(values[alias])
                        except (TypeError, ValueError):
                            pass
                        break
        paper_ids = set(component_maps)
        if any(set(scores) != paper_ids for scores in raw_scores.values()):
            return {}

        normalized = {method: V66SafeFusionRidgeScorer.normalize_scores(scores) for method, scores in raw_scores.items()}
        ranks_v49 = V66SafeFusionRidgeScorer.rank_map(normalized["v4_9_guarded_text_blend"])
        ranks_v61 = V66SafeFusionRidgeScorer.rank_map(normalized["v6_1_neural"])
        ranks_v62 = V66SafeFusionRidgeScorer.rank_map(normalized["v6_2_multitask_neural"])
        margins_v49 = V66SafeFusionRidgeScorer.margin_to_10th(normalized["v4_9_guarded_text_blend"])
        margins_v61 = V66SafeFusionRidgeScorer.margin_to_10th(normalized["v6_1_neural"])
        margins_v62 = V66SafeFusionRidgeScorer.margin_to_10th(normalized["v6_2_multitask_neural"])

        output: dict[int, dict[str, float]] = {}
        for paper_id, values in component_maps.items():
            v49 = float(normalized["v4_9_guarded_text_blend"][paper_id])
            v61 = float(normalized["v6_1_neural"][paper_id])
            v62 = float(normalized["v6_2_multitask_neural"][paper_id])
            output[paper_id] = {
                "bm25_score": float(normalized["bm25"][paper_id]),
                "v3_3_ltr_score": float(normalized["v3_3_ltr"][paper_id]),
                "v4_1_blend_score": float(normalized["v4_1_blend"][paper_id]),
                "v4_9_guarded_text_blend_score": v49,
                "v6_1_neural_score": v61,
                "v6_2_multitask_neural_score": v62,
                "v4_9_minus_v6_1_neural_score": v49 - v61,
                "v4_9_minus_v6_2_multitask_score": v49 - v62,
                "v4_9_minus_v6_1_rank_diff": float(ranks_v61[paper_id] - ranks_v49[paper_id]),
                "v4_9_minus_v6_2_rank_diff": float(ranks_v62[paper_id] - ranks_v49[paper_id]),
                "v4_9_max_ranker_confidence": float(values.get("v4_9_max_ranker_confidence") or v49),
                "v4_9_margin_to_10th": float(margins_v49[paper_id]),
                "v6_1_margin_to_10th": float(margins_v61[paper_id]),
                "v6_2_margin_to_10th": float(margins_v62[paper_id]),
            }
        return output

    def score_feature_values(self, feature_values: dict[str, float]) -> float:
        missing_policy = str(self.config.get("missing_value_policy", {}).get("default") or "zero")
        vector = []
        for name in self.feature_names:
            value = feature_values.get(name)
            if value is None:
                value = 0.0 if missing_policy == "zero" else 0.0
            vector.append(float(value) if math.isfinite(float(value)) else 0.0)
        scaled = [
            (value - mean) / scale
            for value, mean, scale in zip(vector, self.scaler_mean, self.scaler_scale, strict=True)
        ]
        score = self.intercept + sum(coefficient * value for coefficient, value in zip(self.coefficients, scaled, strict=True))
        return float(score)

    def score_component_maps(self, component_maps: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        feature_maps = self.feature_maps_from_components(component_maps)
        if not feature_maps:
            return {}
        return {
            paper_id: {
                "v6_6_ridge_fusion_score": self.score_feature_values(feature_values),
                "v6_6_ridge_feature_values": feature_values,
                "v6_6_ridge_scorer_version": self.model_version,
                "v6_6_ridge_scorer_config_path": str(self.config_path),
            }
            for paper_id, feature_values in feature_maps.items()
        }


_MODEL_CACHE: dict[tuple[str, str], V33LTRModel] = {}
_V41_WEIGHTED_MODEL_CACHE: dict[tuple[str, str], V41WeightedLTRModel] = {}
_V41_BLEND_MODEL_CACHE: dict[tuple[str, str, str, str], V41BlendModel] = {}
_V43_TEXT_MODEL_CACHE: dict[tuple[str, str], V43TextRerankerModel] = {}
_V49_BLEND_MODEL_CACHE: dict[tuple[str, str, str, str, str], V49GuardedTextBlendModel] = {}
_V64_SAFE_FUSION_MODEL_CACHE: dict[str, V64SafeFusionModel] = {}
_V66_RIDGE_SCORER_CACHE: dict[str, V66SafeFusionRidgeScorer] = {}


def numeric_feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "citation_count_log1p":
        return math.log1p(max(float(row.get("citation_count") or 0.0), 0.0))
    if feature == "paper_age_years":
        year = row.get("year")
        return float(max(0, 2026 - int(year))) if year else 0.0
    if feature == "full_text_available_bool":
        return 1.0 if row.get("full_text_available") else 0.0
    value = row.get(feature)
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _resolve_artifact_path(model_dir: Path, path_value: str | None, fallback_name: str) -> Path:
    if path_value:
        candidate = Path(path_value)
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            relative_candidate = model_dir / candidate
            if relative_candidate.exists():
                return relative_candidate
    return model_dir / fallback_name


def load_v3_3_ltr_model(model_dir: Path, model_name: str = V33_MODEL_NAME) -> V33LTRModel:
    cache_key = (str(model_dir.resolve()), model_name)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    index_path = model_dir / "index.json"
    if not index_path.exists():
        raise ValueError(f"V3.3 LTR artifact index was not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact_entry = (index.get("artifacts") or {}).get(model_name)
    if not isinstance(artifact_entry, dict):
        raise ValueError(f"V3.3 LTR artifact '{model_name}' was not listed in {index_path}")

    metadata_path = _resolve_artifact_path(model_dir, artifact_entry.get("metadata_path"), f"{model_name}.metadata.json")
    artifact_path = _resolve_artifact_path(model_dir, artifact_entry.get("artifact_path"), f"{model_name}.joblib")
    if not metadata_path.exists():
        raise ValueError(f"V3.3 LTR model metadata was not found: {metadata_path}")
    if not artifact_path.exists():
        raise ValueError(f"V3.3 LTR model artifact was not found: {artifact_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata.get("feature_columns") or [])
    if metadata.get("uses_v2_7_feature") is True or "v2_7_score" in feature_columns:
        raise ValueError("V3.3 runtime model must not use v2_7_score as a feature.")
    raw_intersection = sorted(RAW_SCORE_FEATURES & set(feature_columns))
    if raw_intersection:
        raise ValueError(f"V3.3 runtime model contains raw score features: {raw_intersection}")

    model = V33LTRModel(
        model_name=model_name,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        metadata=metadata,
        estimator=joblib.load(artifact_path),
    )
    _MODEL_CACHE[cache_key] = model
    return model


def _validate_no_v27_or_raw_features(feature_columns: list[str], label: str) -> None:
    if "v2_7_score" in feature_columns:
        raise ValueError(f"{label} runtime model must not use v2_7_score as a feature.")
    raw_intersection = sorted(RAW_SCORE_FEATURES & set(feature_columns))
    if raw_intersection:
        raise ValueError(f"{label} runtime model contains raw score features: {raw_intersection}")


def load_v4_1_weighted_ltr_model(model_dir: Path, model_name: str = V41_WEIGHTED_MODEL_NAME) -> V41WeightedLTRModel:
    cache_key = (str(model_dir.resolve()), model_name)
    if cache_key in _V41_WEIGHTED_MODEL_CACHE:
        return _V41_WEIGHTED_MODEL_CACHE[cache_key]

    index_path = model_dir / "index.json"
    if not index_path.exists():
        raise ValueError(f"V4.1 weighted LTR artifact index was not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact_entry = (index.get("artifacts") or {}).get(model_name)
    if not isinstance(artifact_entry, dict):
        raise ValueError(f"V4.1 weighted LTR artifact '{model_name}' was not listed in {index_path}")

    metadata_path = _resolve_artifact_path(model_dir, artifact_entry.get("metadata_path"), f"{model_name}.metadata.json")
    artifact_path = _resolve_artifact_path(model_dir, artifact_entry.get("artifact_path"), f"{model_name}.joblib")
    if not metadata_path.exists():
        raise ValueError(f"V4.1 weighted LTR model metadata was not found: {metadata_path}")
    if not artifact_path.exists():
        raise ValueError(f"V4.1 weighted LTR model artifact was not found: {artifact_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = list(metadata.get("feature_columns") or [])
    if metadata.get("uses_v2_7_feature") is True:
        raise ValueError("V4.1 weighted runtime model must not use v2_7_score as a feature.")
    _validate_no_v27_or_raw_features(feature_columns, "V4.1 weighted")

    model = V41WeightedLTRModel(
        model_name=model_name,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        metadata=metadata,
        estimator=joblib.load(artifact_path),
    )
    _V41_WEIGHTED_MODEL_CACHE[cache_key] = model
    return model


def load_v4_1_blend_model(
    *,
    v33_model_dir: Path,
    v41_model_dir: Path,
    blend_config_path: Path,
    v33_model_name: str = V33_MODEL_NAME,
    v41_model_name: str = V41_WEIGHTED_MODEL_NAME,
) -> V41BlendModel:
    cache_key = (
        str(v33_model_dir.resolve()),
        str(v41_model_dir.resolve()),
        str(blend_config_path.resolve()),
        v41_model_name,
    )
    if cache_key in _V41_BLEND_MODEL_CACHE:
        return _V41_BLEND_MODEL_CACHE[cache_key]
    if not blend_config_path.exists():
        raise ValueError(f"V4.1 blend config was not found: {blend_config_path}")

    config = json.loads(blend_config_path.read_text(encoding="utf-8"))
    if config.get("method") != V41_BLEND_NAME:
        raise ValueError(f"Unexpected V4.1 blend method '{config.get('method')}'. Expected '{V41_BLEND_NAME}'.")
    if config.get("best_v41_method") != f"v4_1_{v41_model_name}":
        raise ValueError(
            f"Unexpected V4.1 weighted method '{config.get('best_v41_method')}'. Expected 'v4_1_{v41_model_name}'."
        )
    weights = (config.get("spec") or {}).get("weights") or {}
    if float(weights.get("v3_3_random_forest_no_v27_compact", -1.0)) != 0.75:
        raise ValueError(f"Unexpected V4.1 blend V3.3 weight: {weights.get('v3_3_random_forest_no_v27_compact')}")
    if float(weights.get("__v41__", -1.0)) != 0.25:
        raise ValueError(f"Unexpected V4.1 blend V4.1 weight: {weights.get('__v41__')}")

    v33_model = load_v3_3_ltr_model(v33_model_dir, v33_model_name)
    v41_model = load_v4_1_weighted_ltr_model(v41_model_dir, v41_model_name)
    feature_order = list(config.get("feature_order") or [])
    if feature_order != v41_model.feature_columns:
        raise ValueError("V4.1 blend config feature_order does not match the weighted model metadata feature_columns.")
    if v33_model.feature_columns != v41_model.feature_columns:
        raise ValueError("V4.1 blend requires V3.3 and V4.1 weighted feature columns to match exactly.")

    model = V41BlendModel(v33_model=v33_model, v41_model=v41_model, config_path=blend_config_path, config=config)
    _V41_BLEND_MODEL_CACHE[cache_key] = model
    return model


def load_v4_3_text_reranker_model(
    model_dir: Path,
    score_column: str = V43_TEXT_SCORE_COLUMN,
) -> V43TextRerankerModel:
    cache_key = (str(model_dir.resolve()), score_column)
    if cache_key in _V43_TEXT_MODEL_CACHE:
        return _V43_TEXT_MODEL_CACHE[cache_key]
    index_path = model_dir / "index.json"
    if not index_path.exists():
        raise ValueError(f"V4.3 text reranker index was not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = list(index.get("models") or [])
    entry = next((row for row in entries if row.get("score_column") == score_column), None)
    if not isinstance(entry, dict):
        raise ValueError(f"V4.3 text reranker score column '{score_column}' was not listed in {index_path}")
    metadata_path = _resolve_artifact_path(model_dir, entry.get("metadata_path"), f"{V43_TEXT_MODEL_NAME}.metadata.json")
    artifact_path = _resolve_artifact_path(model_dir, entry.get("artifact_path"), f"{V43_TEXT_MODEL_NAME}.joblib")
    if not metadata_path.exists():
        raise ValueError(f"V4.3 text reranker metadata was not found: {metadata_path}")
    if not artifact_path.exists():
        raise ValueError(f"V4.3 text reranker artifact was not found: {artifact_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("excludes_v2_7_feature") is not True:
        raise ValueError("V4.3 text reranker must exclude v2_7_score.")
    if metadata.get("excludes_raw_score_features") is not True:
        raise ValueError("V4.3 text reranker must exclude raw score features.")
    features = list(metadata.get("numeric_ranker_features") or [])
    if "v2_7_score" in features:
        raise ValueError("V4.3 text reranker runtime features must not include v2_7_score.")
    _validate_no_v27_or_raw_features(features, "V4.3 text reranker")
    payload = joblib.load(artifact_path)
    model = V43TextRerankerModel(
        model_name=str(metadata.get("model_name") or V43_TEXT_MODEL_NAME),
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        metadata=metadata,
        payload=payload,
    )
    _V43_TEXT_MODEL_CACHE[cache_key] = model
    return model


def load_v4_9_guarded_text_blend_model(
    *,
    v33_model_dir: Path,
    v41_model_dir: Path,
    v43_model_dir: Path,
    v41_blend_config_path: Path,
    v49_candidate_config_path: Path,
) -> V49GuardedTextBlendModel:
    cache_key = (
        str(v33_model_dir.resolve()),
        str(v41_model_dir.resolve()),
        str(v43_model_dir.resolve()),
        str(v41_blend_config_path.resolve()),
        str(v49_candidate_config_path.resolve()),
    )
    if cache_key in _V49_BLEND_MODEL_CACHE:
        return _V49_BLEND_MODEL_CACHE[cache_key]
    if not v49_candidate_config_path.exists():
        raise ValueError(f"V4.9 guarded text candidate config was not found: {v49_candidate_config_path}")
    config = json.loads(v49_candidate_config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "v4.9_guarded_text_blend_candidate_config":
        raise ValueError(f"Unexpected V4.9 candidate config schema: {config.get('schema_version')}")
    if config.get("blend_name") != V49_BLEND_NAME:
        raise ValueError(f"Unexpected V4.9 blend name '{config.get('blend_name')}'. Expected '{V49_BLEND_NAME}'.")
    features_used = set(((config.get("formula") or {}).get("features_used") or []))
    allowed = {
        "normalized_v3_3_score",
        "normalized_v4_1_score",
        "normalized_v4_3_text_score",
        "max_ranker_confidence",
    }
    if features_used != allowed:
        raise ValueError(f"Unexpected V4.9 formula features: {sorted(features_used)}")
    if "0.675" not in str((config.get("formula") or {}).get("base_formula") or ""):
        raise ValueError("V4.9 candidate config formula does not contain the expected V3.3 weight.")
    if "0.10" not in str((config.get("formula") or {}).get("text_component") or ""):
        raise ValueError("V4.9 candidate config formula does not contain the expected text weight.")
    v41_blend = load_v4_1_blend_model(
        v33_model_dir=v33_model_dir,
        v41_model_dir=v41_model_dir,
        blend_config_path=v41_blend_config_path,
    )
    text_model = load_v4_3_text_reranker_model(v43_model_dir, V43_TEXT_SCORE_COLUMN)
    model = V49GuardedTextBlendModel(
        v41_blend_model=v41_blend,
        text_model=text_model,
        config_path=v49_candidate_config_path,
        config=config,
    )
    _V49_BLEND_MODEL_CACHE[cache_key] = model
    return model


def load_v6_4_safe_fusion_model(v64_candidate_config_path: Path) -> V64SafeFusionModel:
    cache_key = str(v64_candidate_config_path.resolve())
    if cache_key in _V64_SAFE_FUSION_MODEL_CACHE:
        return _V64_SAFE_FUSION_MODEL_CACHE[cache_key]
    if not v64_candidate_config_path.exists():
        raise ValueError(f"V6.4 safe fusion candidate config was not found: {v64_candidate_config_path}")

    config = json.loads(v64_candidate_config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "v6.4_safe_fusion_candidate":
        raise ValueError(f"Unexpected V6.4 candidate config schema: {config.get('schema_version')}")
    if config.get("method") != V64_BLEND_NAME:
        raise ValueError(f"Unexpected V6.4 method '{config.get('method')}'. Expected '{V64_BLEND_NAME}'.")
    if config.get("formula") != V64_BLEND_NAME:
        raise ValueError(f"Unexpected V6.4 formula '{config.get('formula')}'. Expected '{V64_BLEND_NAME}'.")
    if config.get("acceptance_passed") is not True:
        raise ValueError("V6.4 safe fusion candidate config was not accepted offline.")

    forbidden_runtime_features = {
        "relevance_label",
        "true_relevance_label",
        "relevance_bucket",
        "reading_value_score",
        "topic_match_score",
        "hard_negative",
        "hard_negative_label",
        "section_label",
        "difficulty_label",
        "generic_survey",
        "eval_outcomes",
        "known_hurt_topic",
        "known_improved_topic",
    }
    forbidden_features_used = set(config.get("forbidden_features_used") or [])
    if forbidden_features_used:
        raise ValueError(f"V6.4 candidate config uses forbidden features: {sorted(forbidden_features_used)}")
    inference_safe_features = set(config.get("inference_safe_features") or [])
    forbidden_in_safe_list = sorted(forbidden_runtime_features & inference_safe_features)
    if forbidden_in_safe_list:
        raise ValueError(f"V6.4 candidate config declares forbidden runtime features: {forbidden_in_safe_list}")

    model = V64SafeFusionModel(config_path=v64_candidate_config_path, config=config)
    _V64_SAFE_FUSION_MODEL_CACHE[cache_key] = model
    return model


def load_v6_6_safe_fusion_ridge_scorer(config_path: Path) -> V66SafeFusionRidgeScorer:
    cache_key = str(config_path.resolve())
    if cache_key in _V66_RIDGE_SCORER_CACHE:
        return _V66_RIDGE_SCORER_CACHE[cache_key]
    if not config_path.exists():
        raise ValueError(f"V6.6 safe fusion ridge scorer config was not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != V66_RIDGE_SCORER_SCHEMA:
        raise ValueError(f"Unexpected V6.6 ridge scorer schema: {config.get('schema_version')}")
    feature_names = list(config.get("feature_names") or [])
    coefficients = list(config.get("coefficients") or [])
    scaler = config.get("scaler") or {}
    mean = list(scaler.get("mean") or [])
    scale = list(scaler.get("scale") or [])
    if not feature_names:
        raise ValueError("V6.6 ridge scorer feature_names are empty.")
    if len(coefficients) != len(feature_names):
        raise ValueError("V6.6 ridge scorer coefficient count does not match feature_names.")
    if len(mean) != len(feature_names) or len(scale) != len(feature_names):
        raise ValueError("V6.6 ridge scorer scaler dimensions do not match feature_names.")
    if config.get("intercept") is None:
        raise ValueError("V6.6 ridge scorer intercept is missing.")
    forbidden = {
        "relevance_label",
        "true_relevance_label",
        "relevance_bucket",
        "reading_value_score",
        "topic_match_score",
        "hard_negative",
        "hard_negative_label",
        "section_label",
        "difficulty_label",
        "generic_survey",
        "eval_outcomes",
        "known_hurt_topic",
        "known_improved_topic",
        "topic_hurt_status",
    }
    forbidden_used = sorted(forbidden & set(feature_names))
    if forbidden_used:
        raise ValueError(f"V6.6 ridge scorer contains forbidden runtime features: {forbidden_used}")
    if config.get("forbidden_features_used"):
        raise ValueError(f"V6.6 ridge scorer declared forbidden features: {config.get('forbidden_features_used')}")
    model = V66SafeFusionRidgeScorer(config_path=config_path, config=config)
    _V66_RIDGE_SCORER_CACHE[cache_key] = model
    return model
