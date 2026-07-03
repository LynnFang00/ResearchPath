from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.services.v2_2_learned_ranker import V22FeatureContext, V22RidgeRanker, load_v2_2_ridge_ranker


DEFAULT_FEATURE_NAMES = [
    "bm25_score",
    "tfidf_score",
    "faiss_score",
    "citation_score",
    "influence_score",
    "recency_score",
    "difficulty_fit_score",
    "background_signal",
    "method_signal",
    "narrow_application_score",
    "duplicate_penalty",
    "too_narrow_penalty",
    "personalization_score",
    "saved_similarity",
    "skipped_similarity",
    "too_hard_similarity",
]


@dataclass(frozen=True)
class LinearRanker:
    feature_names: list[str]
    weights: list[float]
    intercept: float = 0.0
    version: str = "manual"
    metadata: dict[str, Any] | None = None

    def score(self, components: dict[str, Any]) -> float:
        raw_score = self.intercept
        for feature_name, weight in zip(self.feature_names, self.weights, strict=False):
            raw_score += weight * _numeric(components.get(feature_name, 0.0))
        return max(0.0, min(1.0, raw_score))

    def diagnostics(self, components: dict[str, Any]) -> dict[str, float | str]:
        score = self.score(components)
        return {
            "learned_ranker_score": round(score, 6),
            "learned_ranker_version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "linear_relevance_reranker",
            "version": self.version,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "intercept": self.intercept,
            "metadata": self.metadata or {},
        }


def load_linear_ranker(path: Path | None) -> LinearRanker | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_names = [str(name) for name in payload.get("feature_names", DEFAULT_FEATURE_NAMES)]
    weights = [float(value) for value in payload.get("weights", [])]
    if len(weights) != len(feature_names):
        raise ValueError(f"Ranker model has {len(weights)} weights for {len(feature_names)} features: {path}")
    return LinearRanker(
        feature_names=feature_names,
        weights=weights,
        intercept=float(payload.get("intercept", 0.0)),
        version=str(payload.get("version", path.stem)),
        metadata=dict(payload.get("metadata", {})),
    )


def load_v2_2_learned_ranker(path: Path | None) -> V22RidgeRanker | None:
    """Load a V2.2 packet-feature Ridge ranker artifact.

    This is intentionally separate from ``load_linear_ranker`` because V2.2
    models require packet-style feature extraction plus StandardScaler state.
    The current production HybridRetriever still uses ``LinearRanker``. To
    deploy V2.2 in production, call ``score_v2_2_packet_row`` after constructing
    the same packet-like row/context used by the benchmark evaluator.
    """
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not str(payload.get("model_type", "")).startswith("v2_2"):
        return None
    return load_v2_2_ridge_ranker(path)


def score_v2_2_packet_row(
    ranker: V22RidgeRanker,
    row: dict[str, Any],
    context: V22FeatureContext,
) -> float:
    return ranker.score_packet_row(row, context)


def save_linear_ranker(ranker: LinearRanker, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ranker.to_dict(), indent=2), encoding="utf-8")


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
