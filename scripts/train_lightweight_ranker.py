import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.learned_ranker import DEFAULT_FEATURE_NAMES, LinearRanker, save_linear_ranker  # noqa: E402


DEFAULT_REVIEW = REPO_ROOT / "data" / "processed" / "evaluations" / "reading_path_manual_review.json"
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "models" / "lightweight_ranker.json"


def load_labels(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("query_id"):
            raise ValueError(f"Line {line_number}: missing query_id.")
        if record.get("paper_id") is None:
            raise ValueError(f"Line {line_number}: missing paper_id.")
        relevance = record.get("relevance_score")
        if relevance is None:
            continue
        if not isinstance(relevance, int) or relevance < 0 or relevance > 3:
            raise ValueError(f"Line {line_number}: relevance_score must be an integer from 0 to 3.")
        labels[(record["query_id"], int(record["paper_id"]))] = record
    return labels


def load_review_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Reading-path review file was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("queries", []))


def build_training_examples(
    *,
    review_rows: list[dict[str, Any]],
    labels: dict[tuple[str, int], dict[str, Any]],
    feature_names: list[str] = DEFAULT_FEATURE_NAMES,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in review_rows:
        query_id = row["query_id"]
        sections = row["reading_path"]["sections"]
        for section_name, papers in sections.items():
            for paper in papers:
                label = labels.get((query_id, int(paper["paper_id"])))
                if label is None:
                    continue
                features = [_feature_value(paper, feature_name) for feature_name in feature_names]
                examples.append(
                    {
                        "query_id": query_id,
                        "paper_id": int(paper["paper_id"]),
                        "section": section_name,
                        "features": features,
                        "target": float(label["relevance_score"]) / 3.0,
                    }
                )
    return examples


def train_linear_ranker(
    examples: list[dict[str, Any]],
    *,
    feature_names: list[str] = DEFAULT_FEATURE_NAMES,
    l2: float = 0.1,
    version: str | None = None,
) -> tuple[LinearRanker, dict[str, Any]]:
    if len(examples) < 2:
        raise ValueError("Need at least two labeled examples to train the lightweight ranker.")

    x = np.array([example["features"] for example in examples], dtype=np.float64)
    y = np.array([example["target"] for example in examples], dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    predictions = np.clip(design @ coefficients, 0.0, 1.0)
    intercept = float(coefficients[0])
    weights = [float(value) for value in coefficients[1:]]
    metrics = {
        "example_count": len(examples),
        "mae": float(np.mean(np.abs(predictions - y))),
        "rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
        "mean_target": float(np.mean(y)),
        "created_at": datetime.now(UTC).isoformat(),
    }
    ranker = LinearRanker(
        feature_names=list(feature_names),
        weights=weights,
        intercept=intercept,
        version=version or datetime.now(UTC).strftime("lightweight_%Y%m%d_%H%M%S"),
        metadata=metrics,
    )
    return ranker, metrics


def train_from_files(
    *,
    review_path: Path = DEFAULT_REVIEW,
    labels_path: Path = DEFAULT_LABELS,
    output_path: Path = DEFAULT_OUTPUT,
    l2: float = 0.1,
) -> dict[str, Any]:
    labels = load_labels(labels_path)
    review_rows = load_review_rows(review_path)
    examples = build_training_examples(review_rows=review_rows, labels=labels)
    ranker, metrics = train_linear_ranker(examples, l2=l2)
    save_linear_ranker(ranker, output_path)
    return {
        "output_path": str(output_path),
        "label_count": len(labels),
        "matched_examples": len(examples),
        "metrics": metrics,
        "feature_names": ranker.feature_names,
        "weights": ranker.weights,
        "intercept": ranker.intercept,
    }


def _feature_value(paper: dict[str, Any], feature_name: str) -> float:
    value = paper.get(feature_name, 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an explainable linear reranker from manual reading-path labels.")
    parser.add_argument("--review", default=str(DEFAULT_REVIEW))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--l2", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_from_files(
        review_path=_resolve_repo_path(args.review),
        labels_path=_resolve_repo_path(args.labels),
        output_path=_resolve_repo_path(args.output),
        l2=args.l2,
    )
    print(f"Lightweight ranker written to: {report['output_path']}")
    print(f"Matched labeled examples: {report['matched_examples']}")
    print(f"MAE: {report['metrics']['mae']:.4f}")
    print(f"RMSE: {report['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()
