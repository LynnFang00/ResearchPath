import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.services.recommendation_service import recommend_reading_path  # noqa: E402


DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "manual_queries.jsonl"
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "evaluations" / "reading_path_manual_review.json"


def load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manual query file was not found: {path}")
    queries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("query_id"):
            raise ValueError(f"Line {line_number}: missing query_id.")
        if not record.get("query"):
            raise ValueError(f"Line {line_number}: missing query.")
        queries.append(record)
    return queries


def load_labels(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    labels: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("query_id"):
            raise ValueError(f"Line {line_number}: missing query_id.")
        if record.get("paper_id") is None:
            raise ValueError(f"Line {line_number}: missing paper_id.")
        relevance = record.get("relevance_score")
        if relevance is not None and (not isinstance(relevance, int) or relevance < 0 or relevance > 3):
            raise ValueError(f"Line {line_number}: relevance_score must be an integer from 0 to 3.")
        for field_name in ("section_correct", "duplicate", "too_advanced", "too_narrow"):
            if field_name in record and record[field_name] is not None and not isinstance(record[field_name], bool):
                raise ValueError(f"Line {line_number}: {field_name} must be true, false, or null.")
        labels.append(record)
    return labels


def _path_to_review_payload(path_response) -> dict[str, Any]:
    payload = path_response.model_dump()
    return {
        "query": payload["query"],
        "method": payload["method"],
        "background_level": payload["background_level"],
        "sections": payload["sections"],
        "manual_scores": {
            "overall": None,
            "background": None,
            "foundational": None,
            "core_methods": None,
            "recent_frontier": None,
            "notes": "",
        },
    }


def evaluate_reading_paths(
    *,
    queries: list[dict[str, Any]],
    labels: list[dict[str, Any]] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    method: str = "hybrid",
    k: int = 4,
    candidate_k: int = 40,
) -> dict[str, Any]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)

    rows: list[dict[str, Any]] = []
    labels = labels or []
    label_lookup = {(label["query_id"], int(label["paper_id"])): label for label in labels}
    with SessionLocal() as db:
        for query in queries:
            path = recommend_reading_path(
                db=db,
                query=query["query"],
                k=k,
                method=query.get("method", method),
                background_level=query.get("background_level", "basic_ml"),
                candidate_k=int(query.get("candidate_k", candidate_k)),
            )
            rows.append(
                {
                    "query_id": query["query_id"],
                    "input": query,
                    "reading_path": _path_to_review_payload(path),
                    "labels": [
                        label
                        for (query_id, _), label in label_lookup.items()
                        if query_id == query["query_id"]
                    ],
                }
            )

    metrics = compute_metrics(rows=rows, labels=labels)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": {
            "name": "reading_path_manual_eval",
            "ranking_method": method,
            "label_source": str(DEFAULT_LABELS),
            "query_source": str(DEFAULT_QUERIES),
            "model_version": method,
        },
        "query_count": len(rows),
        "label_count": len(labels),
        "default_method": method,
        "papers_per_section": k,
        "candidate_k": candidate_k,
        "review_schema_version": 2,
        "metrics": metrics,
        "queries": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def compute_metrics(*, rows: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    average_score_by_section = _average_score_by_section(rows)
    if not labels:
        return {
            "average_relevance": None,
            "section_accuracy": None,
            "duplicate_rate": None,
            "too_narrow_rate": None,
            "too_advanced_rate": None,
            "labeled_precision_at_5": None,
            "labeled_recall_at_10": None,
            "labeled_ndcg_at_10": None,
            "average_score_by_section": average_score_by_section,
        }

    return {
        "average_relevance": _mean(
            label["relevance_score"] for label in labels if label.get("relevance_score") is not None
        ),
        "section_accuracy": _boolean_rate(labels, "section_correct"),
        "duplicate_rate": _boolean_rate(labels, "duplicate"),
        "too_narrow_rate": _boolean_rate(labels, "too_narrow"),
        "too_advanced_rate": _boolean_rate(labels, "too_advanced"),
        **_ranking_metrics(rows=rows, labels=labels),
        "average_score_by_section": average_score_by_section,
    }


def _average_score_by_section(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores_by_section: dict[str, list[float]] = {}
    for row in rows:
        sections = row["reading_path"]["sections"]
        for section, papers in sections.items():
            scores_by_section.setdefault(section, [])
            for paper in papers:
                score = paper.get("final_path_score", paper.get("score"))
                if score is not None:
                    scores_by_section[section].append(float(score))
    return {section: _mean(scores) for section, scores in scores_by_section.items()}


def _ranking_metrics(*, rows: list[dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, float | None]:
    labels_by_query: dict[str, dict[int, int]] = {}
    for label in labels:
        relevance = label.get("relevance_score")
        if relevance is None:
            continue
        labels_by_query.setdefault(label["query_id"], {})[int(label["paper_id"])] = int(relevance)

    precision_at_5: list[float] = []
    recall_at_10: list[float] = []
    ndcg_at_10: list[float] = []
    for row in rows:
        query_id = row.get("query_id")
        if query_id is None:
            continue
        query_labels = labels_by_query.get(query_id, {})
        if not query_labels:
            continue
        retrieved = _flatten_retrieved_paper_ids(row)
        relevant_ids = {paper_id for paper_id, relevance in query_labels.items() if relevance >= 2}
        if not relevant_ids:
            continue
        top_5 = retrieved[:5]
        top_10 = retrieved[:10]
        precision_at_5.append(sum(1 for paper_id in top_5 if query_labels.get(paper_id, 0) >= 2) / max(len(top_5), 1))
        recall_at_10.append(sum(1 for paper_id in top_10 if paper_id in relevant_ids) / len(relevant_ids))
        ndcg_at_10.append(_ndcg(top_10, query_labels, k=10))

    return {
        "labeled_precision_at_5": _mean(precision_at_5),
        "labeled_recall_at_10": _mean(recall_at_10),
        "labeled_ndcg_at_10": _mean(ndcg_at_10),
    }


def _flatten_retrieved_paper_ids(row: dict[str, Any]) -> list[int]:
    paper_ids: list[int] = []
    sections = row["reading_path"]["sections"]
    for section_name in ("background", "foundational", "core_methods", "recent_frontier"):
        for paper in sections.get(section_name, []):
            paper_ids.append(int(paper["paper_id"]))
    return paper_ids


def _ndcg(retrieved: list[int], labels: dict[int, int], *, k: int) -> float:
    gains = [labels.get(paper_id, 0) for paper_id in retrieved[:k]]
    dcg = sum((2**gain - 1) / math_log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(labels.values(), reverse=True)[:k]
    ideal = sum((2**gain - 1) / math_log2(index + 2) for index, gain in enumerate(ideal_gains))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def math_log2(value: float) -> float:
    import math

    return math.log2(value)


def _boolean_rate(labels: list[dict[str, Any]], field_name: str) -> float | None:
    values = [label[field_name] for label in labels if label.get(field_name) is not None]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _mean(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reading paths for manual quality review.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--method", default="hybrid")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_path = Path(args.queries)
    if not query_path.is_absolute():
        query_path = REPO_ROOT / query_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    label_path = Path(args.labels) if args.labels else None
    if label_path is not None and not label_path.is_absolute():
        label_path = REPO_ROOT / label_path

    queries = load_queries(query_path)
    labels = load_labels(label_path)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    report = evaluate_reading_paths(
        queries=queries,
        labels=labels,
        output_path=output_path,
        method=args.method,
        k=args.k,
        candidate_k=args.candidate_k,
    )
    print(f"Manual reading-path review file: {report['output_path']}")
    print(f"Queries: {report['query_count']}")
    print(f"Labels: {report['label_count']}")


if __name__ == "__main__":
    main()
