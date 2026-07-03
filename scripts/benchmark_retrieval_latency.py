import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_JSON = REPO_ROOT / "data" / "processed" / "reports" / "retrieval_latency_v2_50k.json"
DEFAULT_MD = REPO_ROOT / "data" / "processed" / "reports" / "retrieval_latency_v2_50k.md"
DEFAULT_QUERIES = [
    "transformer architecture",
    "retrieval augmented generation",
    "graph neural networks",
    "contrastive learning",
    "Bayesian optimization",
    "large language model agents",
    "recommendation systems",
    "diffusion image generation",
]
DEFAULT_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services import recommendation_service  # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def time_call(db: Any, *, query: str, method: str, k: int) -> tuple[float, int]:
    start = time.perf_counter()
    results = recommendation_service.recommend_from_query(db=db, query=query, k=k, method=method)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, len(results)


def benchmark(*, queries: list[str], methods: list[str], k: int, warm_repeats: int) -> dict[str, Any]:
    settings = get_settings()
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "queries": queries,
        "methods": {},
        "config": {
            "embedding_index_path": settings.embedding_index_path,
            "faiss_index_path": settings.faiss_index_path,
            "faiss_id_map_path": settings.faiss_id_map_path,
        },
    }
    with SessionLocal() as db:
        for method in methods:
            recommendation_service._CORPUS_CACHE.clear()
            recommendation_service._RETRIEVER_CACHE.clear()
            cold_ms, cold_count = time_call(db, query=queries[0], method=method, k=k)

            # Warm measurements reuse the same process-local corpus/retriever cache.
            warm_samples: list[dict[str, Any]] = []
            for _ in range(warm_repeats):
                for query in queries:
                    elapsed_ms, result_count = time_call(db, query=query, method=method, k=k)
                    warm_samples.append(
                        {"query": query, "latency_ms": round(elapsed_ms, 3), "result_count": result_count}
                    )
            latencies = [item["latency_ms"] for item in warm_samples]
            report["methods"][method] = {
                "cold_first_query_ms": round(cold_ms, 3),
                "cold_first_query_result_count": cold_count,
                "warm_avg_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
                "warm_median_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
                "warm_p95_ms": round(percentile(latencies, 95), 3),
                "warm_min_ms": round(min(latencies), 3) if latencies else 0.0,
                "warm_max_ms": round(max(latencies), 3) if latencies else 0.0,
                "samples": warm_samples,
            }
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Retrieval Latency V2 50k",
        "",
        f"- Created at: `{report['created_at']}`",
        f"- Embedding index: `{report['config']['embedding_index_path']}`",
        f"- FAISS index: `{report['config']['faiss_index_path']}`",
        "",
        "| Method | Cold first query ms | Warm avg ms | Warm median ms | Warm p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in report["methods"].items():
        lines.append(
            f"| {method} | {metrics['cold_first_query_ms']:.3f} | {metrics['warm_avg_ms']:.3f} | "
            f"{metrics['warm_median_ms']:.3f} | {metrics['warm_p95_ms']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark warm/cold retrieval latency.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--queries", default=None, help="Optional JSON file containing a string array of queries.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warm-repeats", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = DEFAULT_QUERIES
    if args.queries:
        query_path = Path(args.queries)
        if not query_path.is_absolute():
            query_path = REPO_ROOT / query_path
        queries = json.loads(query_path.read_text(encoding="utf-8"))
    report = benchmark(queries=queries, methods=parse_csv(args.methods), k=args.k, warm_repeats=args.warm_repeats)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    if not md_path.is_absolute():
        md_path = REPO_ROOT / md_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({method: report["methods"][method] for method in report["methods"]}, indent=2)[:4000])
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
