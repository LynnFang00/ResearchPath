import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_JSON = REPO_ROOT / "data" / "processed" / "reports" / "retrieval_sanity_v2_50k.json"
DEFAULT_MD = REPO_ROOT / "data" / "processed" / "reports" / "retrieval_sanity_v2_50k.md"
DEFAULT_QUERIES = [
    {"query_id": "v2_transformer_architecture", "query": "transformer architecture"},
    {"query_id": "v2_retrieval_augmented_generation", "query": "retrieval augmented generation"},
    {"query_id": "v2_graph_neural_networks", "query": "graph neural networks"},
    {"query_id": "v2_contrastive_learning", "query": "contrastive learning"},
    {"query_id": "v2_bayesian_optimization", "query": "Bayesian optimization"},
    {"query_id": "v2_large_language_model_agents", "query": "large language model agents"},
    {"query_id": "v2_recommendation_systems", "query": "recommendation systems"},
    {"query_id": "v2_diffusion_image_generation", "query": "diffusion image generation"},
]
DEFAULT_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "large",
    "of",
    "the",
    "to",
    "with",
}
sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402
from app.services.deduplication import normalize_title  # noqa: E402
from app.services.recommendation_service import recommend_from_query  # noqa: E402


def query_terms(query: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", query) if token.lower() not in STOPWORDS}


def weak_term_overlap(query: str, paper: Paper) -> int:
    terms = query_terms(query)
    text = f"{paper.title} {paper.abstract}".lower()
    return sum(1 for term in terms if term in text)


def source_provenance(db: Any, paper: Paper) -> list[str]:
    sources = {str(paper.source).lower()} if paper.source else set()
    values = " ".join(str(value or "") for value in (paper.external_id, paper.source_url, paper.url, paper.doi)).lower()
    if "arxiv" in values:
        sources.add("arxiv")
    if "openalex" in values:
        sources.add("openalex")
    sources.update(
        str(value).lower()
        for value in db.scalars(select(PaperIdentifier.source).where(PaperIdentifier.paper_id == paper.id)).all()
        if value
    )
    sources.update(
        str(value).lower()
        for value in db.scalars(select(PaperSource.source).where(PaperSource.paper_id == paper.id)).all()
        if value
    )
    if "doi" in sources and len(sources) > 1:
        sources.remove("doi")
    return sorted(sources) if sources else ["unknown"]


def load_queries(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_QUERIES
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            queries.append({"query_id": f"query_{index + 1}", "query": item})
        else:
            queries.append({"query_id": str(item["query_id"]), "query": str(item["query"])})
    return queries


def build_report(*, queries: list[dict[str, str]], methods: list[str], k: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "k": k,
        "methods": methods,
        "queries": {},
    }
    with SessionLocal() as db:
        paper_cache: dict[int, Paper] = {}
        for query_record in queries:
            query = query_record["query"]
            query_payload: dict[str, Any] = {"query": query, "methods": {}}
            for method in methods:
                results = recommend_from_query(db=db, query=query, k=k, method=method)
                rows: list[dict[str, Any]] = []
                title_counts: Counter[str] = Counter()
                source_counts: Counter[str] = Counter()
                weak_overlap_zero = 0
                for rank, result in enumerate(results, start=1):
                    paper = paper_cache.get(result.paper_id)
                    if paper is None:
                        paper = db.get(Paper, result.paper_id)
                        if paper is None:
                            continue
                        paper_cache[result.paper_id] = paper
                    title_key = normalize_title(paper.title)
                    title_counts[title_key] += 1
                    provenance = source_provenance(db, paper)
                    source_counts.update(provenance)
                    overlap = weak_term_overlap(query, paper)
                    if overlap == 0:
                        weak_overlap_zero += 1
                    rows.append(
                        {
                            "rank": rank,
                            "paper_id": paper.id,
                            "title": paper.title,
                            "year": paper.year,
                            "score": result.score,
                            "source_provenance": provenance,
                            "weak_query_term_overlap": overlap,
                        }
                    )
                query_payload["methods"][method] = {
                    "results": rows,
                    "source_counts": dict(source_counts),
                    "duplicate_titles_in_top_k": [
                        {"normalized_title": title, "count": count}
                        for title, count in title_counts.items()
                        if count > 1
                    ],
                    "weak_overlap_zero_count": weak_overlap_zero,
                    "manual_review_needed": weak_overlap_zero > 0 or any(count > 1 for count in title_counts.values()),
                }
            report["queries"][query_record["query_id"]] = query_payload
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Retrieval Sanity V2 50k", "", f"- Created at: `{report['created_at']}`", ""]
    for query_id, query_payload in report["queries"].items():
        lines.extend([f"## {query_id}", "", f"Query: `{query_payload['query']}`", ""])
        for method, method_payload in query_payload["methods"].items():
            lines.extend(
                [
                    f"### {method}",
                    "",
                    f"- Source counts: `{method_payload['source_counts']}`",
                    f"- Duplicate titles in top {report['k']}: `{len(method_payload['duplicate_titles_in_top_k'])}`",
                    f"- Weak-overlap zero count: `{method_payload['weak_overlap_zero_count']}`",
                    "",
                ]
            )
            for row in method_payload["results"]:
                lines.append(f"{row['rank']}. `{row['paper_id']}` {row['title']} ({row['year']})")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export qualitative retrieval sanity report.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--queries", default=None)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = Path(args.queries) if args.queries else None
    if queries_path is not None and not queries_path.is_absolute():
        queries_path = REPO_ROOT / queries_path
    report = build_report(queries=load_queries(queries_path), methods=parse_csv(args.methods), k=args.k)
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    if not md_path.is_absolute():
        md_path = REPO_ROOT / md_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
