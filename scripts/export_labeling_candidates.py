import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402
from app.services.formatting import make_snippet, split_semicolon_field  # noqa: E402
from app.services.recommendation_service import recommend_from_query  # noqa: E402
from sqlalchemy import select  # noqa: E402


DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "query_set_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "eval" / "manual_label_pool_v1.jsonl"
DEFAULT_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
EXPECTED_SECTIONS = {"background", "foundational", "core_methods", "recent_frontier"}


@dataclass
class CandidateAccumulator:
    paper: Paper
    source_methods: list[str] = field(default_factory=list)
    retrieval_ranks_by_method: dict[str, int] = field(default_factory=dict)
    retrieval_scores_by_method: dict[str, float] = field(default_factory=dict)
    source_provenance: list[str] = field(default_factory=list)
    first_seen_index: int = 0


def load_query_set(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Query set was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Query set must be a JSON array.")

    queries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(payload, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Query record {index} must be an object.")
        query_id = str(record.get("query_id", "")).strip()
        query = str(record.get("query", "")).strip()
        target_audience = str(record.get("target_audience", "")).strip()
        expected_sections = record.get("expected_sections")
        if not query_id:
            raise ValueError(f"Query record {index} is missing query_id.")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id: {query_id}")
        if not query:
            raise ValueError(f"Query record {query_id} is missing query.")
        if not target_audience:
            raise ValueError(f"Query record {query_id} is missing target_audience.")
        if not isinstance(expected_sections, list) or set(expected_sections) != EXPECTED_SECTIONS:
            raise ValueError(f"Query record {query_id} must include all expected sections.")
        seen_ids.add(query_id)
        queries.append(
            {
                "query_id": query_id,
                "query": query,
                "target_audience": target_audience,
                "expected_sections": list(expected_sections),
            }
        )
    return queries


def parse_methods(value: str) -> list[str]:
    methods = [method.strip().lower() for method in value.split(",") if method.strip()]
    if not methods:
        raise ValueError("At least one retrieval method is required.")
    return list(dict.fromkeys(methods))


def collect_labeling_candidates(
    *,
    queries: list[dict[str, Any]],
    methods: list[str],
    top_k: int,
    retrieve: Callable[[str, str, int], list[Any]],
    get_paper: Callable[[int], Paper | None],
    get_source_provenance: Callable[[Paper], list[str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_record in queries:
        candidates: dict[int, CandidateAccumulator] = {}
        first_seen = 0
        for method in methods:
            results = retrieve(query_record["query"], method, top_k)
            for rank, result in enumerate(results[:top_k], start=1):
                paper_id = int(getattr(result, "paper_id", getattr(result, "document_id", 0)))
                if paper_id <= 0:
                    continue
                paper = get_paper(paper_id)
                if paper is None:
                    continue
                if paper_id not in candidates:
                    first_seen += 1
                    candidates[paper_id] = CandidateAccumulator(
                        paper=paper,
                        source_provenance=(
                            get_source_provenance(paper) if get_source_provenance else paper_source_provenance(paper)
                        ),
                        first_seen_index=first_seen,
                    )
                accumulator = candidates[paper_id]
                if method not in accumulator.source_methods:
                    accumulator.source_methods.append(method)
                accumulator.retrieval_ranks_by_method[method] = rank
                accumulator.retrieval_scores_by_method[method] = float(getattr(result, "score", 0.0) or 0.0)

        ordered = sorted(
            candidates.values(),
            key=lambda item: (min(item.retrieval_ranks_by_method.values()), item.first_seen_index, item.paper.id),
        )
        for item in ordered:
            rows.append(candidate_to_row(query_record, item))
    return rows


def candidate_to_row(query_record: dict[str, Any], candidate: CandidateAccumulator) -> dict[str, Any]:
    paper = candidate.paper
    source_provenance = candidate.source_provenance or paper_source_provenance(paper)
    ranks = list(candidate.retrieval_ranks_by_method.values())
    scores = candidate.retrieval_scores_by_method
    return {
        "query_id": query_record["query_id"],
        "query": query_record["query"],
        "target_audience": query_record["target_audience"],
        "paper_id": paper.id,
        "title": paper.title,
        "abstract": paper.abstract,
        "abstract_snippet": make_snippet(paper.abstract),
        "authors": split_semicolon_field(paper.authors),
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "external_id": paper.external_id,
        "citation_count": paper.citation_count,
        "source_methods": candidate.source_methods,
        "candidate_source": [f"local_{method}" for method in candidate.source_methods],
        "source_provenance": source_provenance,
        "retrieval_ranks_by_method": candidate.retrieval_ranks_by_method,
        "retrieval_scores_by_method": candidate.retrieval_scores_by_method,
        "raw_bm25_score": scores.get("bm25"),
        "raw_tfidf_score": scores.get("tfidf"),
        "embedding_similarity": scores.get("embedding"),
        "faiss_similarity": scores.get("faiss_embedding"),
        "hybrid_score": scores.get("hybrid"),
        "appears_in_n_methods": len(candidate.source_methods),
        "best_rank": min(ranks) if ranks else None,
        "mean_rank": sum(ranks) / len(ranks) if ranks else None,
        "has_arxiv_full_text": has_arxiv_full_text(paper, source_provenance),
        "has_openalex_metadata": "openalex" in source_provenance,
    }


def paper_source_provenance(paper: Paper) -> list[str]:
    values = " ".join(
        str(value or "")
        for value in (paper.source, paper.external_id, paper.source_url, paper.url, paper.doi)
    ).lower()
    sources: list[str] = []
    if "arxiv" in values:
        sources.append("arxiv")
    if "openalex" in values:
        sources.append("openalex")
    if paper.source and paper.source.lower() not in sources:
        sources.append(paper.source.lower())
    return sources or ["unknown"]


def has_arxiv_full_text(paper: Paper, source_provenance: list[str]) -> bool:
    if "arxiv" in source_provenance:
        return True
    values = " ".join(str(value or "") for value in (paper.external_id, paper.source_url, paper.url, paper.doi)).lower()
    return "arxiv" in values or "10.48550/arxiv" in values


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def export_labeling_candidates(
    *,
    queries_path: Path = DEFAULT_QUERIES,
    output_path: Path = DEFAULT_OUTPUT,
    methods: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    queries = load_query_set(queries_path)
    selected_methods = methods or DEFAULT_METHODS
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)

    skipped_methods: dict[str, str] = {}
    with SessionLocal() as db:
        paper_cache: dict[int, Paper | None] = {}

        def get_paper(paper_id: int) -> Paper | None:
            if paper_id not in paper_cache:
                paper_cache[paper_id] = db.get(Paper, paper_id)
            return paper_cache[paper_id]

        source_cache: dict[int, list[str]] = {}

        def get_source_provenance(paper: Paper) -> list[str]:
            if paper.id in source_cache:
                return source_cache[paper.id]
            sources = set(paper_source_provenance(paper))
            for value in db.scalars(select(PaperIdentifier.source).where(PaperIdentifier.paper_id == paper.id)).all():
                if value:
                    sources.add(str(value).lower())
            for value in db.scalars(select(PaperSource.source).where(PaperSource.paper_id == paper.id)).all():
                if value:
                    sources.add(str(value).lower())
            if "doi" in sources and len(sources) > 1:
                sources.remove("doi")
            source_cache[paper.id] = sorted(sources) if sources else ["unknown"]
            return source_cache[paper.id]

        def retrieve(query: str, method: str, k: int) -> list[Any]:
            if method in skipped_methods:
                return []
            try:
                return recommend_from_query(db=db, query=query, k=k, method=method)
            except Exception as exc:
                skipped_methods[method] = str(exc)
                print(f"Skipping method {method}: {exc}", file=sys.stderr)
                return []

        rows = collect_labeling_candidates(
            queries=queries,
            methods=selected_methods,
            top_k=top_k,
            retrieve=retrieve,
            get_paper=get_paper,
            get_source_provenance=get_source_provenance,
        )

    write_jsonl(rows, output_path)
    return {
        "query_count": len(queries),
        "row_count": len(rows),
        "methods": selected_methods,
        "skipped_methods": skipped_methods,
        "output_path": str(output_path),
    }


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pooled retrieval candidates for manual labeling.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    report = export_labeling_candidates(
        queries_path=resolve_repo_path(args.queries),
        output_path=resolve_repo_path(args.out),
        methods=parse_methods(args.methods),
        top_k=args.top_k,
    )
    print(f"Queries: {report['query_count']}")
    print(f"Candidate rows: {report['row_count']}")
    print(f"Methods: {', '.join(report['methods'])}")
    if report["skipped_methods"]:
        print(f"Skipped methods: {report['skipped_methods']}")
    print(f"Output: {report['output_path']}")


if __name__ == "__main__":
    main()
