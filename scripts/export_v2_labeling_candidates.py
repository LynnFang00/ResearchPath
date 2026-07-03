import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "query_set_v2_seed.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "eval" / "v2_labeling_candidate_pool.jsonl"
DEFAULT_META = DEFAULT_OUTPUT.with_suffix(DEFAULT_OUTPUT.suffix + ".meta.json")
DEFAULT_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
STOPWORDS = {"a", "an", "and", "for", "in", "large", "model", "models", "of", "the", "to", "with"}
sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402
from app.services.deduplication import normalize_title  # noqa: E402
from app.services.formatting import make_snippet, split_semicolon_field  # noqa: E402
from app.services.recommendation_service import recommend_from_query  # noqa: E402


@dataclass
class Candidate:
    paper: Paper
    retrieval_ranks_by_method: dict[str, int] = field(default_factory=dict)
    retrieval_scores_by_method: dict[str, float] = field(default_factory=dict)
    source_methods: list[str] = field(default_factory=list)
    candidate_source: set[str] = field(default_factory=set)
    first_seen: int = 0


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("V2 query file must be a JSON array.")
    return payload


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def query_terms(query: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", query) if token.lower() not in STOPWORDS}


def source_provenance(db: Any, paper_id: int, paper: Paper) -> list[str]:
    sources = {str(paper.source).lower()} if paper.source else set()
    values = " ".join(str(value or "") for value in (paper.external_id, paper.source_url, paper.url, paper.doi)).lower()
    if "arxiv" in values:
        sources.add("arxiv")
    if "openalex" in values:
        sources.add("openalex")
    sources.update(
        str(value).lower()
        for value in db.scalars(select(PaperIdentifier.source).where(PaperIdentifier.paper_id == paper_id)).all()
        if value
    )
    sources.update(
        str(value).lower()
        for value in db.scalars(select(PaperSource.source).where(PaperSource.paper_id == paper_id)).all()
        if value
    )
    if "doi" in sources and len(sources) > 1:
        sources.remove("doi")
    return sorted(sources) if sources else ["unknown"]


def identifiers(db: Any, paper_id: int) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    rows = db.scalars(
        select(PaperIdentifier).where(PaperIdentifier.paper_id == paper_id).order_by(PaperIdentifier.source)
    ).all()
    for row in rows:
        values.setdefault(row.source, []).append(row.identifier)
    return values


def duplicate_title_counts(db: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for title in db.scalars(select(Paper.title)).all():
        key = normalize_title(title)
        if key:
            counts[key] += 1
    return counts


def add_candidate(
    candidates: dict[int, Candidate],
    *,
    paper: Paper,
    candidate_source: str,
    first_seen_counter: list[int],
) -> Candidate:
    if paper.id not in candidates:
        first_seen_counter[0] += 1
        candidates[paper.id] = Candidate(paper=paper, first_seen=first_seen_counter[0])
    candidates[paper.id].candidate_source.add(candidate_source)
    return candidates[paper.id]


def add_random_weak_negatives(
    db: Any,
    candidates: dict[int, Candidate],
    *,
    query: str,
    limit: int,
    first_seen_counter: list[int],
    rng: random.Random,
) -> None:
    terms = query_terms(query)
    if not terms:
        return
    rows = db.scalars(select(Paper).order_by(Paper.id)).all()
    rng.shuffle(rows)
    for paper in rows:
        if len([candidate for candidate in candidates.values() if "random_weak_negative" in candidate.candidate_source]) >= limit:
            return
        if paper.id in candidates:
            continue
        text = f"{paper.title} {paper.abstract}".lower()
        if any(term in text for term in terms):
            add_candidate(candidates, paper=paper, candidate_source="random_weak_negative", first_seen_counter=first_seen_counter)


def candidate_to_row(
    db: Any,
    *,
    query_record: dict[str, Any],
    candidate: Candidate,
    title_counts: Counter[str],
) -> dict[str, Any]:
    paper = candidate.paper
    ranks = list(candidate.retrieval_ranks_by_method.values())
    scores = candidate.retrieval_scores_by_method
    provenance = source_provenance(db, paper.id, paper)
    title_key = normalize_title(paper.title)
    return {
        "query_id": query_record["query_id"],
        "query": query_record["query"],
        "target_audience": query_record.get("target_audience"),
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
        "source_provenance": provenance,
        "identifiers": identifiers(db, paper.id),
        "retrieval_ranks_by_method": candidate.retrieval_ranks_by_method,
        "retrieval_scores_by_method": candidate.retrieval_scores_by_method,
        "source_methods": candidate.source_methods,
        "candidate_source": sorted(candidate.candidate_source),
        "raw_bm25_score": scores.get("bm25"),
        "raw_tfidf_score": scores.get("tfidf"),
        "embedding_similarity": scores.get("embedding"),
        "faiss_similarity": scores.get("faiss_embedding"),
        "hybrid_score": scores.get("hybrid"),
        "appears_in_n_methods": len(candidate.source_methods),
        "best_rank": min(ranks) if ranks else None,
        "mean_rank": sum(ranks) / len(ranks) if ranks else None,
        "duplicate_title_key": title_key,
        "duplicate_title_cluster_count": title_counts.get(title_key, 0),
        "evidence_availability": {
            "has_arxiv_source": "arxiv" in provenance,
            "has_openalex_source": "openalex" in provenance,
        },
    }


def export_candidates(
    *,
    queries_path: Path,
    output_path: Path,
    methods: list[str],
    top_k: int,
    random_negatives_per_query: int,
    seed: int,
) -> dict[str, Any]:
    queries = load_queries(queries_path)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "queries_path": str(queries_path),
        "output_path": str(output_path),
        "methods": methods,
        "top_k": top_k,
        "random_negatives_per_query": random_negatives_per_query,
        "query_count": len(queries),
        "rows_by_query": {},
        "notes": [
            "This is a pre-labeling candidate pool, not final labels.",
            "Candidate source tags intentionally mix retrieval outputs, source strata, canonical seeds, and weak negatives.",
        ],
    }
    with SessionLocal() as db:
        title_counts = duplicate_title_counts(db)
        paper_cache: dict[int, Paper | None] = {}

        def get_paper(paper_id: int) -> Paper | None:
            if paper_id not in paper_cache:
                paper_cache[paper_id] = db.get(Paper, paper_id)
            return paper_cache[paper_id]

        for query_record in queries:
            candidates: dict[int, Candidate] = {}
            first_seen_counter = [0]
            for method in methods:
                results = recommend_from_query(db=db, query=query_record["query"], k=top_k, method=method)
                hybrid_top_ids = {
                    result.paper_id
                    for result in results[:10]
                } if method == "hybrid" else set()
                for rank, result in enumerate(results, start=1):
                    paper = get_paper(result.paper_id)
                    if paper is None:
                        continue
                    candidate = add_candidate(
                        candidates,
                        paper=paper,
                        candidate_source=f"internal_{method}_top" if rank <= 10 else f"internal_{method}_deeper",
                        first_seen_counter=first_seen_counter,
                    )
                    if method == "bm25" and rank > 10 and paper.id not in hybrid_top_ids:
                        candidate.candidate_source.add("hard_negative_candidate")
                    if method not in candidate.source_methods:
                        candidate.source_methods.append(method)
                    candidate.retrieval_ranks_by_method[method] = rank
                    candidate.retrieval_scores_by_method[method] = float(result.score)

            for paper_id in query_record.get("canonical_paper_ids") or []:
                paper = get_paper(int(paper_id))
                if paper is not None:
                    add_candidate(
                        candidates,
                        paper=paper,
                        candidate_source="canonical_seed",
                        first_seen_counter=first_seen_counter,
                    )

            add_random_weak_negatives(
                db,
                candidates,
                query=query_record["query"],
                limit=random_negatives_per_query,
                first_seen_counter=first_seen_counter,
                rng=rng,
            )

            for candidate in candidates.values():
                provenance = source_provenance(db, candidate.paper.id, candidate.paper)
                if "arxiv" in provenance:
                    candidate.candidate_source.add("arxiv_source_candidate")
                if "openalex" in provenance:
                    candidate.candidate_source.add("openalex_source_candidate")

            ordered_candidates = sorted(
                candidates.values(),
                key=lambda item: (
                    min(item.retrieval_ranks_by_method.values()) if item.retrieval_ranks_by_method else 9999,
                    item.first_seen,
                    item.paper.id,
                ),
            )
            summary["rows_by_query"][query_record["query_id"]] = len(ordered_candidates)
            for candidate in ordered_candidates:
                rows.append(candidate_to_row(db, query_record=query_record, candidate=candidate, title_counts=title_counts))

    write_jsonl(rows, output_path)
    summary["row_count"] = len(rows)
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V2 pre-labeling candidate pool.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--random-negatives-per-query", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries_path = Path(args.queries)
    output_path = Path(args.out)
    if not queries_path.is_absolute():
        queries_path = REPO_ROOT / queries_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    summary = export_candidates(
        queries_path=queries_path,
        output_path=output_path,
        methods=parse_csv(args.methods),
        top_k=args.top_k,
        random_negatives_per_query=args.random_negatives_per_query,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
