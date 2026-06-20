import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.services.ingestion import ingest_papers_from_file  # noqa: E402
from scripts.evaluate_retrieval import evaluate_retrievers, load_papers, print_comparison_table  # noqa: E402
from scripts.expand_openalex_references import (  # noqa: E402
    collect_missing_reference_counts,
    expand_reference_records,
    fetch_openalex_work,
    select_reference_ids,
)
from scripts.fetch_openalex_papers import (  # noqa: E402
    DEFAULT_SELECT_FIELDS,
    build_filter,
    fetch_openalex_records,
    write_fetch_metadata,
    write_jsonl,
)
from scripts.generate_weak_labels import generate_examples, load_graph, write_examples  # noqa: E402


DEFAULT_DATASET_NAME = "openalex_cs_ml"
DEFAULT_QUERY = "machine learning artificial intelligence"
RAW_DIR = REPO_ROOT / "data" / "raw"
WEAK_LABEL_DIR = REPO_ROOT / "data" / "processed" / "evaluation_examples"
EVALUATION_DIR = REPO_ROOT / "data" / "processed" / "evaluations"


def dataset_paths(dataset_name: str) -> dict[str, Path]:
    return {
        "seed": RAW_DIR / f"{dataset_name}.jsonl",
        "references": RAW_DIR / f"{dataset_name}_references.jsonl",
        "weak_labels": WEAK_LABEL_DIR / f"{dataset_name}_weak_labels.jsonl",
        "run_summary": REPO_ROOT / "data" / "processed" / "manifests" / f"{dataset_name}_pipeline_summary.json",
    }


def fetch_seed_records(args: argparse.Namespace, seed_path: Path) -> list[dict[str, Any]]:
    filter_value = build_filter(
        from_year=args.from_year,
        to_year=args.to_year,
        work_type=args.work_type,
        min_citations=args.min_citations,
        topic_ids=args.topic_id,
        extra_filter=args.extra_filter,
    )
    records = fetch_openalex_records(
        query=args.query,
        max_results=args.max_results,
        per_page=args.per_page,
        filter_value=filter_value,
        sort=args.sort,
        api_key=args.api_key,
        email=args.email,
        sleep_seconds=args.sleep_seconds,
    )
    write_jsonl(records, seed_path)
    write_fetch_metadata(
        seed_path,
        {
            "dataset_name": args.dataset_name,
            "source": "openalex",
            "date_created": datetime.now(UTC).isoformat(),
            "query": args.query,
            "filter": filter_value,
            "sort": args.sort,
            "max_results": args.max_results,
            "per_page": args.per_page,
            "records_written": len(records),
            "select": DEFAULT_SELECT_FIELDS,
            "notes": "Seed OpenAlex fetch for a larger ResearchPath retrieval dataset.",
        },
    )
    return records


def fetch_reference_records(
    *,
    seed_records: list[dict[str, Any]],
    max_references: int,
    min_reference_frequency: int,
    api_key: str | None,
    email: str | None,
    sleep_seconds: float,
    reference_path: Path,
) -> tuple[list[dict[str, Any]], list[str], int, int]:
    missing_counts = collect_missing_reference_counts(seed_records)
    reference_ids = select_reference_ids(
        missing_counts,
        max_references=max_references,
        min_frequency=min_reference_frequency,
    )

    def fetcher(reference_id: str) -> dict[str, Any] | None:
        work = fetch_openalex_work(reference_id, api_key=api_key, email=email)
        time.sleep(sleep_seconds)
        return work

    records, warnings = expand_reference_records(reference_ids, fetcher)
    write_jsonl(records, reference_path)
    write_fetch_metadata(
        reference_path,
        {
            "dataset_name": reference_path.stem,
            "source": "openalex_reference_expansion",
            "date_created": datetime.now(UTC).isoformat(),
            "seed_records": len(seed_records),
            "candidate_missing_references": len(missing_counts),
            "selected_references": len(reference_ids),
            "records_written": len(records),
            "min_reference_frequency": min_reference_frequency,
            "max_references": max_references,
            "sleep_seconds": sleep_seconds,
            "warnings": warnings,
            "notes": "Reference expansion for weak citation labels.",
        },
    )
    return records, warnings, len(missing_counts), len(reference_ids)


def ingest_file(path: Path, *, dataset_name: str, source: str, notes: str) -> dict[str, Any]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    with SessionLocal() as db:
        return ingest_papers_from_file(
            path,
            db,
            dataset_name=dataset_name,
            source=source,
            notes=notes,
        )


def write_pipeline_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def compact_ingestion_result(result: dict[str, Any], *, max_warnings: int = 50) -> dict[str, Any]:
    warnings = result.get("warnings", [])
    return {
        "inserted": result.get("inserted", 0),
        "skipped": result.get("skipped", 0),
        "citation_edges_inserted": result.get("citation_edges_inserted", 0),
        "errors": result.get("errors", []),
        "warnings_count": len(warnings),
        "warnings_preview": warnings[:max_warnings],
        "manifest_path": result.get("manifest_path"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a larger OpenAlex dataset and optional weak-label evaluation set."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-results", type=int, default=2000)
    parser.add_argument("--max-references", type=int, default=500)
    parser.add_argument("--min-reference-frequency", type=int, default=1)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--from-year", type=int, default=2020)
    parser.add_argument("--to-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--type", default="article", dest="work_type")
    parser.add_argument("--min-citations", type=int, default=0)
    parser.add_argument("--topic-id", action="append", default=[])
    parser.add_argument("--extra-filter", default=None)
    parser.add_argument("--sort", default="cited_by_count:desc")
    parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument("--email", default=os.environ.get("OPENALEX_EMAIL"))
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--query-mode", choices=["title", "title_abstract", "goal"], default="title")
    parser.add_argument("--min-relevant", type=int, default=1)
    parser.add_argument("--max-relevant", type=int, default=20)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = dataset_paths(args.dataset_name)
    summary: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now(UTC).isoformat(),
        "query": args.query,
        "paths": {name: str(path) for name, path in paths.items()},
        "ingestion": {},
        "evaluation": None,
    }

    print(f"Building OpenAlex dataset: {args.dataset_name}")
    seed_records = fetch_seed_records(args, paths["seed"])
    print(f"Seed records fetched: {len(seed_records)}")

    reference_records, reference_warnings, missing_reference_count, selected_reference_count = fetch_reference_records(
        seed_records=seed_records,
        max_references=args.max_references,
        min_reference_frequency=args.min_reference_frequency,
        api_key=args.api_key,
        email=args.email,
        sleep_seconds=args.sleep_seconds,
        reference_path=paths["references"],
    )
    summary["reference_expansion"] = {
        "candidate_missing_references": missing_reference_count,
        "selected_references": selected_reference_count,
        "records_written": len(reference_records),
        "warnings_count": len(reference_warnings),
    }
    print(f"Reference records fetched: {len(reference_records)}")

    if not args.skip_ingest:
        seed_ingestion = ingest_file(
            paths["seed"],
            dataset_name=args.dataset_name,
            source="openalex",
            notes="Large OpenAlex seed dataset for ResearchPath retrieval evaluation.",
        )
        reference_ingestion = ingest_file(
            paths["references"],
            dataset_name=f"{args.dataset_name}_references",
            source="openalex_reference_expansion",
            notes="Fetched referenced papers to improve citation graph coverage.",
        )
        summary["ingestion"] = {
            "seed": compact_ingestion_result(seed_ingestion),
            "references": compact_ingestion_result(reference_ingestion),
        }
        print(
            "Ingested seed/reference records: "
            f"{seed_ingestion['inserted']}/{reference_ingestion['inserted']} inserted, "
            f"{seed_ingestion['citation_edges_inserted'] + reference_ingestion['citation_edges_inserted']} edges inserted"
        )

    if not args.skip_evaluation:
        papers, edges = load_graph()
        examples = generate_examples(
            papers=papers,
            edges=edges,
            query_mode=args.query_mode,
            bidirectional=True,
            min_relevant=args.min_relevant,
            max_relevant=args.max_relevant,
            max_examples=args.max_eval_examples,
        )
        write_examples(examples, paths["weak_labels"])
        report = evaluate_retrievers(
            examples=examples,
            papers=load_papers(),
            output_dir=EVALUATION_DIR,
        )
        summary["evaluation"] = {
            "weak_label_examples": len(examples),
            "report_path": report["output_path"],
            "paper_count": report["paper_count"],
            "query_count": report["query_count"],
            "methods": {
                method: payload["averages"]
                for method, payload in report["methods"].items()
            },
        }
        print_comparison_table(report)

    write_pipeline_summary(paths["run_summary"], summary)
    print(f"Pipeline summary: {paths['run_summary']}")


if __name__ == "__main__":
    main()
