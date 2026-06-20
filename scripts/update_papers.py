import argparse
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.models.citation_edge import CitationEdge  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.services.dataset_manifest import (  # noqa: E402
    build_dataset_manifest,
    latest_manifest_path,
    read_manifest,
    write_dataset_manifest,
)
from app.services.ingestion import ingest_papers_from_file  # noqa: E402
from sqlalchemy import func, select  # noqa: E402


INCREMENTAL_DIR = REPO_ROOT / "data" / "processed" / "incremental_updates"


class PaperFetcher(ABC):
    @abstractmethod
    def fetch_newer_than(self, last_updated: datetime | None) -> list[dict[str, Any]]:
        raise NotImplementedError


class LocalJsonlFetcher(PaperFetcher):
    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch_newer_than(self, last_updated: datetime | None) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"Local update file was not found: {self.path}")

        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if _is_newer_record(record, last_updated):
                records.append(record)
            elif last_updated is not None and not _record_timestamp(record):
                print(f"Warning: line {line_number} skipped because it has no updated_at timestamp.")
        return records


def _record_timestamp(record: dict[str, Any]) -> datetime | None:
    value = record.get("updated_at") or record.get("updated_date") or record.get("published_at")
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _manifest_timestamp(manifest: dict[str, Any] | None) -> datetime | None:
    if not manifest:
        return None
    value = (
        manifest.get("last_updated_timestamp")
        or manifest.get("last_updated")
        or manifest.get("date_created")
        or manifest.get("created_at")
    )
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _is_newer_record(record: dict[str, Any], last_updated: datetime | None) -> bool:
    if last_updated is None:
        return True
    updated_at = _record_timestamp(record)
    return updated_at is not None and updated_at > last_updated


def _write_incremental_file(records: list[dict[str, Any]], dataset_name: str) -> Path:
    INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = INCREMENTAL_DIR / f"{dataset_name}_{timestamp}.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def _counts() -> tuple[int, int]:
    with SessionLocal() as db:
        paper_count = int(db.scalar(select(func.count()).select_from(Paper)) or 0)
        edge_count = int(db.scalar(select(func.count()).select_from(CitationEdge)) or 0)
    return paper_count, edge_count


def _build_fetcher(args: argparse.Namespace) -> PaperFetcher:
    if args.provider == "local_jsonl":
        if args.file is None:
            raise ValueError("--file is required for provider=local_jsonl.")
        path = Path(args.file)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return LocalJsonlFetcher(path)
    raise ValueError(
        f"Provider '{args.provider}' is not implemented yet. Use provider=local_jsonl for this milestone."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally update ResearchPath papers.")
    parser.add_argument("--provider", default="local_jsonl", choices=["local_jsonl", "openalex", "arxiv"])
    parser.add_argument("--file", default=None, help="Local JSONL file containing new paper records.")
    parser.add_argument("--dataset-name", default="researchpath_incremental")
    parser.add_argument("--source", default="local_incremental")
    parser.add_argument("--notes", default="Incremental paper update.")
    parser.add_argument("--keep-empty-manifest", action="store_true", help="Write a manifest even if no records are new.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)

    manifest_path = latest_manifest_path()
    manifest = read_manifest(manifest_path) if manifest_path is not None else None
    last_updated = _manifest_timestamp(manifest)
    fetcher = _build_fetcher(args)
    records = fetcher.fetch_newer_than(last_updated)

    print("Incremental update")
    print(f"Latest manifest: {manifest_path or 'none'}")
    print(f"Last updated: {last_updated.isoformat() if last_updated else 'none'}")
    print(f"Fetched newer records: {len(records)}")

    if records:
        update_file = _write_incremental_file(records, args.dataset_name)
        with SessionLocal() as db:
            result = ingest_papers_from_file(
                update_file,
                db,
                dataset_name=args.dataset_name,
                source=args.source,
                notes=args.notes,
            )
    else:
        paper_count, edge_count = _counts()
        result = {
            "inserted": 0,
            "skipped": 0,
            "citation_edges_inserted": 0,
            "paper_count": paper_count,
            "citation_edge_count": edge_count,
            "errors": [],
            "warnings": [],
            "manifest_path": None,
        }
        if args.keep_empty_manifest:
            empty_manifest = build_dataset_manifest(
                dataset_name=args.dataset_name,
                source=args.source,
                paper_count=paper_count,
                citation_edge_count=edge_count,
                notes=f"{args.notes} No newer records were available.",
            )
            result["manifest_path"] = str(write_dataset_manifest(empty_manifest))

    print(f"Inserted papers: {result['inserted']}")
    print(f"Skipped/merged duplicates: {result['skipped']}")
    print(f"Citation edges inserted: {result['citation_edges_inserted']}")
    print(f"Total papers: {result['paper_count']}")
    print(f"Total citation edges: {result['citation_edge_count']}")
    if result["manifest_path"]:
        print(f"New manifest: {result['manifest_path']}")
    for warning in result["warnings"][:25]:
        print(f"Warning: {warning}")
    if len(result["warnings"]) > 25:
        print(f"Warning: {len(result['warnings']) - 25} additional warnings omitted.")
    for error in result["errors"]:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
