import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import SessionLocal  # noqa: E402
from app.models.citation_edge import CitationEdge  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402


KNOWN_DUPLICATE_MERGES = {
    15800: 5880,
    15801: 5901,
    15802: 15577,
}


def _list_union(serialized: str | None, incoming: str | None) -> str:
    seen: set[str] = set()
    values: list[str] = []
    for item in [*(serialized or "").split(";"), *(incoming or "").split(";")]:
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return "; ".join(values)


def transfer_paper_metadata(target: Paper, duplicate: Paper) -> list[str]:
    changed: list[str] = []
    for field_name in ("doi", "venue", "url", "source_url", "updated_at"):
        current = getattr(target, field_name)
        incoming = getattr(duplicate, field_name)
        if current in (None, "") and incoming not in (None, ""):
            setattr(target, field_name, incoming)
            changed.append(field_name)
    if len(duplicate.abstract or "") > len(target.abstract or ""):
        target.abstract = duplicate.abstract
        changed.append("abstract")
    if target.year is None and duplicate.year is not None:
        target.year = duplicate.year
        changed.append("year")
    for field_name in ("citation_count", "references_count", "influential_citation_count", "abstract_word_count"):
        if (getattr(duplicate, field_name) or 0) > (getattr(target, field_name) or 0):
            setattr(target, field_name, getattr(duplicate, field_name))
            changed.append(field_name)
    merged_authors = _list_union(target.authors, duplicate.authors)
    if merged_authors != (target.authors or ""):
        target.authors = merged_authors
        changed.append("authors")
    merged_categories = _list_union(target.categories, duplicate.categories)
    if merged_categories != (target.categories or ""):
        target.categories = merged_categories
        changed.append("categories")
    return changed


def transfer_identifiers(db: Any, *, duplicate_id: int, target_id: int) -> tuple[int, int]:
    moved = 0
    skipped = 0
    rows = db.scalars(select(PaperIdentifier).where(PaperIdentifier.paper_id == duplicate_id)).all()
    for row in rows:
        existing = db.scalar(
            select(PaperIdentifier).where(
                PaperIdentifier.source == row.source,
                PaperIdentifier.identifier == row.identifier,
            )
        )
        if existing is None:
            row.paper_id = target_id
            moved += 1
            continue
        if existing.id == row.id:
            row.paper_id = target_id
            moved += 1
            continue
        if existing.paper_id == target_id:
            db.delete(row)
            skipped += 1
            continue
        raise ValueError(
            f"Identifier {row.source}:{row.identifier} already belongs to paper_id={existing.paper_id}, "
            f"cannot move duplicate paper_id={duplicate_id} to target paper_id={target_id}."
        )
    return moved, skipped


def transfer_sources(db: Any, *, duplicate_id: int, target_id: int) -> tuple[int, int]:
    moved = 0
    skipped = 0
    rows = db.scalars(select(PaperSource).where(PaperSource.paper_id == duplicate_id)).all()
    for row in rows:
        existing = db.scalar(
            select(PaperSource).where(
                PaperSource.paper_id == target_id,
                PaperSource.source == row.source,
                PaperSource.source_record_id == row.source_record_id,
            )
        )
        if existing is None:
            row.paper_id = target_id
            moved += 1
            continue
        db.delete(row)
        skipped += 1
    return moved, skipped


def transfer_citation_edges(db: Any, *, duplicate_id: int, target_id: int) -> tuple[int, int]:
    moved = 0
    skipped = 0
    rows = db.scalars(
        select(CitationEdge).where(
            (CitationEdge.source_paper_id == duplicate_id) | (CitationEdge.target_paper_id == duplicate_id)
        )
    ).all()
    for row in rows:
        new_source = target_id if row.source_paper_id == duplicate_id else row.source_paper_id
        new_target = target_id if row.target_paper_id == duplicate_id else row.target_paper_id
        if new_source == new_target:
            db.delete(row)
            skipped += 1
            continue
        existing = db.scalar(
            select(CitationEdge).where(
                CitationEdge.source_paper_id == new_source,
                CitationEdge.target_paper_id == new_target,
                CitationEdge.relationship_type == row.relationship_type,
            )
        )
        if existing is None:
            row.source_paper_id = new_source
            row.target_paper_id = new_target
            moved += 1
        else:
            db.delete(row)
            skipped += 1
    return moved, skipped


def cleanup_known_duplicates(*, dry_run: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "started_at": datetime.now(UTC).isoformat(),
        "mappings": [],
        "duplicates_deleted": 0,
    }
    with SessionLocal() as db:
        try:
            for duplicate_id, target_id in KNOWN_DUPLICATE_MERGES.items():
                duplicate = db.get(Paper, duplicate_id)
                target = db.get(Paper, target_id)
                entry: dict[str, Any] = {
                    "duplicate_id": duplicate_id,
                    "target_id": target_id,
                    "duplicate_found": duplicate is not None,
                    "target_found": target is not None,
                }
                if duplicate is None or target is None:
                    report["mappings"].append(entry)
                    continue
                entry.update(
                    {
                        "duplicate_title": duplicate.title,
                        "target_title": target.title,
                        "metadata_changed": transfer_paper_metadata(target, duplicate),
                        "identifiers": {},
                        "sources": {},
                        "citation_edges": {},
                    }
                )
                moved, skipped = transfer_identifiers(db, duplicate_id=duplicate_id, target_id=target_id)
                entry["identifiers"] = {"moved": moved, "skipped_existing": skipped}
                moved, skipped = transfer_sources(db, duplicate_id=duplicate_id, target_id=target_id)
                entry["sources"] = {"moved": moved, "skipped_existing": skipped}
                moved, skipped = transfer_citation_edges(db, duplicate_id=duplicate_id, target_id=target_id)
                entry["citation_edges"] = {"moved": moved, "skipped_existing": skipped}
                db.delete(duplicate)
                report["duplicates_deleted"] += 1
                report["mappings"].append(entry)
            if dry_run:
                db.rollback()
            else:
                db.commit()
        except (IntegrityError, ValueError):
            db.rollback()
            raise
    report["finished_at"] = datetime.now(UTC).isoformat()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the reviewed OpenAlex enrichment duplicate rows.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = cleanup_known_duplicates(dry_run=args.dry_run)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        path = Path(args.json_out)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
