import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

from sqlalchemy import select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402


def normalize_openalex_id(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().rstrip("/").split("/")[-1]
    value = value.removeprefix("openalex:").removeprefix("OpenAlex:")
    return value.upper() if value.lower().startswith("w") else None


def openalex_url(openalex_id: str) -> str:
    return f"https://openalex.org/{openalex_id}"


def backfill_openalex_provenance(*, dry_run: bool) -> dict[str, int]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    now = datetime.now(UTC).isoformat()
    stats = {
        "candidate_papers": 0,
        "identifiers_added": 0,
        "sources_added": 0,
        "identifiers_existing": 0,
        "sources_existing": 0,
    }
    with SessionLocal() as db:
        papers = db.scalars(select(Paper).where(Paper.external_id.ilike("openalex:%"))).all()
        for paper in papers:
            openalex_id = normalize_openalex_id(paper.external_id)
            if not openalex_id:
                continue
            stats["candidate_papers"] += 1
            existing_identifier = db.scalar(
                select(PaperIdentifier).where(
                    PaperIdentifier.source == "openalex",
                    PaperIdentifier.identifier == openalex_id,
                )
            )
            if existing_identifier is None:
                stats["identifiers_added"] += 1
                if not dry_run:
                    db.add(PaperIdentifier(paper_id=paper.id, source="openalex", identifier=openalex_id))
            else:
                stats["identifiers_existing"] += 1

            source_record_id = f"openalex:{openalex_id}"
            existing_source = db.scalar(
                select(PaperSource).where(
                    PaperSource.paper_id == paper.id,
                    PaperSource.source == "openalex",
                    PaperSource.source_record_id == source_record_id,
                )
            )
            if existing_source is None:
                stats["sources_added"] += 1
                if not dry_run:
                    db.add(
                        PaperSource(
                            paper_id=paper.id,
                            source="openalex",
                            source_record_id=source_record_id,
                            source_url=paper.source_url or openalex_url(openalex_id),
                            raw_metadata_json=json.dumps(
                                {
                                    "source": "legacy_openalex_backfill",
                                    "openalex_id": openalex_id,
                                    "paper_id": paper.id,
                                    "backfilled_at": now,
                                    "external_id": paper.external_id,
                                },
                                ensure_ascii=True,
                                sort_keys=True,
                            ),
                        )
                    )
            else:
                stats["sources_existing"] += 1
        if not dry_run:
            db.commit()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill OpenAlex paper_identifiers and paper_sources rows.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing to the database.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = backfill_openalex_provenance(dry_run=args.dry_run)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
