import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

from sqlalchemy import select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import SessionLocal  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402


SEED_PAPER_IDS = {1, 2, 3, 4, 5, 6}


def backfill_seed_provenance(*, dry_run: bool) -> dict[str, int]:
    now = datetime.now(UTC).isoformat()
    stats = {
        "candidate_papers": 0,
        "source_field_updated": 0,
        "identifiers_added": 0,
        "sources_added": 0,
        "identifiers_existing": 0,
        "sources_existing": 0,
    }
    with SessionLocal() as db:
        papers = db.scalars(select(Paper).where(Paper.id.in_(SEED_PAPER_IDS))).all()
        for paper in papers:
            stats["candidate_papers"] += 1
            if paper.source in (None, ""):
                stats["source_field_updated"] += 1
                if not dry_run:
                    paper.source = "seed"

            seed_identifier = f"seed:{paper.id}"
            existing_identifier = db.scalar(
                select(PaperIdentifier).where(
                    PaperIdentifier.source == "seed",
                    PaperIdentifier.identifier == seed_identifier,
                )
            )
            if existing_identifier is None:
                stats["identifiers_added"] += 1
                if not dry_run:
                    db.add(PaperIdentifier(paper_id=paper.id, source="seed", identifier=seed_identifier))
            else:
                stats["identifiers_existing"] += 1

            existing_source = db.scalar(
                select(PaperSource).where(
                    PaperSource.paper_id == paper.id,
                    PaperSource.source == "seed",
                    PaperSource.source_record_id == seed_identifier,
                )
            )
            if existing_source is None:
                stats["sources_added"] += 1
                if not dry_run:
                    db.add(
                        PaperSource(
                            paper_id=paper.id,
                            source="seed",
                            source_record_id=seed_identifier,
                            source_url=paper.source_url or paper.url,
                            raw_metadata_json=json.dumps(
                                {
                                    "source": "legacy_seed_backfill",
                                    "paper_id": paper.id,
                                    "title": paper.title,
                                    "backfilled_at": now,
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
    parser = argparse.ArgumentParser(description="Backfill provenance for the original local seed papers.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = backfill_seed_provenance(dry_run=args.dry_run)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
