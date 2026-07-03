import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import func, select


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_JSON = REPO_ROOT / "data" / "processed" / "reports" / "corpus_provenance_validation_v2_15k.json"
DEFAULT_MD = REPO_ROOT / "data" / "processed" / "reports" / "corpus_provenance_validation_v2_15k.md"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import SessionLocal  # noqa: E402
from app.models.citation_edge import CitationEdge  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.models.paper_identifier import PaperIdentifier  # noqa: E402
from app.models.paper_source import PaperSource  # noqa: E402
from app.services.deduplication import normalize_title  # noqa: E402


def rows_to_dict(rows: list[tuple[Any, int]]) -> dict[str, int]:
    return {str(key): int(value) for key, value in rows}


def duplicate_identifier_report(db: Any) -> list[dict[str, Any]]:
    rows = db.execute(
        select(PaperIdentifier.source, PaperIdentifier.identifier, func.count(PaperIdentifier.paper_id))
        .group_by(PaperIdentifier.source, PaperIdentifier.identifier)
        .having(func.count(PaperIdentifier.paper_id) > 1)
        .order_by(func.count(PaperIdentifier.paper_id).desc(), PaperIdentifier.source, PaperIdentifier.identifier)
    ).all()
    return [
        {"source": source, "identifier": identifier, "count": int(count)}
        for source, identifier, count in rows
    ]


def duplicate_title_report(db: Any, *, limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    for paper_id, title, year, external_id in db.execute(select(Paper.id, Paper.title, Paper.year, Paper.external_id)):
        key = normalize_title(title)
        if not key:
            continue
        counts[key] += 1
        examples.setdefault(key, [])
        if len(examples[key]) < 5:
            examples[key].append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "external_id": external_id,
                }
            )
    duplicates = [
        {"normalized_title": key, "count": count, "examples": examples[key]}
        for key, count in counts.most_common()
        if count > 1
    ]
    return duplicates[:limit]


def enrichment_openalex_only_inserts(db: Any) -> list[dict[str, Any]]:
    arxiv_paper_ids = select(PaperIdentifier.paper_id).where(PaperIdentifier.source == "arxiv")
    rows = (
        db.query(Paper, PaperSource)
        .join(PaperSource, PaperSource.paper_id == Paper.id)
        .filter(PaperSource.source == "openalex")
        .filter(PaperSource.raw_metadata_json.like("%matched_arxiv_id%"))
        .filter(~Paper.id.in_(arxiv_paper_ids))
        .order_by(Paper.id)
        .all()
    )
    results: list[dict[str, Any]] = []
    for paper, source in rows:
        raw = json.loads(source.raw_metadata_json or "{}")
        matched_arxiv_id = raw.get("matched_arxiv_id")
        if matched_arxiv_id is None and isinstance(raw.get("work"), dict):
            matched_arxiv_id = raw.get("work", {}).get("matched_arxiv_id")
        results.append(
            {
                "paper_id": paper.id,
                "title": paper.title,
                "year": paper.year,
                "doi": paper.doi,
                "external_id": paper.external_id,
                "openalex_source_record_id": source.source_record_id,
                "matched_arxiv_id": matched_arxiv_id,
            }
        )
    return results


def build_report(*, duplicate_title_limit: int) -> dict[str, Any]:
    with SessionLocal() as db:
        arxiv_ids = select(PaperIdentifier.paper_id).where(PaperIdentifier.source == "arxiv").subquery()
        openalex_ids = select(PaperIdentifier.paper_id).where(PaperIdentifier.source == "openalex").subquery()
        missing_identifiers = (
            db.query(Paper)
            .outerjoin(PaperIdentifier, PaperIdentifier.paper_id == Paper.id)
            .filter(PaperIdentifier.id.is_(None))
            .count()
        )
        missing_source_provenance = (
            db.query(Paper)
            .outerjoin(PaperSource, PaperSource.paper_id == Paper.id)
            .filter(PaperSource.id.is_(None))
            .count()
        )
        duplicate_titles = duplicate_title_report(db, limit=duplicate_title_limit)
        return {
            "paper_count": db.query(Paper).count(),
            "citation_edge_count": db.query(CitationEdge).count(),
            "paper_identifier_count": db.query(PaperIdentifier).count(),
            "paper_source_record_count": db.query(PaperSource).count(),
            "paper_count_by_source": rows_to_dict(
                db.query(Paper.source, func.count(Paper.id)).group_by(Paper.source).order_by(Paper.source).all()
            ),
            "identifier_count_by_type": rows_to_dict(
                db.query(PaperIdentifier.source, func.count(PaperIdentifier.id))
                .group_by(PaperIdentifier.source)
                .order_by(PaperIdentifier.source)
                .all()
            ),
            "papers_with_arxiv_and_openalex": int(
                db.execute(
                    select(func.count(func.distinct(arxiv_ids.c.paper_id))).join(
                        openalex_ids,
                        arxiv_ids.c.paper_id == openalex_ids.c.paper_id,
                    )
                ).scalar_one()
            ),
            "papers_missing_identifiers": int(missing_identifiers),
            "papers_missing_source_provenance": int(missing_source_provenance),
            "duplicate_identifiers": duplicate_identifier_report(db),
            "duplicate_normalized_title_count_shown": len(duplicate_titles),
            "duplicate_normalized_titles": duplicate_titles,
            "openalex_only_inserts_from_enrichment": enrichment_openalex_only_inserts(db),
        }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Corpus Provenance Validation",
        "",
        f"- Paper count: `{report['paper_count']}`",
        f"- Citation edges: `{report['citation_edge_count']}`",
        f"- Paper identifiers: `{report['paper_identifier_count']}`",
        f"- Paper source provenance records: `{report['paper_source_record_count']}`",
        f"- Papers with arXiv + OpenAlex identifiers: `{report['papers_with_arxiv_and_openalex']}`",
        f"- Papers missing identifiers: `{report['papers_missing_identifiers']}`",
        f"- Papers missing source provenance: `{report['papers_missing_source_provenance']}`",
        "",
        "## Paper Count By Source",
        "",
        "| Source | Count |",
        "|---|---:|",
    ]
    for source, count in report["paper_count_by_source"].items():
        lines.append(f"| {source} | {count} |")
    lines.extend(["", "## Identifier Count By Type", "", "| Type | Count |", "|---|---:|"])
    for source, count in report["identifier_count_by_type"].items():
        lines.append(f"| {source} | {count} |")
    lines.extend(
        [
            "",
            "## Duplicate Identifiers",
            "",
            f"Total duplicate identifier keys: `{len(report['duplicate_identifiers'])}`",
            "",
        ]
    )
    for item in report["duplicate_identifiers"][:20]:
        lines.append(f"- `{item['source']}:{item['identifier']}` appears `{item['count']}` times")
    lines.extend(
        [
            "",
            "## OpenAlex-Only Inserts From Enrichment",
            "",
            f"Count: `{len(report['openalex_only_inserts_from_enrichment'])}`",
            "",
        ]
    )
    for item in report["openalex_only_inserts_from_enrichment"]:
        lines.append(
            f"- paper_id `{item['paper_id']}` | matched_arxiv_id `{item['matched_arxiv_id']}` | {item['title']}"
        )
    lines.extend(
        [
            "",
            "## Duplicate Normalized Titles",
            "",
            f"Shown: `{len(report['duplicate_normalized_titles'])}`",
            "",
        ]
    )
    for item in report["duplicate_normalized_titles"][:20]:
        lines.append(f"- count `{item['count']}` | {item['examples'][0]['title']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate corpus source provenance and identifier health.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--duplicate-title-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    if not md_path.is_absolute():
        md_path = REPO_ROOT / md_path
    report = build_report(duplicate_title_limit=args.duplicate_title_limit)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report, indent=2)[:4000])
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
