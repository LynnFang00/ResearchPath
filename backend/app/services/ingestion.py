import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.citation_edge import CitationEdge
from app.models.paper import Paper
from app.schemas.paper import CitationEdgeCreate, PaperCreate
from app.services.dataset_manifest import (
    DEFAULT_MANIFEST_DIR,
    build_dataset_manifest,
    write_dataset_manifest as write_manifest_file,
)
from app.services.deduplication import (
    completeness_score,
    duplicate_match,
    lookup_keys_for_record,
    normalize_identifier,
    normalize_title,
)


SUPPORTED_EXTENSIONS = {".jsonl", ".csv"}
REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use JSONL or CSV.")

    if path.suffix.lower() == ".jsonl":
        dataframe = pd.read_json(path, lines=True)
    else:
        dataframe = pd.read_csv(path)

    records = dataframe.astype(object).where(pd.notnull(dataframe), None).to_dict(orient="records")
    return [{key: _clean_missing_values(value) for key, value in record.items()} for record in records]


def _clean_missing_values(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, list):
        return [_clean_missing_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_missing_values(item) for key, item in value.items()}
    return value


def _serialize_list(values: list[str]) -> str:
    return "; ".join(values)


def _paper_lookup_keys(paper: Paper) -> list[str]:
    return lookup_keys_for_record(paper)


def _record_reference_key(reference: str) -> str:
    reference = reference.strip()
    lower = reference.lower()
    if lower.startswith("doi:"):
        return f"doi:{lower.removeprefix('doi:').strip()}"
    if lower.startswith("title:"):
        return f"title:{normalize_title(reference.split(':', 1)[1])}"
    if lower.startswith("external_id:"):
        return f"external_id:{lower.removeprefix('external_id:').strip()}"
    if ":" in reference:
        return f"external_id:{lower}"
    if "/" in reference or "." in reference:
        return f"doi:{lower}"
    if " " in reference:
        return f"title:{normalize_title(reference)}"
    return f"external_id:{lower}"


def _count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", text))


def parse_paper_record(record: dict[str, Any], row_number: int) -> PaperCreate:
    if record.get("url") is None and record.get("source_url") is not None:
        record = {**record, "url": record["source_url"]}
    if record.get("source_url") is None and record.get("url") is not None:
        record = {**record, "source_url": record["url"]}

    try:
        return PaperCreate(**record)
    except ValidationError as exc:
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise ValueError(f"Row {row_number}: {detail}") from exc


def parse_citation_edge_record(record: dict[str, Any], row_number: int) -> CitationEdgeCreate:
    try:
        return CitationEdgeCreate(**record)
    except ValidationError as exc:
        detail = "; ".join(error["msg"] for error in exc.errors())
        raise ValueError(f"Row {row_number}: {detail}") from exc


def _dedupe_key(parsed: PaperCreate) -> tuple[str, str]:
    if parsed.external_id:
        return ("external_id", normalize_identifier(parsed.external_id) or "")
    if parsed.doi:
        return ("doi", normalize_identifier(parsed.doi) or "")
    return ("title", normalize_title(parsed.title))


def _load_existing_lookup(db: Session) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for paper in db.scalars(select(Paper)).all():
        for key in _paper_lookup_keys(paper):
            lookup[key] = paper.id
    return lookup


def _load_existing_papers(db: Session) -> list[Paper]:
    return list(db.scalars(select(Paper).order_by(Paper.id)).all())


def write_dataset_manifest(
    *,
    dataset_name: str,
    source: str,
    result: dict[str, Any],
    fields_available: list[str],
    filters_used: dict[str, Any] | None = None,
    notes: str | None = None,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
) -> Path:
    manifest = build_dataset_manifest(
        dataset_name=dataset_name,
        source=source,
        paper_count=result["paper_count"],
        citation_edge_count=result["citation_edge_count"],
        fields_available=fields_available,
        filters_used=filters_used,
        notes=notes,
        inserted=result["inserted"],
        skipped=result["skipped"],
        citation_edges_inserted=result["citation_edges_inserted"],
        errors=result["errors"],
        warnings=result["warnings"],
    )
    return write_manifest_file(manifest, manifest_dir=manifest_dir)


def _list_union(left: list[str], right: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*left, *right]:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def _merge_parsed_records(primary: PaperCreate, duplicate: PaperCreate) -> PaperCreate:
    if completeness_score(duplicate) > completeness_score(primary):
        primary, duplicate = duplicate, primary

    updates: dict[str, Any] = {}
    for field_name in ("venue", "source_url", "external_id", "source", "doi", "url", "updated_at"):
        primary_value = getattr(primary, field_name)
        duplicate_value = getattr(duplicate, field_name)
        if primary_value in (None, "") and duplicate_value not in (None, ""):
            updates[field_name] = duplicate_value

    if len(duplicate.abstract or "") > len(primary.abstract or ""):
        updates["abstract"] = duplicate.abstract
    if duplicate.year is not None and primary.year is None:
        updates["year"] = duplicate.year
    for count_field in (
        "citation_count",
        "references_count",
        "influential_citation_count",
        "abstract_word_count",
    ):
        updates[count_field] = max(getattr(primary, count_field) or 0, getattr(duplicate, count_field) or 0)

    updates["authors"] = _list_union(primary.authors, duplicate.authors)
    updates["categories"] = _list_union(primary.categories, duplicate.categories)
    updates["references"] = _list_union(primary.references, duplicate.references)
    updates["citations"] = _list_union(primary.citations, duplicate.citations)
    return primary.model_copy(update=updates)


def _dedupe_parsed_records(
    parsed_rows: list[tuple[int, PaperCreate]],
    warnings: list[str],
) -> tuple[list[tuple[int, PaperCreate]], int]:
    unique_rows: list[tuple[int, PaperCreate]] = []
    skipped = 0
    for row_number, parsed in parsed_rows:
        match_index: int | None = None
        match_kind = ""
        for index, (_, existing) in enumerate(unique_rows):
            match = duplicate_match(parsed, existing)
            if match is not None:
                match_index = index
                match_kind = match.kind
                break
        if match_index is None:
            unique_rows.append((row_number, parsed))
            continue

        original_row_number, existing = unique_rows[match_index]
        unique_rows[match_index] = (original_row_number, _merge_parsed_records(existing, parsed))
        warnings.append(
            f"Row {row_number}: merged duplicate paper by {match_kind} with row {original_row_number}."
        )
        skipped += 1
    return unique_rows, skipped


def _find_existing_duplicate(
    parsed: PaperCreate,
    *,
    lookup: dict[str, int],
    existing_papers: list[Paper],
) -> tuple[Paper | None, str]:
    for key in lookup_keys_for_record(parsed):
        existing_id = lookup.get(key)
        if existing_id is not None:
            for paper in existing_papers:
                if paper.id == existing_id:
                    return paper, key.split(":", 1)[0]

    for paper in existing_papers:
        match = duplicate_match(parsed, paper)
        if match is not None:
            return paper, match.kind
    return None, ""


def _split_serialized_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def _merge_into_existing_paper(paper: Paper, parsed: PaperCreate) -> bool:
    changed = False
    for field_name in ("venue", "source_url", "external_id", "source", "doi", "url", "updated_at"):
        current = getattr(paper, field_name)
        incoming = getattr(parsed, field_name)
        if current in (None, "") and incoming not in (None, ""):
            setattr(paper, field_name, incoming)
            changed = True

    if len(parsed.abstract or "") > len(paper.abstract or ""):
        paper.abstract = parsed.abstract
        changed = True
    if paper.year is None and parsed.year is not None:
        paper.year = parsed.year
        changed = True

    for count_field in (
        "citation_count",
        "references_count",
        "influential_citation_count",
        "abstract_word_count",
    ):
        current_count = getattr(paper, count_field) or 0
        incoming_count = getattr(parsed, count_field) or 0
        if incoming_count > current_count:
            setattr(paper, count_field, incoming_count)
            changed = True

    merged_authors = _serialize_list(_list_union(_split_serialized_list(paper.authors), parsed.authors))
    if merged_authors != paper.authors:
        paper.authors = merged_authors
        changed = True

    merged_categories = _serialize_list(_list_union(_split_serialized_list(paper.categories), parsed.categories))
    if merged_categories != paper.categories:
        paper.categories = merged_categories
        changed = True

    return changed


def _insert_pending_edges(
    *,
    db: Session,
    pending_edges: list[tuple[int, str, str, str | None]],
    lookup: dict[str, int],
    warnings: list[str],
) -> int:
    inserted = 0
    existing_edges = {
        (edge.source_paper_id, edge.target_paper_id, edge.relationship_type)
        for edge in db.scalars(select(CitationEdge)).all()
    }

    for source_paper_id, target_reference, relationship_type, source in pending_edges:
        target_key = _record_reference_key(target_reference)
        target_paper_id = lookup.get(target_key)
        if target_paper_id is None:
            warnings.append(
                f"Skipped citation edge from paper_id={source_paper_id}: target '{target_reference}' was not found."
            )
            continue
        if source_paper_id == target_paper_id:
            warnings.append(f"Skipped self citation edge for paper_id={source_paper_id}.")
            continue

        edge_key = (source_paper_id, target_paper_id, relationship_type)
        if edge_key in existing_edges:
            warnings.append(
                f"Skipped duplicate citation edge {source_paper_id}->{target_paper_id} ({relationship_type})."
            )
            continue

        parsed_edge = parse_citation_edge_record(
            {
                "source_paper_id": source_paper_id,
                "target_paper_id": target_paper_id,
                "relationship_type": relationship_type,
                "source": source,
            },
            row_number=source_paper_id,
        )
        db.add(CitationEdge(**parsed_edge.model_dump()))
        existing_edges.add(edge_key)
        inserted += 1

    return inserted


def ingest_papers_from_file(
    path: Path,
    db: Session,
    *,
    dataset_name: str | None = None,
    source: str | None = None,
    filters_used: dict[str, Any] | None = None,
    notes: str | None = None,
    write_manifest: bool = True,
) -> dict[str, Any]:
    records = _read_records(path)
    inserted = 0
    skipped = 0
    citation_edges_inserted = 0
    errors: list[str] = []
    warnings: list[str] = []
    lookup = _load_existing_lookup(db)
    existing_papers = _load_existing_papers(db)
    pending_edges: list[tuple[int, str, str, str | None]] = []
    fields_available = sorted({field for record in records for field in record.keys()})
    parsed_rows: list[tuple[int, PaperCreate]] = []

    for row_number, record in enumerate(records, start=1):
        try:
            parsed = parse_paper_record(record, row_number)
        except ValueError as exc:
            errors.append(str(exc))
            skipped += 1
            continue
        parsed_rows.append((row_number, parsed))

    parsed_rows, merged_duplicate_count = _dedupe_parsed_records(parsed_rows, warnings)
    skipped += merged_duplicate_count

    for row_number, parsed in parsed_rows:
        dedupe_kind, dedupe_value = _dedupe_key(parsed)
        existing_duplicate, duplicate_kind = _find_existing_duplicate(
            parsed,
            lookup=lookup,
            existing_papers=existing_papers,
        )
        if existing_duplicate is not None:
            changed = _merge_into_existing_paper(existing_duplicate, parsed)
            db.add(existing_duplicate)
            for key in _paper_lookup_keys(existing_duplicate):
                lookup[key] = existing_duplicate.id
            for reference in parsed.references:
                pending_edges.append((existing_duplicate.id, reference, "reference", parsed.source or source))
            for citation in parsed.citations:
                pending_edges.append((existing_duplicate.id, citation, "citation", parsed.source or source))
            warnings.append(
                f"Row {row_number}: skipped duplicate paper by {duplicate_kind or dedupe_kind}='{dedupe_value}'"
                f"{' after merging metadata' if changed else ''}."
            )
            skipped += 1
            continue

        abstract_word_count = parsed.abstract_word_count or _count_words(parsed.abstract)
        paper_kwargs = {
            "title": parsed.title.strip(),
            "abstract": parsed.abstract.strip(),
            "authors": _serialize_list(parsed.authors),
            "year": parsed.year,
            "venue": parsed.venue,
            "categories": _serialize_list(parsed.categories),
            "citation_count": parsed.citation_count,
            "source_url": parsed.source_url,
            "external_id": parsed.external_id,
            "source": parsed.source or source,
            "doi": parsed.doi,
            "url": parsed.url or parsed.source_url,
            "references_count": parsed.references_count or len(parsed.references),
            "influential_citation_count": parsed.influential_citation_count,
            "abstract_word_count": abstract_word_count,
        }
        if parsed.updated_at is not None:
            paper_kwargs["updated_at"] = parsed.updated_at
        paper = Paper(**paper_kwargs)
        db.add(paper)
        db.flush()
        for key in _paper_lookup_keys(paper):
            lookup[key] = paper.id
        existing_papers.append(paper)
        for reference in parsed.references:
            pending_edges.append((paper.id, reference, "reference", parsed.source or source))
        for citation in parsed.citations:
            pending_edges.append((paper.id, citation, "citation", parsed.source or source))
        inserted += 1

    citation_edges_inserted = _insert_pending_edges(
        db=db,
        pending_edges=pending_edges,
        lookup=lookup,
        warnings=warnings,
    )
    db.commit()

    paper_count = len(_load_existing_papers(db))
    citation_edge_count = len(list(db.scalars(select(CitationEdge)).all()))

    result = {
        "inserted": inserted,
        "skipped": skipped,
        "citation_edges_inserted": citation_edges_inserted,
        "paper_count": paper_count,
        "citation_edge_count": citation_edge_count,
        "errors": errors,
        "warnings": warnings,
        "manifest_path": None,
    }
    if write_manifest:
        manifest_path = write_dataset_manifest(
            dataset_name=dataset_name or path.stem,
            source=source or "local_file",
            result=result,
            fields_available=fields_available,
            filters_used=filters_used,
            notes=notes,
        )
        result["manifest_path"] = str(manifest_path)

    return result
