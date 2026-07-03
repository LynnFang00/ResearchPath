import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.citation_edge import CitationEdge
from app.models.paper import Paper
from app.models.paper_identifier import PaperIdentifier
from app.models.paper_source import PaperSource
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


def _normalize_arxiv_identifier(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^10\.48550/arxiv\.", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", value, flags=re.IGNORECASE).strip()
    value = value.removesuffix(".pdf")
    value = re.sub(r"v[0-9]+$", "", value, flags=re.IGNORECASE)
    return value.lower()


def _normalize_openalex_identifier(value: str) -> str:
    value = value.strip()
    value = value.rstrip("/").split("/")[-1]
    value = re.sub(r"^openalex:", "", value, flags=re.IGNORECASE).strip()
    return value.upper() if value.lower().startswith("w") else value.lower()


def _normalize_doi_identifier(value: str) -> str:
    value = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :].strip()
            break
    return value.lower()


def _extract_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    patterns = [
        r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
        r"10\.48550/arxiv\.([A-Za-z\-]+/[0-9]{7})(v[0-9]+)?",
        r"arxiv:([0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
        r"arxiv:([A-Za-z\-]+/[0-9]{7})(v[0-9]+)?",
        r"arxiv\.org/(?:abs|pdf)/([A-Za-z\-]+/[0-9]{7}|[0-9]{4}\.[0-9]{4,5})(v[0-9]+)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return _normalize_arxiv_identifier(match.group(1))
    return None


def _identifier_lookup_key(source: str, identifier: str) -> str:
    normalized_source = source.strip().lower()
    if normalized_source == "doi":
        return f"doi:{_normalize_doi_identifier(identifier)}"
    if normalized_source == "arxiv":
        return f"external_id:arxiv:{_normalize_arxiv_identifier(identifier)}"
    if normalized_source == "openalex":
        return f"external_id:openalex:{_normalize_openalex_identifier(identifier).lower()}"
    return f"external_id:{normalized_source}:{identifier.strip().lower()}"


def _identifier_rows_for_record(record: dict[str, Any], parsed: PaperCreate) -> list[tuple[str, str]]:
    identifiers: list[tuple[str, str]] = []

    def add(source: str, identifier: Any) -> None:
        if identifier is None:
            return
        source = source.strip().lower()
        value = str(identifier).strip()
        if not source or not value:
            return
        if source == "doi":
            value = _normalize_doi_identifier(value)
        elif source == "arxiv":
            value = _normalize_arxiv_identifier(value)
        elif source == "openalex":
            value = _normalize_openalex_identifier(value)
        key = (source, value)
        if key not in identifiers:
            identifiers.append(key)

    raw_identifiers = record.get("identifiers") or record.get("source_identifiers") or {}
    if isinstance(raw_identifiers, dict):
        for source, identifier in raw_identifiers.items():
            if isinstance(identifier, list):
                for item in identifier:
                    add(str(source), item)
            else:
                add(str(source), identifier)
    elif isinstance(raw_identifiers, list):
        for item in raw_identifiers:
            if isinstance(item, dict):
                add(str(item.get("source") or ""), item.get("identifier"))

    add("doi", parsed.doi)
    add("arxiv", record.get("arxiv_id") or _extract_arxiv_id(parsed.external_id) or _extract_arxiv_id(parsed.doi))
    add("arxiv", _extract_arxiv_id(parsed.url) or _extract_arxiv_id(parsed.source_url))

    external_id = parsed.external_id or ""
    if external_id.lower().startswith("openalex:") or "openalex.org/" in external_id.lower():
        add("openalex", external_id)
    if external_id.lower().startswith("arxiv:") or "arxiv.org/" in external_id.lower():
        add("arxiv", external_id)
    if record.get("openalex_id"):
        add("openalex", record["openalex_id"])

    return identifiers


def _lookup_keys_for_record_with_identifiers(record: dict[str, Any], parsed: PaperCreate) -> list[str]:
    keys = list(lookup_keys_for_record(parsed))
    for source, identifier in _identifier_rows_for_record(record, parsed):
        key = _identifier_lookup_key(source, identifier)
        if key not in keys:
            keys.append(key)
    return keys


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
    for identifier in db.scalars(select(PaperIdentifier)).all():
        lookup[_identifier_lookup_key(identifier.source, identifier.identifier)] = identifier.paper_id
    return lookup


def _load_existing_identifier_keys(db: Session) -> set[tuple[str, str]]:
    return {
        (identifier.source, identifier.identifier)
        for identifier in db.scalars(select(PaperIdentifier)).all()
    }


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
    parsed_rows: list[tuple[int, dict[str, Any], PaperCreate]],
    warnings: list[str],
    *,
    allow_near_title_dedupe: bool = True,
) -> tuple[list[tuple[int, dict[str, Any], PaperCreate]], int]:
    unique_rows: list[tuple[int, dict[str, Any], PaperCreate]] = []
    unique_lookup: dict[str, int] = {}
    skipped = 0
    for row_number, record, parsed in parsed_rows:
        match_index: int | None = None
        match_kind = ""
        for key in _lookup_keys_for_record_with_identifiers(record, parsed):
            if key in unique_lookup:
                match_index = unique_lookup[key]
                match_kind = key.split(":", 1)[0]
                break
        if match_index is None and allow_near_title_dedupe:
            for index, (_, _, existing) in enumerate(unique_rows):
                match = duplicate_match(parsed, existing)
                if match is not None:
                    match_index = index
                    match_kind = match.kind
                    break
        if match_index is None:
            unique_lookup.update(
                {key: len(unique_rows) for key in _lookup_keys_for_record_with_identifiers(record, parsed)}
            )
            unique_rows.append((row_number, record, parsed))
            continue

        original_row_number, original_record, existing = unique_rows[match_index]
        merged_record = {**record, **original_record}
        unique_rows[match_index] = (original_row_number, merged_record, _merge_parsed_records(existing, parsed))
        unique_lookup.update(
            {
                key: match_index
                for key in _lookup_keys_for_record_with_identifiers(
                    unique_rows[match_index][1],
                    unique_rows[match_index][2],
                )
            }
        )
        warnings.append(
            f"Row {row_number}: merged duplicate paper by {match_kind} with row {original_row_number}."
        )
        skipped += 1
    return unique_rows, skipped


def _find_existing_duplicate(
    record: dict[str, Any],
    parsed: PaperCreate,
    *,
    lookup: dict[str, int],
    existing_papers: list[Paper],
    existing_by_id: dict[int, Paper],
    allow_near_title_dedupe: bool = True,
) -> tuple[Paper | None, str]:
    for key in _lookup_keys_for_record_with_identifiers(record, parsed):
        existing_id = lookup.get(key)
        if existing_id is not None:
            paper = existing_by_id.get(existing_id)
            if paper is not None:
                return paper, key.split(":", 1)[0]

    if not allow_near_title_dedupe:
        return None, ""

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


def _source_record_id(record: dict[str, Any], parsed: PaperCreate, source: str | None) -> str:
    if parsed.external_id:
        return parsed.external_id
    if parsed.doi:
        return f"doi:{_normalize_doi_identifier(parsed.doi)}"
    arxiv_id = record.get("arxiv_id") or _extract_arxiv_id(parsed.url) or _extract_arxiv_id(parsed.source_url)
    if arxiv_id:
        return f"arxiv:{_normalize_arxiv_identifier(str(arxiv_id))}"
    return f"{source or parsed.source or 'local_file'}:{normalize_title(parsed.title)}"


def _upsert_identifier(
    *,
    db: Session,
    paper_id: int,
    source: str,
    identifier: str,
    lookup: dict[str, int],
    identifier_keys: set[tuple[str, str]],
) -> None:
    identifier_key = (source, identifier)
    if identifier_key in identifier_keys:
        return
    lookup_key = _identifier_lookup_key(source, identifier)
    existing_paper_id = lookup.get(lookup_key)
    if existing_paper_id is not None and existing_paper_id != paper_id:
        return
    db.add(PaperIdentifier(paper_id=paper_id, source=source, identifier=identifier))
    lookup[lookup_key] = paper_id
    identifier_keys.add(identifier_key)


def _upsert_source_record(
    *,
    db: Session,
    paper_id: int,
    record: dict[str, Any],
    parsed: PaperCreate,
    source: str | None,
) -> None:
    source_name = str(parsed.source or source or record.get("source") or "local_file").strip().lower()
    source_record_id = _source_record_id(record, parsed, source_name)
    existing = db.scalar(
        select(PaperSource).where(
            PaperSource.paper_id == paper_id,
            PaperSource.source == source_name,
            PaperSource.source_record_id == source_record_id,
        )
    )
    raw_metadata = record.get("raw_source_metadata") or record.get("source_record") or record
    raw_metadata_json = json.dumps(raw_metadata, ensure_ascii=True, sort_keys=True, default=str)
    if existing is not None:
        existing.source_url = parsed.source_url or parsed.url or existing.source_url
        existing.raw_metadata_json = raw_metadata_json
        db.add(existing)
        return
    db.add(
        PaperSource(
            paper_id=paper_id,
            source=source_name,
            source_record_id=source_record_id,
            source_url=parsed.source_url or parsed.url,
            raw_metadata_json=raw_metadata_json,
        )
    )


def _record_identity_metadata(
    *,
    db: Session,
    paper_id: int,
    record: dict[str, Any],
    parsed: PaperCreate,
    source: str | None,
    lookup: dict[str, int],
    identifier_keys: set[tuple[str, str]],
) -> None:
    for identifier_source, identifier in _identifier_rows_for_record(record, parsed):
        _upsert_identifier(
            db=db,
            paper_id=paper_id,
            source=identifier_source,
            identifier=identifier,
            lookup=lookup,
            identifier_keys=identifier_keys,
        )
    _upsert_source_record(db=db, paper_id=paper_id, record=record, parsed=parsed, source=source)


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
    allow_near_title_dedupe: bool = True,
) -> dict[str, Any]:
    records = _read_records(path)
    inserted = 0
    skipped = 0
    citation_edges_inserted = 0
    errors: list[str] = []
    warnings: list[str] = []
    lookup = _load_existing_lookup(db)
    identifier_keys = _load_existing_identifier_keys(db)
    existing_papers = _load_existing_papers(db)
    existing_by_id = {paper.id: paper for paper in existing_papers}
    pending_edges: list[tuple[int, str, str, str | None]] = []
    fields_available = sorted({field for record in records for field in record.keys()})
    parsed_rows: list[tuple[int, dict[str, Any], PaperCreate]] = []

    for row_number, record in enumerate(records, start=1):
        try:
            parsed = parse_paper_record(record, row_number)
        except ValueError as exc:
            errors.append(str(exc))
            skipped += 1
            continue
        parsed_rows.append((row_number, record, parsed))

    parsed_rows, merged_duplicate_count = _dedupe_parsed_records(
        parsed_rows,
        warnings,
        allow_near_title_dedupe=allow_near_title_dedupe,
    )
    skipped += merged_duplicate_count

    for row_number, record, parsed in parsed_rows:
        dedupe_kind, dedupe_value = _dedupe_key(parsed)
        existing_duplicate, duplicate_kind = _find_existing_duplicate(
            record,
            parsed,
            lookup=lookup,
            existing_papers=existing_papers,
            existing_by_id=existing_by_id,
            allow_near_title_dedupe=allow_near_title_dedupe,
        )
        if existing_duplicate is not None:
            changed = _merge_into_existing_paper(existing_duplicate, parsed)
            db.add(existing_duplicate)
            for key in _paper_lookup_keys(existing_duplicate):
                lookup[key] = existing_duplicate.id
            _record_identity_metadata(
                db=db,
                paper_id=existing_duplicate.id,
                record=record,
                parsed=parsed,
                source=source,
                lookup=lookup,
                identifier_keys=identifier_keys,
            )
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
        _record_identity_metadata(
            db=db,
            paper_id=paper.id,
            record=record,
            parsed=parsed,
            source=source,
            lookup=lookup,
            identifier_keys=identifier_keys,
        )
        existing_papers.append(paper)
        existing_by_id[paper.id] = paper
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
