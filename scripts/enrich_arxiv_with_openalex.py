import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "arxiv_ml_ai_papers.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "arxiv_openalex_enrichment.jsonl"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SELECT_FIELDS = [
    "id",
    "doi",
    "display_name",
    "title",
    "abstract_inverted_index",
    "authorships",
    "publication_year",
    "primary_location",
    "cited_by_count",
    "referenced_works",
    "topics",
    "keywords",
    "updated_date",
    "type",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_title(value: str) -> str:
    return " ".join(value.lower().split())


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


def fetch_json(url: str) -> dict[str, Any] | None:
    request = Request(url, headers={"User-Agent": "ResearchPath/0.1 OpenAlex enrichment"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def openalex_work_by_doi(doi: str, *, api_key: str | None, email: str | None) -> dict[str, Any] | None:
    params = {"select": ",".join(SELECT_FIELDS)}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["mailto"] = email
    url = f"{OPENALEX_WORKS_URL}/doi:{quote(doi, safe='')}?{urlencode(params)}"
    payload = fetch_json(url)
    return payload if payload and payload.get("id") else None


def openalex_work_by_title(
    title: str,
    *,
    year: int | None,
    api_key: str | None,
    email: str | None,
    min_similarity: float,
) -> dict[str, Any] | None:
    params = {
        "search": title,
        "per_page": "5",
        "select": ",".join(SELECT_FIELDS),
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["mailto"] = email
    payload = fetch_json(f"{OPENALEX_WORKS_URL}?{urlencode(params)}")
    if not payload:
        return None
    best_work: dict[str, Any] | None = None
    best_score = 0.0
    for work in payload.get("results") or []:
        work_title = str(work.get("display_name") or work.get("title") or "")
        score = title_similarity(title, work_title)
        work_year = work.get("publication_year")
        if year and work_year and abs(int(work_year) - int(year)) > 1:
            score -= 0.1
        if score > best_score:
            best_score = score
            best_work = work
    if best_work is None or best_score < min_similarity:
        return None
    best_work["_researchpath_title_similarity"] = best_score
    return best_work


def openalex_id(uri: str | None) -> str | None:
    if not uri:
        return None
    return uri.rstrip("/").split("/")[-1]


def external_work_id(uri: str | None) -> str | None:
    work_id = openalex_id(uri)
    return f"openalex:{work_id}" if work_id else None


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.lower().startswith(prefix):
            return value[len(prefix) :].strip().lower()
    return value.lower()


def normalize_openalex_updated_at(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if "T" in value:
        if value.endswith("Z"):
            return f"{value[:-1]}+00:00"
        if value.endswith("+00:00"):
            return value
        return f"{value}+00:00"
    return f"{value}T00:00:00+00:00"


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, token_positions in inverted_index.items():
        for position in token_positions:
            positions.append((position, token))
    return " ".join(token for _, token in sorted(positions))


def extract_authors(authorships: list[dict[str, Any]] | None) -> list[str]:
    authors: list[str] = []
    for authorship in authorships or []:
        raw_name = authorship.get("raw_author_name")
        display_name = (authorship.get("author") or {}).get("display_name")
        name = raw_name or display_name
        if name:
            authors.append(name)
    return authors


def extract_venue(work: dict[str, Any]) -> str | None:
    source = ((work.get("primary_location") or {}).get("source") or {})
    return source.get("display_name")


def extract_url(work: dict[str, Any]) -> str | None:
    primary_location = work.get("primary_location") or {}
    return primary_location.get("landing_page_url") or work.get("doi") or work.get("id")


def extract_categories(work: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for topic in work.get("topics") or []:
        display_name = topic.get("display_name")
        if display_name:
            categories.append(display_name)
    for keyword in work.get("keywords") or []:
        display_name = keyword.get("display_name")
        if display_name:
            categories.append(display_name)
    seen: set[str] = set()
    deduped: list[str] = []
    for category in categories:
        normalized = category.lower()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(category)
    return deduped


def abstract_word_count(abstract: str) -> int:
    return len([token for token in abstract.split() if token.strip()])


def openalex_work_to_record(work: dict[str, Any], arxiv_record: dict[str, Any]) -> dict[str, Any]:
    abstract = reconstruct_abstract(work.get("abstract_inverted_index")) or str(arxiv_record.get("abstract") or "")
    references = [
        reference_id
        for reference_id in (external_work_id(uri) for uri in work.get("referenced_works") or [])
        if reference_id
    ]
    updated_at = normalize_openalex_updated_at(work.get("updated_date"))
    openalex_external_id = external_work_id(work.get("id"))
    arxiv_id = ((arxiv_record.get("identifiers") or {}).get("arxiv") or "").strip()
    doi = normalize_doi(work.get("doi")) or normalize_doi(arxiv_record.get("doi"))
    return {
        "external_id": openalex_external_id,
        "source": "openalex",
        "openalex_id": openalex_id(work.get("id")),
        "doi": doi,
        "title": work.get("display_name") or work.get("title") or arxiv_record.get("title") or "",
        "abstract": abstract,
        "authors": extract_authors(work.get("authorships")) or arxiv_record.get("authors") or [],
        "year": work.get("publication_year") or arxiv_record.get("year"),
        "venue": extract_venue(work) or arxiv_record.get("venue"),
        "categories": extract_categories(work) or arxiv_record.get("categories") or [],
        "citation_count": work.get("cited_by_count") or 0,
        "source_url": work.get("id"),
        "url": extract_url(work),
        "references_count": len(references),
        "influential_citation_count": 0,
        "abstract_word_count": abstract_word_count(abstract),
        "updated_at": updated_at,
        "references": references,
        "identifiers": {
            "openalex": openalex_id(work.get("id")),
            **({"arxiv": arxiv_id} if arxiv_id else {}),
            **({"doi": doi} if doi else {}),
        },
        "raw_source_metadata": {
            "source": "openalex",
            "openalex_id": openalex_id(work.get("id")),
            "matched_arxiv_id": arxiv_id or None,
            "match_title_similarity": work.get("_researchpath_title_similarity"),
            "work": work,
        },
    }


def enrich_records(
    arxiv_rows: list[dict[str, Any]],
    *,
    max_records: int,
    api_key: str | None,
    email: str | None,
    sleep_seconds: float,
    min_title_similarity: float,
    doi_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    seen_openalex_ids: set[str] = set()
    for row in arxiv_rows:
        if len(enriched) >= max_records:
            break
        doi = normalize_doi(row.get("doi"))
        work = openalex_work_by_doi(doi, api_key=api_key, email=email) if doi else None
        match_method = "doi" if work else ""
        if work is None and not doi_only:
            work = openalex_work_by_title(
                str(row.get("title") or ""),
                year=row.get("year"),
                api_key=api_key,
                email=email,
                min_similarity=min_title_similarity,
            )
            match_method = "title" if work else ""
        time.sleep(sleep_seconds)
        if work is None:
            misses.append({"arxiv_id": (row.get("identifiers") or {}).get("arxiv"), "title": row.get("title")})
            continue
        work_id = openalex_id(work.get("id")) or ""
        if work_id in seen_openalex_ids:
            continue
        seen_openalex_ids.add(work_id)
        record = openalex_work_to_record(work, row)
        record["raw_source_metadata"]["match_method"] = match_method
        enriched.append(record)
    return enriched, misses


def enrich_one_record(
    row: dict[str, Any],
    *,
    api_key: str | None,
    email: str | None,
    min_title_similarity: float,
    doi_only: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    doi = normalize_doi(row.get("doi"))
    work = openalex_work_by_doi(doi, api_key=api_key, email=email) if doi else None
    match_method = "doi" if work else ""
    if work is None and not doi_only:
        work = openalex_work_by_title(
            str(row.get("title") or ""),
            year=row.get("year"),
            api_key=api_key,
            email=email,
            min_similarity=min_title_similarity,
        )
        match_method = "title" if work else ""
    if work is None:
        return None, {"arxiv_id": (row.get("identifiers") or {}).get("arxiv"), "title": row.get("title")}
    record = openalex_work_to_record(work, row)
    record["raw_source_metadata"]["match_method"] = match_method
    return record, None


def enrich_records_parallel(
    arxiv_rows: list[dict[str, Any]],
    *,
    max_records: int,
    api_key: str | None,
    email: str | None,
    workers: int,
    min_title_similarity: float,
    doi_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = arxiv_rows[:max_records]
    enriched_by_index: dict[int, dict[str, Any]] = {}
    misses_by_index: dict[int, dict[str, Any]] = {}
    seen_openalex_ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                enrich_one_record,
                row,
                api_key=api_key,
                email=email,
                min_title_similarity=min_title_similarity,
                doi_only=doi_only,
            ): index
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            index = futures[future]
            record, miss = future.result()
            if miss is not None:
                misses_by_index[index] = miss
                continue
            if record is None:
                continue
            work_id = str((record.get("identifiers") or {}).get("openalex") or "")
            if work_id in seen_openalex_ids:
                continue
            seen_openalex_ids.add(work_id)
            enriched_by_index[index] = record
    return (
        [enriched_by_index[index] for index in sorted(enriched_by_index)],
        [misses_by_index[index] for index in sorted(misses_by_index)],
    )


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match arXiv records to OpenAlex and write enrichment JSONL.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--min-title-similarity", type=float, default=0.92)
    parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument("--email", default=os.environ.get("OPENALEX_EMAIL"))
    parser.add_argument("--doi-only", action="store_true", help="Only use DOI endpoint; skip title-search fallback.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel OpenAlex workers. Use conservatively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    arxiv_rows = load_jsonl(input_path)
    if args.workers > 1:
        enriched, misses = enrich_records_parallel(
            arxiv_rows,
            max_records=args.max_records,
            api_key=args.api_key,
            email=args.email,
            workers=args.workers,
            min_title_similarity=args.min_title_similarity,
            doi_only=args.doi_only,
        )
    else:
        enriched, misses = enrich_records(
            arxiv_rows,
            max_records=args.max_records,
            api_key=args.api_key,
            email=args.email,
            sleep_seconds=args.sleep_seconds,
            min_title_similarity=args.min_title_similarity,
            doi_only=args.doi_only,
        )
    write_jsonl(enriched, output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "input": str(input_path),
                "output": str(output_path),
                "input_records": len(arxiv_rows),
                "matched_records": len(enriched),
                "misses": len(misses),
                "min_title_similarity": args.min_title_similarity,
                "doi_only": args.doi_only,
                "workers": args.workers,
                "notes": "OpenAlex enrichment records are intended to merge with arXiv records through identifiers.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    misses_path = output_path.with_suffix(output_path.suffix + ".misses.json")
    misses_path.write_text(json.dumps(misses[:1000], indent=2), encoding="utf-8")
    print(f"Input arXiv records: {len(arxiv_rows)}")
    print(f"Matched OpenAlex records: {len(enriched)}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Miss sample: {misses_path}")


if __name__ == "__main__":
    main()
