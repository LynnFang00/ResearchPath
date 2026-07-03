import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "openalex_papers.jsonl"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_SELECT_FIELDS = [
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


def abstract_word_count(abstract: str) -> int:
    return len([token for token in abstract.split() if token.strip()])


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


def openalex_work_to_paper_record(work: dict[str, Any]) -> dict[str, Any]:
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    references = [
        reference_id
        for reference_id in (external_work_id(uri) for uri in work.get("referenced_works") or [])
        if reference_id
    ]
    updated_at = normalize_openalex_updated_at(work.get("updated_date"))

    return {
        "external_id": external_work_id(work.get("id")),
        "source": "openalex",
        "doi": normalize_doi(work.get("doi")),
        "title": work.get("display_name") or work.get("title") or "",
        "abstract": abstract,
        "authors": extract_authors(work.get("authorships")),
        "year": work.get("publication_year"),
        "venue": extract_venue(work),
        "categories": extract_categories(work),
        "citation_count": work.get("cited_by_count") or 0,
        "source_url": work.get("id"),
        "url": extract_url(work),
        "references_count": len(references),
        "influential_citation_count": 0,
        "abstract_word_count": abstract_word_count(abstract),
        "updated_at": updated_at,
        "references": references,
    }


def build_filter(
    *,
    from_year: int | None,
    to_year: int | None,
    work_type: str | None,
    min_citations: int,
    topic_ids: list[str],
    extra_filter: str | None,
) -> str:
    filters: list[str] = []
    if work_type:
        filters.append(f"type:{work_type}")
    if from_year and to_year:
        filters.append(f"publication_year:{from_year}-{to_year}")
    elif from_year:
        filters.append(f"publication_year:>{from_year - 1}")
    elif to_year:
        filters.append(f"publication_year:<{to_year + 1}")
    if min_citations > 0:
        filters.append(f"cited_by_count:>{min_citations - 1}")
    if topic_ids:
        filters.append("topics.id:" + "|".join(topic_ids))
    if extra_filter:
        filters.append(extra_filter)
    return ",".join(filters)


def fetch_openalex_page(params: dict[str, str]) -> dict[str, Any]:
    url = f"{OPENALEX_WORKS_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "ResearchPath/0.1 (local portfolio project)"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_openalex_records(
    *,
    query: str,
    max_results: int,
    per_page: int,
    filter_value: str,
    sort: str,
    api_key: str | None,
    email: str | None,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    cursor = "*"
    records: list[dict[str, Any]] = []

    while len(records) < max_results:
        params = {
            "search": query,
            "filter": filter_value,
            "sort": sort,
            "per_page": str(min(per_page, 100, max_results - len(records))),
            "cursor": cursor,
            "select": ",".join(DEFAULT_SELECT_FIELDS),
        }
        if api_key:
            params["api_key"] = api_key
        if email:
            params["mailto"] = email

        payload = fetch_openalex_page(params)
        works = payload.get("results") or []
        if not works:
            break

        for work in works:
            record = openalex_work_to_paper_record(work)
            if record["title"] and record["abstract"]:
                records.append(record)

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(sleep_seconds)

    return records[:max_results]


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_fetch_metadata(output_path: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OpenAlex works into ResearchPath JSONL format.")
    parser.add_argument("--query", default="machine learning artificial intelligence")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-results", type=int, default=1000)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--from-year", type=int, default=2020)
    parser.add_argument("--to-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--type", default="article", dest="work_type")
    parser.add_argument("--min-citations", type=int, default=0)
    parser.add_argument("--topic-id", action="append", default=[], help="OpenAlex topic ID, e.g. T12345.")
    parser.add_argument("--extra-filter", default=None, help="Additional OpenAlex filter fragment.")
    parser.add_argument("--sort", default="cited_by_count:desc")
    parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument("--email", default=os.environ.get("OPENALEX_EMAIL"))
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    filter_value = build_filter(
        from_year=args.from_year,
        to_year=args.to_year,
        work_type=args.work_type,
        min_citations=args.min_citations,
        topic_ids=args.topic_id,
        extra_filter=args.extra_filter,
    )
    metadata = {
        "dataset_name": output_path.stem,
        "source": "openalex",
        "date_created": datetime.now(UTC).isoformat(),
        "query": args.query,
        "filter": filter_value,
        "sort": args.sort,
        "max_results": args.max_results,
        "per_page": args.per_page,
        "select": DEFAULT_SELECT_FIELDS,
        "notes": "Raw OpenAlex fetch converted to ResearchPath ingestion JSONL.",
    }

    preview_params = {
        "search": args.query,
        "filter": filter_value,
        "sort": args.sort,
        "per_page": str(args.per_page),
        "cursor": "*",
        "select": ",".join(DEFAULT_SELECT_FIELDS),
    }
    if args.dry_run:
        print(f"{OPENALEX_WORKS_URL}?{urlencode(preview_params)}")
        return

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
    write_jsonl(records, output_path)
    metadata["records_written"] = len(records)
    metadata_path = write_fetch_metadata(output_path, metadata)

    print(f"Fetched records: {len(records)}")
    print(f"JSONL: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
