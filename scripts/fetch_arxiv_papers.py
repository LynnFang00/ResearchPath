import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "arxiv_ml_ai_papers.jsonl"
ARXIV_OAI_URL = "https://export.arxiv.org/oai2"
USER_AGENT = "ResearchPath/0.1 arXiv metadata harvester (local research corpus; contact: local)"
OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arxiv": "http://arxiv.org/OAI/arXiv/"}
DEFAULT_SETS = ["cs", "stat", "eess"]
DEFAULT_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.IR",
    "cs.LG",
    "cs.NE",
    "cs.RO",
    "stat.ML",
    "eess.IV",
]


def text_or_empty(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return normalize_whitespace(element.text)


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def record_arxiv_id(record: ET.Element) -> str:
    metadata_id = text_or_empty(record.find(".//arxiv:id", OAI_NS))
    if metadata_id:
        return metadata_id
    header_id = text_or_empty(record.find("./oai:header/oai:identifier", OAI_NS))
    return header_id.rsplit(":", 1)[-1] if header_id else ""


def record_authors(record: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in record.findall(".//arxiv:authors/arxiv:author", OAI_NS):
        keyname = text_or_empty(author.find("./arxiv:keyname", OAI_NS))
        forenames = text_or_empty(author.find("./arxiv:forenames", OAI_NS))
        suffix = text_or_empty(author.find("./arxiv:suffix", OAI_NS))
        parts = [part for part in (forenames, keyname, suffix) if part]
        if parts:
            authors.append(" ".join(parts))
    return authors


def categories_for_record(record: ET.Element) -> list[str]:
    categories = text_or_empty(record.find(".//arxiv:categories", OAI_NS))
    return [category for category in categories.split() if category.strip()]


def abstract_word_count(abstract: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", abstract))


def updated_at(record: ET.Element) -> str | None:
    updated = text_or_empty(record.find(".//arxiv:updated", OAI_NS))
    if updated:
        return f"{updated}T00:00:00+00:00" if "T" not in updated else updated
    datestamp = text_or_empty(record.find("./oai:header/oai:datestamp", OAI_NS))
    if datestamp:
        return f"{datestamp}T00:00:00+00:00"
    return None


def arxiv_record_to_paper(record: ET.Element) -> dict[str, Any] | None:
    arxiv_id = record_arxiv_id(record)
    title = text_or_empty(record.find(".//arxiv:title", OAI_NS))
    abstract = text_or_empty(record.find(".//arxiv:abstract", OAI_NS))
    if not arxiv_id or not title or not abstract:
        return None

    created = text_or_empty(record.find(".//arxiv:created", OAI_NS))
    doi = text_or_empty(record.find(".//arxiv:doi", OAI_NS)) or None
    journal_ref = text_or_empty(record.find(".//arxiv:journal-ref", OAI_NS)) or None
    categories = categories_for_record(record)
    source_url = f"https://arxiv.org/abs/{arxiv_id}"
    year = int(created[:4]) if len(created) >= 4 and created[:4].isdigit() else None

    return {
        "external_id": f"arxiv:{arxiv_id}",
        "source": "arxiv",
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": record_authors(record),
        "year": year,
        "venue": journal_ref,
        "categories": categories,
        "citation_count": 0,
        "source_url": source_url,
        "url": source_url,
        "references_count": 0,
        "influential_citation_count": 0,
        "abstract_word_count": abstract_word_count(abstract),
        "updated_at": updated_at(record),
        "references": [],
        "identifiers": {
            "arxiv": arxiv_id,
            **({"doi": doi} if doi else {}),
        },
        "raw_source_metadata": {
            "source": "arxiv_oai",
            "arxiv_id": arxiv_id,
            "created": created,
            "updated": text_or_empty(record.find(".//arxiv:updated", OAI_NS)),
            "datestamp": text_or_empty(record.find("./oai:header/oai:datestamp", OAI_NS)),
            "categories": categories,
            "journal_ref": journal_ref,
            "doi": doi,
        },
    }


def fetch_xml(params: dict[str, str]) -> ET.Element:
    url = f"{ARXIV_OAI_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return ET.fromstring(response.read())


def iter_oai_records(
    *,
    set_spec: str,
    from_date: str | None,
    until_date: str | None,
    sleep_seconds: float,
) -> Any:
    params = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": set_spec,
    }
    if from_date:
        params["from"] = from_date
    if until_date:
        params["until"] = until_date

    while True:
        root = fetch_xml(params)
        error = root.find(".//oai:error", OAI_NS)
        if error is not None:
            code = error.attrib.get("code", "unknown")
            message = text_or_empty(error)
            raise RuntimeError(f"arXiv OAI error for set={set_spec}: {code}: {message}")

        for record in root.findall(".//oai:ListRecords/oai:record", OAI_NS):
            header = record.find("./oai:header", OAI_NS)
            if header is not None and header.attrib.get("status") == "deleted":
                continue
            yield record

        token = text_or_empty(root.find(".//oai:resumptionToken", OAI_NS))
        if not token:
            break
        params = {"verb": "ListRecords", "resumptionToken": token}
        time.sleep(sleep_seconds)


def fetch_arxiv_records(
    *,
    set_specs: list[str],
    categories: set[str],
    max_records: int,
    from_date: str | None,
    until_date: str | None,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_arxiv_ids: set[str] = set()
    for set_spec in set_specs:
        for record in iter_oai_records(
            set_spec=set_spec,
            from_date=from_date,
            until_date=until_date,
            sleep_seconds=sleep_seconds,
        ):
            paper = arxiv_record_to_paper(record)
            if paper is None:
                continue
            arxiv_id = str(paper["identifiers"]["arxiv"])
            if arxiv_id in seen_arxiv_ids:
                continue
            paper_categories = set(paper.get("categories") or [])
            if categories and not paper_categories.intersection(categories):
                continue
            seen_arxiv_ids.add(arxiv_id)
            records.append(paper)
            if len(records) >= max_records:
                return records
        time.sleep(sleep_seconds)
    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_metadata(output_path: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch arXiv OAI metadata into ResearchPath JSONL format.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--sets", default=",".join(DEFAULT_SETS), help="Comma-separated arXiv OAI sets, e.g. cs,stat,eess.")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--from-date", default=None, help="OAI from date, YYYY-MM-DD.")
    parser.add_argument("--until-date", default=None, help="OAI until date, YYYY-MM-DD.")
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    set_specs = parse_csv(args.sets)
    categories = set(parse_csv(args.categories))
    metadata = {
        "dataset_name": output_path.stem,
        "source": "arxiv_oai",
        "created_at": datetime.now(UTC).isoformat(),
        "sets": set_specs,
        "categories": sorted(categories),
        "from_date": args.from_date,
        "until_date": args.until_date,
        "max_records": args.max_records,
        "notes": "arXiv OAI metadata normalized to ResearchPath ingestion JSONL.",
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    records = fetch_arxiv_records(
        set_specs=set_specs,
        categories=categories,
        max_records=args.max_records,
        from_date=args.from_date,
        until_date=args.until_date,
        sleep_seconds=args.sleep_seconds,
    )
    write_jsonl(records, output_path)
    metadata["records_written"] = len(records)
    metadata_path = write_metadata(output_path, metadata)
    print(f"Fetched records: {len(records)}")
    print(f"JSONL: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
