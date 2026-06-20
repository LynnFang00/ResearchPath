import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fetch_openalex_papers import (  # noqa: E402
    DEFAULT_SELECT_FIELDS,
    OPENALEX_WORKS_URL,
    openalex_work_to_paper_record,
    write_fetch_metadata,
    write_jsonl,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "openalex_agents_scidisc.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "openalex_agents_scidisc_expanded.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file was not found: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def collect_missing_reference_counts(records: list[dict[str, Any]]) -> Counter[str]:
    existing_ids = {record.get("external_id") for record in records if record.get("external_id")}
    counts: Counter[str] = Counter()
    for record in records:
        for reference in record.get("references") or []:
            if reference and reference not in existing_ids:
                counts[reference] += 1
    return counts


def select_reference_ids(
    counts: Counter[str],
    *,
    max_references: int,
    min_frequency: int,
) -> list[str]:
    return [
        reference_id
        for reference_id, count in counts.most_common()
        if count >= min_frequency
    ][:max_references]


def openalex_path_id(external_id: str) -> str:
    if external_id.startswith("openalex:"):
        return external_id.removeprefix("openalex:")
    return external_id


def fetch_openalex_work(
    external_id: str,
    *,
    api_key: str | None,
    email: str | None,
) -> dict[str, Any] | None:
    work_id = openalex_path_id(external_id)
    params = {"select": ",".join(DEFAULT_SELECT_FIELDS)}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["mailto"] = email
    url = f"{OPENALEX_WORKS_URL}/{work_id}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "ResearchPath/0.1 (local portfolio project)"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def expand_reference_records(
    reference_ids: list[str],
    fetcher: Callable[[str], dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for reference_id in reference_ids:
        try:
            work = fetcher(reference_id)
        except Exception as exc:  # pragma: no cover - network behavior is smoke-tested manually.
            warnings.append(f"Failed to fetch {reference_id}: {exc}")
            continue
        if not work:
            warnings.append(f"Skipped {reference_id}: OpenAlex returned no work.")
            continue

        record = openalex_work_to_paper_record(work)
        if not record["title"] or not record["abstract"]:
            warnings.append(f"Skipped {reference_id}: missing title or abstract.")
            continue
        records.append(record)
    return records, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch missing OpenAlex references from a seed JSONL file.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Seed OpenAlex JSONL file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL for expanded reference papers.")
    parser.add_argument("--max-references", type=int, default=100, help="Maximum missing references to fetch.")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum reference frequency in the seed set.")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument("--email", default=os.environ.get("OPENALEX_EMAIL"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    seed_records = load_jsonl(input_path)
    missing_counts = collect_missing_reference_counts(seed_records)
    reference_ids = select_reference_ids(
        missing_counts,
        max_references=args.max_references,
        min_frequency=args.min_frequency,
    )

    def fetcher(reference_id: str) -> dict[str, Any] | None:
        work = fetch_openalex_work(reference_id, api_key=args.api_key, email=args.email)
        time.sleep(args.sleep_seconds)
        return work

    expanded_records, warnings = expand_reference_records(reference_ids, fetcher)
    write_jsonl(expanded_records, output_path)
    metadata_path = write_fetch_metadata(
        output_path,
        {
            "dataset_name": output_path.stem,
            "source": "openalex_reference_expansion",
            "date_created": datetime.now(UTC).isoformat(),
            "input": str(input_path),
            "output": str(output_path),
            "seed_records": len(seed_records),
            "candidate_missing_references": len(missing_counts),
            "selected_references": len(reference_ids),
            "records_written": len(expanded_records),
            "min_frequency": args.min_frequency,
            "max_references": args.max_references,
            "warnings": warnings,
        },
    )

    print(f"Seed records: {len(seed_records)}")
    print(f"Missing reference candidates: {len(missing_counts)}")
    print(f"Selected references: {len(reference_ids)}")
    print(f"Fetched records: {len(expanded_records)}")
    print(f"JSONL: {output_path}")
    print(f"Metadata: {metadata_path}")
    for warning in warnings[:25]:
        print(f"Warning: {warning}")
    if len(warnings) > 25:
        print(f"Warning: {len(warnings) - 25} additional warnings omitted. See metadata for details.")


if __name__ == "__main__":
    main()
