import argparse
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_v2_1_labels import load_jsonl, load_packet_keys, validate_labels  # noqa: E402

DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"


def read_input_rows(input_path: Path | None) -> list[dict[str, Any]]:
    if input_path is not None:
        return load_jsonl(input_path)
    text = sys.stdin.read()
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    try:
        return load_jsonl(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "_line_number"}
            handle.write(json.dumps(clean, ensure_ascii=True) + "\n")


def merge_labels(existing: list[dict[str, Any]], incoming: list[dict[str, Any]], *, replace: bool) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, int]] = []
    for row in existing:
        key = label_key(row)
        merged_by_key[key] = row
        order.append(key)

    duplicates = [label_key(row) for row in incoming if label_key(row) in merged_by_key]
    if duplicates and not replace:
        duplicate_text = ", ".join(f"{query_id}/{paper_id}" for query_id, paper_id in duplicates[:10])
        raise ValueError(f"Refusing to append duplicate labels without --replace: {duplicate_text}")

    for row in incoming:
        key = label_key(row)
        if key not in merged_by_key:
            order.append(key)
        merged_by_key[key] = row
    return [merged_by_key[key] for key in order]


def label_key(row: dict[str, Any]) -> tuple[str, int]:
    query_id = row.get("query_id")
    paper_id = row.get("paper_id")
    if not isinstance(query_id, str) or not isinstance(paper_id, int):
        raise ValueError(f"Label row is missing valid query_id/paper_id: {row}")
    return query_id, paper_id


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append validated V2.1 label JSONL rows.")
    parser.add_argument("--input", default=None, help="Input JSONL file. If omitted, reads stdin.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--replace", action="store_true", help="Replace existing labels with matching query_id/paper_id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input) if args.input else None
    labels_path = resolve_repo_path(args.labels)
    packet_path = resolve_repo_path(args.packet)

    incoming = read_input_rows(input_path)
    packet_keys = load_packet_keys(packet_path)
    incoming_report = validate_labels(incoming, packet_keys=packet_keys)
    if not incoming_report["is_valid"]:
        print(json.dumps(incoming_report, indent=2, ensure_ascii=True))
        raise SystemExit(1)

    existing = load_jsonl(labels_path) if labels_path.exists() else []
    merged = merge_labels(existing, incoming, replace=args.replace)
    merged_report = validate_labels(merged, packet_keys=packet_keys)
    if not merged_report["is_valid"]:
        print(json.dumps(merged_report, indent=2, ensure_ascii=True))
        raise SystemExit(1)

    write_jsonl(merged, labels_path)
    print(
        json.dumps(
            {
                "labels_path": str(labels_path),
                "incoming_labels": len(incoming),
                "total_labels": len(merged),
                "labels_remaining": max(0, len(packet_keys) - len(merged)),
                "replace": args.replace,
                "warnings": merged_report["warning_count"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
