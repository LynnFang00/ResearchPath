import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "query_id",
    "query",
    "paper_id",
    "title",
    "relevance",
    "best_section",
    "section_fit",
    "difficulty",
    "difficulty_fit_for_beginner",
    "too_narrow",
    "too_advanced",
    "duplicate",
    "would_recommend",
    "notes",
}
SECTIONS = {"background", "foundational", "core_methods", "recent_frontier"}
DIFFICULTIES = {"beginner", "intermediate", "advanced"}
FULLTEXT_FIELDS = {"evidence_level", "full_text_available", "label_confidence"}
EVIDENCE_LEVELS = {"abstract_only", "full_text_skim", "full_text_read"}
LABEL_CONFIDENCE = {"low", "medium", "high"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File was not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: row must be a JSON object.")
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def load_pool_keys(path: Path | None) -> set[tuple[str, int]]:
    if path is None:
        return set()
    return {
        (str(row.get("query_id", "")), int(row.get("paper_id", -1)))
        for row in load_jsonl(path)
        if row.get("query_id") and row.get("paper_id") is not None
    }


def validate_labels(
    labels: list[dict[str, Any]],
    pool_keys: set[tuple[str, int]] | None = None,
    *,
    require_fulltext_fields: bool = False,
) -> dict[str, Any]:
    pool_keys = pool_keys or set()
    missing_required_fields: list[dict[str, Any]] = []
    invalid_values: list[dict[str, Any]] = []
    duplicate_query_paper_rows: list[dict[str, Any]] = []
    labels_not_in_pool: list[dict[str, Any]] = []
    seen: dict[tuple[str, int], int] = {}

    labels_per_query: Counter[str] = Counter()
    relevance_distribution: Counter[int] = Counter()
    section_distribution: Counter[str] = Counter()
    difficulty_distribution: Counter[str] = Counter()
    would_recommend_count = 0
    duplicate_count = 0

    for index, label in enumerate(labels, start=1):
        line_number = int(label.get("_line_number", index))
        missing = sorted(REQUIRED_FIELDS - set(label))
        if require_fulltext_fields:
            missing = sorted(set(missing) | (FULLTEXT_FIELDS - set(label)))
        if missing:
            missing_required_fields.append({"line": line_number, "missing": missing})

        query_id = str(label.get("query_id", ""))
        paper_id = _safe_int(label.get("paper_id"))
        if query_id and paper_id is not None:
            key = (query_id, paper_id)
            if key in seen:
                duplicate_query_paper_rows.append(
                    {"query_id": query_id, "paper_id": paper_id, "first_line": seen[key], "line": line_number}
                )
            else:
                seen[key] = line_number
            if pool_keys and key not in pool_keys:
                labels_not_in_pool.append({"query_id": query_id, "paper_id": paper_id, "line": line_number})
            labels_per_query[query_id] += 1

        _validate_int_range(label, "relevance", 0, 3, line_number, invalid_values)
        _validate_int_range(label, "section_fit", 0, 3, line_number, invalid_values)
        _validate_int_range(label, "difficulty_fit_for_beginner", 0, 3, line_number, invalid_values)
        _validate_allowed(label, "best_section", SECTIONS, line_number, invalid_values)
        _validate_allowed(label, "difficulty", DIFFICULTIES, line_number, invalid_values)
        for field_name in ("too_narrow", "too_advanced", "duplicate", "would_recommend"):
            _validate_bool(label, field_name, line_number, invalid_values)
        if "evidence_level" in label:
            _validate_allowed(label, "evidence_level", EVIDENCE_LEVELS, line_number, invalid_values)
        if "label_confidence" in label:
            _validate_allowed(label, "label_confidence", LABEL_CONFIDENCE, line_number, invalid_values)
        if "full_text_available" in label:
            _validate_bool(label, "full_text_available", line_number, invalid_values)

        if isinstance(label.get("relevance"), int):
            relevance_distribution[int(label["relevance"])] += 1
        if isinstance(label.get("best_section"), str):
            section_distribution[str(label["best_section"])] += 1
        if isinstance(label.get("difficulty"), str):
            difficulty_distribution[str(label["difficulty"])] += 1
        if label.get("would_recommend") is True:
            would_recommend_count += 1
        if label.get("duplicate") is True:
            duplicate_count += 1

    return {
        "label_count": len(labels),
        "labels_per_query": dict(sorted(labels_per_query.items())),
        "relevance_distribution": dict(sorted(relevance_distribution.items())),
        "section_distribution": dict(sorted(section_distribution.items())),
        "difficulty_distribution": dict(sorted(difficulty_distribution.items())),
        "would_recommend_count": would_recommend_count,
        "duplicate_count": duplicate_count,
        "missing_required_fields": missing_required_fields,
        "invalid_values": invalid_values,
        "duplicate_query_paper_rows": duplicate_query_paper_rows,
        "labels_not_in_pool": labels_not_in_pool,
        "is_valid": not (
            missing_required_fields or invalid_values or duplicate_query_paper_rows or labels_not_in_pool
        ),
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Labels: {report['label_count']}")
    print(f"Labels per query: {report['labels_per_query']}")
    print(f"Relevance distribution: {report['relevance_distribution']}")
    print(f"Section distribution: {report['section_distribution']}")
    print(f"Difficulty distribution: {report['difficulty_distribution']}")
    print(f"Would recommend count: {report['would_recommend_count']}")
    print(f"Duplicate count: {report['duplicate_count']}")
    print(f"Missing required fields: {report['missing_required_fields']}")
    print(f"Invalid values: {report['invalid_values']}")
    print(f"Duplicate query_id plus paper_id rows: {report['duplicate_query_paper_rows']}")
    print(f"Labels not in candidate pool: {report['labels_not_in_pool']}")
    print(f"Valid: {report['is_valid']}")


def _validate_int_range(
    label: dict[str, Any],
    field_name: str,
    minimum: int,
    maximum: int,
    line_number: int,
    invalid_values: list[dict[str, Any]],
) -> None:
    value = label.get(field_name)
    if field_name not in label:
        return
    if not isinstance(value, int) or value < minimum or value > maximum:
        invalid_values.append(
            {"line": line_number, "field": field_name, "value": value, "expected": f"integer {minimum} to {maximum}"}
        )


def _validate_allowed(
    label: dict[str, Any],
    field_name: str,
    allowed: set[str],
    line_number: int,
    invalid_values: list[dict[str, Any]],
) -> None:
    value = label.get(field_name)
    if field_name not in label:
        return
    if value not in allowed:
        invalid_values.append(
            {"line": line_number, "field": field_name, "value": value, "expected": sorted(allowed)}
        )


def _validate_bool(
    label: dict[str, Any],
    field_name: str,
    line_number: int,
    invalid_values: list[dict[str, Any]],
) -> None:
    if field_name not in label:
        return
    value = label.get(field_name)
    if not isinstance(value, bool):
        invalid_values.append({"line": line_number, "field": field_name, "value": value, "expected": "boolean"})


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ResearchPath manual label JSONL files.")
    parser.add_argument("--labels", default="data/eval/manual_labels_v1.jsonl")
    parser.add_argument("--pool", default=None)
    parser.add_argument("--require-fulltext-fields", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    pool_path = Path(args.pool) if args.pool else None
    labels = load_jsonl(labels_path)
    pool_keys = load_pool_keys(pool_path)
    report = validate_labels(labels, pool_keys=pool_keys, require_fulltext_fields=args.require_fulltext_fields)
    print_report(report)
    if not report["is_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
