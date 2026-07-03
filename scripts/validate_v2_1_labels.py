import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"

SCHEMA_VERSION = "v2.1"
IDENTITY_FIELDS = {"schema_version", "query_id", "query", "paper_id", "title"}
REQUIRED_SCORE_FIELDS = {
    "topic_match_score",
    "reading_value_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
}
INTENT_SCORE_FIELDS = {
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
}
REQUIRED_FIELDS = IDENTITY_FIELDS | REQUIRED_SCORE_FIELDS | {
    "intent_scores",
    "primary_role",
    "secondary_roles",
    "duplicate_status",
    "duplicate_of_paper_id",
    "evidence_level",
    "full_text_available",
    "label_confidence",
    "notes",
}
ROLE_VALUES = {
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
    "negative",
    "duplicate",
    "uncertain",
}
DUPLICATE_STATUS_VALUES = {"none", "near_duplicate", "exact_duplicate", "uncertain"}
EVIDENCE_LEVELS = {"title_only", "title_abstract", "title_abstract_intro_conclusion", "fulltext_available"}
LABEL_CONFIDENCE_VALUES = {"low", "medium", "high"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File was not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected JSON object.")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def load_packet_keys(path: Path | None) -> set[tuple[str, int]]:
    if path is None:
        return set()
    keys: set[tuple[str, int]] = set()
    for row in load_jsonl(path):
        paper_id = safe_int(row.get("paper_id"))
        query_id = row.get("query_id")
        if isinstance(query_id, str) and paper_id is not None:
            keys.add((query_id, paper_id))
    return keys


def validate_labels(
    labels: list[dict[str, Any]],
    *,
    packet_keys: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    packet_keys = packet_keys or set()
    missing_required_fields: list[dict[str, Any]] = []
    invalid_values: list[dict[str, Any]] = []
    duplicate_query_paper_rows: list[dict[str, Any]] = []
    labels_not_in_packet: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen: dict[tuple[str, int], int] = {}

    for index, label in enumerate(labels, start=1):
        line_number = int(label.get("_line_number", index))
        missing = sorted(REQUIRED_FIELDS - set(label))
        if missing:
            missing_required_fields.append({"line": line_number, "missing": missing})

        query_id = label.get("query_id")
        paper_id = safe_int(label.get("paper_id"))
        if isinstance(query_id, str) and paper_id is not None:
            key = (query_id, paper_id)
            if key in seen:
                duplicate_query_paper_rows.append(
                    {"query_id": query_id, "paper_id": paper_id, "first_line": seen[key], "line": line_number}
                )
            else:
                seen[key] = line_number
            if packet_keys and key not in packet_keys:
                labels_not_in_packet.append({"query_id": query_id, "paper_id": paper_id, "line": line_number})

        validate_identity(label, line_number, invalid_values)
        for field_name in sorted(REQUIRED_SCORE_FIELDS):
            validate_score(label, field_name, line_number, invalid_values)
        validate_intent_scores(label, line_number, invalid_values)
        validate_enums(label, line_number, invalid_values)
        add_soft_consistency_warnings(label, line_number, warnings)

    per_topic = per_topic_summary(labels)
    return {
        "label_count": len(labels),
        "missing_required_fields": missing_required_fields,
        "invalid_values": invalid_values,
        "duplicate_query_paper_rows": duplicate_query_paper_rows,
        "labels_not_in_packet": labels_not_in_packet,
        "warnings": warnings,
        "warning_count": len(warnings),
        "per_topic": per_topic,
        "is_valid": not (
            missing_required_fields or invalid_values or duplicate_query_paper_rows or labels_not_in_packet
        ),
    }


def validate_identity(label: dict[str, Any], line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if label.get("schema_version") != SCHEMA_VERSION:
        invalid_values.append(
            {
                "line": line_number,
                "field": "schema_version",
                "value": label.get("schema_version"),
                "expected": SCHEMA_VERSION,
            }
        )
    if "paper_id" in label and safe_int(label.get("paper_id")) is None:
        invalid_values.append({"line": line_number, "field": "paper_id", "value": label.get("paper_id"), "expected": "integer"})
    for field_name in ("query_id", "query", "title", "notes"):
        if field_name in label and not isinstance(label.get(field_name), str):
            invalid_values.append(
                {"line": line_number, "field": field_name, "value": label.get(field_name), "expected": "string"}
            )


def validate_score(
    label: dict[str, Any],
    field_name: str,
    line_number: int,
    invalid_values: list[dict[str, Any]],
) -> None:
    if field_name not in label:
        return
    value = label.get(field_name)
    if not is_score(value):
        invalid_values.append(
            {"line": line_number, "field": field_name, "value": value, "expected": "number in [0, 1]"}
        )


def validate_intent_scores(label: dict[str, Any], line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if "intent_scores" not in label:
        return
    scores = label.get("intent_scores")
    if not isinstance(scores, dict):
        invalid_values.append(
            {"line": line_number, "field": "intent_scores", "value": scores, "expected": "object"}
        )
        return
    missing = sorted(INTENT_SCORE_FIELDS - set(scores))
    if missing:
        invalid_values.append(
            {"line": line_number, "field": "intent_scores", "value": missing, "expected": "all intent score fields"}
        )
    for field_name in sorted(INTENT_SCORE_FIELDS):
        if field_name not in scores:
            continue
        if not is_score(scores[field_name]):
            invalid_values.append(
                {
                    "line": line_number,
                    "field": f"intent_scores.{field_name}",
                    "value": scores[field_name],
                    "expected": "number in [0, 1]",
                }
            )


def validate_enums(label: dict[str, Any], line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    validate_allowed(label, "primary_role", ROLE_VALUES, line_number, invalid_values)
    validate_allowed(label, "duplicate_status", DUPLICATE_STATUS_VALUES, line_number, invalid_values)
    validate_allowed(label, "evidence_level", EVIDENCE_LEVELS, line_number, invalid_values)
    validate_allowed(label, "label_confidence", LABEL_CONFIDENCE_VALUES, line_number, invalid_values)

    if "secondary_roles" in label:
        roles = label.get("secondary_roles")
        if not isinstance(roles, list) or any(not isinstance(role, str) or role not in ROLE_VALUES for role in roles):
            invalid_values.append(
                {
                    "line": line_number,
                    "field": "secondary_roles",
                    "value": roles,
                    "expected": sorted(ROLE_VALUES),
                }
            )
    if "full_text_available" in label and not isinstance(label.get("full_text_available"), bool):
        invalid_values.append(
            {
                "line": line_number,
                "field": "full_text_available",
                "value": label.get("full_text_available"),
                "expected": "boolean",
            }
        )
    if "duplicate_of_paper_id" in label and label.get("duplicate_of_paper_id") is not None:
        if safe_int(label.get("duplicate_of_paper_id")) is None:
            invalid_values.append(
                {
                    "line": line_number,
                    "field": "duplicate_of_paper_id",
                    "value": label.get("duplicate_of_paper_id"),
                    "expected": "integer or null",
                }
            )


def validate_allowed(
    label: dict[str, Any],
    field_name: str,
    allowed: set[str],
    line_number: int,
    invalid_values: list[dict[str, Any]],
) -> None:
    if field_name not in label:
        return
    value = label.get(field_name)
    if value not in allowed:
        invalid_values.append(
            {"line": line_number, "field": field_name, "value": value, "expected": sorted(allowed)}
        )


def add_soft_consistency_warnings(label: dict[str, Any], line_number: int, warnings: list[dict[str, Any]]) -> None:
    topic_match = numeric(label.get("topic_match_score"))
    reading_value = numeric(label.get("reading_value_score"))
    beginner_fit = numeric(label.get("beginner_fit_score"))
    expert_fit = numeric(label.get("expert_fit_score"))
    primary_role = label.get("primary_role")
    duplicate_status = label.get("duplicate_status")
    notes = str(label.get("notes") or "")

    if reading_value is not None and topic_match is not None and reading_value >= 0.7 and topic_match < 0.5:
        warn(warnings, line_number, "high_reading_value_low_topic_match", "reading_value_score is high while topic_match_score is low.")
    if primary_role == "negative" and reading_value is not None and reading_value > 0.25:
        warn(warnings, line_number, "negative_role_high_reading_value", "primary_role is negative but reading_value_score is above 0.25.")
    if primary_role == "duplicate" and duplicate_status == "none":
        warn(warnings, line_number, "duplicate_role_without_duplicate_status", "primary_role is duplicate but duplicate_status is none.")
    if duplicate_status in {"near_duplicate", "exact_duplicate"} and label.get("duplicate_of_paper_id") is None:
        warn(warnings, line_number, "duplicate_missing_target", "duplicate_status requires duplicate_of_paper_id when known.")
    roles = {str(primary_role)}
    secondary_roles = label.get("secondary_roles")
    if isinstance(secondary_roles, list):
        roles.update(str(role) for role in secondary_roles)
    broad_reference_roles = {"background", "foundational", "evaluation_benchmark"}
    if (
        beginner_fit is not None
        and expert_fit is not None
        and beginner_fit >= 0.8
        and expert_fit >= 0.8
        and not roles.intersection(broad_reference_roles)
    ):
        warn(warnings, line_number, "high_beginner_and_expert_fit", "Both beginner and expert fit are high.")
    is_duplicate = duplicate_status in {"near_duplicate", "exact_duplicate"} or primary_role == "duplicate"
    if (
        topic_match is not None
        and reading_value is not None
        and topic_match >= 0.8
        and reading_value <= 0.2
        and not is_duplicate
    ):
        warn(warnings, line_number, "high_topic_match_low_reading_value", "Strong topic match but very low reading value.")
    if label.get("label_confidence") == "high" and len(notes.strip()) < 20:
        warn(warnings, line_number, "high_confidence_short_notes", "High-confidence labels should include a short rationale.")


def warn(warnings: list[dict[str, Any]], line_number: int, code: str, message: str) -> None:
    warnings.append({"line": line_number, "code": code, "message": message})


def per_topic_summary(labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        grouped[str(label.get("query_id", ""))].append(label)

    summaries: dict[str, dict[str, Any]] = {}
    for query_id, rows in sorted(grouped.items()):
        summaries[query_id] = {
            "label_count": len(rows),
            "primary_role_distribution": dict(sorted(Counter(str(row.get("primary_role")) for row in rows).items())),
            "duplicate_status_distribution": dict(
                sorted(Counter(str(row.get("duplicate_status")) for row in rows).items())
            ),
            "label_confidence_distribution": dict(
                sorted(Counter(str(row.get("label_confidence")) for row in rows).items())
            ),
            "score_averages": {
                field_name: round(mean_score(rows, field_name), 4) for field_name in sorted(REQUIRED_SCORE_FIELDS)
            },
            "intent_score_averages": {
                field_name: round(mean_intent_score(rows, field_name), 4) for field_name in sorted(INTENT_SCORE_FIELDS)
            },
            "topic_match_buckets": score_buckets(row.get("topic_match_score") for row in rows),
            "reading_value_buckets": score_buckets(row.get("reading_value_score") for row in rows),
        }
    return summaries


def score_buckets(values: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        score = numeric(value)
        if score is None:
            counter["invalid_or_missing"] += 1
        elif score < 0.34:
            counter["low"] += 1
        elif score < 0.67:
            counter["medium"] += 1
        else:
            counter["high"] += 1
    return dict(sorted(counter.items()))


def mean_score(rows: list[dict[str, Any]], field_name: str) -> float:
    scores = [score for row in rows if (score := numeric(row.get(field_name))) is not None]
    return sum(scores) / len(scores) if scores else 0.0


def mean_intent_score(rows: list[dict[str, Any]], field_name: str) -> float:
    scores: list[float] = []
    for row in rows:
        intent_scores = row.get("intent_scores")
        if isinstance(intent_scores, dict):
            score = numeric(intent_scores.get(field_name))
            if score is not None:
                scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def is_score(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0


def numeric(value: Any) -> float | None:
    return float(value) if is_score(value) else None


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    print(f"Labels: {report['label_count']}")
    print(f"Valid: {report['is_valid']}")
    print(f"Warnings: {report['warning_count']}")
    print(f"Missing required fields: {len(report['missing_required_fields'])}")
    print(f"Invalid values: {len(report['invalid_values'])}")
    print(f"Duplicate query-paper rows: {len(report['duplicate_query_paper_rows'])}")
    print(f"Labels not in packet: {len(report['labels_not_in_packet'])}")
    print("Per-topic distribution summaries:")
    for query_id, summary in report["per_topic"].items():
        print(
            f"- {query_id}: labels={summary['label_count']}, "
            f"roles={summary['primary_role_distribution']}, "
            f"topic_match={summary['topic_match_buckets']}, "
            f"reading_value={summary['reading_value_buckets']}"
        )


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate V2.1 continuous-score manual labels.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_jsonl(resolve_repo_path(args.labels))
    packet_path = resolve_repo_path(args.packet) if args.packet else None
    packet_keys = load_packet_keys(packet_path)
    report = validate_labels(labels, packet_keys=packet_keys)
    print_report(report)
    if args.json_out:
        write_json(report, resolve_repo_path(args.json_out))
    if not report["is_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
