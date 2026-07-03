import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "eval" / "v2_5_hard_negative_labeling_packet.jsonl"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_SELECTED_240 = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"

EXPECTED_V21_LABELS_SHA256 = "0ECC4A0F638E77E76D44913281EC6431ADDF1873474C3B80A48531B9529359F7"
EXPECTED_SELECTED_240_SHA256 = "6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F"

SCHEMA_VERSION_VALUES = {"v2.1", "v2.5_hard_negative_label", "v2.5_manual_label"}
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


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in rows:
        query_id = row.get("query_id")
        paper_id = safe_int(row.get("paper_id"))
        if isinstance(query_id, str) and paper_id is not None:
            keys.add((query_id, paper_id))
    return keys


def validate_labels(
    *,
    labels: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    v21_labels: list[dict[str, Any]],
    v21_labels_path: Path,
    selected_240_path: Path,
) -> dict[str, Any]:
    candidate_keys = load_keys(candidates)
    v21_keys = load_keys(v21_labels)
    valid_query_ids = {query_id for query_id, _ in candidate_keys}

    seen: dict[tuple[str, int], int] = {}
    missing_required_fields: list[dict[str, Any]] = []
    invalid_values: list[dict[str, Any]] = []
    duplicate_query_paper_rows: list[dict[str, Any]] = []
    labels_not_in_candidates: list[dict[str, Any]] = []
    accidental_v21_duplicates: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for index, label in enumerate(labels, start=1):
        line_number = int(label.get("_line_number", index))
        query_id = label.get("query_id")
        paper_id = safe_int(label.get("paper_id"))
        missing = sorted(REQUIRED_FIELDS - set(label))
        if missing:
            missing_required_fields.append({"line": line_number, "missing": missing})
        if isinstance(query_id, str) and paper_id is not None:
            key = (query_id, paper_id)
            if key in seen:
                duplicate_query_paper_rows.append(
                    {"query_id": query_id, "paper_id": paper_id, "first_line": seen[key], "line": line_number}
                )
            else:
                seen[key] = line_number
            if key not in candidate_keys:
                labels_not_in_candidates.append({"query_id": query_id, "paper_id": paper_id, "line": line_number})
            if key in v21_keys:
                accidental_v21_duplicates.append({"query_id": query_id, "paper_id": paper_id, "line": line_number})
            if query_id not in valid_query_ids:
                invalid_values.append(
                    {"line": line_number, "field": "query_id", "value": query_id, "expected": "V2.5 candidate topic"}
                )
        else:
            invalid_values.append(
                {"line": line_number, "field": "query_id/paper_id", "value": [query_id, label.get("paper_id")], "expected": "string/integer"}
            )

        validate_identity(label, line_number, invalid_values)
        for field_name in sorted(REQUIRED_SCORE_FIELDS):
            validate_score(label, field_name, line_number, invalid_values)
        validate_intent_scores(label, line_number, invalid_values)
        validate_enums(label, line_number, invalid_values)
        add_soft_consistency_warnings(label, line_number, warnings)

    labeled_keys = set(seen)
    unlabeled_candidates = sorted(candidate_keys - labeled_keys)
    extra_labeled_keys = sorted(labeled_keys - candidate_keys)
    per_topic = per_topic_summary(labels, valid_query_ids)
    v21_labels_hash = file_sha256(v21_labels_path)
    selected_240_hash = file_sha256(selected_240_path)
    schema_valid = not (
        missing_required_fields
        or invalid_values
        or duplicate_query_paper_rows
        or labels_not_in_candidates
        or accidental_v21_duplicates
    )
    complete = len(labeled_keys & candidate_keys) == len(candidate_keys) and not extra_labeled_keys
    return {
        "label_count": len(labels),
        "candidate_count": len(candidate_keys),
        "labeled_candidate_count": len(labeled_keys & candidate_keys),
        "unlabeled_candidate_count": len(unlabeled_candidates),
        "unlabeled_candidate_examples": [
            {"query_id": query_id, "paper_id": paper_id}
            for query_id, paper_id in unlabeled_candidates[:20]
        ],
        "missing_required_fields": missing_required_fields,
        "invalid_values": invalid_values,
        "duplicate_query_paper_rows": duplicate_query_paper_rows,
        "labels_not_in_candidates": labels_not_in_candidates,
        "accidental_v2_1_label_duplicates": accidental_v21_duplicates,
        "warnings": warnings,
        "warning_count": len(warnings),
        "per_topic": per_topic,
        "hash_checks": {
            "manual_labels_v2_1_sha256": v21_labels_hash,
            "manual_labels_v2_1_unchanged": v21_labels_hash == EXPECTED_V21_LABELS_SHA256,
            "selected_240_sha256": selected_240_hash,
            "selected_240_unchanged": selected_240_hash == EXPECTED_SELECTED_240_SHA256,
        },
        "schema_valid": schema_valid,
        "complete": complete,
        "is_valid": schema_valid and complete,
    }


def validate_identity(label: dict[str, Any], line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if "schema_version" in label and label.get("schema_version") not in SCHEMA_VERSION_VALUES:
        invalid_values.append(
            {
                "line": line_number,
                "field": "schema_version",
                "value": label.get("schema_version"),
                "expected": sorted(SCHEMA_VERSION_VALUES),
            }
        )
    if "paper_id" in label and safe_int(label.get("paper_id")) is None:
        invalid_values.append({"line": line_number, "field": "paper_id", "value": label.get("paper_id"), "expected": "integer"})
    for field_name in ("query_id", "query", "title", "notes"):
        if field_name in label and not isinstance(label.get(field_name), str):
            invalid_values.append(
                {"line": line_number, "field": field_name, "value": label.get(field_name), "expected": "string"}
            )


def validate_score(label: dict[str, Any], field_name: str, line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if field_name not in label:
        return
    value = label.get(field_name)
    if not is_score(value):
        invalid_values.append({"line": line_number, "field": field_name, "value": value, "expected": "number in [0, 1]"})


def validate_intent_scores(label: dict[str, Any], line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if "intent_scores" not in label:
        return
    scores = label.get("intent_scores")
    if not isinstance(scores, dict):
        invalid_values.append({"line": line_number, "field": "intent_scores", "value": scores, "expected": "object"})
        return
    missing = sorted(INTENT_SCORE_FIELDS - set(scores))
    if missing:
        invalid_values.append({"line": line_number, "field": "intent_scores", "value": missing, "expected": "all intent score fields"})
    for field_name in sorted(INTENT_SCORE_FIELDS):
        if field_name in scores and not is_score(scores[field_name]):
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
            invalid_values.append({"line": line_number, "field": "secondary_roles", "value": roles, "expected": sorted(ROLE_VALUES)})
    if "full_text_available" in label and not isinstance(label.get("full_text_available"), bool):
        invalid_values.append({"line": line_number, "field": "full_text_available", "value": label.get("full_text_available"), "expected": "boolean"})
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
    if field_name in label and label.get(field_name) not in allowed:
        invalid_values.append({"line": line_number, "field": field_name, "value": label.get(field_name), "expected": sorted(allowed)})


def add_soft_consistency_warnings(label: dict[str, Any], line_number: int, warnings: list[dict[str, Any]]) -> None:
    topic_match = numeric(label.get("topic_match_score"))
    reading_value = numeric(label.get("reading_value_score"))
    primary_role = label.get("primary_role")
    duplicate_status = label.get("duplicate_status")
    if reading_value is not None and topic_match is not None and reading_value >= 0.7 and topic_match < 0.5:
        warnings.append({"line": line_number, "code": "high_reading_value_low_topic_match"})
    if primary_role == "negative" and reading_value is not None and reading_value > 0.25:
        warnings.append({"line": line_number, "code": "negative_role_high_reading_value"})
    if primary_role == "duplicate" and duplicate_status == "none":
        warnings.append({"line": line_number, "code": "duplicate_role_without_duplicate_status"})
    if duplicate_status in {"near_duplicate", "exact_duplicate"} and label.get("duplicate_of_paper_id") is None:
        warnings.append({"line": line_number, "code": "duplicate_missing_target"})


def per_topic_summary(labels: list[dict[str, Any]], valid_query_ids: set[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        grouped[str(label.get("query_id", ""))].append(label)
    summaries: dict[str, dict[str, Any]] = {}
    for query_id in sorted(valid_query_ids | set(grouped)):
        rows = grouped.get(query_id, [])
        summaries[query_id] = {
            "label_count": len(rows),
            "primary_role_distribution": dict(sorted(Counter(str(row.get("primary_role")) for row in rows).items())),
            "duplicate_status_distribution": dict(sorted(Counter(str(row.get("duplicate_status")) for row in rows).items())),
        }
    return summaries


def is_score(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def numeric(value: Any) -> float | None:
    return float(value) if is_score(value) else None


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    print(f"Labels: {report['label_count']}")
    print(f"Candidates: {report['candidate_count']}")
    print(f"Labeled candidates: {report['labeled_candidate_count']}")
    print(f"Unlabeled candidates: {report['unlabeled_candidate_count']}")
    print(f"Schema valid: {report['schema_valid']}")
    print(f"Complete: {report['complete']}")
    print(f"Valid: {report['is_valid']}")
    print(f"Missing required fields: {len(report['missing_required_fields'])}")
    print(f"Invalid values: {len(report['invalid_values'])}")
    print(f"Duplicate query-paper rows: {len(report['duplicate_query_paper_rows'])}")
    print(f"Labels not in V2.5 candidates: {len(report['labels_not_in_candidates'])}")
    print(f"Accidental V2.1 duplicates: {len(report['accidental_v2_1_label_duplicates'])}")
    print(f"V2.1 labels unchanged: {report['hash_checks']['manual_labels_v2_1_unchanged']}")
    print(f"Selected 240 unchanged: {report['hash_checks']['selected_240_unchanged']}")
    print("Per-topic label counts:")
    for query_id, summary in report["per_topic"].items():
        print(f"- {query_id}: {summary['label_count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate V2.5 hard-negative manual labels.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240))
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = resolve_repo_path(args.labels)
    candidates_path = resolve_repo_path(args.candidates)
    v21_labels_path = resolve_repo_path(args.v21_labels)
    selected_240_path = resolve_repo_path(args.selected_240)
    report = validate_labels(
        labels=load_jsonl(labels_path, missing_ok=True),
        candidates=load_jsonl(candidates_path),
        v21_labels=load_jsonl(v21_labels_path),
        v21_labels_path=v21_labels_path,
        selected_240_path=selected_240_path,
    )
    print_report(report)
    if args.json_out:
        write_json(report, resolve_repo_path(args.json_out))
    if not report["schema_valid"] or (not args.allow_incomplete and not report["complete"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
