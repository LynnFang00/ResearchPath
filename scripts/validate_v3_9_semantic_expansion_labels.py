import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl"
DEFAULT_CANDIDATES = REPO_ROOT / "data" / "eval" / "labeling" / "v3_9_semantic_expansion_candidates.jsonl"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V32_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
DEFAULT_V35_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_5_unjudged_audit.jsonl"
DEFAULT_SELECTED_240 = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_9_semantic_expansion_label_progress.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_9_semantic_expansion_label_progress.md"

EXPECTED_V21_LABELS_SHA256 = "0ECC4A0F638E77E76D44913281EC6431ADDF1873474C3B80A48531B9529359F7"
EXPECTED_V25_LABELS_SHA256 = "F3CEFD7ED5C89D79796AD487C255879A8246DE87692289C9A3A6C67157F7453C"
EXPECTED_V32_LABELS_SHA256 = "84AFFD9480FB28758A258B85D89AE7A19C8C9620708AAD9050C72D6D34DD0A24"
EXPECTED_V35_LABELS_SHA256 = "55A593C3CFA3EC3148FCB9AA38F0081607065D2E506D7C057C9804A71B05113B"
EXPECTED_SELECTED_240_SHA256 = "6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F"

TARGET_TOTAL_LABELS = 2400
MIN_NEW_LABELS = 1569
SCHEMA_VERSION_VALUES = {"v3.9_semantic_expansion_manual_label"}
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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def is_score(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def load_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in rows:
        query_id = row.get("query_id")
        paper_id = safe_int(row.get("paper_id"))
        if isinstance(query_id, str) and paper_id is not None:
            keys.add((query_id, paper_id))
    return keys


def validate_identity(label: dict[str, Any], line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if "schema_version" in label and label.get("schema_version") not in SCHEMA_VERSION_VALUES:
        invalid_values.append({"line": line_number, "field": "schema_version", "value": label.get("schema_version"), "expected": sorted(SCHEMA_VERSION_VALUES)})
    if "paper_id" in label and safe_int(label.get("paper_id")) is None:
        invalid_values.append({"line": line_number, "field": "paper_id", "value": label.get("paper_id"), "expected": "integer"})
    for field_name in ("query_id", "query", "title", "notes"):
        if field_name in label and not isinstance(label.get(field_name), str):
            invalid_values.append({"line": line_number, "field": field_name, "value": label.get(field_name), "expected": "string"})


def validate_score(label: dict[str, Any], field_name: str, line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if field_name in label and not is_score(label.get(field_name)):
        invalid_values.append({"line": line_number, "field": field_name, "value": label.get(field_name), "expected": "number in [0, 1]"})


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
            invalid_values.append({"line": line_number, "field": f"intent_scores.{field_name}", "value": scores[field_name], "expected": "number in [0, 1]"})


def validate_enums(label: dict[str, Any], line_number: int, invalid_values: list[dict[str, Any]]) -> None:
    if "primary_role" in label and label.get("primary_role") not in ROLE_VALUES:
        invalid_values.append({"line": line_number, "field": "primary_role", "value": label.get("primary_role"), "expected": sorted(ROLE_VALUES)})
    if "secondary_roles" in label:
        secondary_roles = label.get("secondary_roles")
        if not isinstance(secondary_roles, list):
            invalid_values.append({"line": line_number, "field": "secondary_roles", "value": secondary_roles, "expected": "list"})
        else:
            for role in secondary_roles:
                if role not in ROLE_VALUES:
                    invalid_values.append({"line": line_number, "field": "secondary_roles", "value": role, "expected": sorted(ROLE_VALUES)})
    if "duplicate_status" in label and label.get("duplicate_status") not in DUPLICATE_STATUS_VALUES:
        invalid_values.append({"line": line_number, "field": "duplicate_status", "value": label.get("duplicate_status"), "expected": sorted(DUPLICATE_STATUS_VALUES)})
    if "evidence_level" in label and label.get("evidence_level") not in EVIDENCE_LEVELS:
        invalid_values.append({"line": line_number, "field": "evidence_level", "value": label.get("evidence_level"), "expected": sorted(EVIDENCE_LEVELS)})
    if "label_confidence" in label and label.get("label_confidence") not in LABEL_CONFIDENCE_VALUES:
        invalid_values.append({"line": line_number, "field": "label_confidence", "value": label.get("label_confidence"), "expected": sorted(LABEL_CONFIDENCE_VALUES)})
    if "full_text_available" in label and not isinstance(label.get("full_text_available"), bool):
        invalid_values.append({"line": line_number, "field": "full_text_available", "value": label.get("full_text_available"), "expected": "boolean"})


def protected_hashes(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v35_labels_path: Path,
    selected_240_path: Path,
) -> dict[str, Any]:
    v21_hash = file_sha256(v21_labels_path)
    v25_hash = file_sha256(v25_labels_path)
    v32_hash = file_sha256(v32_labels_path)
    v35_hash = file_sha256(v35_labels_path)
    selected_hash = file_sha256(selected_240_path)
    return {
        "v2_1_labels_sha256": v21_hash,
        "v2_1_labels_hash_unchanged": v21_hash == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": v25_hash,
        "v2_5_labels_hash_unchanged": v25_hash == EXPECTED_V25_LABELS_SHA256,
        "v3_2_labels_sha256": v32_hash,
        "v3_2_labels_hash_unchanged": v32_hash == EXPECTED_V32_LABELS_SHA256,
        "v3_5_labels_sha256": v35_hash,
        "v3_5_labels_hash_unchanged": v35_hash == EXPECTED_V35_LABELS_SHA256,
        "selected_240_sha256": selected_hash,
        "selected_240_hash_unchanged": selected_hash == EXPECTED_SELECTED_240_SHA256,
    }


def per_topic_progress(labels: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    labels_by_topic = Counter(str(row.get("query_id")) for row in labels)
    candidates_by_topic = Counter(str(row.get("query_id")) for row in candidates)
    output: dict[str, dict[str, Any]] = {}
    for query_id in sorted(candidates_by_topic):
        candidate_count = candidates_by_topic[query_id]
        labeled_count = labels_by_topic.get(query_id, 0)
        output[query_id] = {
            "candidate_count": candidate_count,
            "labeled_count": labeled_count,
            "remaining_count": max(0, candidate_count - labeled_count),
            "progress_percent": labeled_count / candidate_count if candidate_count else 0.0,
        }
    return output


def per_batch_progress(labels: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidate_batch: dict[tuple[str, int], int] = {
        (str(row["query_id"]), int(row["paper_id"])): int(row.get("batch_index") or 0)
        for row in candidates
    }
    batch_candidates = Counter(int(row.get("batch_index") or 0) for row in candidates)
    batch_labels: Counter[int] = Counter()
    for label in labels:
        key = (str(label.get("query_id")), int(label.get("paper_id") or -1))
        batch_index = candidate_batch.get(key)
        if batch_index is not None:
            batch_labels[batch_index] += 1
    output: dict[str, dict[str, Any]] = {}
    for batch_index in sorted(batch_candidates):
        candidate_count = batch_candidates[batch_index]
        labeled_count = batch_labels.get(batch_index, 0)
        output[str(batch_index)] = {
            "candidate_count": candidate_count,
            "labeled_count": labeled_count,
            "remaining_count": max(0, candidate_count - labeled_count),
            "progress_percent": labeled_count / candidate_count if candidate_count else 0.0,
        }
    return output


def validate_labels(
    *,
    labels: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    existing_labels: list[dict[str, Any]],
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v35_labels_path: Path,
    selected_240_path: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    candidate_keys = load_keys(candidates)
    existing_keys = load_keys(existing_labels)
    seen: dict[tuple[str, int], int] = {}
    missing_required_fields: list[dict[str, Any]] = []
    invalid_values: list[dict[str, Any]] = []
    duplicate_query_paper_rows: list[dict[str, Any]] = []
    labels_not_in_candidates: list[dict[str, Any]] = []
    overlap_existing_labels: list[dict[str, Any]] = []

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
                duplicate_query_paper_rows.append({"query_id": query_id, "paper_id": paper_id, "first_line": seen[key], "line": line_number})
            else:
                seen[key] = line_number
            if key not in candidate_keys:
                labels_not_in_candidates.append({"query_id": query_id, "paper_id": paper_id, "line": line_number})
            if key in existing_keys:
                overlap_existing_labels.append({"query_id": query_id, "paper_id": paper_id, "line": line_number})
        else:
            invalid_values.append({"line": line_number, "field": "query_id/paper_id", "value": [query_id, label.get("paper_id")], "expected": "string/integer"})
        validate_identity(label, line_number, invalid_values)
        for field_name in sorted(REQUIRED_SCORE_FIELDS):
            validate_score(label, field_name, line_number, invalid_values)
        validate_intent_scores(label, line_number, invalid_values)
        validate_enums(label, line_number, invalid_values)

    labeled_candidate_count = len(set(seen) & candidate_keys)
    projected_total = len(existing_keys) + len(candidate_keys)
    current_total = len(existing_keys) + labeled_candidate_count
    hash_checks = protected_hashes(
        v21_labels_path=v21_labels_path,
        v25_labels_path=v25_labels_path,
        v32_labels_path=v32_labels_path,
        v35_labels_path=v35_labels_path,
        selected_240_path=selected_240_path,
    )
    schema_valid = not (
        missing_required_fields
        or invalid_values
        or duplicate_query_paper_rows
        or labels_not_in_candidates
        or overlap_existing_labels
    )
    complete = current_total >= TARGET_TOTAL_LABELS and labeled_candidate_count >= MIN_NEW_LABELS
    is_valid = schema_valid and all(hash_checks.values()) and (complete if require_complete else True)
    return {
        "schema_version": "v3.9_semantic_expansion_label_progress",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_modified_by_this_script": False,
        "existing_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "auto_labeled_by_this_script": False,
        "require_complete": require_complete,
        "is_valid": is_valid,
        "schema_valid": schema_valid,
        "complete": complete,
        "label_file_rows": len(labels),
        "candidate_count": len(candidate_keys),
        "existing_judged_rows": len(existing_keys),
        "labeled_candidate_count": labeled_candidate_count,
        "unlabeled_candidate_count": max(0, len(candidate_keys) - labeled_candidate_count),
        "current_total_judged_rows": current_total,
        "projected_total_judged_rows": projected_total,
        "target_total_judged_rows": TARGET_TOTAL_LABELS,
        "minimum_new_labels": MIN_NEW_LABELS,
        "progress_toward_new_labels": labeled_candidate_count / MIN_NEW_LABELS,
        "progress_toward_total": current_total / TARGET_TOTAL_LABELS,
        "projected_target_reached": projected_total >= TARGET_TOTAL_LABELS,
        "missing_required_fields": missing_required_fields,
        "invalid_values": invalid_values,
        "duplicate_query_paper_rows": duplicate_query_paper_rows,
        "labels_not_in_candidates": labels_not_in_candidates,
        "overlap_existing_labels": overlap_existing_labels,
        "progress_by_topic": per_topic_progress(labels, candidates),
        "progress_by_batch": per_batch_progress(labels, candidates),
        "hash_checks": hash_checks,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3.9 Semantic Expansion Label Progress",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Valid: `{report['is_valid']}`",
        f"- Schema valid: `{report['schema_valid']}`",
        f"- Complete: `{report['complete']}`",
        f"- Label file rows: `{report['label_file_rows']}`",
        f"- Candidate count: `{report['candidate_count']}`",
        f"- Existing judged rows: `{report['existing_judged_rows']}`",
        f"- Current total judged rows: `{report['current_total_judged_rows']}`",
        f"- Projected total judged rows: `{report['projected_total_judged_rows']}`",
        f"- Projected target reached: `{report['projected_target_reached']}`",
        "",
        "## Progress By Topic",
        "",
        "| topic | labeled | candidates | remaining | progress |",
        "|---|---:|---:|---:|---:|",
    ]
    for topic, row in report["progress_by_topic"].items():
        lines.append(
            f"| `{topic}` | {row['labeled_count']} | {row['candidate_count']} | {row['remaining_count']} | {row['progress_percent']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Progress By Batch",
            "",
            "| batch | labeled | candidates | remaining | progress |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for batch, row in report["progress_by_batch"].items():
        lines.append(
            f"| {batch} | {row['labeled_count']} | {row['candidate_count']} | {row['remaining_count']} | {row['progress_percent']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Validation Issues",
            "",
            f"- Missing fields: `{len(report['missing_required_fields'])}`",
            f"- Invalid values: `{len(report['invalid_values'])}`",
            f"- Duplicate query-paper rows: `{len(report['duplicate_query_paper_rows'])}`",
            f"- Labels not in candidates: `{len(report['labels_not_in_candidates'])}`",
            f"- Overlap existing labels: `{len(report['overlap_existing_labels'])}`",
            "",
            "## Protected Hashes",
            "",
            f"- V2.1 labels unchanged: `{report['hash_checks']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels unchanged: `{report['hash_checks']['v2_5_labels_hash_unchanged']}`",
            f"- V3.2 labels unchanged: `{report['hash_checks']['v3_2_labels_hash_unchanged']}`",
            f"- V3.5 labels unchanged: `{report['hash_checks']['v3_5_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['hash_checks']['selected_240_hash_unchanged']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate V3.9 semantic expansion labels.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = resolve_repo_path(args.labels)
    candidates_path = resolve_repo_path(args.candidates)
    v21_labels_path = resolve_repo_path(args.v21_labels)
    v25_labels_path = resolve_repo_path(args.v25_labels)
    v32_labels_path = resolve_repo_path(args.v32_labels)
    v35_labels_path = resolve_repo_path(args.v35_labels)
    selected_240_path = resolve_repo_path(args.selected_240)
    labels = load_jsonl(labels_path, missing_ok=True)
    candidates = load_jsonl(candidates_path)
    existing_labels = (
        load_jsonl(v21_labels_path)
        + load_jsonl(v25_labels_path)
        + load_jsonl(v32_labels_path)
        + load_jsonl(v35_labels_path)
    )
    report = validate_labels(
        labels=labels,
        candidates=candidates,
        existing_labels=existing_labels,
        v21_labels_path=v21_labels_path,
        v25_labels_path=v25_labels_path,
        v32_labels_path=v32_labels_path,
        v35_labels_path=v35_labels_path,
        selected_240_path=selected_240_path,
        require_complete=args.require_complete,
    )
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    print(f"V3.9 label validation valid: {report['is_valid']}")
    print(f"Label rows: {report['label_file_rows']}")
    print(f"Current total judged rows: {report['current_total_judged_rows']}")
    print(f"Projected total judged rows: {report['projected_total_judged_rows']}")
    if not report["is_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
