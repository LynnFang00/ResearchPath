import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_0_ltr_dataset_2400 import (  # noqa: E402
    DEFAULT_DATASET_CSV,
    DEFAULT_DATASET_JSONL,
    DEFAULT_REPORT_JSON,
    DEFAULT_REPORT_MD,
    DEFAULT_SPLITS_OUT,
    EXPECTED_ROW_COUNT,
    EXPECTED_TOPIC_COUNT,
    REQUIRED_LABEL_FIELDS,
    load_jsonl,
    row_key,
    score_scale_violations,
)
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes,
)


EXPECTED_SOURCE_COUNTS = {"v2_1": 240, "v2_5": 96, "v3_2": 464, "v3_5": 31, "v3_9": 1569}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in Counter(row_key(row) for row in rows).items()
        if count > 1
    ]


def validate_dataset(
    *,
    dataset_path: Path,
    csv_path: Path,
    splits_path: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v35_labels_path: Path,
    selected_240_path: Path,
) -> dict[str, Any]:
    rows = load_jsonl(dataset_path)
    csv_rows = max(0, len(csv_path.read_text(encoding="utf-8").splitlines()) - 1) if csv_path.exists() else 0
    splits = load_json(splits_path)
    source_counts = dict(sorted(Counter(str(row.get("judged_source")) for row in rows).items()))
    topic_counts = dict(sorted(Counter(str(row.get("query_id")) for row in rows).items()))
    duplicates = duplicate_rows(rows)
    missing_fields = [
        {"query_id": row.get("query_id"), "paper_id": row.get("paper_id"), "missing": sorted(REQUIRED_LABEL_FIELDS - set(row))}
        for row in rows
        if REQUIRED_LABEL_FIELDS - set(row)
    ]
    hashes = protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path, v35_labels_path, selected_240_path)
    split = splits.get("train_dev_test", {})
    train = set(split.get("train_topics", []))
    dev = set(split.get("dev_topics", []))
    test = set(split.get("test_topics", []))
    split_valid = not (train & dev or train & test or dev & test)
    report = {
        "schema_version": "v4.0_ltr_dataset_2400_validation",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "dataset_path": str(dataset_path),
        "csv_path": str(csv_path),
        "splits_path": str(splits_path),
        "row_count": len(rows),
        "csv_row_count": csv_rows,
        "expected_row_count": EXPECTED_ROW_COUNT,
        "topic_count": len(topic_counts),
        "expected_topic_count": EXPECTED_TOPIC_COUNT,
        "rows_by_judged_source": source_counts,
        "expected_source_counts": EXPECTED_SOURCE_COUNTS,
        "rows_per_topic": topic_counts,
        "duplicate_query_paper_rows": duplicates,
        "missing_required_fields": missing_fields,
        "score_scale_violations": score_scale_violations(rows),
        "split_validation": {
            "train_dev_test_no_topic_leakage": split_valid,
            "leave_topic_out_fold_count": len(splits.get("leave_topic_out", [])),
        },
        "protected_hashes": hashes,
    }
    report["valid"] = (
        report["row_count"] == EXPECTED_ROW_COUNT
        and report["csv_row_count"] == EXPECTED_ROW_COUNT
        and report["topic_count"] == EXPECTED_TOPIC_COUNT
        and source_counts == EXPECTED_SOURCE_COUNTS
        and not duplicates
        and not missing_fields
        and not report["score_scale_violations"]
        and split_valid
        and all(hashes.values())
    )
    return report


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.0 LTR Dataset 2400 Validation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Valid: `{report['valid']}`",
        f"- Rows: `{report['row_count']}`",
        f"- CSV rows: `{report['csv_row_count']}`",
        f"- Topics: `{report['topic_count']}`",
        f"- Duplicate rows: `{len(report['duplicate_query_paper_rows'])}`",
        f"- Missing required fields: `{len(report['missing_required_fields'])}`",
        f"- Score scale violations: `{len(report['score_scale_violations'])}`",
        "",
        "## Source Counts",
        "",
        f"`{report['rows_by_judged_source']}`",
        "",
        "## Protected Hashes",
        "",
        f"- V2.1 unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
        f"- V2.5 unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
        f"- V3.2 unchanged: `{report['protected_hashes']['v3_2_labels_hash_unchanged']}`",
        f"- V3.5 unchanged: `{report['protected_hashes']['v3_5_labels_hash_unchanged']}`",
        f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the V4.0 2,400-row LTR dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSONL.relative_to(REPO_ROOT)))
    parser.add_argument("--csv", default=str(DEFAULT_DATASET_CSV.relative_to(REPO_ROOT)))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240.relative_to(REPO_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(
        dataset_path=resolve_repo_path(args.dataset),
        csv_path=resolve_repo_path(args.csv),
        splits_path=resolve_repo_path(args.splits),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        v35_labels_path=resolve_repo_path(args.v35_labels),
        selected_240_path=resolve_repo_path(args.selected_240),
    )
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    print(f"V4.0 dataset validation valid: {report['valid']}")
    print(f"Rows: {report['row_count']}")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
