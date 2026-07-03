import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from build_v4_3_text_reranker_dataset import (  # noqa: E402
    DEFAULT_DATASET_JSONL,
    DEFAULT_REPORT_JSON,
    EXPECTED_ROW_COUNT,
    EXPECTED_TOPIC_COUNT,
    NUMERIC_RANKER_FEATURES,
    duplicate_keys,
    load_jsonl,
    protected_status,
    resolve_repo_path,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


ALLOWED_SCORE_VALUES = {0.0, 0.5, 1.0}
REQUIRED_FIELDS = {
    "schema_version",
    "query_id",
    "query",
    "paper_id",
    "title",
    "abstract",
    "text_input",
    "split",
    "judged_source",
    "relevance_label",
    "topic_match_score",
    "reading_value_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
    "intent_scores",
    "primary_role",
    "secondary_roles",
    "duplicate_status",
    "evidence_level",
    "full_text_available",
    "positive",
    "hard_negative",
    "large_score_disagreement",
    "v3_3_promoted_v4_demoted",
    "v4_fixed_v3_hard_negative",
    "weak_topic",
    "numeric_ranker_features",
}
SCORE_FIELDS = {
    "topic_match_score",
    "reading_value_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_score(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return float(value) in ALLOWED_SCORE_VALUES
    except (TypeError, ValueError):
        return False


def validate_rows(rows: list[dict[str, Any]], report: dict[str, Any] | None, v39_labels: Path) -> dict[str, Any]:
    missing_fields = []
    invalid_values = []
    missing_text_rows = []
    for index, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            missing_fields.append({"line": index, "query_id": row.get("query_id"), "paper_id": row.get("paper_id"), "fields": missing})
        if row.get("schema_version") != "v4.3_text_reranker_dataset_row":
            invalid_values.append({"line": index, "field": "schema_version", "value": row.get("schema_version")})
        if row.get("split") not in {"train", "dev", "test"}:
            invalid_values.append({"line": index, "field": "split", "value": row.get("split")})
        for field in SCORE_FIELDS:
            if field in row and not valid_score(row.get(field)):
                invalid_values.append({"line": index, "field": field, "value": row.get(field)})
        for field in ["positive", "hard_negative", "large_score_disagreement", "v3_3_promoted_v4_demoted", "v4_fixed_v3_hard_negative", "weak_topic"]:
            if field in row and not isinstance(row.get(field), bool):
                invalid_values.append({"line": index, "field": field, "value": row.get(field), "expected": "boolean"})
        numeric_features = row.get("numeric_ranker_features")
        if not isinstance(numeric_features, dict):
            invalid_values.append({"line": index, "field": "numeric_ranker_features", "value": numeric_features, "expected": "object"})
        else:
            missing_numeric = sorted(set(NUMERIC_RANKER_FEATURES) - set(numeric_features))
            if missing_numeric:
                invalid_values.append({"line": index, "field": "numeric_ranker_features", "value": missing_numeric, "expected": "all numeric ranker features"})
            if "v2_7_score" in numeric_features:
                invalid_values.append({"line": index, "field": "numeric_ranker_features.v2_7_score", "value": numeric_features.get("v2_7_score"), "expected": "excluded"})
            raw_features = sorted(feature for feature in numeric_features if feature.endswith("_raw_score"))
            if raw_features:
                invalid_values.append({"line": index, "field": "numeric_ranker_features", "value": raw_features, "expected": "raw scores excluded"})
        if not str(row.get("title") or "").strip() or not str(row.get("text_input") or "").strip():
            missing_text_rows.append({"line": index, "query_id": row.get("query_id"), "paper_id": row.get("paper_id")})
    topics = Counter(str(row.get("query_id")) for row in rows)
    splits = Counter(str(row.get("split")) for row in rows)
    disagreement_count = sum(
        1
        for row in rows
        if row.get("disagreement_kind")
        or row.get("large_score_disagreement")
        or row.get("v3_3_promoted_v4_demoted")
        or row.get("v4_fixed_v3_hard_negative")
    )
    protected = protected_status(v39_labels)
    result = {
        "valid": (
            len(rows) == EXPECTED_ROW_COUNT
            and len(topics) == EXPECTED_TOPIC_COUNT
            and not duplicate_keys(rows)
            and not missing_fields
            and not invalid_values
            and not missing_text_rows
            and disagreement_count > 0
            and all(value for key, value in protected.items() if key.endswith("_hash_unchanged"))
        ),
        "row_count": len(rows),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "topic_count": len(topics),
        "expected_topic_count": EXPECTED_TOPIC_COUNT,
        "rows_by_split": dict(sorted(splits.items())),
        "duplicate_query_paper_rows": duplicate_keys(rows),
        "missing_fields": missing_fields,
        "invalid_values": invalid_values,
        "missing_text_rows": missing_text_rows,
        "disagreement_marked_rows": disagreement_count,
        "protected_hashes": protected,
        "report_row_count_matches_dataset": None if report is None else report.get("dataset", {}).get("row_count") == len(rows),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_JSONL))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = resolve_repo_path(args.dataset)
    report_path = resolve_repo_path(args.report)
    report = load_json(report_path) if report_path.exists() else None
    result = validate_rows(load_jsonl(dataset), report, resolve_repo_path(args.v39_labels))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
