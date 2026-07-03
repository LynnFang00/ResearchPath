import argparse
import csv
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from evaluate_v2_4_true_production_candidate_pool import target_gain, write_json, write_text  # noqa: E402
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes as protected_hashes_v35,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_V33_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v3_3_ltr_dataset_800.jsonl"
DEFAULT_V35_CANDIDATES = REPO_ROOT / "data" / "eval" / "labeling" / "v3_5_unjudged_top10_audit_candidates.jsonl"
DEFAULT_V39_CANDIDATES = REPO_ROOT / "data" / "eval" / "labeling" / "v3_9_semantic_expansion_candidates.jsonl"
DEFAULT_TRAINING_DIR = REPO_ROOT / "data" / "processed" / "training" / "v4_0_ltr_2400"
DEFAULT_DATASET_JSONL = DEFAULT_TRAINING_DIR / "ltr_dataset_2400.jsonl"
DEFAULT_DATASET_CSV = DEFAULT_TRAINING_DIR / "ltr_dataset_2400.csv"
DEFAULT_SPLITS_OUT = DEFAULT_TRAINING_DIR / "splits_2400.json"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v4_0_ltr_dataset_2400_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v4_0_ltr_dataset_2400_report.md"

EXPECTED_ROW_COUNT = 2400
EXPECTED_TOPIC_COUNT = 16
BASELINE_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
SCORE_METHODS = BASELINE_METHODS + ["old_v2_2b", "v2_6", "v2_7"]
REQUIRED_LABEL_FIELDS = {
    "query_id",
    "query",
    "paper_id",
    "title",
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
    "duplicate_of_paper_id",
    "evidence_level",
    "full_text_available",
    "label_confidence",
    "notes",
}
CSV_COLUMNS = [
    "query_id",
    "query",
    "paper_id",
    "title",
    "year",
    "venue",
    "judged_source",
    "relevance_label",
    "topic_match_score",
    "reading_value_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
    "primary_role",
    "duplicate_status",
    "evidence_level",
    "full_text_available",
    "label_confidence",
    "positive",
    "hard_negative",
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
    "v2_7_score",
    "bm25_rank",
    "tfidf_rank",
    "embedding_rank",
    "faiss_embedding_rank",
    "hybrid_rank",
    "bm25_reciprocal_rank",
    "tfidf_reciprocal_rank",
    "embedding_reciprocal_rank",
    "faiss_embedding_reciprocal_rank",
    "hybrid_reciprocal_rank",
]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["query_id"]), int(row["paper_id"])


def reciprocal_rank(rank: Any) -> float:
    try:
        rank_value = int(rank)
    except (TypeError, ValueError):
        return 1.000001000001e-06
    return 1.0 / rank_value if rank_value > 0 and rank_value < 999999 else 1.000001000001e-06


def quantize_score(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric <= 0.25:
        return 0.0
    if numeric < 0.75:
        return 0.5
    return 1.0


def normalized_label(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    for field in [
        "topic_match_score",
        "reading_value_score",
        "beginner_fit_score",
        "intermediate_fit_score",
        "advanced_fit_score",
        "expert_fit_score",
    ]:
        copied[field] = quantize_score(copied.get(field))
    copied["intent_scores"] = {
        key: quantize_score(value)
        for key, value in (copied.get("intent_scores") or {}).items()
    }
    return copied


def score_value(candidate: dict[str, Any], method: str) -> float:
    scores = candidate.get("scores") or {}
    retrieval = candidate.get("retrieval_scores_by_method") or {}
    aliases = {"v3_3_ltr": "v3_3_random_forest_no_v27_compact"}
    value = scores.get(method, scores.get(aliases.get(method, ""), retrieval.get(method, 0.0)))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rank_value(candidate: dict[str, Any], method: str) -> int:
    ranks = candidate.get("ranks") or {}
    retrieval = candidate.get("retrieval_ranks_by_method") or candidate.get("retrieval_ranks") or {}
    aliases = {"v3_3_ltr": "v3_3_random_forest_no_v27_compact"}
    value = ranks.get(method, ranks.get(aliases.get(method, ""), retrieval.get(method, 999999)))
    try:
        return int(value or 999999)
    except (TypeError, ValueError):
        return 999999


def is_positive(label: dict[str, Any]) -> bool:
    return float(label.get("reading_value_score") or 0.0) >= 0.5 and float(label.get("topic_match_score") or 0.0) >= 0.5


def is_hard_negative(label: dict[str, Any]) -> bool:
    return str(label.get("primary_role")) == "negative" or (
        float(label.get("reading_value_score") or 0.0) <= 0.0 and float(label.get("topic_match_score") or 0.0) <= 0.0
    )


def normalize_v33_row(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["schema_version"] = "v4.0_ltr_2400_row"
    copied["judged_source"] = str(copied["judged_source"])
    copied["v4_0_source_group"] = "v3_3_800_carryforward"
    for field in [
        "topic_match_score",
        "reading_value_score",
        "beginner_fit_score",
        "intermediate_fit_score",
        "advanced_fit_score",
        "expert_fit_score",
    ]:
        copied[field] = quantize_score(copied.get(field))
    copied["intent_scores"] = {
        key: quantize_score(value)
        for key, value in (copied.get("intent_scores") or {}).items()
    }
    copied["difficulty_labels"] = {
        "beginner": copied.get("beginner_fit_score"),
        "intermediate": copied.get("intermediate_fit_score"),
        "advanced": copied.get("advanced_fit_score"),
        "expert": copied.get("expert_fit_score"),
    }
    copied["notes"] = copied.get("notes") or copied.get("label_notes") or ""
    copied["relevance_label"] = target_gain(copied)
    copied["positive"] = is_positive(copied)
    copied["hard_negative"] = is_hard_negative(copied)
    return copied


def build_candidate_row(label: dict[str, Any], candidate: dict[str, Any], source: str, source_group: str) -> dict[str, Any]:
    label = normalized_label(label)
    row: dict[str, Any] = {
        "schema_version": "v4.0_ltr_2400_row",
        "query_id": str(label["query_id"]),
        "query": label.get("query") or candidate.get("query"),
        "topic": candidate.get("topic") or str(label["query_id"]),
        "paper_id": int(label["paper_id"]),
        "title": label.get("title") or candidate.get("title"),
        "abstract": candidate.get("abstract"),
        "year": candidate.get("year"),
        "authors": candidate.get("authors") or [],
        "venue": candidate.get("venue"),
        "citation_count": candidate.get("citation_count") or 0,
        "identifiers": candidate.get("identifiers") or {},
        "source_url": candidate.get("source_url"),
        "pdf_url": candidate.get("pdf_url"),
        "sources_provenance": candidate.get("sources_provenance") or [],
        "judged_source": source,
        "v4_0_source_group": source_group,
        "relevance_label": target_gain(label),
        "topic_match_score": label.get("topic_match_score"),
        "reading_value_score": label.get("reading_value_score"),
        "beginner_fit_score": label.get("beginner_fit_score"),
        "intermediate_fit_score": label.get("intermediate_fit_score"),
        "advanced_fit_score": label.get("advanced_fit_score"),
        "expert_fit_score": label.get("expert_fit_score"),
        "difficulty_labels": {
            "beginner": label.get("beginner_fit_score"),
            "intermediate": label.get("intermediate_fit_score"),
            "advanced": label.get("advanced_fit_score"),
            "expert": label.get("expert_fit_score"),
        },
        "intent_scores": label.get("intent_scores") or {},
        "primary_role": label.get("primary_role"),
        "secondary_roles": label.get("secondary_roles") or [],
        "duplicate_status": label.get("duplicate_status"),
        "duplicate_of_paper_id": label.get("duplicate_of_paper_id"),
        "evidence_level": label.get("evidence_level"),
        "full_text_available": bool(label.get("full_text_available")),
        "label_confidence": label.get("label_confidence"),
        "label_notes": label.get("notes"),
        "notes": label.get("notes") or "",
        "hard_negative": is_hard_negative(label),
        "hidden_positive": "hidden_positive_candidate" in (candidate.get("diagnostic_reasons") or []),
        "positive": is_positive(label),
        "source_pool_info": {
            "source_pool": source_group,
            "diagnostic_reasons": candidate.get("diagnostic_reasons") or [],
            "candidate_source_methods": candidate.get("candidate_source_methods") or candidate.get("source_methods") or [],
            "priority_score": candidate.get("priority_score"),
            "packet_index": candidate.get("packet_index"),
            "batch_index": candidate.get("batch_index"),
        },
        "v3_2_raw_scores_available": False,
    }
    for method in BASELINE_METHODS:
        rank = rank_value(candidate, method)
        row[f"{method}_score"] = score_value(candidate, method)
        row[f"{method}_raw_score"] = float((candidate.get("retrieval_scores_by_method") or {}).get(method, 0.0) or 0.0)
        row[f"{method}_rank"] = rank
        row[f"{method}_reciprocal_rank"] = reciprocal_rank(rank)
    row["old_v2_2b_score"] = score_value(candidate, "old_v2_2b")
    row["v2_6_score"] = score_value(candidate, "v2_6")
    row["v2_7_score"] = score_value(candidate, "v2_7")
    row["old_minus_v26"] = row["old_v2_2b_score"] - row["v2_6_score"]
    row["old_minus_v26_positive"] = max(row["old_minus_v26"], 0.0)
    row["v26_minus_old_positive"] = max(-row["old_minus_v26"], 0.0)
    return row


def build_topic_splits(topics: list[str]) -> dict[str, Any]:
    topics = sorted(topics)
    dev_topics = topics[10:13]
    test_topics = topics[13:16]
    train_topics = [topic for topic in topics if topic not in set(dev_topics + test_topics)]
    return {
        "schema_version": "v4.0_topic_splits_2400",
        "created_at": datetime.now(UTC).isoformat(),
        "split_strategy": "topic_grouped_no_query_id_leakage",
        "topics": topics,
        "train_dev_test": {
            "train_topics": train_topics,
            "dev_topics": dev_topics,
            "test_topics": test_topics,
        },
        "leave_topic_out": [
            {
                "fold_id": f"loto_{index:02d}",
                "test_topics": [topic],
                "dev_topics": [],
                "train_topics": [other for other in topics if other != topic],
            }
            for index, topic in enumerate(topics, start=1)
        ],
    }


def score_scale_violations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {0.0, 0.5, 1.0}
    fields = [
        "topic_match_score",
        "reading_value_score",
        "beginner_fit_score",
        "intermediate_fit_score",
        "advanced_fit_score",
        "expert_fit_score",
    ]
    violations = []
    for row in rows:
        for field in fields:
            if float(row.get(field) or 0.0) not in allowed:
                violations.append({"query_id": row["query_id"], "paper_id": row["paper_id"], "field": field, "value": row.get(field)})
        for field, value in (row.get("intent_scores") or {}).items():
            if float(value or 0.0) not in allowed:
                violations.append({"query_id": row["query_id"], "paper_id": row["paper_id"], "field": f"intent_scores.{field}", "value": value})
    return violations


def build_dataset(
    *,
    v33_dataset_path: Path,
    v35_labels_path: Path,
    v35_candidates_path: Path,
    v39_labels_path: Path,
    v39_candidates_path: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    selected_240_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    carried = [normalize_v33_row(row) for row in load_jsonl(v33_dataset_path)]
    v35_lookup = {row_key(row): row for row in load_jsonl(v35_candidates_path)}
    v39_lookup = {row_key(row): row for row in load_jsonl(v39_candidates_path)}
    missing_candidates = []
    added = []
    for label in load_jsonl(v35_labels_path):
        candidate = v35_lookup.get(row_key(label))
        if candidate is None:
            missing_candidates.append({"source": "v3_5", "query_id": label["query_id"], "paper_id": label["paper_id"]})
            continue
        added.append(build_candidate_row(label, candidate, "v3_5", "v3_5_unjudged_audit"))
    for label in load_jsonl(v39_labels_path):
        candidate = v39_lookup.get(row_key(label))
        if candidate is None:
            missing_candidates.append({"source": "v3_9", "query_id": label["query_id"], "paper_id": label["paper_id"]})
            continue
        added.append(build_candidate_row(label, candidate, "v3_9", "v3_9_semantic_expansion"))
    rows = sorted(carried + added, key=lambda row: (str(row["query_id"]), int(row["paper_id"])))
    duplicates = [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in Counter(row_key(row) for row in rows).items()
        if count > 1
    ]
    missing_fields = [
        {"query_id": row["query_id"], "paper_id": row["paper_id"], "missing": sorted(REQUIRED_LABEL_FIELDS - set(row))}
        for row in rows
        if REQUIRED_LABEL_FIELDS - set(row)
    ]
    topics = sorted({str(row["query_id"]) for row in rows})
    splits = build_topic_splits(topics)
    hashes = protected_hashes_v35(v21_labels_path, v25_labels_path, v32_labels_path, v35_labels_path, selected_240_path)
    report = {
        "schema_version": "v4.0_ltr_dataset_2400_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "row_count": len(rows),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "valid": len(rows) == EXPECTED_ROW_COUNT and not duplicates and not missing_candidates and not missing_fields and len(topics) == EXPECTED_TOPIC_COUNT and all(hashes.values()),
        "topic_count": len(topics),
        "expected_topic_count": EXPECTED_TOPIC_COUNT,
        "rows_by_judged_source": dict(sorted(Counter(str(row["judged_source"]) for row in rows).items())),
        "rows_per_topic": dict(sorted(Counter(str(row["query_id"]) for row in rows).items())),
        "duplicate_query_paper_rows": duplicates,
        "missing_candidate_rows": missing_candidates,
        "missing_required_fields": missing_fields,
        "score_scale_violations": score_scale_violations(rows),
        "positive_count": sum(1 for row in rows if row.get("positive")),
        "hard_negative_count": sum(1 for row in rows if row.get("hard_negative")),
        "protected_hashes": hashes,
        "outputs": {
            "jsonl": str(DEFAULT_DATASET_JSONL),
            "csv": str(DEFAULT_DATASET_CSV),
            "splits": str(DEFAULT_SPLITS_OUT),
        },
    }
    return rows, splits, report


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.0 LTR Dataset 2400 Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Valid: `{report['valid']}`",
        f"- Rows: `{report['row_count']}` / `{report['expected_row_count']}`",
        f"- Topics: `{report['topic_count']}` / `{report['expected_topic_count']}`",
        f"- Duplicate query-paper rows: `{len(report['duplicate_query_paper_rows'])}`",
        f"- Missing candidate rows: `{len(report['missing_candidate_rows'])}`",
        f"- Missing required fields: `{len(report['missing_required_fields'])}`",
        f"- Score scale violations: `{len(report['score_scale_violations'])}`",
        "",
        "## Counts By Source",
        "",
        "| source | rows |",
        "|---|---:|",
    ]
    for source, count in report["rows_by_judged_source"].items():
        lines.append(f"| `{source}` | {count} |")
    lines.extend(["", "## Counts By Topic", "", "| topic | rows |", "|---|---:|"])
    for topic, count in report["rows_per_topic"].items():
        lines.append(f"| `{topic}` | {count} |")
    lines.extend(
        [
            "",
            "## Protected Hashes",
            "",
            f"- V2.1 unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
            f"- V3.2 unchanged: `{report['protected_hashes']['v3_2_labels_hash_unchanged']}`",
            f"- V3.5 unchanged: `{report['protected_hashes']['v3_5_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V4.0 2,400-row LTR dataset.")
    parser.add_argument("--v33-dataset", default=str(DEFAULT_V33_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-candidates", default=str(DEFAULT_V35_CANDIDATES.relative_to(REPO_ROOT)))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v39-candidates", default=str(DEFAULT_V39_CANDIDATES.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240.relative_to(REPO_ROOT)))
    parser.add_argument("--jsonl-out", default=str(DEFAULT_DATASET_JSONL.relative_to(REPO_ROOT)))
    parser.add_argument("--csv-out", default=str(DEFAULT_DATASET_CSV.relative_to(REPO_ROOT)))
    parser.add_argument("--splits-out", default=str(DEFAULT_SPLITS_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON.relative_to(REPO_ROOT)))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD.relative_to(REPO_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, splits, report = build_dataset(
        v33_dataset_path=resolve_repo_path(args.v33_dataset),
        v35_labels_path=resolve_repo_path(args.v35_labels),
        v35_candidates_path=resolve_repo_path(args.v35_candidates),
        v39_labels_path=resolve_repo_path(args.v39_labels),
        v39_candidates_path=resolve_repo_path(args.v39_candidates),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        selected_240_path=resolve_repo_path(args.selected_240),
    )
    if not report["valid"]:
        raise ValueError("V4.0 dataset validation failed before write.")
    write_jsonl(resolve_repo_path(args.jsonl_out), rows)
    write_csv(resolve_repo_path(args.csv_out), rows)
    write_json(resolve_repo_path(args.splits_out), splits)
    write_json(resolve_repo_path(args.report_json), report)
    write_text(resolve_repo_path(args.report_md), build_markdown(report))
    print("V4.0 LTR dataset build complete")
    print(f"Rows: {report['row_count']}")
    print(f"Rows by source: {report['rows_by_judged_source']}")
    print(f"Report: {resolve_repo_path(args.report_json)}")


if __name__ == "__main__":
    main()
