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

from evaluate_v2_4_true_production_candidate_pool import file_sha256, write_json, write_text  # noqa: E402
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_V40_DATASET = REPO_ROOT / "data" / "processed" / "training" / "v4_0_ltr_2400" / "ltr_dataset_2400.jsonl"
DEFAULT_V40_SPLITS = REPO_ROOT / "data" / "processed" / "training" / "v4_0_ltr_2400" / "splits_2400.json"
DEFAULT_DISAGREEMENT_EXAMPLES = REPO_ROOT / "data" / "eval" / "results" / "v4_1_v4_0_vs_v3_3_disagreement_examples.jsonl"
DEFAULT_ERROR_ANALYSIS = REPO_ROOT / "data" / "eval" / "results" / "v4_1_v4_0_vs_v3_3_error_analysis.json"
DEFAULT_BLEND_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_calibrated_blend_report.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "training" / "v4_3_text_reranker"
DEFAULT_DATASET_JSONL = DEFAULT_OUTPUT_DIR / "v4_3_text_reranker_dataset.jsonl"
DEFAULT_DATASET_CSV = DEFAULT_OUTPUT_DIR / "v4_3_text_reranker_dataset.csv"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_dataset_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_dataset_report.md"

EXPECTED_ROW_COUNT = 2400
EXPECTED_TOPIC_COUNT = 16
BASELINE_SCORE_FIELDS = [
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
    "v2_7_score",
]
RANK_FIELDS = [
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
NUMERIC_RANKER_FEATURES = [
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
    *RANK_FIELDS,
    "year",
    "paper_age_years",
    "citation_count_log1p",
    "full_text_available_bool",
]
WEAK_TOPICS = {
    "v2_ai_for_scientific_discovery",
    "v2_causal_representation_learning",
    "v2_large_language_model_agents",
    "v2_llm_evaluation",
    "v2_retrieval_augmented_generation",
    "v2_transformer_architecture",
}
CSV_COLUMNS = [
    "query_id",
    "query",
    "paper_id",
    "title",
    "text_input",
    "split",
    "judged_source",
    "relevance_label",
    "topic_match_score",
    "reading_value_score",
    "primary_role",
    "positive",
    "hard_negative",
    "large_score_disagreement",
    "v3_3_promoted_v4_demoted",
    "v4_fixed_v3_hard_negative",
    "weak_topic",
    "disagreement_kind",
    "v3_3_score",
    "v4_0_score",
    "v4_1_weighted_score",
    "v4_1_blend_score",
    *BASELINE_SCORE_FIELDS,
]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{line_number}")
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


def split_lookup(splits: dict[str, Any]) -> dict[str, str]:
    output = {}
    for split_name, topics in splits["train_dev_test"].items():
        for topic in topics:
            output[str(topic)] = split_name.replace("_topics", "")
    return output


def text_input(row: dict[str, Any]) -> str:
    parts = [
        f"Query: {row.get('query') or row.get('query_id')}",
        f"Title: {row.get('title') or ''}",
        f"Abstract: {row.get('abstract') or ''}",
    ]
    return "\n".join(parts)


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_rows(
    *,
    v40_rows: list[dict[str, Any]],
    split_by_topic: dict[str, str],
    disagreement_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    disagreement_by_key = {row_key(row): row for row in disagreement_examples}
    rows = []
    for row in v40_rows:
        key = row_key(row)
        disagreement = disagreement_by_key.get(key, {})
        feature_values = disagreement.get("feature_values") or {}
        v33_score = safe_float(disagreement.get("v3_3_score"))
        v40_score = safe_float(disagreement.get("v4_0_score"))
        copied = dict(row)
        copied["schema_version"] = "v4.3_text_reranker_dataset_row"
        copied["split"] = split_by_topic[str(copied["query_id"])]
        copied["text_input"] = text_input(copied)
        copied["disagreement_kind"] = disagreement.get("kind")
        copied["v3_3_score"] = v33_score
        copied["v4_0_score"] = v40_score
        copied["v4_1_weighted_score"] = None
        copied["v4_1_blend_score"] = None
        copied["large_score_disagreement"] = bool(v33_score is not None and v40_score is not None and abs(v33_score - v40_score) >= 0.05)
        copied["v3_3_promoted_v4_demoted"] = str(disagreement.get("kind", "")).startswith("v3_3")
        copied["v4_fixed_v3_hard_negative"] = str(disagreement.get("kind")) == "v4_0_fixes_v3_3_hard_negative"
        copied["weak_topic"] = str(copied["query_id"]) in WEAK_TOPICS
        copied["numeric_ranker_features"] = {
            feature: safe_float(copied.get(feature, feature_values.get(feature)))
            for feature in NUMERIC_RANKER_FEATURES
        }
        rows.append(copied)
    return sorted(rows, key=lambda item: (str(item["query_id"]), int(item["paper_id"])))


def duplicate_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row_key(row) for row in rows)
    return [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in sorted(counts.items())
        if count > 1
    ]


def protected_status(v39_labels_path: Path) -> dict[str, Any]:
    status = protected_hashes(
        DEFAULT_V21_LABELS,
        DEFAULT_V25_LABELS,
        DEFAULT_V32_LABELS,
        DEFAULT_V35_LABELS,
        DEFAULT_SELECTED_240,
    )
    v39_hash = file_sha256(v39_labels_path)
    status["v3_9_labels_sha256"] = v39_hash
    status["v3_9_labels_hash_unchanged"] = v39_hash == "60D5649F6A402A804BF58832239378FDF16A265922E53C68F83945B98C3641B7"
    return status


def build_report(rows: list[dict[str, Any]], disagreement_examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    missing_abstract = sum(1 for row in rows if not str(row.get("abstract") or "").strip())
    by_source = Counter(str(row.get("judged_source")) for row in rows)
    by_topic = Counter(str(row.get("query_id")) for row in rows)
    by_split = Counter(str(row.get("split")) for row in rows)
    return {
        "schema_version": "v4.3_text_reranker_dataset_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "dataset": {
            "row_count": len(rows),
            "expected_row_count": EXPECTED_ROW_COUNT,
            "topic_count": len(by_topic),
            "expected_topic_count": EXPECTED_TOPIC_COUNT,
            "duplicate_query_paper_rows": duplicate_keys(rows),
            "missing_abstract_rows": missing_abstract,
            "rows_by_source": dict(sorted(by_source.items())),
            "rows_by_topic": dict(sorted(by_topic.items())),
            "rows_by_split": dict(sorted(by_split.items())),
            "disagreement_example_rows_matched": sum(1 for row in rows if row.get("disagreement_kind")),
            "large_score_disagreement_rows": sum(1 for row in rows if row["large_score_disagreement"]),
            "v3_3_promoted_v4_demoted_rows": sum(1 for row in rows if row["v3_3_promoted_v4_demoted"]),
            "v4_fixed_v3_hard_negative_rows": sum(1 for row in rows if row["v4_fixed_v3_hard_negative"]),
            "weak_topic_rows": sum(1 for row in rows if row["weak_topic"]),
        },
        "source_inputs": {
            "v4_0_dataset": str(resolve_repo_path(args.v40_dataset)),
            "v4_0_splits": str(resolve_repo_path(args.v40_splits)),
            "disagreement_examples": str(resolve_repo_path(args.disagreement_examples)),
            "disagreement_example_count": len(disagreement_examples),
            "error_analysis": str(resolve_repo_path(args.error_analysis)),
            "v4_1_blend_report": str(resolve_repo_path(args.blend_report)),
        },
        "target": {
            "primary_target": "relevance_label",
            "auxiliary_targets": ["reading_value_score", "topic_match_score", "positive", "hard_negative"],
        },
        "feature_contract": {
            "text_input": "Query, title, and abstract concatenated with field prefixes.",
            "numeric_ranker_features": NUMERIC_RANKER_FEATURES,
            "excludes_raw_score_features": True,
            "excludes_v2_7_score_from_numeric_ranker_features": True,
        },
        "outputs": {
            "jsonl": str(resolve_repo_path(args.jsonl_out)),
            "csv": str(resolve_repo_path(args.csv_out)),
        },
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    lines = [
        "# V4.3 Text Reranker Dataset Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Rows: `{dataset['row_count']}` / `{dataset['expected_row_count']}`",
        f"- Topics: `{dataset['topic_count']}` / `{dataset['expected_topic_count']}`",
        f"- Duplicate query-paper rows: `{len(dataset['duplicate_query_paper_rows'])}`",
        f"- Missing abstract rows: `{dataset['missing_abstract_rows']}`",
        f"- Matched disagreement example rows: `{dataset['disagreement_example_rows_matched']}`",
        f"- Large disagreement rows: `{dataset['large_score_disagreement_rows']}`",
        f"- V3.3 promoted / V4 demoted rows: `{dataset['v3_3_promoted_v4_demoted_rows']}`",
        f"- V4 fixed V3 hard-negative rows: `{dataset['v4_fixed_v3_hard_negative_rows']}`",
        "",
        "## Split Counts",
        "",
    ]
    for split, count in dataset["rows_by_split"].items():
        lines.append(f"- `{split}`: `{count}`")
    lines.extend(["", "## Topic Counts", ""])
    for topic, count in dataset["rows_by_topic"].items():
        lines.append(f"- `{topic}`: `{count}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v40-dataset", default=str(DEFAULT_V40_DATASET))
    parser.add_argument("--v40-splits", default=str(DEFAULT_V40_SPLITS))
    parser.add_argument("--disagreement-examples", default=str(DEFAULT_DISAGREEMENT_EXAMPLES))
    parser.add_argument("--error-analysis", default=str(DEFAULT_ERROR_ANALYSIS))
    parser.add_argument("--blend-report", default=str(DEFAULT_BLEND_REPORT))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--jsonl-out", default=str(DEFAULT_DATASET_JSONL))
    parser.add_argument("--csv-out", default=str(DEFAULT_DATASET_CSV))
    parser.add_argument("--json-report-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-report-out", default=str(DEFAULT_REPORT_MD))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v40_rows = load_jsonl(resolve_repo_path(args.v40_dataset))
    splits = load_json(resolve_repo_path(args.v40_splits))
    disagreement_examples = load_jsonl(resolve_repo_path(args.disagreement_examples))
    rows = build_rows(
        v40_rows=v40_rows,
        split_by_topic=split_lookup(splits),
        disagreement_examples=disagreement_examples,
    )
    write_jsonl(resolve_repo_path(args.jsonl_out), rows)
    write_csv(resolve_repo_path(args.csv_out), rows)
    report = build_report(rows, disagreement_examples, args)
    write_json(resolve_repo_path(args.json_report_out), report)
    write_text(resolve_repo_path(args.md_report_out), render_markdown(report))
    print(f"Wrote {len(rows)} V4.3 text reranker rows to {resolve_repo_path(args.jsonl_out)}")


if __name__ == "__main__":
    main()
