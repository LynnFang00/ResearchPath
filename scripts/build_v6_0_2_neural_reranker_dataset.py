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

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from build_v6_0_neural_reranker_dataset import (  # noqa: E402
    DEFAULT_EXAMPLES_OUT as DEFAULT_V60_EXAMPLES,
    DEFAULT_V39_LABELS,
    DIFFICULTY_FIELDS,
    SECTION_FIELDS,
    build_pairwise_train,
    distribution,
    safe_float,
    split_diagnostics,
    sufficiency_report,
    text_word_count,
    write_jsonl,
)
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402


DEFAULT_V601_RECOMMENDATIONS = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_1_targeted_labeling_recommendations.jsonl"
DEFAULT_V601_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_0_1_neural_split_balance_report.json"
DEFAULT_V602_LABELS = REPO_ROOT / "data" / "eval" / "neural_reranker" / "manual_labels_v6_0_2_targeted_neural.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "neural_reranker"
DEFAULT_EXAMPLES_OUT = DEFAULT_OUTPUT_DIR / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_TRAIN_OUT = DEFAULT_OUTPUT_DIR / "v6_0_2_train.jsonl"
DEFAULT_VAL_OUT = DEFAULT_OUTPUT_DIR / "v6_0_2_val.jsonl"
DEFAULT_TEST_OUT = DEFAULT_OUTPUT_DIR / "v6_0_2_test.jsonl"
DEFAULT_PAIRWISE_TRAIN_OUT = DEFAULT_OUTPUT_DIR / "v6_0_2_pairwise_train.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_0_2_neural_reranker_dataset_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_0_2_neural_reranker_dataset_report.md"

EXPECTED_V602_LABELS = 80
EXPECTED_BASE_EXAMPLES = 2400


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["query_id"]), int(row["paper_id"])


def difficulty_scores(label: str) -> dict[str, float]:
    parts = {part.strip() for part in str(label or "").split("+") if part.strip()}
    return {field: 1.0 if field in parts else 0.0 for field in DIFFICULTY_FIELDS}


def section_scores(label: str) -> dict[str, float]:
    return {field: 1.0 if field == label else 0.0 for field in SECTION_FIELDS}


def preflight_v602_labels(labels: list[dict[str, Any]], recommendations: list[dict[str, Any]], base_examples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [row_key(row) for row in labels]
    recommendation_keys = {row_key(row) for row in recommendations}
    base_keys = {row_key(row) for row in base_examples}
    invalid_scores = []
    invalid_schema = []
    for row in labels:
        if row.get("schema_version") != "v6.0.2_targeted_neural_suggested_label":
            invalid_schema.append(row_key(row))
        label = row.get("relevance_label") or {}
        bucket = label.get("bucket")
        reading = label.get("reading_value_score")
        topic = label.get("topic_match_score")
        if bucket not in {0, 1, 2} or not (0.0 <= safe_float(reading, -1.0) <= 1.0) or not (0.0 <= safe_float(topic, -1.0) <= 1.0):
            invalid_scores.append(row_key(row))
    return {
        "row_count": len(labels),
        "expected_row_count": EXPECTED_V602_LABELS,
        "valid_row_count": len(labels) == EXPECTED_V602_LABELS,
        "duplicate_query_paper_rows": sorted([list(key) for key, count in Counter(keys).items() if count > 1]),
        "rows_not_in_v6_0_1_recommendations": sorted([list(key) for key in set(keys) - recommendation_keys]),
        "overlap_with_v6_0_examples": sorted([list(key) for key in set(keys) & base_keys]),
        "invalid_schema_rows": sorted([list(key) for key in invalid_schema]),
        "invalid_score_rows": sorted([list(key) for key in invalid_scores]),
        "bucket_distribution": dict(sorted(Counter(int(row["relevance_label"]["bucket"]) for row in labels).items())),
        "hard_negative_count": sum(1 for row in labels if bool(row.get("hard_negative"))),
        "generic_survey_count": sum(1 for row in labels if bool(row.get("generic_survey"))),
        "average_reading_value_score": sum(safe_float(row["relevance_label"]["reading_value_score"]) for row in labels) / len(labels) if labels else 0.0,
        "average_topic_match_score": sum(safe_float(row["relevance_label"]["topic_match_score"]) for row in labels) / len(labels) if labels else 0.0,
        "by_topic": {
            topic: {
                "row_count": sum(1 for row in labels if row["query_id"] == topic),
                "bucket_distribution": dict(sorted(Counter(int(row["relevance_label"]["bucket"]) for row in labels if row["query_id"] == topic).items())),
                "hard_negative_count": sum(1 for row in labels if row["query_id"] == topic and bool(row.get("hard_negative"))),
            }
            for topic in sorted({row["query_id"] for row in labels})
        },
    }


def split_topics_from_v601_report(path: Path) -> dict[str, list[str]]:
    report = load_json(path)
    return {
        split: list(report["split_strategy"]["balanced_split_topics"][split])
        for split in ("train", "val", "test")
    }


def split_for_query(query_id: str, split_topics: dict[str, list[str]]) -> str:
    for split, topics in split_topics.items():
        if query_id in topics:
            return split
    return "train"


def normalize_v602_label(row: dict[str, Any], split_topics: dict[str, list[str]]) -> dict[str, Any]:
    relevance = row["relevance_label"]
    bucket = int(relevance["bucket"])
    reading = safe_float(relevance["reading_value_score"])
    topic = safe_float(relevance["topic_match_score"])
    query_id = str(row["query_id"])
    split = split_for_query(query_id, split_topics)
    difficulty = str(row.get("difficulty_label") or "unknown")
    section = str(row.get("section_label") or "unknown")
    return {
        "schema_version": "v6.0.2_neural_reranker_example",
        "query_id": query_id,
        "query": str(row.get("query") or query_id.replace("v2_", "").replace("_", " ")),
        "paper_id": int(row["paper_id"]),
        "title": str(row.get("title") or ""),
        "abstract": str(row.get("abstract") or ""),
        "abstract_word_count": text_word_count(row.get("abstract")),
        "year": row.get("year"),
        "venue": row.get("venue"),
        "source": row.get("source"),
        "source_url": row.get("source_url"),
        "relevance_bucket": bucket,
        "classification_target": bucket,
        "reading_value_score": reading,
        "topic_match_score": topic,
        "regression_target_reading": reading,
        "regression_target_topic": topic,
        "section_label": section,
        "section_scores": section_scores(section),
        "difficulty_label": difficulty,
        "difficulty_scores": difficulty_scores(difficulty),
        "hard_negative": bool(row.get("hard_negative")),
        "generic_survey": bool(row.get("generic_survey")),
        "short_or_missing_abstract": text_word_count(row.get("abstract")) < 50,
        "label_source": "v6_0_2",
        "label_annotation_source": str(row.get("label_source") or ""),
        "label_schema_version": str(row.get("schema_version") or ""),
        "label_notes": str(row.get("label_notes") or ""),
        "v4_8_override": False,
        "v6_0_2_targeted_label": True,
        "source_v6_0_query_grouped_split": None,
        "query_grouped_split": split,
        "v6_0_2_grouped_split": split,
        "within_topic_debug_split": "",
        "within_topic_debug_split_leakage_prone": True,
        "baseline_rankings": row.get("candidate_rankings") or {},
        "analysis_only_ranker_metadata": True,
        "original_reason_selected": row.get("original_reason_selected") or [],
        "original_suggested_labeling_focus": row.get("original_suggested_labeling_focus"),
        "do_not_modify_official_labels": bool(row.get("do_not_modify_official_labels")),
    }


def normalize_base_example(row: dict[str, Any], split_topics: dict[str, list[str]]) -> dict[str, Any]:
    copied = dict(row)
    split = split_for_query(str(copied["query_id"]), split_topics)
    copied["schema_version"] = "v6.0.2_neural_reranker_example"
    copied["source_v6_0_query_grouped_split"] = copied.get("query_grouped_split")
    copied["query_grouped_split"] = split
    copied["v6_0_2_grouped_split"] = split
    copied["v6_0_2_targeted_label"] = False
    return copied


def apply_within_topic_debug_split(examples: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for row in examples:
        query_id = row["query_id"]
        index = seen.get(query_id, 0)
        mod = index % 10
        row["within_topic_debug_split"] = "train" if mod < 7 else "val" if mod < 8 else "test"
        seen[query_id] = index + 1


def split_examples(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    splits = {
        split: [row for row in examples if row["query_grouped_split"] == split]
        for split in ("train", "val", "test")
    }
    for split in splits:
        splits[split].sort(key=lambda row: (row["query_id"], int(row["paper_id"])))
    return splits


def build_examples(
    *,
    base_examples: list[dict[str, Any]],
    v602_labels: list[dict[str, Any]],
    split_topics: dict[str, list[str]],
) -> list[dict[str, Any]]:
    examples = [normalize_base_example(row, split_topics) for row in base_examples]
    examples.extend(normalize_v602_label(row, split_topics) for row in v602_labels)
    examples.sort(key=lambda row: (row["query_id"], int(row["paper_id"])))
    apply_within_topic_debug_split(examples)
    return examples


def build_report(
    *,
    examples: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
    preflight: dict[str, Any],
    sufficiency: dict[str, Any],
    split_topics: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": "v6.0.2_neural_reranker_dataset_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "v6_0_outputs_overwritten": False,
        "v6_0_1_outputs_overwritten": False,
        "dataset": {
            "base_example_count": EXPECTED_BASE_EXAMPLES,
            "v6_0_2_label_count": preflight["row_count"],
            "example_count": len(examples),
            "pairwise_train_count": len(pairs),
            "query_count": len({row["query_id"] for row in examples}),
            "v6_0_2_labels_integrated": True,
        },
        "preflight": preflight,
        "outputs": {
            "examples": str(resolve_repo_path(args.examples_out)),
            "train": str(resolve_repo_path(args.train_out)),
            "val": str(resolve_repo_path(args.val_out)),
            "test": str(resolve_repo_path(args.test_out)),
            "pairwise_train": str(resolve_repo_path(args.pairwise_train_out)),
        },
        "split_strategy": {
            "main": "v6_0_1_balanced_query_topic_grouped",
            "split_topics": split_topics,
        },
        "split_diagnostics": {split: split_diagnostics(rows) for split, rows in splits.items()},
        "overall_relevance_distribution": distribution(examples, "relevance_bucket"),
        "overall_hard_negative_count": sum(1 for row in examples if row["hard_negative"]),
        "label_sources": distribution(examples, "label_source"),
        "pairwise_distribution": distribution(pairs, "pair_type"),
        "label_sufficiency": sufficiency,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.0.2 Neural Reranker Dataset Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Neural examples: `{report['dataset']['example_count']}`",
        f"- V6.0.2 labels integrated: `{report['dataset']['v6_0_2_label_count']}`",
        f"- Pairwise train examples: `{report['dataset']['pairwise_train_count']}`",
        f"- Overall relevance distribution: `{report['overall_relevance_distribution']}`",
        f"- Overall hard negatives: `{report['overall_hard_negative_count']}`",
        f"- Label sources: `{report['label_sources']}`",
        "",
        "## Split Diagnostics",
        "",
    ]
    for split, row in report["split_diagnostics"].items():
        lines.append(
            f"- `{split}`: `{row['example_count']}` examples, `{row['query_count']}` topics, "
            f"relevance `{row['relevance_distribution']}`, hard negatives `{row['hard_negative_count']}`"
        )
    lines.extend(
        [
            "",
            "## Label Sufficiency",
            "",
            f"- More labels recommended before V6.1 training: `{report['label_sufficiency']['recommend_more_labels_before_v6_1_training']}`",
            f"- Insufficient scopes: `{report['label_sufficiency']['insufficient_scopes']}`",
            "",
            "## Protected Hashes",
            "",
        ]
    )
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-examples", default=str(DEFAULT_V60_EXAMPLES))
    parser.add_argument("--v6-0-1-recommendations", default=str(DEFAULT_V601_RECOMMENDATIONS))
    parser.add_argument("--v6-0-1-report", default=str(DEFAULT_V601_REPORT))
    parser.add_argument("--v6-0-2-labels", default=str(DEFAULT_V602_LABELS))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--examples-out", default=str(DEFAULT_EXAMPLES_OUT))
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--val-out", default=str(DEFAULT_VAL_OUT))
    parser.add_argument("--test-out", default=str(DEFAULT_TEST_OUT))
    parser.add_argument("--pairwise-train-out", default=str(DEFAULT_PAIRWISE_TRAIN_OUT))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_examples = load_jsonl(resolve_repo_path(args.base_examples))
    v602_labels = load_jsonl(resolve_repo_path(args.v6_0_2_labels))
    recommendations = load_jsonl(resolve_repo_path(args.v6_0_1_recommendations))
    split_topics = split_topics_from_v601_report(resolve_repo_path(args.v6_0_1_report))
    preflight = preflight_v602_labels(v602_labels, recommendations, base_examples)
    blocking_issues = [
        preflight["valid_row_count"] is not True,
        bool(preflight["duplicate_query_paper_rows"]),
        bool(preflight["rows_not_in_v6_0_1_recommendations"]),
        bool(preflight["overlap_with_v6_0_examples"]),
        bool(preflight["invalid_schema_rows"]),
        bool(preflight["invalid_score_rows"]),
    ]
    if any(blocking_issues):
        raise SystemExit(f"V6.0.2 preflight failed: {json.dumps(preflight, sort_keys=True)}")
    examples = build_examples(base_examples=base_examples, v602_labels=v602_labels, split_topics=split_topics)
    splits = split_examples(examples)
    pairs = build_pairwise_train(splits["train"])
    sufficiency = sufficiency_report(examples, splits)
    write_jsonl(resolve_repo_path(args.examples_out), examples)
    write_jsonl(resolve_repo_path(args.train_out), splits["train"])
    write_jsonl(resolve_repo_path(args.val_out), splits["val"])
    write_jsonl(resolve_repo_path(args.test_out), splits["test"])
    write_jsonl(resolve_repo_path(args.pairwise_train_out), pairs)
    report = build_report(
        examples=examples,
        splits=splits,
        pairs=pairs,
        preflight=preflight,
        sufficiency=sufficiency,
        split_topics=split_topics,
        args=args,
    )
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.0.2 neural reranker dataset report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
