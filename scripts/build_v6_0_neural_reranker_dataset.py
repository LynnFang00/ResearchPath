import argparse
from collections import Counter, defaultdict
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

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
)
from evaluate_v4_8_targeted_contrastive import (  # noqa: E402
    DEFAULT_V48_LABELS,
    combine_with_v48_overrides,
    load_all_base_labels,
    load_jsonl,
    row_key,
    v48_preflight,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_V40_DATASET = REPO_ROOT / "data" / "processed" / "training" / "v4_0_ltr_2400" / "ltr_dataset_2400.jsonl"
DEFAULT_V47_CANDIDATES = REPO_ROOT / "data" / "eval" / "results" / "v4_7_targeted_labeling_candidates.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "neural_reranker"
DEFAULT_EXAMPLES_OUT = DEFAULT_OUTPUT_DIR / "v6_0_neural_reranker_examples.jsonl"
DEFAULT_TRAIN_OUT = DEFAULT_OUTPUT_DIR / "v6_0_train.jsonl"
DEFAULT_VAL_OUT = DEFAULT_OUTPUT_DIR / "v6_0_val.jsonl"
DEFAULT_TEST_OUT = DEFAULT_OUTPUT_DIR / "v6_0_test.jsonl"
DEFAULT_PAIRWISE_TRAIN_OUT = DEFAULT_OUTPUT_DIR / "v6_0_pairwise_train.jsonl"
DEFAULT_TARGETED_RECOMMENDATIONS_OUT = DEFAULT_OUTPUT_DIR / "v6_0_targeted_labeling_recommendations.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_0_neural_reranker_dataset_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_0_neural_reranker_dataset_report.md"

EXPECTED_ROW_COUNT = 2400
SHORT_ABSTRACT_WORDS = 50
SECTION_FIELDS = ["background", "foundational", "core_methods", "recent_frontier", "evaluation_benchmark", "application"]
DIFFICULTY_FIELDS = ["beginner", "intermediate", "advanced", "expert"]
BASELINE_METHODS = ["bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"]
QUERY_GROUPED_SPLIT_TOPICS = {
    "train": [
        "v2_ai_for_scientific_discovery",
        "v2_bayesian_optimization",
        "v2_causal_representation_learning",
        "v2_contrastive_learning",
        "v2_diffusion_image_generation",
        "v2_efficient_transformers",
        "v2_graph_neural_networks",
        "v2_graph_recommendation",
        "v2_large_language_model_agents",
        "v2_llm_evaluation",
        "v2_recommendation_systems",
        "v2_retrieval_augmented_generation",
    ],
    "val": ["v2_multimodal_learning", "v2_robot_learning"],
    "test": ["v2_self_supervised_vision", "v2_transformer_architecture"],
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def text_word_count(value: str | None) -> int:
    return len(str(value or "").split())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def relevance_bucket(row: dict[str, Any]) -> int:
    reading = safe_float(row.get("reading_value_score"))
    topic = safe_float(row.get("topic_match_score"))
    if reading >= 0.75 and topic >= 0.75:
        return 2
    if reading >= 0.35 and topic >= 0.35:
        return 1
    return 0


def is_hard_negative(row: dict[str, Any]) -> bool:
    return bool(row.get("hard_negative")) or str(row.get("primary_role")) == "negative" or (
        safe_float(row.get("reading_value_score")) <= 0.25 and safe_float(row.get("topic_match_score")) <= 0.35
    )


def is_generic_survey(row: dict[str, Any]) -> bool:
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or "").lower()
    return any(token in title for token in ["survey", "review", "overview"]) or "we survey" in abstract or "this survey" in abstract


def section_label(row: dict[str, Any]) -> str:
    return str(row.get("primary_role") or "unknown")


def difficulty_label(row: dict[str, Any]) -> str:
    scores = {
        "beginner": safe_float(row.get("beginner_fit_score")),
        "intermediate": safe_float(row.get("intermediate_fit_score")),
        "advanced": safe_float(row.get("advanced_fit_score")),
        "expert": safe_float(row.get("expert_fit_score")),
    }
    best = max(scores, key=lambda key: (scores[key], key))
    return best if scores[best] > 0 else "unknown"


def query_grouped_split(query_id: str) -> str:
    for split, topics in QUERY_GROUPED_SPLIT_TOPICS.items():
        if query_id in topics:
            return split
    return "train"


def within_topic_debug_split(row_index: int) -> str:
    mod = row_index % 10
    if mod < 7:
        return "train"
    if mod < 8:
        return "val"
    return "test"


def difficulty_scores(row: dict[str, Any]) -> dict[str, float]:
    return {
        "beginner": safe_float(row.get("beginner_fit_score")),
        "intermediate": safe_float(row.get("intermediate_fit_score")),
        "advanced": safe_float(row.get("advanced_fit_score")),
        "expert": safe_float(row.get("expert_fit_score")),
    }


def role_scores(row: dict[str, Any]) -> dict[str, float]:
    intents = row.get("intent_scores") or {}
    return {field: safe_float(intents.get(field), 1.0 if section_label(row) == field else 0.0) for field in SECTION_FIELDS}


def baseline_value(metadata: dict[str, Any], field: str) -> float | int | None:
    value = metadata.get(field)
    if value is None:
        return None
    if field.endswith("_rank"):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return safe_float(value)


def load_label_view(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    base_labels = load_all_base_labels(
        {
            "v2_1": resolve_repo_path(args.v21_labels),
            "v2_5": resolve_repo_path(args.v25_labels),
            "v3_2": resolve_repo_path(args.v32_labels),
            "v3_5": resolve_repo_path(args.v35_labels),
            "v3_9": resolve_repo_path(args.v39_labels),
        }
    )
    v48_rows = load_jsonl(resolve_repo_path(args.v48_labels))
    v47_rows = load_jsonl(resolve_repo_path(args.v47_candidates))
    labels, override_summary = combine_with_v48_overrides(base_labels, v48_rows)
    return labels, override_summary, v48_preflight(v48_rows, v47_rows)


def build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    labels, override_summary, preflight = load_label_view(args)
    labels_by_key = {row_key(row): row for row in labels}
    metadata_by_key = {row_key(row): row for row in load_jsonl(resolve_repo_path(args.v40_dataset))}
    examples: list[dict[str, Any]] = []
    for key in sorted(labels_by_key):
        label = labels_by_key[key]
        metadata = metadata_by_key.get(key, {})
        merged = {**metadata, **label}
        abstract = str(metadata.get("abstract") or label.get("abstract") or "")
        bucket = relevance_bucket(merged)
        hard_negative = is_hard_negative(merged)
        query_id = str(merged["query_id"])
        paper_id = int(merged["paper_id"])
        example = {
            "schema_version": "v6.0_neural_reranker_example",
            "query_id": query_id,
            "query": str(merged.get("query") or query_id.replace("v2_", "").replace("_", " ")),
            "paper_id": paper_id,
            "title": str(merged.get("title") or metadata.get("title") or ""),
            "abstract": abstract,
            "abstract_word_count": text_word_count(abstract),
            "year": metadata.get("year") or merged.get("year"),
            "venue": metadata.get("venue") or metadata.get("source") or merged.get("venue"),
            "source": metadata.get("source") or (metadata.get("sources_provenance") or [None])[0],
            "source_url": metadata.get("source_url"),
            "relevance_bucket": bucket,
            "classification_target": bucket,
            "reading_value_score": safe_float(merged.get("reading_value_score")),
            "topic_match_score": safe_float(merged.get("topic_match_score")),
            "regression_target_reading": safe_float(merged.get("reading_value_score")),
            "regression_target_topic": safe_float(merged.get("topic_match_score")),
            "section_label": section_label(merged),
            "section_scores": role_scores(merged),
            "difficulty_label": difficulty_label(merged),
            "difficulty_scores": difficulty_scores(merged),
            "hard_negative": hard_negative,
            "generic_survey": bool(merged.get("generic_survey")) or is_generic_survey(metadata) or is_generic_survey(merged),
            "short_or_missing_abstract": text_word_count(abstract) < SHORT_ABSTRACT_WORDS,
            "label_source": str(merged.get("judged_source") or "unknown"),
            "label_schema_version": str(merged.get("schema_version") or ""),
            "v4_8_override": str(merged.get("judged_source")) == "v4_8",
            "query_grouped_split": query_grouped_split(query_id),
            "within_topic_debug_split": "",
            "within_topic_debug_split_leakage_prone": True,
            "baseline_rankings": {
                "bm25": {"rank": baseline_value(metadata, "bm25_rank"), "score": baseline_value(metadata, "bm25_score")},
                "v3_3_ltr": {"rank": baseline_value(metadata, "v3_3_rank"), "score": baseline_value(metadata, "v3_3_score")},
                "v4_1_blend": {"rank": baseline_value(metadata, "v4_1_blend_rank"), "score": baseline_value(metadata, "v4_1_blend_score")},
                "v4_9_guarded_text_blend": {"rank": baseline_value(metadata, "v4_9_guarded_text_blend_rank"), "score": baseline_value(metadata, "v4_9_guarded_text_blend_score")},
            },
            "analysis_only_ranker_metadata": True,
        }
        examples.append(example)
    by_topic_seen: dict[str, int] = defaultdict(int)
    for row in examples:
        index = by_topic_seen[row["query_id"]]
        row["within_topic_debug_split"] = within_topic_debug_split(index)
        by_topic_seen[row["query_id"]] += 1
    return examples, override_summary, preflight


def split_examples(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [row for row in examples if row["query_grouped_split"] == split]
        for split in ["train", "val", "test"]
    }


def pair_type(preferred: dict[str, Any], rejected: dict[str, Any]) -> str:
    if rejected["hard_negative"]:
        return "positive_over_hard_negative"
    if preferred["relevance_bucket"] == 2 and rejected["relevance_bucket"] == 1:
        return "strong_positive_over_partial"
    if preferred["relevance_bucket"] >= 1 and rejected["relevance_bucket"] == 0:
        return "positive_over_negative"
    return "positive_over_weak"


def build_pairwise_train(train_rows: list[dict[str, Any]], limit_per_topic: int = 600) -> list[dict[str, Any]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_topic[row["query_id"]].append(row)
    pairs = []
    for query_id, rows in sorted(by_topic.items()):
        sorted_rows = sorted(rows, key=lambda row: (-row["relevance_bucket"], -row["reading_value_score"], int(row["paper_id"])))
        positives = [row for row in sorted_rows if row["relevance_bucket"] >= 1]
        rejects = [row for row in sorted_rows if row["relevance_bucket"] <= 1 or row["hard_negative"]]
        count = 0
        for preferred in positives:
            for rejected in rejects:
                if preferred["paper_id"] == rejected["paper_id"]:
                    continue
                if preferred["relevance_bucket"] <= rejected["relevance_bucket"] and preferred["reading_value_score"] <= rejected["reading_value_score"]:
                    continue
                pair = {
                    "schema_version": "v6.0_neural_reranker_pairwise_example",
                    "query_id": query_id,
                    "query": preferred["query"],
                    "preferred_paper_id": preferred["paper_id"],
                    "rejected_paper_id": rejected["paper_id"],
                    "preferred_relevance_bucket": preferred["relevance_bucket"],
                    "rejected_relevance_bucket": rejected["relevance_bucket"],
                    "preferred_reading_value_score": preferred["reading_value_score"],
                    "rejected_reading_value_score": rejected["reading_value_score"],
                    "preferred_topic_match_score": preferred["topic_match_score"],
                    "rejected_topic_match_score": rejected["topic_match_score"],
                    "hard_negative_pair": bool(rejected["hard_negative"]),
                    "pair_type": pair_type(preferred, rejected),
                    "preferred_title": preferred["title"],
                    "rejected_title": rejected["title"],
                }
                pairs.append(pair)
                count += 1
                if count >= limit_per_topic:
                    break
            if count >= limit_per_topic:
                break
    return pairs


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def split_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "example_count": len(rows),
        "query_count": len({row["query_id"] for row in rows}),
        "query_ids": sorted({row["query_id"] for row in rows}),
        "relevance_distribution": distribution(rows, "relevance_bucket"),
        "hard_negative_count": sum(1 for row in rows if row["hard_negative"]),
        "section_distribution": distribution(rows, "section_label"),
        "difficulty_distribution": distribution(rows, "difficulty_label"),
        "average_abstract_word_count": sum(row["abstract_word_count"] for row in rows) / len(rows) if rows else 0.0,
        "missing_or_short_abstract_count": sum(1 for row in rows if row["short_or_missing_abstract"]),
    }


def sufficiency_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    negatives = sum(1 for row in rows if row["relevance_bucket"] == 0)
    partials = sum(1 for row in rows if row["relevance_bucket"] == 1)
    hard_negatives = sum(1 for row in rows if row["hard_negative"])
    foundational_core = sum(1 for row in rows if row["section_label"] in {"foundational", "core_methods"})
    applications = sum(1 for row in rows if row["section_label"] == "application")
    short = sum(1 for row in rows if row["short_or_missing_abstract"])
    buckets = Counter(row["relevance_bucket"] for row in rows)
    max_class_share = max(buckets.values()) / total if total else 0.0
    flags = {
        "too_few_total_labels": total < 100,
        "too_few_relevance_0_negatives": negatives < 15,
        "too_few_relevance_1_partials": partials < 15,
        "too_few_hard_negatives": hard_negatives < 5,
        "too_few_foundational_core_papers": foundational_core < 20,
        "too_many_application_only_papers": applications / total > 0.55 if total else False,
        "missing_or_short_abstracts": short / total > 0.10 if total else False,
        "severe_class_imbalance": max_class_share > 0.75,
    }
    return {
        "total_labels": total,
        "relevance_0_negatives": negatives,
        "relevance_1_partials": partials,
        "hard_negatives": hard_negatives,
        "foundational_core_papers": foundational_core,
        "application_only_papers": applications,
        "missing_or_short_abstracts": short,
        "max_class_share": max_class_share,
        "flags": flags,
        "appears_sufficient": not any(flags.values()),
    }


def sufficiency_report(examples: list[dict[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_split = {split: sufficiency_for_rows(rows) for split, rows in splits.items()}
    by_topic = {
        topic: sufficiency_for_rows([row for row in examples if row["query_id"] == topic])
        for topic in sorted({row["query_id"] for row in examples})
    }
    needs_more = [scope for scope, row in {**{f"split:{k}": v for k, v in by_split.items()}, **{f"topic:{k}": v for k, v in by_topic.items()}}.items() if not row["appears_sufficient"]]
    return {
        "by_split": by_split,
        "by_topic": by_topic,
        "recommend_more_labels_before_v6_1_training": bool(needs_more),
        "insufficient_scopes": needs_more,
        "recommendation": "More targeted labels are recommended before V6.1 neural training." if needs_more else "Label set appears sufficient for V6.1 neural training.",
    }


def targeted_recommendations(examples: list[dict[str, Any]], sufficiency: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for topic, status in sufficiency["by_topic"].items():
        failed = [key for key, value in status["flags"].items() if value]
        if not failed:
            continue
        topic_rows = [row for row in examples if row["query_id"] == topic]
        rows.append(
            {
                "schema_version": "v6.0_targeted_labeling_recommendation",
                "query_id": topic,
                "query": topic_rows[0]["query"] if topic_rows else topic.replace("v2_", "").replace("_", " "),
                "recommendation_reasons": failed,
                "suggested_label_count": 50 if status["total_labels"] >= 100 else 100,
                "candidate_strategy": "sample additional unjudged candidates emphasizing hard negatives, relevance=0 negatives, partial matches, and foundational/core papers",
                "do_not_modify_official_labels": True,
            }
        )
    return rows


def build_report(
    *,
    examples: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
    override_summary: dict[str, Any],
    preflight: dict[str, Any],
    sufficiency: dict[str, Any],
    recommendations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    protected = protected_status(resolve_repo_path(args.v39_labels))
    return {
        "schema_version": "v6.0_neural_reranker_dataset_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "dataset": {
            "example_count": len(examples),
            "expected_example_count": EXPECTED_ROW_COUNT,
            "pairwise_train_count": len(pairs),
            "query_count": len({row["query_id"] for row in examples}),
            "v4_8_overrides_included": True,
            "v4_8_override_count": sum(1 for row in examples if row["v4_8_override"]),
            "v4_8_preflight": preflight,
            "v4_8_override_behavior": override_summary,
        },
        "outputs": {
            "examples": str(resolve_repo_path(args.examples_out)),
            "train": str(resolve_repo_path(args.train_out)),
            "val": str(resolve_repo_path(args.val_out)),
            "test": str(resolve_repo_path(args.test_out)),
            "pairwise_train": str(resolve_repo_path(args.pairwise_train_out)),
            "targeted_recommendations": str(resolve_repo_path(args.targeted_recommendations_out)),
        },
        "split_strategy": {
            "main": "query_topic_grouped",
            "main_split_topics": QUERY_GROUPED_SPLIT_TOPICS,
            "within_topic_debug_split": "deterministic per-topic 70/10/20 row split; leakage-prone and not the main reported split",
        },
        "split_diagnostics": {split: split_diagnostics(rows) for split, rows in splits.items()},
        "overall_relevance_distribution": distribution(examples, "relevance_bucket"),
        "overall_hard_negative_count": sum(1 for row in examples if row["hard_negative"]),
        "label_sources": distribution(examples, "label_source"),
        "pairwise_distribution": distribution(pairs, "pair_type"),
        "label_sufficiency": sufficiency,
        "targeted_labeling_recommendation_count": len(recommendations),
        "protected_hashes": protected,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V6.0 Neural Reranker Dataset Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Neural examples: `{report['dataset']['example_count']}`",
        f"- Pairwise train examples: `{report['dataset']['pairwise_train_count']}`",
        f"- V4.8 overrides included: `{report['dataset']['v4_8_overrides_included']}`",
        f"- V4.8 override count: `{report['dataset']['v4_8_override_count']}`",
        f"- Overall relevance distribution: `{report['overall_relevance_distribution']}`",
        f"- Overall hard negatives: `{report['overall_hard_negative_count']}`",
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
            f"- Recommendation: {report['label_sufficiency']['recommendation']}",
            f"- Insufficient scopes: `{report['label_sufficiency']['insufficient_scopes']}`",
            f"- Targeted recommendation rows: `{report['targeted_labeling_recommendation_count']}`",
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
    parser.add_argument("--v40-dataset", default=str(DEFAULT_V40_DATASET))
    parser.add_argument("--v48-labels", default=str(DEFAULT_V48_LABELS))
    parser.add_argument("--v47-candidates", default=str(DEFAULT_V47_CANDIDATES))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240))
    parser.add_argument("--examples-out", default=str(DEFAULT_EXAMPLES_OUT))
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN_OUT))
    parser.add_argument("--val-out", default=str(DEFAULT_VAL_OUT))
    parser.add_argument("--test-out", default=str(DEFAULT_TEST_OUT))
    parser.add_argument("--pairwise-train-out", default=str(DEFAULT_PAIRWISE_TRAIN_OUT))
    parser.add_argument("--targeted-recommendations-out", default=str(DEFAULT_TARGETED_RECOMMENDATIONS_OUT))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, override_summary, preflight = build_examples(args)
    splits = split_examples(examples)
    pairs = build_pairwise_train(splits["train"])
    sufficiency = sufficiency_report(examples, splits)
    recommendations = targeted_recommendations(examples, sufficiency)
    write_jsonl(resolve_repo_path(args.examples_out), examples)
    write_jsonl(resolve_repo_path(args.train_out), splits["train"])
    write_jsonl(resolve_repo_path(args.val_out), splits["val"])
    write_jsonl(resolve_repo_path(args.test_out), splits["test"])
    write_jsonl(resolve_repo_path(args.pairwise_train_out), pairs)
    write_jsonl(resolve_repo_path(args.targeted_recommendations_out), recommendations)
    report = build_report(
        examples=examples,
        splits=splits,
        pairs=pairs,
        override_summary=override_summary,
        preflight=preflight,
        sufficiency=sufficiency,
        recommendations=recommendations,
        args=args,
    )
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.0 neural reranker dataset report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
