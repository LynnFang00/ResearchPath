import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import itertools
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from build_v6_0_neural_reranker_dataset import (  # noqa: E402
    DEFAULT_EXAMPLES_OUT,
    DEFAULT_V39_LABELS,
    build_pairwise_train,
    distribution,
    split_diagnostics,
    sufficiency_for_rows,
    text_word_count,
    write_jsonl,
)
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402


DEFAULT_CANDIDATE_POOL = REPO_ROOT / "data" / "eval" / "v2_labeling_candidate_pool.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "neural_reranker"
DEFAULT_TRAIN_OUT = DEFAULT_OUTPUT_DIR / "v6_0_1_train.jsonl"
DEFAULT_VAL_OUT = DEFAULT_OUTPUT_DIR / "v6_0_1_val.jsonl"
DEFAULT_TEST_OUT = DEFAULT_OUTPUT_DIR / "v6_0_1_test.jsonl"
DEFAULT_PAIRWISE_TRAIN_OUT = DEFAULT_OUTPUT_DIR / "v6_0_1_pairwise_train.jsonl"
DEFAULT_TARGETED_RECOMMENDATIONS_OUT = DEFAULT_OUTPUT_DIR / "v6_0_1_targeted_labeling_recommendations.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_0_1_neural_split_balance_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_0_1_neural_split_balance_report.md"

PRIORITY_LABELING_TOPICS = {
    "v2_graph_neural_networks",
    "v2_multimodal_learning",
    "v2_recommendation_systems",
    "v2_robot_learning",
}
TARGET_VAL_TEST_HARD_NEGATIVES = 70
TARGETED_RECOMMENDATION_LIMIT = 80
TOPIC_SPLIT_SIZE = 2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["query_id"]), int(row["paper_id"])


def split_rows_by_topics(examples: list[dict[str, Any]], split_topics: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    split_for_topic = {
        topic: split
        for split, topics in split_topics.items()
        for topic in topics
    }
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in examples:
        split = split_for_topic[row["query_id"]]
        copied = dict(row)
        copied["v6_0_1_grouped_split"] = split
        rows_by_split[split].append(copied)
    for split in rows_by_split:
        rows_by_split[split].sort(key=lambda row: (row["query_id"], int(row["paper_id"])))
    return rows_by_split


def topic_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = split_diagnostics(rows)
    stats["sections_present"] = sorted({row["section_label"] for row in rows})
    stats["has_relevance_0_1_2"] = all(stats["relevance_distribution"].get(str(bucket), 0) > 0 for bucket in (0, 1, 2))
    stats["has_hard_negatives"] = stats["hard_negative_count"] > 0
    stats["has_core_foundational_application_diversity"] = bool(
        {"foundational", "core_methods", "application"} & set(stats["sections_present"])
    )
    return stats


def original_split_topics(examples: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        by_split[row["query_grouped_split"]].add(row["query_id"])
    return {split: sorted(topics) for split, topics in by_split.items()}


def split_score(
    *,
    val_topics: tuple[str, ...],
    test_topics: tuple[str, ...],
    topic_rows: dict[str, list[dict[str, Any]]],
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    val_stats = topic_stats([row for topic in val_topics for row in topic_rows[topic]])
    test_stats = topic_stats([row for topic in test_topics for row in topic_rows[topic]])
    penalty = 0.0
    for stats in (val_stats, test_stats):
        if not stats["has_relevance_0_1_2"]:
            penalty += 100_000.0
        if not stats["has_hard_negatives"]:
            penalty += 100_000.0
        if not stats["has_core_foundational_application_diversity"]:
            penalty += 1_000.0
    penalty += abs(val_stats["hard_negative_count"] - TARGET_VAL_TEST_HARD_NEGATIVES)
    penalty += abs(test_stats["hard_negative_count"] - TARGET_VAL_TEST_HARD_NEGATIVES)
    penalty += abs(val_stats["hard_negative_count"] - test_stats["hard_negative_count"]) * 1.5

    eval_topics = set(val_topics) | set(test_topics)
    penalty -= len(eval_topics & PRIORITY_LABELING_TOPICS) * 8.0
    if "v2_multimodal_learning" in eval_topics:
        penalty -= 10.0
    if "v2_robot_learning" in eval_topics:
        penalty -= 10.0
    return penalty, val_stats, test_stats


def choose_balanced_split(examples: list[dict[str, Any]]) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    topics = sorted({row["query_id"] for row in examples})
    topic_rows = {topic: [row for row in examples if row["query_id"] == topic] for topic in topics}
    candidates = []
    for val_topics in itertools.combinations(topics, TOPIC_SPLIT_SIZE):
        remaining = [topic for topic in topics if topic not in val_topics]
        for test_topics in itertools.combinations(remaining, TOPIC_SPLIT_SIZE):
            train_topics = tuple(topic for topic in topics if topic not in set(val_topics) | set(test_topics))
            score, val_stats, test_stats = split_score(
                val_topics=val_topics,
                test_topics=test_topics,
                topic_rows=topic_rows,
            )
            split_topics = {
                "train": list(train_topics),
                "val": list(val_topics),
                "test": list(test_topics),
            }
            candidates.append(
                {
                    "score": score,
                    "split_topics": split_topics,
                    "val_hard_negative_count": val_stats["hard_negative_count"],
                    "test_hard_negative_count": test_stats["hard_negative_count"],
                    "val_relevance_distribution": val_stats["relevance_distribution"],
                    "test_relevance_distribution": test_stats["relevance_distribution"],
                    "priority_topics_in_val_test": sorted((set(val_topics) | set(test_topics)) & PRIORITY_LABELING_TOPICS),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["score"],
            item["split_topics"]["val"],
            item["split_topics"]["test"],
        )
    )
    return candidates[0]["split_topics"], candidates[:10]


def topic_sufficiency_audit(examples: list[dict[str, Any]]) -> dict[str, Any]:
    audit = {}
    for topic in sorted({row["query_id"] for row in examples}):
        rows = [row for row in examples if row["query_id"] == topic]
        sufficiency = sufficiency_for_rows(rows)
        failed = [key for key, value in sufficiency["flags"].items() if value]
        audit[topic] = {
            "total_labels": len(rows),
            "relevance_distribution": distribution(rows, "relevance_bucket"),
            "hard_negative_count": sum(1 for row in rows if row["hard_negative"]),
            "section_distribution": distribution(rows, "section_label"),
            "difficulty_distribution": distribution(rows, "difficulty_label"),
            "missing_or_short_abstract_count": sum(1 for row in rows if row["short_or_missing_abstract"]),
            "weak_for_neural_training_or_evaluation": bool(failed),
            "weakness_reasons": failed,
        }
    return audit


def rank_value(row: dict[str, Any], method: str) -> int | None:
    ranks = row.get("retrieval_ranks_by_method") or {}
    value = ranks.get(method)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def score_value(row: dict[str, Any], method: str) -> float | None:
    scores = row.get("retrieval_scores_by_method") or {}
    value = scores.get(method)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_generic_survey_candidate(row: dict[str, Any]) -> bool:
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or "").lower()
    return any(token in title for token in ("survey", "review", "overview")) or "we survey" in abstract or "this survey" in abstract


def candidate_reason_and_focus(row: dict[str, Any], eval_topics: set[str]) -> tuple[list[str], str, float]:
    topic = str(row["query_id"])
    ranks = [rank for rank in (rank_value(row, method) for method in ("bm25", "tfidf", "embedding", "hybrid")) if rank is not None]
    best_rank = min(ranks) if ranks else 999_999
    rank_gap = max(ranks) - min(ranks) if len(ranks) >= 2 else 0
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or "").lower()
    application_like = any(
        token in title or token in abstract[:500]
        for token in ("detection", "diagnosis", "forecasting", "enhancement", "tourism", "robot", "traffic", "medical")
    )
    generic_survey = is_generic_survey_candidate(row)

    reasons = []
    score = 0.0
    if topic in PRIORITY_LABELING_TOPICS:
        reasons.append("topic_needs_more_labels")
        score += 100.0
    if topic in eval_topics:
        reasons.append("validation_or_test_topic")
        score += 20.0
    if 8 <= best_rank <= 35:
        reasons.append("top10_top30_boundary_candidate")
        score += 25.0
    elif best_rank <= 30:
        reasons.append("high_rank_unjudged_candidate")
        score += 15.0
    if rank_gap >= 20:
        reasons.append("ranker_disagreement_candidate")
        score += 20.0
    if application_like:
        reasons.append("application_overpromotion_candidate")
        score += 10.0
    if generic_survey:
        reasons.append("generic_survey_candidate")
        score += 10.0
    if text_word_count(row.get("abstract")) < 50:
        reasons.append("short_abstract_evidence_gap")
        score += 5.0
    if not reasons:
        reasons.append("boundary_case_candidate")

    if generic_survey:
        focus = "generic_survey_check"
    elif application_like:
        focus = "application_overpromotion_check"
    elif rank_gap >= 20:
        focus = "hard_negative_check"
    elif 8 <= best_rank <= 35:
        focus = "partial_match_check"
    else:
        focus = "boundary_case_check"
    return reasons, focus, score - best_rank * 0.01


def candidate_rank_score_metadata(row: dict[str, Any]) -> dict[str, dict[str, float | int | None]]:
    return {
        "bm25": {"rank": rank_value(row, "bm25"), "score": score_value(row, "bm25")},
        "v3_3_ltr": {"rank": None, "score": None},
        "v4_1_blend": {"rank": None, "score": None},
        "v4_9_guarded_text_blend": {"rank": None, "score": None},
    }


def targeted_labeling_recommendations(
    *,
    examples: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    split_topics: dict[str, list[str]],
    topic_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    judged_keys = {row_key(row) for row in examples}
    eval_topics = set(split_topics["val"]) | set(split_topics["test"])
    weak_topics = {
        topic
        for topic, row in topic_audit.items()
        if row["weak_for_neural_training_or_evaluation"]
    }
    selected = []
    seen: set[tuple[str, int]] = set()
    for row in candidate_pool:
        key = row_key(row)
        if key in judged_keys or key in seen:
            continue
        topic = str(row["query_id"])
        if topic not in (PRIORITY_LABELING_TOPICS | eval_topics | weak_topics):
            continue
        reasons, focus, priority = candidate_reason_and_focus(row, eval_topics)
        selected.append((priority, str(row.get("title") or ""), row, reasons, focus))
        seen.add(key)
    selected.sort(key=lambda item: (-item[0], item[2]["query_id"], int(item[2]["paper_id"])))

    rows = []
    for _, _, row, reasons, focus in selected[:TARGETED_RECOMMENDATION_LIMIT]:
        rows.append(
            {
                "schema_version": "v6.0.1_targeted_labeling_recommendation",
                "query_id": str(row["query_id"]),
                "query": str(row.get("query") or str(row["query_id"]).replace("v2_", "").replace("_", " ")),
                "paper_id": int(row["paper_id"]),
                "title": str(row.get("title") or ""),
                "abstract": str(row.get("abstract") or ""),
                "abstract_word_count": text_word_count(row.get("abstract")),
                "year": row.get("year"),
                "venue": row.get("venue"),
                "source": (row.get("source_provenance") or [None])[0],
                "current_label": None,
                "candidate_rankings": candidate_rank_score_metadata(row),
                "reason_selected": reasons,
                "suggested_labeling_focus": focus,
                "do_not_modify_official_labels": True,
            }
        )
    return rows


def compare_split_diagnostics(original: dict[str, list[dict[str, Any]]], balanced: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "original": {split: split_diagnostics(rows) for split, rows in original.items()},
        "balanced": {split: split_diagnostics(rows) for split, rows in balanced.items()},
        "hard_negative_delta": {
            split: sum(1 for row in balanced[split] if row["hard_negative"]) - sum(1 for row in original[split] if row["hard_negative"])
            for split in ("train", "val", "test")
        },
    }


def build_report(
    *,
    examples: list[dict[str, Any]],
    original_splits: dict[str, list[dict[str, Any]]],
    balanced_splits: dict[str, list[dict[str, Any]]],
    balanced_topics: dict[str, list[str]],
    candidate_splits: list[dict[str, Any]],
    topic_audit: dict[str, Any],
    recommendations: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_comparison = compare_split_diagnostics(original_splits, balanced_splits)
    original_val_hard = split_comparison["original"]["val"]["hard_negative_count"]
    balanced_val_hard = split_comparison["balanced"]["val"]["hard_negative_count"]
    original_test_hard = split_comparison["original"]["test"]["hard_negative_count"]
    balanced_test_hard = split_comparison["balanced"]["test"]["hard_negative_count"]
    weak_topics = [topic for topic, row in topic_audit.items() if row["weak_for_neural_training_or_evaluation"]]
    return {
        "schema_version": "v6.0.1_neural_split_balance_report",
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
        "source_examples": str(resolve_repo_path(args.examples)),
        "outputs": {
            "train": str(resolve_repo_path(args.train_out)),
            "val": str(resolve_repo_path(args.val_out)),
            "test": str(resolve_repo_path(args.test_out)),
            "pairwise_train": str(resolve_repo_path(args.pairwise_train_out)),
            "targeted_recommendations": str(resolve_repo_path(args.targeted_recommendations_out)),
        },
        "split_strategy": {
            "main": "query_topic_grouped",
            "topic_group_separation_preserved": True,
            "topics_per_split": {split: len(topics) for split, topics in balanced_topics.items()},
            "balanced_split_topics": balanced_topics,
            "candidate_split_count_considered": len(list(itertools.combinations(sorted({row["query_id"] for row in examples}), TOPIC_SPLIT_SIZE))),
            "top_candidate_splits": candidate_splits,
        },
        "split_comparison": split_comparison,
        "grouped_balance_improved": balanced_val_hard > original_val_hard and balanced_test_hard >= original_test_hard,
        "balance_constraints": {
            "validation_hard_negatives_meaningfully_higher": balanced_val_hard > original_val_hard,
            "test_hard_negatives_remain_meaningful": balanced_test_hard > 0,
            "val_test_have_relevance_0_1_2": all(
                all(split_comparison["balanced"][split]["relevance_distribution"].get(str(bucket), 0) > 0 for bucket in (0, 1, 2))
                for split in ("val", "test")
            ),
            "topic_grouped_constraint_makes_per_topic_class_balance_impossible": True,
            "constraint_note": "Several individual topics have too few hard negatives or too much class imbalance; split balancing can improve validation/test coverage but cannot repair per-topic label gaps.",
        },
        "topic_sufficiency_audit": topic_audit,
        "more_labels_recommended_before_v6_1_training": bool(weak_topics),
        "topics_needing_labels_most": weak_topics,
        "targeted_labeling_recommendation_count": len(recommendations),
        "pairwise_train_count": len(pairs),
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    original = report["split_comparison"]["original"]
    balanced = report["split_comparison"]["balanced"]
    lines = [
        "# V6.0.1 Neural Split Balance Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Grouped split balance improved: `{report['grouped_balance_improved']}`",
        f"- More labels recommended before V6.1 training: `{report['more_labels_recommended_before_v6_1_training']}`",
        f"- Targeted labeling candidates exported: `{report['targeted_labeling_recommendation_count']}`",
        f"- Pairwise train examples: `{report['pairwise_train_count']}`",
        "",
        "## Original Vs Balanced",
        "",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"- `{split}` original: `{original[split]['example_count']}` examples, "
            f"relevance `{original[split]['relevance_distribution']}`, hard negatives `{original[split]['hard_negative_count']}`"
        )
        lines.append(
            f"- `{split}` balanced: `{balanced[split]['example_count']}` examples, "
            f"relevance `{balanced[split]['relevance_distribution']}`, hard negatives `{balanced[split]['hard_negative_count']}`"
        )
    lines.extend(
        [
            "",
            "## Balanced Topics",
            "",
        ]
    )
    for split, topics in report["split_strategy"]["balanced_split_topics"].items():
        lines.append(f"- `{split}`: `{topics}`")
    lines.extend(
        [
            "",
            "## Label Sufficiency",
            "",
            f"- Topics needing labels most: `{report['topics_needing_labels_most']}`",
            f"- Constraint note: {report['balance_constraints']['constraint_note']}",
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
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES_OUT))
    parser.add_argument("--candidate-pool", default=str(DEFAULT_CANDIDATE_POOL))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
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
    examples = load_jsonl(resolve_repo_path(args.examples))
    examples.sort(key=lambda row: (row["query_id"], int(row["paper_id"])))
    original_splits = {
        split: [row for row in examples if row["query_grouped_split"] == split]
        for split in ("train", "val", "test")
    }
    balanced_topics, candidate_splits = choose_balanced_split(examples)
    balanced_splits = split_rows_by_topics(examples, balanced_topics)
    pairs = build_pairwise_train(balanced_splits["train"])
    topic_audit = topic_sufficiency_audit(examples)
    recommendations = targeted_labeling_recommendations(
        examples=examples,
        candidate_pool=load_jsonl(resolve_repo_path(args.candidate_pool)),
        split_topics=balanced_topics,
        topic_audit=topic_audit,
    )
    write_jsonl(resolve_repo_path(args.train_out), balanced_splits["train"])
    write_jsonl(resolve_repo_path(args.val_out), balanced_splits["val"])
    write_jsonl(resolve_repo_path(args.test_out), balanced_splits["test"])
    write_jsonl(resolve_repo_path(args.pairwise_train_out), pairs)
    write_jsonl(resolve_repo_path(args.targeted_recommendations_out), recommendations)
    report = build_report(
        examples=examples,
        original_splits=original_splits,
        balanced_splits=balanced_splits,
        balanced_topics=balanced_topics,
        candidate_splits=candidate_splits,
        topic_audit=topic_audit,
        recommendations=recommendations,
        pairs=pairs,
        args=args,
    )
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.0.1 neural split balance report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
