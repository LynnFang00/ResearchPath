import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from build_v4_3_text_reranker_dataset import WEAK_TOPICS, protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    K,
    labels_by_query,
    rank_from_scores,
    resolve_repo_path,
    write_json,
    write_jsonl,
    write_text,
)
from evaluate_v4_0_production_pool_2400 import V33_METHOD  # noqa: E402
from evaluate_v4_4_text_blend_sensitivity import (  # noqa: E402
    BLEND_SPECS,
    DEFAULT_V43_TRAINING_REPORT,
    DEFAULT_JSON_OUT as DEFAULT_SENSITIVITY_JSON,
    ACCEPTANCE_TOLERANCE,
    add_sensitivity_blends,
    aggregate_metric_rows,
    build_report as build_sensitivity_report,
    hard_negative_suppression,
    is_hard_negative,
    is_positive,
    load_jsonl,
    score_context,
)
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
)
from evaluate_v4_1_calibrated_blends import DEFAULT_WEIGHTED_REPORT, DEFAULT_V41_MODEL_DIR  # noqa: E402
from evaluate_v4_3_text_reranker_production_pool import DEFAULT_V43_MODEL_DIR  # noqa: E402
from build_v4_3_text_reranker_dataset import DEFAULT_DATASET_JSONL as DEFAULT_V43_DATASET  # noqa: E402
from evaluate_v4_3_text_reranker_production_pool import V41_BLEND_METHOD  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_4_text_blend_robustness_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_4_text_blend_robustness_report.md"
DEFAULT_FAILURE_EXAMPLES_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_4_text_blend_failure_examples.jsonl"
DEFAULT_CANDIDATE_CONFIG_OUT = REPO_ROOT / "data" / "processed" / "models" / "v4_4_text_blend_candidate.json"
SEVERE_REGRESSION_THRESHOLD = -0.05


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * pct)))
    return sorted_values[index]


def per_topic_deltas(evaluation: dict[str, Any], selected_method: str, baseline_method: str) -> list[dict[str, Any]]:
    selected = {row["query_id"]: row for row in evaluation["per_topic"][selected_method]}
    baseline = {row["query_id"]: row for row in evaluation["per_topic"][baseline_method]}
    rows = []
    for query_id in sorted(set(selected) & set(baseline)):
        rows.append(
            {
                "query_id": query_id,
                "weak_topic": query_id in WEAK_TOPICS,
                "reading_delta": selected[query_id]["reading_value_ndcg"] - baseline[query_id]["reading_value_ndcg"],
                "topic_delta": selected[query_id]["topic_match_ndcg"] - baseline[query_id]["topic_match_ndcg"],
                "hard_negative_delta": selected[query_id]["hard_negative_rate_at_10"] - baseline[query_id]["hard_negative_rate_at_10"],
                "judged_delta": selected[query_id]["judged_at_10"] - baseline[query_id]["judged_at_10"],
                "unjudged_delta": selected[query_id]["unjudged_at_10"] - baseline[query_id]["unjudged_at_10"],
            }
        )
    return rows


def bootstrap_query_deltas(evaluation: dict[str, Any], selected_method: str, baseline_method: str, *, iterations: int = 500) -> dict[str, Any]:
    rng = random.Random(17)
    selected = {row["query_id"]: row for row in evaluation["per_topic"][selected_method]}
    baseline = {row["query_id"]: row for row in evaluation["per_topic"][baseline_method]}
    topics = sorted(set(selected) & set(baseline))
    reading_deltas = []
    topic_deltas = []
    hard_negative_deltas = []
    for _ in range(iterations):
        sample = [rng.choice(topics) for _ in topics]
        reading_deltas.append(sum(selected[q]["reading_value_ndcg"] - baseline[q]["reading_value_ndcg"] for q in sample) / len(sample))
        topic_deltas.append(sum(selected[q]["topic_match_ndcg"] - baseline[q]["topic_match_ndcg"] for q in sample) / len(sample))
        hard_negative_deltas.append(sum(selected[q]["hard_negative_rate_at_10"] - baseline[q]["hard_negative_rate_at_10"] for q in sample) / len(sample))
    return {
        "iterations": iterations,
        "reading_delta_mean": sum(reading_deltas) / len(reading_deltas),
        "reading_delta_p05": percentile(reading_deltas, 0.05),
        "reading_delta_p95": percentile(reading_deltas, 0.95),
        "topic_delta_mean": sum(topic_deltas) / len(topic_deltas),
        "topic_delta_p05": percentile(topic_deltas, 0.05),
        "topic_delta_p95": percentile(topic_deltas, 0.95),
        "hard_negative_delta_mean": sum(hard_negative_deltas) / len(hard_negative_deltas),
        "hard_negative_delta_p05": percentile(hard_negative_deltas, 0.05),
        "hard_negative_delta_p95": percentile(hard_negative_deltas, 0.95),
    }


def abstract_word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("abstract") or "").split())


def generic_survey(row: dict[str, Any]) -> bool:
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or "").lower()
    return any(token in title for token in ["survey", "review", "overview"]) or "we survey" in abstract or "this survey" in abstract


def promotion_bias(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    selected_method: str,
    baseline_method: str,
) -> dict[str, Any]:
    selected_counts = Counter()
    baseline_counts = Counter()
    selected_lengths = []
    baseline_lengths = []
    examples = []
    for query_id, rows in candidate_rows_by_q.items():
        row_by_id = {int(row["paper_id"]): row for row in rows}
        selected_top = rank_from_scores(method_scores[selected_method][query_id])[:K]
        baseline_top = rank_from_scores(method_scores[baseline_method][query_id])[:K]
        for paper_id in selected_top:
            row = row_by_id[paper_id]
            length = abstract_word_count(row)
            selected_lengths.append(length)
            selected_counts["missing_or_short_abstract"] += int(length < 50)
            selected_counts["generic_survey"] += int(generic_survey(row))
        for paper_id in baseline_top:
            row = row_by_id[paper_id]
            length = abstract_word_count(row)
            baseline_lengths.append(length)
            baseline_counts["missing_or_short_abstract"] += int(length < 50)
            baseline_counts["generic_survey"] += int(generic_survey(row))
        for paper_id in selected_top:
            if paper_id not in baseline_top:
                row = row_by_id[paper_id]
                length = abstract_word_count(row)
                if length < 50 or generic_survey(row):
                    examples.append(
                        {
                            "query_id": query_id,
                            "paper_id": paper_id,
                            "title": row.get("title"),
                            "abstract_word_count": length,
                            "generic_survey": generic_survey(row),
                            "selected_score": method_scores[selected_method][query_id][paper_id],
                            "baseline_score": method_scores[baseline_method][query_id].get(paper_id),
                        }
                    )
    total = max(len(candidate_rows_by_q) * K, 1)
    return {
        "selected_missing_or_short_abstract_rate": selected_counts["missing_or_short_abstract"] / total,
        "baseline_missing_or_short_abstract_rate": baseline_counts["missing_or_short_abstract"] / total,
        "selected_generic_survey_rate": selected_counts["generic_survey"] / total,
        "baseline_generic_survey_rate": baseline_counts["generic_survey"] / total,
        "selected_mean_abstract_words": sum(selected_lengths) / len(selected_lengths) if selected_lengths else 0.0,
        "baseline_mean_abstract_words": sum(baseline_lengths) / len(baseline_lengths) if baseline_lengths else 0.0,
        "flagged_new_promotions": examples[:50],
    }


def scoped_candidate_metrics(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    selected_method: str,
    scope: str,
) -> dict[str, Any]:
    label_lookup = labels_by_query(labels)
    rows = []
    for query_id, candidates in candidate_rows_by_q.items():
        ranked = rank_from_scores(method_scores[selected_method][query_id])
        top = ranked[:K]
        for paper_id in top:
            label = label_lookup.get(query_id, {}).get(paper_id)
            candidate = next((row for row in candidates if int(row["paper_id"]) == paper_id), None)
            include = False
            if scope == "weak_topics":
                include = query_id in WEAK_TOPICS
            elif scope == "hard_negative_candidates":
                include = bool(label and is_hard_negative(label))
            elif scope == "missing_or_short_abstract":
                include = bool(candidate and abstract_word_count(candidate) < 50)
            elif scope == "generic_survey":
                include = bool(candidate and generic_survey(candidate))
            if include:
                rows.append(
                    {
                        "query_id": query_id,
                        "paper_id": paper_id,
                        "label_present": label is not None,
                        "positive": bool(label and is_positive(label)),
                        "hard_negative": bool(label and is_hard_negative(label)),
                    }
                )
    return {
        "scope": scope,
        "top10_row_count": len(rows),
        "judged_count": sum(1 for row in rows if row["label_present"]),
        "positive_count": sum(1 for row in rows if row["positive"]),
        "hard_negative_count": sum(1 for row in rows if row["hard_negative"]),
    }


def failure_examples(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    selected_method: str,
    baseline_method: str,
    per_topic_vs_baseline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    label_lookup = labels_by_query(labels)
    bad_topics = {row["query_id"] for row in per_topic_vs_baseline if row["reading_delta"] < 0 or row["topic_delta"] < 0}
    examples = []
    for query_id in sorted(bad_topics):
        row_by_id = {int(row["paper_id"]): row for row in candidate_rows_by_q[query_id]}
        selected_top = rank_from_scores(method_scores[selected_method][query_id])[:K]
        baseline_top = rank_from_scores(method_scores[baseline_method][query_id])[:K]
        for paper_id in selected_top:
            if paper_id in baseline_top:
                continue
            label = label_lookup.get(query_id, {}).get(paper_id)
            candidate = row_by_id[paper_id]
            examples.append(
                {
                    "query_id": query_id,
                    "paper_id": paper_id,
                    "title": candidate.get("title"),
                    "abstract_word_count": abstract_word_count(candidate),
                    "generic_survey": generic_survey(candidate),
                    "selected_method": selected_method,
                    "baseline_method": baseline_method,
                    "selected_score": method_scores[selected_method][query_id][paper_id],
                    "baseline_score": method_scores[baseline_method][query_id].get(paper_id),
                    "selected_rank": selected_top.index(paper_id) + 1,
                    "baseline_rank": baseline_top.index(paper_id) + 1 if paper_id in baseline_top else None,
                    "label_present": label is not None,
                    "reading_value_score": label.get("reading_value_score") if label else None,
                    "topic_match_score": label.get("topic_match_score") if label else None,
                    "primary_role": label.get("primary_role") if label else None,
                }
            )
    return examples[:100]


def acceptance_check(
    *,
    sensitivity_report: dict[str, Any],
    selected_method: str,
    per_topic_vs_v41: list[dict[str, Any]],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    selected = next(row for row in sensitivity_report["sensitivity"] if row["method"] == selected_method)
    baseline = sensitivity_report["baseline_v4_1_blend"]
    severe_weak_regressions = [
        row
        for row in per_topic_vs_v41
        if row["weak_topic"] and (row["reading_delta"] < SEVERE_REGRESSION_THRESHOLD or row["topic_delta"] < SEVERE_REGRESSION_THRESHOLD)
    ]
    stable = sensitivity_report["selection"]["acceptable_count"] >= 2 and bootstrap["reading_delta_p05"] > -0.02 and bootstrap["topic_delta_p05"] > -0.02
    criteria = {
        "reading_ndcg_at_10_gte_v4_1_blend": selected["reading_value_ndcg"] >= baseline["reading_value_ndcg"] - ACCEPTANCE_TOLERANCE,
        "topic_ndcg_at_10_gte_v4_1_blend": selected["topic_match_ndcg"] >= baseline["topic_match_ndcg"] - ACCEPTANCE_TOLERANCE,
        "hard_negative_at_10_lte_v4_1_blend": selected["hard_negative_rate_at_10"] <= baseline["hard_negative_rate_at_10"] + ACCEPTANCE_TOLERANCE,
        "judged_at_10_close": selected["judged_at_10"] >= 9.75,
        "unjudged_at_10_close": selected["unjudged_at_10"] <= 0.25,
        "no_severe_weak_topic_regression": not severe_weak_regressions,
        "text_weight_sensitivity_not_brittle": stable,
    }
    return {
        "passes": all(criteria.values()),
        "criteria": criteria,
        "severe_weak_topic_regressions": severe_weak_regressions,
    }


def write_candidate_config(path: Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    selected = report["selected_blend"]
    config = {
        "schema_version": "v4.4_text_blend_candidate_config",
        "created_at": datetime.now(UTC).isoformat(),
        "blend_name": selected["method"],
        "formula": selected["weights"],
        "component_methods": report["components"],
        "component_model_paths": {
            "v3_3_model_dir": str(resolve_repo_path(args.v33_model_dir)),
            "v4_1_model_dir": str(resolve_repo_path(args.v41_model_dir)),
            "v4_3_text_model_dir": str(resolve_repo_path(args.v43_model_dir)),
        },
        "text_preprocessing": {
            "text_input": "Query, title, and abstract concatenated with field prefixes.",
            "selected_text_score": report["components"]["v4_3_text_method"],
        },
        "evaluation_summary": selected,
        "warning": "Offline-only candidate. Not runtime-integrated and not a default ranker.",
    }
    write_json(path, config)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sensitivity_path = resolve_repo_path(args.sensitivity_report)
    if sensitivity_path.exists():
        sensitivity_report = load_json(sensitivity_path)
    else:
        sensitivity_report = build_sensitivity_report(args)
    context = score_context(args)
    method_scores = add_sensitivity_blends(context["method_scores"], context["best_v41_method"], context["best_v43_method"])
    selected_method = sensitivity_report["selection"]["best"]["method"]
    evaluation = sensitivity_report["evaluation"]
    per_topic_v41 = per_topic_deltas(evaluation, selected_method, V41_BLEND_METHOD)
    per_topic_v33 = per_topic_deltas(evaluation, selected_method, V33_METHOD)
    bootstrap = bootstrap_query_deltas(evaluation, selected_method, V41_BLEND_METHOD, iterations=args.bootstrap_iterations)
    bias = promotion_bias(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        method_scores=method_scores,
        selected_method=selected_method,
        baseline_method=V41_BLEND_METHOD,
    )
    hard_negative = hard_negative_suppression(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        methods=[selected_method, V41_BLEND_METHOD, V33_METHOD],
    )
    selected_blend = next(row for row in sensitivity_report["sensitivity"] if row["method"] == selected_method)
    acceptance = acceptance_check(
        sensitivity_report=sensitivity_report,
        selected_method=selected_method,
        per_topic_vs_v41=per_topic_v41,
        bootstrap=bootstrap,
    )
    examples = failure_examples(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        selected_method=selected_method,
        baseline_method=V41_BLEND_METHOD,
        per_topic_vs_baseline=per_topic_v41,
    )
    report = {
        "schema_version": "v4.4_text_blend_robustness_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "components": sensitivity_report["components"],
        "selected_blend": selected_blend,
        "baseline_v4_1_blend": sensitivity_report["baseline_v4_1_blend"],
        "acceptance": acceptance,
        "recommendation": "future_opt_in_candidate" if acceptance["passes"] else "keep_offline_only",
        "per_topic_wins_losses_vs_v4_1_blend": per_topic_v41,
        "per_topic_wins_losses_vs_v3_3": per_topic_v33,
        "severe_topic_regressions_vs_v4_1_blend": [
            row for row in per_topic_v41 if row["reading_delta"] < SEVERE_REGRESSION_THRESHOLD or row["topic_delta"] < SEVERE_REGRESSION_THRESHOLD
        ],
        "bootstrap_query_resampling_vs_v4_1_blend": bootstrap,
        "disagreement_pool_accuracy": sensitivity_report["disagreement_pool_accuracy"].get(selected_method),
        "hard_negative_suppression": hard_negative,
        "weak_topic_metrics": aggregate_metric_rows([row for row in evaluation["per_topic"][selected_method] if row["query_id"] in WEAK_TOPICS]),
        "scope_checks": {
            scope: scoped_candidate_metrics(
                candidate_rows_by_q=context["candidate_rows_by_q"],
                labels=context["labels"],
                method_scores=method_scores,
                selected_method=selected_method,
                scope=scope,
            )
            for scope in ["weak_topics", "hard_negative_candidates", "missing_or_short_abstract", "generic_survey"]
        },
        "promotion_bias_checks": bias,
        "unjudged_top10_delta_vs_v4_1_blend": selected_blend["unjudged_at_10"] - sensitivity_report["baseline_v4_1_blend"]["unjudged_at_10"],
        "failure_examples_path": str(resolve_repo_path(args.failure_examples_out)),
        "candidate_config_created": False,
        "candidate_config_path": str(resolve_repo_path(args.candidate_config_out)),
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }
    if acceptance["passes"]:
        write_candidate_config(resolve_repo_path(args.candidate_config_out), report, args)
        report["candidate_config_created"] = True
    return report, examples


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["selected_blend"]
    lines = [
        "# V4.4 Text Blend Robustness Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Decision",
        "",
        f"- Selected blend: `{selected['method']}`",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Acceptance passed: `{report['acceptance']['passes']}`",
        f"- Candidate config created: `{report['candidate_config_created']}`",
        "",
        "## Selected Metrics",
        "",
        f"- Reading NDCG@10: `{selected['reading_value_ndcg']:.3f}`",
        f"- Topic NDCG@10: `{selected['topic_match_ndcg']:.3f}`",
        f"- Judged@10: `{selected['judged_at_10']:.3f}`",
        f"- Unjudged@10: `{selected['unjudged_at_10']:.3f}`",
        f"- Hard-neg@10: `{selected['hard_negative_rate_at_10']:.3f}`",
        "",
        "## Acceptance Criteria",
        "",
    ]
    for key, value in report["acceptance"]["criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Per-Topic Deltas Vs V4.1 Blend", "", "| Topic | Weak | Reading Delta | Topic Delta | Hard-neg Delta |", "|---|---:|---:|---:|---:|"])
    for row in report["per_topic_wins_losses_vs_v4_1_blend"]:
        lines.append(
            f"| `{row['query_id']}` | `{row['weak_topic']}` | `{row['reading_delta']:.3f}` | "
            f"`{row['topic_delta']:.3f}` | `{row['hard_negative_delta']:.3f}` |"
        )
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR))
    parser.add_argument("--v41-model-dir", default=str(DEFAULT_V41_MODEL_DIR))
    parser.add_argument("--weighted-report", default=str(DEFAULT_WEIGHTED_REPORT))
    parser.add_argument("--v43-model-dir", default=str(DEFAULT_V43_MODEL_DIR))
    parser.add_argument("--v43-training-report", default=str(DEFAULT_V43_TRAINING_REPORT))
    parser.add_argument("--v43-dataset", default=str(DEFAULT_V43_DATASET))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sensitivity-report", default=str(DEFAULT_SENSITIVITY_JSON))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--failure-examples-out", default=str(DEFAULT_FAILURE_EXAMPLES_OUT))
    parser.add_argument("--candidate-config-out", default=str(DEFAULT_CANDIDATE_CONFIG_OUT))
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, examples = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    write_jsonl(resolve_repo_path(args.failure_examples_out), examples)
    print(f"Wrote V4.4 text blend robustness report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
