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
from diagnose_v4_5_text_blend_regressions import (  # noqa: E402
    TARGET_TOPIC,
    TEXT_05,
    TEXT_10,
    abstract_word_count,
    filtered_weighted_blend,
    generic_survey,
)
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
    evaluate_rankings,
    labels_by_query,
    rank_from_scores,
    resolve_repo_path,
    write_json,
    write_jsonl,
    write_text,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
)
from evaluate_v4_0_production_pool_2400 import V33_METHOD  # noqa: E402
from evaluate_v4_1_calibrated_blends import DEFAULT_WEIGHTED_REPORT, DEFAULT_V41_MODEL_DIR, weighted_blend  # noqa: E402
from evaluate_v4_3_text_reranker_production_pool import DEFAULT_V43_MODEL_DIR, V41_BLEND_METHOD  # noqa: E402
from evaluate_v4_4_text_blend_sensitivity import (  # noqa: E402
    DEFAULT_V43_TRAINING_REPORT,
    add_sensitivity_blends,
    is_hard_negative,
    is_positive,
    score_context,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_6_guarded_text_blend_validation_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_6_guarded_text_blend_validation_report.md"
DEFAULT_LEAKAGE_AUDIT_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_6_guardrail_leakage_audit.json"
DEFAULT_FAILURE_EXAMPLES_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_6_guardrail_failure_examples.jsonl"
DEFAULT_CANDIDATE_CONFIG_OUT = REPO_ROOT / "data" / "processed" / "models" / "v4_6_guarded_text_blend_candidate.json"

GUARDRAIL_METHOD = "v4_6_guardrail_text_when_rankers_not_confident"
TOPIC_HURT_METHOD = "v4_5_topic_hurt_reduced_non_runtime_safe"
SEVERE_REGRESSION_THRESHOLD = -0.05
MEANINGFUL_JUDGED_DROP = 0.25
ALLOWED_FEATURES = {
    "v3_3_score",
    "v4_1_score",
    "v4_3_text_score",
    "v3_3_rank",
    "v4_1_rank",
    "v4_3_text_rank",
    "v3_3_v4_1_margin",
    "v3_3_v4_1_gap",
    "v3_3_text_gap",
    "v4_1_text_gap",
    "normalized_v3_3_score",
    "normalized_v4_1_score",
    "normalized_v4_3_text_score",
    "max_ranker_confidence",
}
FORBIDDEN_FEATURES = {
    "true_relevance_label",
    "reading_value_score",
    "topic_match_score",
    "hard_negative_label",
    "primary_role",
    "section_label",
    "difficulty_label",
    "per_topic_known_hurt_status",
    "per_topic_known_improved_status",
    "weak_topic_label",
    "judged_source",
}
GUARDRAIL_FORMULA = {
    "name": GUARDRAIL_METHOD,
    "base_formula": "0.675 * normalized_v3_3_score + 0.225 * normalized_v4_1_score + gated_text_component",
    "text_component": "0.10 * normalized_v4_3_text_score only when max(normalized_v3_3_score, normalized_v4_1_score) < 0.85; otherwise 0.0",
    "features_used": [
        "normalized_v3_3_score",
        "normalized_v4_1_score",
        "normalized_v4_3_text_score",
        "max_ranker_confidence",
    ],
}


def guardrail_feature_audit(extra_features: set[str] | None = None) -> dict[str, Any]:
    features = set(GUARDRAIL_FORMULA["features_used"]) | set(extra_features or set())
    forbidden_used = sorted(features & FORBIDDEN_FEATURES)
    unknown_used = sorted(features - ALLOWED_FEATURES - FORBIDDEN_FEATURES)
    return {
        "schema_version": "v4.6_guardrail_leakage_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "guardrail_method": GUARDRAIL_METHOD,
        "formula": GUARDRAIL_FORMULA,
        "allowed_features": sorted(ALLOWED_FEATURES),
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "features_used": sorted(features),
        "forbidden_features_used": forbidden_used,
        "unknown_features_used": unknown_used,
        "uses_only_inference_time_features": not forbidden_used and not unknown_used,
        "non_runtime_safe_methods": {
            TOPIC_HURT_METHOD: {
                "runtime_safe": False,
                "reason": "Requires per-topic known hurt status from validation/evaluation outcomes.",
                "forbidden_feature": "per_topic_known_hurt_status",
            }
        },
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def bootstrap_metric_ci(evaluation: dict[str, Any], method: str, *, iterations: int = 1000) -> dict[str, Any]:
    rng = random.Random(23)
    rows = evaluation["per_topic"][method]
    reading = []
    topic = []
    for _ in range(iterations):
        sample = [rng.choice(rows) for _ in rows]
        reading.append(sum(float(row["reading_value_ndcg"]) for row in sample) / len(sample))
        topic.append(sum(float(row["topic_match_ndcg"]) for row in sample) / len(sample))
    return {
        "iterations": iterations,
        "reading_value_ndcg_mean": sum(reading) / len(reading),
        "reading_value_ndcg_p05": percentile(reading, 0.05),
        "reading_value_ndcg_p95": percentile(reading, 0.95),
        "topic_match_ndcg_mean": sum(topic) / len(topic),
        "topic_match_ndcg_p05": percentile(topic, 0.05),
        "topic_match_ndcg_p95": percentile(topic, 0.95),
    }


def aggregate_topic_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = [
        "reading_value_ndcg",
        "topic_match_ndcg",
        "mrr_at_10",
        "positive_recall_at_10",
        "hard_negative_rate_at_10",
        "judged_at_10",
        "unjudged_at_10",
    ]
    return {field: sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0 for field in fields}


def per_topic_deltas(evaluation: dict[str, Any], method: str, baseline: str) -> list[dict[str, Any]]:
    left = {row["query_id"]: row for row in evaluation["per_topic"][method]}
    right = {row["query_id"]: row for row in evaluation["per_topic"][baseline]}
    output = []
    for query_id in sorted(set(left) & set(right)):
        output.append(
            {
                "query_id": query_id,
                "weak_topic": query_id in WEAK_TOPICS,
                "reading_delta": left[query_id]["reading_value_ndcg"] - right[query_id]["reading_value_ndcg"],
                "topic_delta": left[query_id]["topic_match_ndcg"] - right[query_id]["topic_match_ndcg"],
                "mrr_delta": left[query_id]["mrr_at_10"] - right[query_id]["mrr_at_10"],
                "recall_delta": left[query_id]["positive_recall_at_10"] - right[query_id]["positive_recall_at_10"],
                "hard_negative_delta": left[query_id]["hard_negative_rate_at_10"] - right[query_id]["hard_negative_rate_at_10"],
                "judged_delta": left[query_id]["judged_at_10"] - right[query_id]["judged_at_10"],
                "unjudged_delta": left[query_id]["unjudged_at_10"] - right[query_id]["unjudged_at_10"],
            }
        )
    return output


def win_loss_tie(deltas: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [float(row[field]) for row in deltas]
    return {
        "wins": sum(value > 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
    }


def leave_one_topic_sensitivity(evaluation: dict[str, Any], method: str, baseline: str) -> list[dict[str, Any]]:
    method_rows = {row["query_id"]: row for row in evaluation["per_topic"][method]}
    baseline_rows = {row["query_id"]: row for row in evaluation["per_topic"][baseline]}
    topics = sorted(set(method_rows) & set(baseline_rows))
    output = []
    for omitted in topics:
        remaining = [topic for topic in topics if topic != omitted]
        reading_delta = sum(method_rows[t]["reading_value_ndcg"] - baseline_rows[t]["reading_value_ndcg"] for t in remaining) / len(remaining)
        topic_delta = sum(method_rows[t]["topic_match_ndcg"] - baseline_rows[t]["topic_match_ndcg"] for t in remaining) / len(remaining)
        output.append({"omitted_topic": omitted, "reading_delta_vs_v4_1_blend": reading_delta, "topic_delta_vs_v4_1_blend": topic_delta})
    return output


def rank_diagnostics(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    method: str,
    baseline: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_lookup = labels_by_query(labels)
    summary = Counter()
    examples = []
    for query_id, rows in candidate_rows_by_q.items():
        row_by_id = {int(row["paper_id"]): row for row in rows}
        method_top = rank_from_scores(method_scores[method][query_id])[:K]
        baseline_top = rank_from_scores(method_scores[baseline][query_id])[:K]
        method_only = [paper_id for paper_id in method_top if paper_id not in baseline_top]
        baseline_only = [paper_id for paper_id in baseline_top if paper_id not in method_top]
        summary["top10_swaps"] += len(method_only)
        for paper_id in baseline_only:
            label = label_lookup.get(query_id, {}).get(paper_id)
            if label and is_positive(label):
                summary["good_papers_demoted"] += 1
                examples.append(example_row("good_paper_demoted", query_id, paper_id, row_by_id[paper_id], label, method_scores, method, baseline))
        for paper_id in method_only:
            label = label_lookup.get(query_id, {}).get(paper_id)
            row = row_by_id[paper_id]
            if label and not is_positive(label):
                summary["bad_or_weak_papers_promoted"] += 1
            if label and is_hard_negative(label):
                summary["hard_negatives_promoted"] += 1
            if generic_survey(row):
                summary["generic_surveys_promoted"] += 1
            if abstract_word_count(row) < 50:
                summary["short_or_missing_abstract_promotions"] += 1
            examples.append(example_row("paper_promoted", query_id, paper_id, row, label, method_scores, method, baseline))

            baseline_confidence = max(
                normalized_score(method_scores[V33_METHOD][query_id], paper_id),
                normalized_score(method_scores[baseline][query_id], paper_id),
            )
            if baseline_confidence >= 0.85:
                summary["text_changed_rank_despite_high_ranker_confidence"] += 1
    return dict(summary), examples[:200]


def normalized_score(scores: dict[int, float], paper_id: int) -> float:
    values = list(scores.values())
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return 0.5
    return (scores.get(paper_id, low) - low) / (high - low)


def example_row(
    kind: str,
    query_id: str,
    paper_id: int,
    paper: dict[str, Any],
    label: dict[str, Any] | None,
    method_scores: dict[str, dict[str, dict[int, float]]],
    method: str,
    baseline: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "query_id": query_id,
        "paper_id": paper_id,
        "title": paper.get("title"),
        "abstract_word_count": abstract_word_count(paper),
        "generic_survey": generic_survey(paper),
        "method_score": method_scores[method][query_id].get(paper_id),
        "baseline_score": method_scores[baseline][query_id].get(paper_id),
        "reading_value_score": label.get("reading_value_score") if label else None,
        "topic_match_score": label.get("topic_match_score") if label else None,
        "primary_role": label.get("primary_role") if label else None,
        "positive": bool(label and is_positive(label)),
        "hard_negative": bool(label and is_hard_negative(label)),
    }


def create_candidate_config(path: Path, report: dict[str, Any], args: argparse.Namespace) -> None:
    config = {
        "schema_version": "v4.6_guarded_text_blend_candidate_config",
        "created_at": datetime.now(UTC).isoformat(),
        "blend_name": GUARDRAIL_METHOD,
        "formula": GUARDRAIL_FORMULA,
        "component_model_paths": {
            "v3_3_model_dir": str(resolve_repo_path(args.v33_model_dir)),
            "v4_1_model_dir": str(resolve_repo_path(args.v41_model_dir)),
            "v4_3_text_model_dir": str(resolve_repo_path(args.v43_model_dir)),
        },
        "evaluation_summary": report["method_averages"][GUARDRAIL_METHOD],
        "acceptance": report["acceptance"],
        "warning": "Offline-only candidate. Not runtime-integrated and not a default ranker.",
    }
    write_json(path, config)


def acceptance_check(
    *,
    evaluation: dict[str, Any],
    leakage_audit: dict[str, Any],
    protected: dict[str, Any],
) -> dict[str, Any]:
    guarded = evaluation["method_averages"][GUARDRAIL_METHOD]
    baseline = evaluation["method_averages"][V41_BLEND_METHOD]
    deltas = per_topic_deltas(evaluation, GUARDRAIL_METHOD, V41_BLEND_METHOD)
    severe_non_weak = [
        row
        for row in deltas
        if not row["weak_topic"] and row["reading_delta"] < SEVERE_REGRESSION_THRESHOLD
    ]
    multimodal_delta = next(row for row in deltas if row["query_id"] == TARGET_TOPIC)
    criteria = {
        "guardrail_uses_only_inference_time_features": leakage_audit["uses_only_inference_time_features"],
        "reading_ndcg_at_10_gte_v4_1_blend": guarded["reading_value_ndcg"] >= baseline["reading_value_ndcg"] - 1e-12,
        "topic_ndcg_at_10_gte_v4_1_blend": guarded["topic_match_ndcg"] >= baseline["topic_match_ndcg"] - 1e-12,
        "hard_negative_at_10_lte_v4_1_blend": guarded["hard_negative_rate_at_10"] <= baseline["hard_negative_rate_at_10"] + 1e-12,
        "no_severe_non_weak_topic_regression": not severe_non_weak,
        "multimodal_reading_delta_gte_minus_0_01": multimodal_delta["reading_delta"] >= -0.01,
        "judged_at_10_not_meaningfully_decreased": guarded["judged_at_10"] >= baseline["judged_at_10"] - MEANINGFUL_JUDGED_DROP,
        "protected_hashes_unchanged": all(value for key, value in protected.items() if key.endswith("_hash_unchanged")),
    }
    return {
        "passes": all(criteria.values()),
        "criteria": criteria,
        "severe_non_weak_topic_regressions": severe_non_weak,
        "multimodal_delta_vs_v4_1_blend": multimodal_delta,
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    leakage_audit = guardrail_feature_audit()
    context = score_context(args)
    method_scores = add_sensitivity_blends(context["method_scores"], context["best_v41_method"], context["best_v43_method"])
    method_scores[GUARDRAIL_METHOD] = filtered_weighted_blend(
        method_scores,
        v41_method=context["best_v41_method"],
        v43_method=context["best_v43_method"],
        text_weight=0.10,
        v33_weight=0.675,
        v41_weight=0.225,
        mode="text_only_when_rankers_not_confident",
    )
    method_scores[TOPIC_HURT_METHOD] = filtered_weighted_blend(
        method_scores,
        v41_method=context["best_v41_method"],
        v43_method=context["best_v43_method"],
        text_weight=0.10,
        v33_weight=0.675,
        v41_weight=0.225,
        mode="topic_hurt_reduced",
        hurt_topics={TARGET_TOPIC},
    )
    methods = [
        V33_METHOD,
        context["best_v41_method"],
        V41_BLEND_METHOD,
        TEXT_05,
        TEXT_10,
        GUARDRAIL_METHOD,
        TOPIC_HURT_METHOD,
    ]
    evaluation = evaluate_rankings(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores={method: method_scores[method] for method in methods},
        k=K,
    )
    protected = protected_status(resolve_repo_path(args.v39_labels))
    acceptance = acceptance_check(evaluation=evaluation, leakage_audit=leakage_audit, protected=protected)
    rank_summary, examples = rank_diagnostics(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        method=GUARDRAIL_METHOD,
        baseline=V41_BLEND_METHOD,
    )
    deltas_v41 = per_topic_deltas(evaluation, GUARDRAIL_METHOD, V41_BLEND_METHOD)
    deltas_v33 = per_topic_deltas(evaluation, GUARDRAIL_METHOD, V33_METHOD)
    report = {
        "schema_version": "v4.6_guarded_text_blend_validation_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "guardrail_method": GUARDRAIL_METHOD,
        "components": {
            "v3_3_method": V33_METHOD,
            "v4_1_weighted_method": context["best_v41_method"],
            "v4_1_blend_method": V41_BLEND_METHOD,
            "v4_3_text_method": context["best_v43_method"],
            "topic_hurt_reduced_method": TOPIC_HURT_METHOD,
        },
        "method_averages": evaluation["method_averages"],
        "per_topic": evaluation["per_topic"],
        "weak_topic_metrics": {
            method: aggregate_topic_rows([row for row in evaluation["per_topic"][method] if row["query_id"] in WEAK_TOPICS])
            for method in methods
        },
        "per_topic_deltas_vs_v4_1_blend": deltas_v41,
        "per_topic_deltas_vs_v3_3": deltas_v33,
        "topic_win_loss_tie_vs_v4_1_blend": {
            "reading": win_loss_tie(deltas_v41, "reading_delta"),
            "topic": win_loss_tie(deltas_v41, "topic_delta"),
            "hard_negative_improve_or_tie": sum(row["hard_negative_delta"] <= 1e-12 for row in deltas_v41),
            "topic_count": len(deltas_v41),
        },
        "topic_win_loss_tie_vs_v3_3": {
            "reading": win_loss_tie(deltas_v33, "reading_delta"),
            "topic": win_loss_tie(deltas_v33, "topic_delta"),
            "hard_negative_improve_or_tie": sum(row["hard_negative_delta"] <= 1e-12 for row in deltas_v33),
            "topic_count": len(deltas_v33),
        },
        "severe_regression_threshold": SEVERE_REGRESSION_THRESHOLD,
        "severe_regressions_vs_v4_1_blend": acceptance["severe_non_weak_topic_regressions"],
        "bootstrap_confidence_intervals": {
            method: bootstrap_metric_ci(evaluation, method, iterations=args.bootstrap_iterations)
            for method in [V41_BLEND_METHOD, TEXT_05, TEXT_10, GUARDRAIL_METHOD]
        },
        "leave_one_topic_out_sensitivity_vs_v4_1_blend": leave_one_topic_sensitivity(evaluation, GUARDRAIL_METHOD, V41_BLEND_METHOD),
        "rank_level_diagnostics": rank_summary,
        "acceptance": acceptance,
        "leakage_audit_path": str(resolve_repo_path(args.leakage_audit_out)),
        "failure_examples_path": str(resolve_repo_path(args.failure_examples_out)),
        "candidate_config_path": str(resolve_repo_path(args.candidate_config_out)),
        "candidate_config_created": False,
        "recommendation": "future_opt_in_candidate" if acceptance["passes"] else "keep_offline_only",
        "protected_hashes": protected,
    }
    if acceptance["passes"]:
        create_candidate_config(resolve_repo_path(args.candidate_config_out), report, args)
        report["candidate_config_created"] = True
    return report, leakage_audit, examples


def render_markdown(report: dict[str, Any]) -> str:
    guarded = report["method_averages"][GUARDRAIL_METHOD]
    multimodal = report["acceptance"]["multimodal_delta_vs_v4_1_blend"]
    lines = [
        "# V4.6 Guarded Text Blend Validation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Decision",
        "",
        f"- Inference-safe: `{report['acceptance']['criteria']['guardrail_uses_only_inference_time_features']}`",
        f"- Acceptance passed: `{report['acceptance']['passes']}`",
        f"- Candidate config created: `{report['candidate_config_created']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "## Guarded Blend Metrics",
        "",
        f"- Reading NDCG@10: `{guarded['reading_value_ndcg']:.3f}`",
        f"- Topic NDCG@10: `{guarded['topic_match_ndcg']:.3f}`",
        f"- Hard-neg@10: `{guarded['hard_negative_rate_at_10']:.3f}`",
        f"- Judged@10: `{guarded['judged_at_10']:.3f}`",
        f"- Unjudged@10: `{guarded['unjudged_at_10']:.3f}`",
        f"- Multimodal reading delta vs V4.1 blend: `{multimodal['reading_delta']:.3f}`",
        "",
        "## Acceptance Criteria",
        "",
    ]
    for key, value in report["acceptance"]["criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
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
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--leakage-audit-out", default=str(DEFAULT_LEAKAGE_AUDIT_OUT))
    parser.add_argument("--failure-examples-out", default=str(DEFAULT_FAILURE_EXAMPLES_OUT))
    parser.add_argument("--candidate-config-out", default=str(DEFAULT_CANDIDATE_CONFIG_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, leakage_audit, examples = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    write_json(resolve_repo_path(args.leakage_audit_out), leakage_audit)
    write_jsonl(resolve_repo_path(args.failure_examples_out), examples)
    print(f"Wrote V4.6 guarded text blend validation report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
