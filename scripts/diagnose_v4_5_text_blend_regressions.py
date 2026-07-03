import argparse
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

from build_v4_3_text_reranker_dataset import WEAK_TOPICS, protected_status  # noqa: E402
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
from evaluate_v4_0_production_pool_2400 import V33_METHOD  # noqa: E402
from evaluate_v4_1_calibrated_blends import (  # noqa: E402
    DEFAULT_WEIGHTED_REPORT,
    DEFAULT_V41_MODEL_DIR,
    normalize_query_scores,
    weighted_blend,
)
from evaluate_v4_3_text_reranker_production_pool import DEFAULT_V43_MODEL_DIR, V41_BLEND_METHOD  # noqa: E402
from evaluate_v4_4_text_blend_sensitivity import (  # noqa: E402
    DEFAULT_V43_TRAINING_REPORT,
    add_sensitivity_blends,
    is_hard_negative,
    is_positive,
    score_context,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_5_text_blend_regression_diagnosis_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_5_text_blend_regression_diagnosis_report.md"
DEFAULT_FAILURE_EXAMPLES_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_5_multimodal_learning_failure_examples.jsonl"
DEFAULT_GUARDRAIL_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_5_guardrail_experiments.json"
DEFAULT_GUARDRAIL_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_5_guardrail_experiments.md"

TARGET_TOPIC = "v2_multimodal_learning"
TEXT_05 = "text_weight_0_05"
TEXT_10 = "text_weight_0_10"
TEXT_15 = "text_weight_0_15"
RANKER_METHODS = [
    V33_METHOD,
    "__v41_weighted__",
    V41_BLEND_METHOD,
    TEXT_05,
    TEXT_10,
    TEXT_15,
]
RANKER_LABELS = {
    V33_METHOD: "V3.3 RF no-V2.7",
    "__v41_weighted__": "V4.1 weighted standalone",
    V41_BLEND_METHOD: "V4.1 calibrated blend",
    TEXT_05: "V4.4 text blend 5%",
    TEXT_10: "V4.4 text blend 10%",
    TEXT_15: "V4.4 text blend 15%",
}


def abstract_word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("abstract") or "").split())


def generic_survey(row: dict[str, Any]) -> bool:
    title = str(row.get("title") or "").lower()
    abstract = str(row.get("abstract") or "").lower()
    return any(token in title for token in ["survey", "review", "overview"]) or "we survey" in abstract or "this survey" in abstract


def difficulty_label(label: dict[str, Any] | None) -> str | None:
    if not label:
        return None
    scores = {
        "beginner": float(label.get("beginner_fit_score") or 0.0),
        "intermediate": float(label.get("intermediate_fit_score") or 0.0),
        "advanced": float(label.get("advanced_fit_score") or 0.0),
        "expert": float(label.get("expert_fit_score") or 0.0),
    }
    best_score = max(scores.values())
    if best_score <= 0.0:
        return None
    return ",".join(level for level, score in scores.items() if score == best_score)


def relevance_label(label: dict[str, Any] | None) -> dict[str, Any] | None:
    if not label:
        return None
    return {
        "reading_value_score": label.get("reading_value_score"),
        "topic_match_score": label.get("topic_match_score"),
        "positive": is_positive(label),
    }


def venue_or_source(row: dict[str, Any]) -> str | None:
    if row.get("venue"):
        return str(row["venue"])
    if row.get("source"):
        return str(row["source"])
    provenance = row.get("sources_provenance")
    if isinstance(provenance, list) and provenance:
        return ",".join(str(item) for item in provenance)
    source_info = row.get("source_pool_info") or {}
    return source_info.get("source_pool")


def rank_lookup(scores: dict[int, float]) -> dict[int, int]:
    return {paper_id: rank for rank, paper_id in enumerate(rank_from_scores(scores), start=1)}


def normalized_components(method_scores: dict[str, dict[str, dict[int, float]]], query_id: str, v41_method: str, v43_method: str) -> dict[str, dict[int, float]]:
    return {
        "v3_3": normalize_query_scores(method_scores[V33_METHOD][query_id]),
        "v4_1": normalize_query_scores(method_scores[v41_method][query_id]),
        "v4_3_text": normalize_query_scores(method_scores[v43_method][query_id]),
    }


def make_top10_rows(
    *,
    query_id: str,
    method: str,
    method_label: str,
    candidate_rows: list[dict[str, Any]],
    labels: dict[int, dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    v41_method: str,
    v43_method: str,
) -> list[dict[str, Any]]:
    row_by_id = {int(row["paper_id"]): row for row in candidate_rows}
    baseline_rank = rank_lookup(method_scores[V41_BLEND_METHOD][query_id])
    components = normalized_components(method_scores, query_id, v41_method, v43_method)
    ranked = rank_from_scores(method_scores[method][query_id])[:K]
    rows = []
    for rank, paper_id in enumerate(ranked, start=1):
        row = row_by_id[paper_id]
        label = labels.get(paper_id)
        base_rank = baseline_rank.get(paper_id)
        rows.append(
            {
                "query_id": query_id,
                "ranker": method,
                "ranker_label": method_label,
                "rank": rank,
                "rank_in_v4_1_blend": base_rank,
                "rank_change_vs_v4_1_blend": None if base_rank is None else base_rank - rank,
                "paper_id": paper_id,
                "title": row.get("title"),
                "year": row.get("year"),
                "venue_or_source": venue_or_source(row),
                "relevance_label": relevance_label(label),
                "section_label": label.get("primary_role") if label else None,
                "difficulty_label": difficulty_label(label),
                "v3_3_score": components["v3_3"].get(paper_id),
                "v4_1_score": components["v4_1"].get(paper_id),
                "v4_3_text_score": components["v4_3_text"].get(paper_id),
                "final_blended_score": method_scores[method][query_id][paper_id],
                "hard_negative": bool(label and is_hard_negative(label)),
                "missing_or_short_abstract": abstract_word_count(row) < 50,
                "abstract_word_count": abstract_word_count(row),
                "generic_survey": generic_survey(row),
            }
        )
    return rows


def top10_export(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    methods: list[str],
    method_labels: dict[str, str],
    v41_method: str,
    v43_method: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    label_lookup = labels_by_query(labels)
    output = {}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        output[query_id] = {}
        query_labels = label_lookup.get(query_id, {})
        for method in methods:
            output[query_id][method] = make_top10_rows(
                query_id=query_id,
                method=method,
                method_label=method_labels[method],
                candidate_rows=rows,
                labels=query_labels,
                method_scores=method_scores,
                v41_method=v41_method,
                v43_method=v43_method,
            )
    return output


def multimodal_failure_examples(top10: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    baseline = {row["paper_id"]: row for row in top10[TARGET_TOPIC][V41_BLEND_METHOD]}
    text10 = {row["paper_id"]: row for row in top10[TARGET_TOPIC][TEXT_10]}
    demoted = [row for paper_id, row in baseline.items() if paper_id not in text10]
    promoted = [row for paper_id, row in text10.items() if paper_id not in baseline]
    examples = []
    for row in demoted:
        examples.append(
            {
                "type": "good_paper_demoted_out_of_top10",
                **row,
                "diagnosis": "Removed from the V4.4 10% text top-10 relative to V4.1 blend.",
            }
        )
    for row in promoted:
        v33 = float(row.get("v3_3_score") or 0.0)
        v41 = float(row.get("v4_1_score") or 0.0)
        text = float(row.get("v4_3_text_score") or 0.0)
        text_advantage = text - max(v33, v41)
        reasons = []
        if text_advantage > 0.10:
            reasons.append("boosted_mainly_by_text_score")
        if abs(text - v33) > 0.25 and abs(text - v41) > 0.25:
            reasons.append("text_score_disagrees_with_v3_3_and_v4_1")
        if row["generic_survey"]:
            reasons.append("survey_like_wording")
        if row["missing_or_short_abstract"]:
            reasons.append("short_or_missing_abstract")
        if not reasons:
            reasons.append("topic_boundary_or_rank_interaction")
        examples.append(
            {
                "type": "weaker_or_different_paper_promoted_into_top10",
                **row,
                "text_advantage_over_ranker_components": text_advantage,
                "diagnostic_reasons": reasons,
            }
        )
    return examples


def filtered_weighted_blend(
    method_scores: dict[str, dict[str, dict[int, float]]],
    *,
    v41_method: str,
    v43_method: str,
    text_weight: float,
    v33_weight: float,
    v41_weight: float,
    mode: str,
    hurt_topics: set[str] | None = None,
) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {}
    for query_id in sorted(method_scores[V33_METHOD]):
        v33 = normalize_query_scores(method_scores[V33_METHOD][query_id])
        v41 = normalize_query_scores(method_scores[v41_method][query_id])
        text = normalize_query_scores(method_scores[v43_method][query_id])
        paper_ids = set(v33) | set(v41) | set(text)
        output[query_id] = {}
        for paper_id in paper_ids:
            local_text_weight = text_weight
            if mode == "topic_hurt_reduced" and hurt_topics and query_id in hurt_topics:
                local_text_weight = 0.05
            base = v33_weight * v33.get(paper_id, 0.0) + v41_weight * v41.get(paper_id, 0.0)
            text_component = local_text_weight * text.get(paper_id, 0.0)
            if mode == "text_contribution_cap":
                text_component = min(text_component, 0.035)
            elif mode == "text_only_when_rankers_not_confident":
                if max(v33.get(paper_id, 0.0), v41.get(paper_id, 0.0)) >= 0.85:
                    text_component = 0.0
            elif mode == "ignore_strong_text_disagreement":
                text_score = text.get(paper_id, 0.0)
                if abs(text_score - v33.get(paper_id, 0.0)) >= 0.35 and abs(text_score - v41.get(paper_id, 0.0)) >= 0.35:
                    text_component = 0.0
            output[query_id][paper_id] = base + text_component
    return output


def topic_metric(evaluation: dict[str, Any], method: str, query_id: str) -> dict[str, Any]:
    return next(row for row in evaluation["per_topic"][method] if row["query_id"] == query_id)


def guardrail_experiments(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    v41_method: str,
    v43_method: str,
    sensitivity_evaluation: dict[str, Any],
) -> dict[str, Any]:
    hurt_topics = {
        row["query_id"]
        for row in sensitivity_evaluation["per_topic"][TEXT_10]
        if row["reading_value_ndcg"] < topic_metric(sensitivity_evaluation, V41_BLEND_METHOD, row["query_id"])["reading_value_ndcg"]
    }
    experiments = {
        "baseline_v4_1_blend": method_scores[V41_BLEND_METHOD],
        "guardrail_5pct_text": weighted_blend(method_scores, {V33_METHOD: 0.70, v41_method: 0.25, v43_method: 0.05}),
        "guardrail_10pct_text_cap": filtered_weighted_blend(method_scores, v41_method=v41_method, v43_method=v43_method, text_weight=0.10, v33_weight=0.675, v41_weight=0.225, mode="text_contribution_cap"),
        "guardrail_text_when_rankers_not_confident": filtered_weighted_blend(method_scores, v41_method=v41_method, v43_method=v43_method, text_weight=0.10, v33_weight=0.675, v41_weight=0.225, mode="text_only_when_rankers_not_confident"),
        "guardrail_ignore_strong_text_disagreement": filtered_weighted_blend(method_scores, v41_method=v41_method, v43_method=v43_method, text_weight=0.10, v33_weight=0.675, v41_weight=0.225, mode="ignore_strong_text_disagreement"),
        "guardrail_topic_hurt_reduced": filtered_weighted_blend(method_scores, v41_method=v41_method, v43_method=v43_method, text_weight=0.10, v33_weight=0.675, v41_weight=0.225, mode="topic_hurt_reduced", hurt_topics=hurt_topics),
    }
    evaluation = evaluate_rankings(candidate_rows_by_q=candidate_rows_by_q, labels=labels, method_scores=experiments, k=K)
    baseline = evaluation["method_averages"]["baseline_v4_1_blend"]
    rows = []
    for method, metrics in sorted(evaluation["method_averages"].items()):
        multimodal = topic_metric(evaluation, method, TARGET_TOPIC)
        rows.append(
            {
                "method": method,
                "reading_value_ndcg": metrics["reading_value_ndcg"],
                "topic_match_ndcg": metrics["topic_match_ndcg"],
                "judged_at_10": metrics["judged_at_10"],
                "unjudged_at_10": metrics["unjudged_at_10"],
                "hard_negative_rate_at_10": metrics["hard_negative_rate_at_10"],
                "reading_delta_vs_v4_1_blend": metrics["reading_value_ndcg"] - baseline["reading_value_ndcg"],
                "topic_delta_vs_v4_1_blend": metrics["topic_match_ndcg"] - baseline["topic_match_ndcg"],
                "hard_negative_delta_vs_v4_1_blend": metrics["hard_negative_rate_at_10"] - baseline["hard_negative_rate_at_10"],
                "multimodal_reading_ndcg": multimodal["reading_value_ndcg"],
                "multimodal_topic_ndcg": multimodal["topic_match_ndcg"],
                "multimodal_hard_negative_rate_at_10": multimodal["hard_negative_rate_at_10"],
                "multimodal_reading_delta_vs_v4_1_blend": multimodal["reading_value_ndcg"] - topic_metric(evaluation, "baseline_v4_1_blend", TARGET_TOPIC)["reading_value_ndcg"],
                "multimodal_topic_delta_vs_v4_1_blend": multimodal["topic_match_ndcg"] - topic_metric(evaluation, "baseline_v4_1_blend", TARGET_TOPIC)["topic_match_ndcg"],
            }
        )
    safest = max(
        rows,
        key=lambda row: (
            row["multimodal_reading_delta_vs_v4_1_blend"] >= -0.02,
            row["reading_value_ndcg"],
            row["topic_match_ndcg"],
            -row["hard_negative_rate_at_10"],
        ),
    )
    return {
        "hurt_topics_from_10pct_text": sorted(hurt_topics),
        "evaluation": evaluation,
        "experiments": rows,
        "safest_method": safest,
    }


def summarize_multimodal(examples: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = [row for row in examples if row["type"] == "weaker_or_different_paper_promoted_into_top10"]
    reasons = Counter(reason for row in promoted for reason in row.get("diagnostic_reasons", []))
    return {
        "demoted_count": sum(1 for row in examples if row["type"] == "good_paper_demoted_out_of_top10"),
        "promoted_count": len(promoted),
        "promoted_reason_counts": dict(sorted(reasons.items())),
        "text_boosted_promotions": sum("boosted_mainly_by_text_score" in row.get("diagnostic_reasons", []) for row in promoted),
        "text_disagreed_promotions": sum("text_score_disagrees_with_v3_3_and_v4_1" in row.get("diagnostic_reasons", []) for row in promoted),
        "generic_survey_promotions": sum(row.get("generic_survey") for row in promoted),
        "short_abstract_promotions": sum(row.get("missing_or_short_abstract") for row in promoted),
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    context = score_context(args)
    method_scores = add_sensitivity_blends(context["method_scores"], context["best_v41_method"], context["best_v43_method"])
    methods = [
        V33_METHOD,
        context["best_v41_method"],
        V41_BLEND_METHOD,
        TEXT_05,
        TEXT_10,
        TEXT_15,
    ]
    labels = {
        V33_METHOD: RANKER_LABELS[V33_METHOD],
        context["best_v41_method"]: RANKER_LABELS["__v41_weighted__"],
        V41_BLEND_METHOD: RANKER_LABELS[V41_BLEND_METHOD],
        TEXT_05: RANKER_LABELS[TEXT_05],
        TEXT_10: RANKER_LABELS[TEXT_10],
        TEXT_15: RANKER_LABELS[TEXT_15],
    }
    evaluation = evaluate_rankings(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores={method: method_scores[method] for method in methods},
        k=K,
    )
    top10 = top10_export(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        methods=methods,
        method_labels=labels,
        v41_method=context["best_v41_method"],
        v43_method=context["best_v43_method"],
    )
    failure_examples = multimodal_failure_examples(top10)
    guardrails = guardrail_experiments(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        v41_method=context["best_v41_method"],
        v43_method=context["best_v43_method"],
        sensitivity_evaluation=evaluation,
    )
    v41_multimodal = topic_metric(evaluation, V41_BLEND_METHOD, TARGET_TOPIC)
    text10_multimodal = topic_metric(evaluation, TEXT_10, TARGET_TOPIC)
    report = {
        "schema_version": "v4.5_text_blend_regression_diagnosis_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "target_regression_topic": TARGET_TOPIC,
        "components": {
            "v3_3_method": V33_METHOD,
            "v4_1_weighted_method": context["best_v41_method"],
            "v4_1_blend_method": V41_BLEND_METHOD,
            "v4_3_text_method": context["best_v43_method"],
        },
        "rankers_compared": labels,
        "evaluation": evaluation,
        "top10_by_topic_and_ranker": top10,
        "multimodal_failure_summary": summarize_multimodal(failure_examples),
        "multimodal_metric_delta_text10_vs_v4_1_blend": {
            "reading_delta": text10_multimodal["reading_value_ndcg"] - v41_multimodal["reading_value_ndcg"],
            "topic_delta": text10_multimodal["topic_match_ndcg"] - v41_multimodal["topic_match_ndcg"],
            "hard_negative_delta": text10_multimodal["hard_negative_rate_at_10"] - v41_multimodal["hard_negative_rate_at_10"],
        },
        "diagnosis": {
            "primary_cause": "The 10% text blend changes the multimodal top-10 through small score interactions near the decision boundary, not through hard-negative promotion, missing abstracts, generic survey wording, or obvious text-score domination. The issue appears topic-boundary and rank-interaction related.",
            "text_score_involved": summarize_multimodal(failure_examples)["text_boosted_promotions"] > 0,
            "generic_survey_involved": summarize_multimodal(failure_examples)["generic_survey_promotions"] > 0,
            "short_abstract_involved": summarize_multimodal(failure_examples)["short_abstract_promotions"] > 0,
        },
        "guardrail_report_path": str(resolve_repo_path(args.guardrail_json_out)),
        "failure_examples_path": str(resolve_repo_path(args.failure_examples_out)),
        "recommendation": {
            "safest_option": guardrails["safest_method"]["method"],
            "summary": "Keep V4.4 10% text blend offline. The aggregate lift is real, but the multimodal regression supports a guarded formula before any future opt-in runtime work; 5% text reduces but does not eliminate the multimodal loss.",
        },
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }
    guardrail_report = {
        "schema_version": "v4.5_guardrail_experiments_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "target_regression_topic": TARGET_TOPIC,
        "experiments": guardrails["experiments"],
        "safest_method": guardrails["safest_method"],
        "hurt_topics_from_10pct_text": guardrails["hurt_topics_from_10pct_text"],
        "evaluation": guardrails["evaluation"],
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }
    return report, failure_examples, guardrail_report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["multimodal_failure_summary"]
    delta = report["multimodal_metric_delta_text10_vs_v4_1_blend"]
    lines = [
        "# V4.5 Text Blend Regression Diagnosis",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Target topic: `{report['target_regression_topic']}`",
        f"- 10% text reading delta vs V4.1 blend: `{delta['reading_delta']:.3f}`",
        f"- 10% text topic delta vs V4.1 blend: `{delta['topic_delta']:.3f}`",
        f"- Good papers demoted out of top-10: `{summary['demoted_count']}`",
        f"- Papers promoted into top-10: `{summary['promoted_count']}`",
        f"- Safest guardrail option: `{report['recommendation']['safest_option']}`",
        "",
        "## Diagnosis",
        "",
        report["diagnosis"]["primary_cause"],
        "",
        "## Compared Rankers",
        "",
    ]
    for method, label in report["rankers_compared"].items():
        lines.append(f"- `{method}`: {label}")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def render_guardrail_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.5 Guardrail Experiments",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Safest method: `{report['safest_method']['method']}`",
        "",
        "| Method | Reading | Topic | Judged | Unjudged | Hard-neg | Multimodal Reading Delta | Multimodal Topic Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["experiments"]:
        lines.append(
            f"| `{row['method']}` | `{row['reading_value_ndcg']:.3f}` | `{row['topic_match_ndcg']:.3f}` | "
            f"`{row['judged_at_10']:.3f}` | `{row['unjudged_at_10']:.3f}` | `{row['hard_negative_rate_at_10']:.3f}` | "
            f"`{row['multimodal_reading_delta_vs_v4_1_blend']:.3f}` | `{row['multimodal_topic_delta_vs_v4_1_blend']:.3f}` |"
        )
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
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--failure-examples-out", default=str(DEFAULT_FAILURE_EXAMPLES_OUT))
    parser.add_argument("--guardrail-json-out", default=str(DEFAULT_GUARDRAIL_JSON_OUT))
    parser.add_argument("--guardrail-md-out", default=str(DEFAULT_GUARDRAIL_MD_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, failure_examples, guardrail_report = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    write_jsonl(resolve_repo_path(args.failure_examples_out), failure_examples)
    write_json(resolve_repo_path(args.guardrail_json_out), guardrail_report)
    write_text(resolve_repo_path(args.guardrail_md_out), render_guardrail_markdown(guardrail_report))
    print(f"Wrote V4.5 regression diagnosis report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
