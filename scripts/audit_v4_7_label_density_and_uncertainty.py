import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
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
    TEXT_05,
    TEXT_10,
    abstract_word_count,
    filtered_weighted_blend,
    generic_survey,
    venue_or_source,
)
from evaluate_v2_4_true_production_candidate_pool import ndcg_at_k, score_map  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
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
from evaluate_v4_1_calibrated_blends import DEFAULT_WEIGHTED_REPORT, DEFAULT_V41_MODEL_DIR, normalize_query_scores  # noqa: E402
from evaluate_v4_3_text_reranker_production_pool import DEFAULT_V43_MODEL_DIR, V41_BLEND_METHOD  # noqa: E402
from evaluate_v4_4_text_blend_sensitivity import (  # noqa: E402
    DEFAULT_V43_TRAINING_REPORT,
    add_sensitivity_blends,
    is_hard_negative,
    is_positive,
    score_context,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402
from validate_v4_6_guarded_text_blend import GUARDRAIL_METHOD  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_7_label_density_uncertainty_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_7_label_density_uncertainty_report.md"
DEFAULT_BOUNDARY_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_7_contrastive_learning_boundary_cases.jsonl"
DEFAULT_CANDIDATES_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_7_targeted_labeling_candidates.jsonl"

CONTRASTIVE_TOPIC = "v2_contrastive_learning"
MULTIMODAL_TOPIC = "v2_multimodal_learning"
METHODS = [V33_METHOD, V41_BLEND_METHOD, TEXT_05, TEXT_10, GUARDRAIL_METHOD]
METHOD_LABELS = {
    V33_METHOD: "V3.3 RF no-V2.7",
    V41_BLEND_METHOD: "V4.1 calibrated blend",
    TEXT_05: "V4.4 5% text blend",
    TEXT_10: "V4.4 10% text blend",
    GUARDRAIL_METHOD: "V4.6 guarded text blend",
}


def relevance_bucket(label: dict[str, Any]) -> int:
    if is_positive(label):
        return 2
    if float(label.get("reading_value_score") or 0.0) >= 0.5 or float(label.get("topic_match_score") or 0.0) >= 0.5:
        return 1
    return 0


def difficulty_bucket(label: dict[str, Any]) -> str:
    levels = {
        "beginner": float(label.get("beginner_fit_score") or 0.0),
        "intermediate": float(label.get("intermediate_fit_score") or 0.0),
        "advanced": float(label.get("advanced_fit_score") or 0.0),
        "expert": float(label.get("expert_fit_score") or 0.0),
    }
    best = max(levels.values())
    if best <= 0:
        return "none"
    return "+".join(level for level, score in levels.items() if score == best)


def rank_lookup(scores: dict[int, float]) -> dict[int, int]:
    return {paper_id: rank for rank, paper_id in enumerate(rank_from_scores(scores), start=1)}


def normalized_components(method_scores: dict[str, dict[str, dict[int, float]]], query_id: str, v43_method: str) -> dict[str, dict[int, float]]:
    return {
        "v3_3": normalize_query_scores(method_scores[V33_METHOD][query_id]),
        "v4_1": normalize_query_scores(method_scores[V41_BLEND_METHOD][query_id]),
        "v4_3_text": normalize_query_scores(method_scores[v43_method][query_id]),
        "v4_6": normalize_query_scores(method_scores[GUARDRAIL_METHOD][query_id]),
    }


def label_density_by_topic(
    labels: list[dict[str, Any]],
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    method_scores: dict[str, dict[str, dict[int, float]]],
) -> dict[str, dict[str, Any]]:
    lookup = labels_by_query(labels)
    judged_counts = {query_id: len(rows) for query_id, rows in lookup.items()}
    median_judged = sorted(judged_counts.values())[len(judged_counts) // 2]
    output = {}
    for query_id, query_labels in sorted(lookup.items()):
        relevance = Counter(relevance_bucket(label) for label in query_labels.values())
        section = Counter(str(label.get("primary_role")) for label in query_labels.values())
        difficulty = Counter(difficulty_bucket(label) for label in query_labels.values())
        hard_negative_count = sum(1 for label in query_labels.values() if is_hard_negative(label))
        judged_top10 = {}
        unjudged_top10 = {}
        strong_positive_near = {}
        for method in METHODS:
            ranked = rank_from_scores(method_scores[method][query_id])
            top10 = ranked[:K]
            top20 = ranked[:20]
            top50 = ranked[:50]
            judged_top10[method] = sum(1 for paper_id in top10 if paper_id in query_labels)
            unjudged_top10[method] = sum(1 for paper_id in top10 if paper_id not in query_labels)
            strong_positive_near[method] = {
                "top20": sum(1 for paper_id in top20 if paper_id in query_labels and relevance_bucket(query_labels[paper_id]) == 2),
                "top50": sum(1 for paper_id in top50 if paper_id in query_labels and relevance_bucket(query_labels[paper_id]) == 2),
            }
        output[query_id] = {
            "total_judged_papers": len(query_labels),
            "relevance_2_count": relevance[2],
            "relevance_1_count": relevance[1],
            "relevance_0_count": relevance[0],
            "hard_negative_count": hard_negative_count,
            "section_label_distribution": dict(sorted(section.items())),
            "difficulty_label_distribution": dict(sorted(difficulty.items())),
            "judged_top10_count_by_method": judged_top10,
            "unjudged_top10_count_by_method": unjudged_top10,
            "strong_positives_available_near_top": strong_positive_near,
            "label_sparse_compared_with_topics": len(query_labels) < median_judged,
            "candidate_pool_size": len(candidate_rows_by_q[query_id]),
        }
    return output


def score_gap_around_rank(scores: dict[int, float], rank: int = 10) -> float | None:
    ranked = rank_from_scores(scores)
    if len(ranked) <= rank:
        return None
    return abs(scores[ranked[rank - 1]] - scores[ranked[rank]])


def topic_ndcg(labels: dict[int, dict[str, Any]], ranked: list[int], field: str, k: int = K) -> float:
    return ndcg_at_k(ranked, score_map(labels, field), k)


def ndcg_swap_sensitivity(labels: dict[int, dict[str, Any]], baseline_top: list[int], method_top: list[int]) -> dict[str, Any]:
    reading_delta = topic_ndcg(labels, method_top, "reading_value_score") - topic_ndcg(labels, baseline_top, "reading_value_score")
    topic_delta = topic_ndcg(labels, method_top, "topic_match_score") - topic_ndcg(labels, baseline_top, "topic_match_score")
    method_only = [paper_id for paper_id in method_top[:K] if paper_id not in baseline_top[:K]]
    baseline_only = [paper_id for paper_id in baseline_top[:K] if paper_id not in method_top[:K]]
    return {
        "reading_delta": reading_delta,
        "topic_delta": topic_delta,
        "top10_swap_count": len(method_only),
        "method_only_top10": method_only,
        "baseline_only_top10": baseline_only,
        "caused_by_one_or_two_swaps": len(method_only) <= 2,
    }


def perturbation_instability(scores: dict[int, float], *, epsilon: float = 0.005) -> dict[str, Any]:
    ranked = rank_from_scores(scores)
    if len(ranked) < 15:
        window = ranked[7:15]
    else:
        window = ranked[7:15]
    gaps = [abs(scores[ranked[index]] - scores[ranked[index + 1]]) for index in range(7, min(14, len(ranked) - 1))]
    boundary = set(ranked[:K])
    unstable = [paper_id for paper_id in window if abs(scores[paper_id] - scores[ranked[K - 1]]) <= epsilon]
    return {
        "epsilon": epsilon,
        "rank_8_15_paper_ids": window,
        "min_adjacent_gap_rank_8_15": min(gaps) if gaps else None,
        "rank10_gap": score_gap_around_rank(scores, K),
        "papers_within_epsilon_of_rank10": unstable,
        "small_perturbation_could_change_top10": any(paper_id not in boundary for paper_id in unstable) or (score_gap_around_rank(scores, K) is not None and score_gap_around_rank(scores, K) <= epsilon),
    }


def top10_jaccard(method_scores: dict[str, dict[str, dict[int, float]]], query_id: str, left: str, right: str) -> float:
    left_top = set(rank_from_scores(method_scores[left][query_id])[:K])
    right_top = set(rank_from_scores(method_scores[right][query_id])[:K])
    return len(left_top & right_top) / len(left_top | right_top) if left_top or right_top else 1.0


def bootstrap_topic_delta(labels: dict[int, dict[str, Any]], method_top: list[int], baseline_top: list[int], *, iterations: int = 1000) -> dict[str, Any]:
    judged_ids = sorted(set(labels) & (set(method_top[:50]) | set(baseline_top[:50])))
    if not judged_ids:
        return {"iterations": iterations, "reading_delta_p05": 0.0, "reading_delta_p95": 0.0, "topic_delta_p05": 0.0, "topic_delta_p95": 0.0}
    rng = random.Random(47)
    reading = []
    topic = []
    for _ in range(iterations):
        sampled_ids = [rng.choice(judged_ids) for _ in judged_ids]
        sampled = {paper_id: labels[paper_id] for paper_id in sampled_ids}
        reading.append(topic_ndcg(sampled, method_top, "reading_value_score") - topic_ndcg(sampled, baseline_top, "reading_value_score"))
        topic.append(topic_ndcg(sampled, method_top, "topic_match_score") - topic_ndcg(sampled, baseline_top, "topic_match_score"))
    return {
        "iterations": iterations,
        "reading_delta_mean": sum(reading) / len(reading),
        "reading_delta_p05": percentile(reading, 0.05),
        "reading_delta_p95": percentile(reading, 0.95),
        "topic_delta_mean": sum(topic) / len(topic),
        "topic_delta_p05": percentile(topic, 0.05),
        "topic_delta_p95": percentile(topic, 0.95),
    }


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def confidence_interval_spans_zero(interval: dict[str, Any]) -> bool:
    return (
        float(interval["reading_delta_p05"]) <= 0.0 <= float(interval["reading_delta_p95"])
        or float(interval["topic_delta_p05"]) <= 0.0 <= float(interval["topic_delta_p95"])
    )


def boundary_row(
    *,
    query_id: str,
    paper_id: int,
    paper: dict[str, Any],
    label: dict[str, Any] | None,
    components: dict[str, dict[int, float]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    ranks: dict[str, dict[int, int]],
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "paper_id": paper_id,
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "abstract_word_count": abstract_word_count(paper),
        "year": paper.get("year"),
        "venue_or_source": venue_or_source(paper),
        "relevance_label": None if not label else {"reading_value_score": label.get("reading_value_score"), "topic_match_score": label.get("topic_match_score"), "bucket": relevance_bucket(label)},
        "section_label": label.get("primary_role") if label else None,
        "difficulty_label": difficulty_bucket(label) if label else None,
        "hard_negative": bool(label and is_hard_negative(label)),
        "generic_survey": generic_survey(paper),
        "short_or_missing_abstract": abstract_word_count(paper) < 50,
        "v3_3_score": components["v3_3"].get(paper_id),
        "v4_1_score": components["v4_1"].get(paper_id),
        "v4_3_text_score": components["v4_3_text"].get(paper_id),
        "v4_6_final_score": method_scores[GUARDRAIL_METHOD][query_id].get(paper_id),
        "v4_1_rank": ranks[V41_BLEND_METHOD].get(paper_id),
        "v4_6_rank": ranks[GUARDRAIL_METHOD].get(paper_id),
        "rank_delta_vs_v4_1_blend": None if paper_id not in ranks[V41_BLEND_METHOD] or paper_id not in ranks[GUARDRAIL_METHOD] else ranks[V41_BLEND_METHOD][paper_id] - ranks[GUARDRAIL_METHOD][paper_id],
        "score_margin_around_rank10_v4_6": score_gap_around_rank(method_scores[GUARDRAIL_METHOD][query_id], K),
    }


def contrastive_boundary_cases(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    v43_method: str,
) -> list[dict[str, Any]]:
    query_id = CONTRASTIVE_TOPIC
    rows = candidate_rows_by_q[query_id]
    row_by_id = {int(row["paper_id"]): row for row in rows}
    query_labels = labels_by_query(labels).get(query_id, {})
    components = normalized_components(method_scores, query_id, v43_method)
    ranks = {method: rank_lookup(method_scores[method][query_id]) for method in METHODS}
    v41_top20 = rank_from_scores(method_scores[V41_BLEND_METHOD][query_id])[:20]
    v46_top20 = rank_from_scores(method_scores[GUARDRAIL_METHOD][query_id])[:20]
    swapped = sorted((set(v41_top20[:K]) ^ set(v46_top20[:K])) | set(v41_top20) | set(v46_top20))
    return [
        boundary_row(
            query_id=query_id,
            paper_id=paper_id,
            paper=row_by_id[paper_id],
            label=query_labels.get(paper_id),
            components=components,
            method_scores=method_scores,
            ranks=ranks,
        )
        for paper_id in swapped
    ]


def targeted_label_candidates(boundary_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = {}
    for row in boundary_cases:
        reasons = []
        v41_rank = row.get("v4_1_rank") or 999999
        v46_rank = row.get("v4_6_rank") or 999999
        if 8 <= min(v41_rank, v46_rank) <= 30:
            reasons.append("near_top10_boundary_rank_8_30")
        if row.get("relevance_label") is None:
            reasons.append("unjudged_boundary_candidate")
        elif row["relevance_label"]["bucket"] <= 1 and v46_rank <= 10:
            reasons.append("v4_6_promoted_weak_or_partial_candidate")
        if v41_rank <= 10 and v46_rank > 10:
            reasons.append("demoted_from_v4_1_top10")
        v33 = row.get("v3_3_score") or 0.0
        v41 = row.get("v4_1_score") or 0.0
        text = row.get("v4_3_text_score") or 0.0
        if abs(text - v33) >= 0.25 and abs(text - v41) >= 0.25:
            reasons.append("text_disagrees_with_ranker_scores")
        if (row.get("score_margin_around_rank10_v4_6") or 1.0) <= 0.01:
            reasons.append("small_rank10_score_gap")
        if reasons:
            candidate = dict(row)
            candidate["reason_selected"] = sorted(set(reasons))
            candidates[int(row["paper_id"])] = candidate
    return sorted(candidates.values(), key=lambda item: (min(item.get("v4_1_rank") or 999999, item.get("v4_6_rank") or 999999), int(item["paper_id"])))


def instability_report(
    *,
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
) -> dict[str, Any]:
    query_labels = labels_by_query(labels).get(CONTRASTIVE_TOPIC, {})
    v41_ranked = rank_from_scores(method_scores[V41_BLEND_METHOD][CONTRASTIVE_TOPIC])
    v46_ranked = rank_from_scores(method_scores[GUARDRAIL_METHOD][CONTRASTIVE_TOPIC])
    return {
        "rank_boundary_v4_1": perturbation_instability(method_scores[V41_BLEND_METHOD][CONTRASTIVE_TOPIC]),
        "rank_boundary_v4_6": perturbation_instability(method_scores[GUARDRAIL_METHOD][CONTRASTIVE_TOPIC]),
        "top10_jaccard_v4_6_vs_v4_1": top10_jaccard(method_scores, CONTRASTIVE_TOPIC, GUARDRAIL_METHOD, V41_BLEND_METHOD),
        "top10_jaccard_v4_6_vs_v3_3": top10_jaccard(method_scores, CONTRASTIVE_TOPIC, GUARDRAIL_METHOD, V33_METHOD),
        "ndcg_swap_sensitivity_v4_6_vs_v4_1": ndcg_swap_sensitivity(query_labels, v41_ranked, v46_ranked),
        "bootstrap_topic_delta_v4_6_vs_v4_1": bootstrap_topic_delta(query_labels, v46_ranked, v41_ranked),
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    density = label_density_by_topic(context["labels"], context["candidate_rows_by_q"], method_scores)
    boundary = contrastive_boundary_cases(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        v43_method=context["best_v43_method"],
    )
    candidates = targeted_label_candidates(boundary)
    instability = instability_report(labels=context["labels"], method_scores=method_scores)
    protected = protected_status(resolve_repo_path(args.v39_labels))
    contrastive_sparse = density[CONTRASTIVE_TOPIC]["label_sparse_compared_with_topics"]
    uncertain = (
        instability["rank_boundary_v4_6"]["small_perturbation_could_change_top10"]
        or instability["rank_boundary_v4_1"]["small_perturbation_could_change_top10"]
        or instability["ndcg_swap_sensitivity_v4_6_vs_v4_1"]["caused_by_one_or_two_swaps"]
        or confidence_interval_spans_zero(instability["bootstrap_topic_delta_v4_6_vs_v4_1"])
        or density[CONTRASTIVE_TOPIC]["unjudged_top10_count_by_method"][GUARDRAIL_METHOD] > 0
    )
    report = {
        "schema_version": "v4.7_label_density_uncertainty_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "methods_compared": METHOD_LABELS,
        "focus_topics": [CONTRASTIVE_TOPIC, MULTIMODAL_TOPIC],
        "label_density_by_topic": density,
        "contrastive_learning_boundary_diagnostics": {
            "instability": instability,
            "boundary_case_count": len(boundary),
            "targeted_label_candidate_count": len(candidates),
            "appears_label_sparse": contrastive_sparse,
            "failure_appears_uncertain": uncertain,
            "additional_labels_recommended": uncertain or contrastive_sparse,
        },
        "multimodal_learning_density": density[MULTIMODAL_TOPIC],
        "protected_hashes": protected,
    }
    return report, boundary, candidates


def render_markdown(report: dict[str, Any]) -> str:
    contrastive = report["contrastive_learning_boundary_diagnostics"]
    density = report["label_density_by_topic"][CONTRASTIVE_TOPIC]
    instability = contrastive["instability"]
    lines = [
        "# V4.7 Label Density and Uncertainty Audit",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Focus topic: `{CONTRASTIVE_TOPIC}`",
        f"- Judged papers: `{density['total_judged_papers']}`",
        f"- Strong positives: `{density['relevance_2_count']}`",
        f"- Label-sparse compared with topics: `{contrastive['appears_label_sparse']}`",
        f"- Failure appears uncertain: `{contrastive['failure_appears_uncertain']}`",
        f"- Additional labels recommended: `{contrastive['additional_labels_recommended']}`",
        f"- Targeted label candidates: `{contrastive['targeted_label_candidate_count']}`",
        f"- V4.6/V4.1 top-10 Jaccard: `{instability['top10_jaccard_v4_6_vs_v4_1']:.3f}`",
        f"- Swap count driving NDCG delta: `{instability['ndcg_swap_sensitivity_v4_6_vs_v4_1']['top10_swap_count']}`",
        "",
        "## Protected Hashes",
        "",
    ]
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
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--boundary-out", default=str(DEFAULT_BOUNDARY_OUT))
    parser.add_argument("--candidates-out", default=str(DEFAULT_CANDIDATES_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, boundary, candidates = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    write_jsonl(resolve_repo_path(args.boundary_out), boundary)
    write_jsonl(resolve_repo_path(args.candidates_out), candidates)
    print(f"Wrote V4.7 label-density uncertainty report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
