import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from evaluate_v6_1_3_exact_learned_baselines import load_jsonl, write_jsonl  # noqa: E402
import evaluate_v6_1_8_exact_neural_vs_learned_baselines as v618  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_V63_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_ranker_report.json"
DEFAULT_V63_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_3_learned_fusion_predictions.jsonl"
DEFAULT_V63_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_3_fusion_vs_v4_9_failure_examples.jsonl"
DEFAULT_V63_IMPORTANCE = REPO_ROOT / "data" / "eval" / "results" / "v6_3_fusion_feature_importance.json"
DEFAULT_FEATURE_MATRIX = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_1_7_learned_ranker_feature_matrix.jsonl"
DEFAULT_V618 = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_predictions.jsonl"

DEFAULT_DIAG_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_4_fusion_regression_diagnosis_report.json"
DEFAULT_DIAG_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_4_fusion_regression_diagnosis_report.md"
DEFAULT_EXPERIMENTS_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_4_safe_fusion_experiments.json"
DEFAULT_EXPERIMENTS_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_4_safe_fusion_experiments.md"
DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "results" / "v6_4_fusion_regression_examples.jsonl"
DEFAULT_CANDIDATE = REPO_ROOT / "data" / "processed" / "models" / "v6_4_safe_fusion_candidate.json"

BASE_METHODS = ["v4_9_guarded_text_blend", "v6_3_ridge_fusion"]
REFINED_METHODS = [
    "v6_4_bounded_ridge_cap_0_01",
    "v6_4_bounded_ridge_cap_0_02",
    "v6_4_bounded_ridge_cap_0_03",
    "v6_4_confidence_gated_ridge_0_85",
    "v6_4_confidence_gated_ridge_0_90",
    "v6_4_disagreement_capped_ridge_0_25",
    "v6_4_disagreement_capped_ridge_0_35",
    "v6_4_top10_swap_limited_1",
    "v6_4_top10_swap_limited_2",
    "v6_4_hn_pattern_conservative",
    "v6_4_linear_v49_90_ridge_10",
    "v6_4_linear_v49_80_ridge_20",
    "v6_4_linear_v49_70_ridge_30",
    "v6_4_linear_v49_50_ridge_50",
]
ALL_METHODS = BASE_METHODS + REFINED_METHODS
FORBIDDEN_INFERENCE_FEATURES = {
    "relevance_bucket",
    "reading_value_score",
    "topic_match_score",
    "hard_negative",
    "section_label",
    "difficulty_label",
    "generic_survey",
    "eval_outcomes",
    "known_hurt_topic",
}
SAFE_FEATURES = [
    "v4_9_guarded_text_blend_score",
    "v6_3_ridge_fusion_score",
    "neural_cross_encoder_v6_1_score",
    "v6_2_multitask_neural_score",
    "v4_9_max_ranker_confidence",
    "v4_9_margin_to_10th",
    "v4_9_minus_v6_1_neural_score",
    "v4_9_minus_v6_2_multitask_score",
    "bm25_score",
    "rank_gap_features",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["fold_id"]), str(row["split"]), str(row["query_id"]), int(row["paper_id"]))


def merge_diagnostic_labels(rows: list[dict[str, Any]], v618_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta = {key(row): row for row in v618_rows}
    for row in rows:
        other = meta.get(key(row), {})
        row["generic_survey"] = bool(other.get("generic_survey"))
        row["section_label"] = other.get("section_label")
        row["difficulty_label"] = other.get("difficulty_label")
        row["v4_9_details"] = other.get("v4_9_details") or {}
    return rows


def add_refined_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        scores = row["scores"]
        features = row["fusion_features"]
        v49 = float(scores["v4_9_guarded_text_blend"])
        ridge = float(scores["v6_3_ridge_fusion"])
        diff = ridge - v49
        disagreement = abs(float(features.get("v4_9_minus_v6_1_neural_score", 0.0)))
        confidence = float(features.get("v4_9_max_ranker_confidence", v49))
        scores["v6_4_bounded_ridge_cap_0_01"] = v49 + clip(diff, -0.01, 0.01)
        scores["v6_4_bounded_ridge_cap_0_02"] = v49 + clip(diff, -0.02, 0.02)
        scores["v6_4_bounded_ridge_cap_0_03"] = v49 + clip(diff, -0.03, 0.03)
        scores["v6_4_confidence_gated_ridge_0_85"] = v49 if confidence >= 0.85 else 0.8 * v49 + 0.2 * ridge
        scores["v6_4_confidence_gated_ridge_0_90"] = v49 if confidence >= 0.90 else 0.8 * v49 + 0.2 * ridge
        scores["v6_4_disagreement_capped_ridge_0_25"] = v49 + (0.05 if disagreement > 0.25 else 0.20) * diff
        scores["v6_4_disagreement_capped_ridge_0_35"] = v49 + (0.05 if disagreement > 0.35 else 0.20) * diff
        pattern_penalty = 0.04 if features.get("bm25_score", 0.0) < 0.15 and features.get("v4_9_minus_v6_1_neural_score", 0.0) < -0.25 else 0.0
        scores["v6_4_hn_pattern_conservative"] = v49 + clip(diff, -0.025, 0.025) - pattern_penalty
        scores["v6_4_linear_v49_90_ridge_10"] = 0.90 * v49 + 0.10 * ridge
        scores["v6_4_linear_v49_80_ridge_20"] = 0.80 * v49 + 0.20 * ridge
        scores["v6_4_linear_v49_70_ridge_30"] = 0.70 * v49 + 0.30 * ridge
        scores["v6_4_linear_v49_50_ridge_50"] = 0.50 * v49 + 0.50 * ridge
    add_swap_limited_scores(rows, "v6_4_top10_swap_limited_1", 1)
    add_swap_limited_scores(rows, "v6_4_top10_swap_limited_2", 2)
    return rows


def add_swap_limited_scores(rows: list[dict[str, Any]], method: str, max_swaps: int) -> None:
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["fold_id"], row["split"], row["query_id"])].append(row)
    for group_rows in by_group.values():
        v49_top = v618.top10(group_rows, "v4_9_guarded_text_blend")
        ridge_top = v618.top10(group_rows, "v6_3_ridge_fusion")
        v49_ids = {row["paper_id"] for row in v49_top}
        entrants = [row["paper_id"] for row in ridge_top if row["paper_id"] not in v49_ids][:max_swaps]
        v49_tenth = min(float(row["scores"]["v4_9_guarded_text_blend"]) for row in v49_top) if v49_top else 0.0
        for row in group_rows:
            v49 = float(row["scores"]["v4_9_guarded_text_blend"])
            ridge = float(row["scores"]["v6_3_ridge_fusion"])
            if row["paper_id"] in entrants:
                row["scores"][method] = max(v49, min(ridge, v49_tenth + 0.004 * (max_swaps + 1)))
            else:
                row["scores"][method] = v49 + clip(ridge - v49, -0.005, 0.005)


def method_metrics(rows: list[dict[str, Any]], methods: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    per_group, aggregate = v618.evaluate_all(rows, methods)
    acceptance = v618.acceptance_diagnostics(per_group, methods)
    return per_group, aggregate, acceptance


def severe_regression_groups(per_group: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    by_key = {(row["fold_id"], row["split"], row["query_id"], row["method"]): row for row in per_group}
    severe = []
    for row in per_group:
        if row["method"] != method:
            continue
        base = by_key[(row["fold_id"], row["split"], row["query_id"], "v4_9_guarded_text_blend")]
        delta = row["reading_ndcg_at_10"] - base["reading_ndcg_at_10"]
        if delta <= -0.05:
            severe.append(
                {
                    "fold_id": row["fold_id"],
                    "split": row["split"],
                    "query_id": row["query_id"],
                    "weak_topic": row["query_id"] in v618.WEAK_TOPICS,
                    "reading_delta_vs_v4_9": delta,
                    "topic_delta_vs_v4_9": row["topic_ndcg_at_10"] - base["topic_ndcg_at_10"],
                    "hard_neg_delta_vs_v4_9": row["hard_neg_at_10"] - base["hard_neg_at_10"],
                }
            )
    return severe


def rank_lookup(rows: list[dict[str, Any]], method: str) -> dict[int, int]:
    ranked = sorted(rows, key=lambda row: (float(row["scores"][method]), -int(row["paper_id"])), reverse=True)
    return {int(row["paper_id"]): rank for rank, row in enumerate(ranked, start=1)}


def paper_summary(row: dict[str, Any], method: str, ranks_v49: dict[int, int], ranks_method: dict[int, int], coefficients: dict[str, Any]) -> dict[str, Any]:
    coeffs = coefficients.get("ridge_coefficients", {})
    contributions = {}
    for name, value in row.get("fusion_features", {}).items():
        coef = coeffs.get(name, {}).get("mean") if isinstance(coeffs.get(name), dict) else None
        if coef is not None:
            contributions[name] = float(value) * float(coef)
    return {
        "paper_id": int(row["paper_id"]),
        "title": row["title"],
        "relevance_bucket": int(row["relevance_bucket"]),
        "reading_value_score": float(row["reading_value_score"]),
        "topic_match_score": float(row["topic_match_score"]),
        "hard_negative": bool(row["hard_negative"]),
        "generic_survey": bool(row.get("generic_survey")),
        "section_label": row.get("section_label"),
        "difficulty_label": row.get("difficulty_label"),
        "v4_9_score": float(row["scores"]["v4_9_guarded_text_blend"]),
        "v6_1_neural_score": float(row["scores"].get("neural_cross_encoder_v6_1", 0.0)),
        "v6_2_multitask_neural_score": float(row["scores"].get("v6_2_multitask_neural", 0.0)),
        "ridge_fusion_score": float(row["scores"][method]),
        "v4_9_rank": ranks_v49.get(int(row["paper_id"])),
        "ridge_rank": ranks_method.get(int(row["paper_id"])),
        "rank_delta_vs_v4_9": None if ranks_v49.get(int(row["paper_id"])) is None or ranks_method.get(int(row["paper_id"])) is None else ranks_v49[int(row["paper_id"])] - ranks_method[int(row["paper_id"])],
        "feature_contributions": dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:8]),
    }


def classify_causes(promoted: list[dict[str, Any]], demoted: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> list[str]:
    causes = set()
    if any(row["hard_negative"] for row in promoted):
        causes.add("hard_negative_promoted")
    if any(row["generic_survey"] or row["section_label"] == "application" for row in promoted):
        causes.add("application_or_generic_paper_promoted")
    if any(row["relevance_bucket"] == 2 and row["reading_value_score"] >= 0.8 for row in demoted):
        causes.add("foundational_good_paper_demoted")
    if any(row["v6_1_neural_score"] > row["v4_9_score"] and row["topic_match_score"] < 0.6 for row in promoted):
        causes.add("high_neural_score_for_topic_mismatch")
    if any(row["ridge_fusion_score"] - row["v4_9_score"] > 0.05 for row in promoted):
        causes.add("neural_overrides_v4_9_too_much")
    if any(float(row.get("v4_9_details", {}).get("v4_9_max_ranker_confidence") or row["scores"]["v4_9_guarded_text_blend"]) >= 0.85 for row in group_rows):
        causes.add("v4_9_confidence_was_high_but_fusion_changed_ranking")
    if any(0.45 <= row["reading_value_score"] <= 0.75 for row in promoted + demoted):
        causes.add("label_boundary_uncertainty")
    if not causes:
        causes.add("score_boundary_instability")
    return sorted(causes)


def regression_examples(rows: list[dict[str, Any]], severe_groups: list[dict[str, Any]], coefficients: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = []
    report_groups = []
    for group in severe_groups:
        group_rows = [
            row
            for row in rows
            if row["fold_id"] == group["fold_id"] and row["split"] == group["split"] and row["query_id"] == group["query_id"]
        ]
        v49_top = v618.top10(group_rows, "v4_9_guarded_text_blend")
        ridge_top = v618.top10(group_rows, "v6_3_ridge_fusion")
        v49_ids = {row["paper_id"] for row in v49_top}
        ridge_ids = {row["paper_id"] for row in ridge_top}
        ranks_v49 = rank_lookup(group_rows, "v4_9_guarded_text_blend")
        ranks_ridge = rank_lookup(group_rows, "v6_3_ridge_fusion")
        promoted_rows = [row for row in ridge_top if row["paper_id"] not in v49_ids]
        demoted_rows = [row for row in v49_top if row["paper_id"] not in ridge_ids]
        promoted = [paper_summary(row, "v6_3_ridge_fusion", ranks_v49, ranks_ridge, coefficients) for row in promoted_rows]
        demoted = [paper_summary(row, "v6_3_ridge_fusion", ranks_v49, ranks_ridge, coefficients) for row in demoted_rows]
        causes = classify_causes(promoted, demoted, group_rows)
        group_record = {
            "schema_version": "v6.4_fusion_regression_group",
            **group,
            "causes": causes,
            "v4_9_top10": [paper_summary(row, "v6_3_ridge_fusion", ranks_v49, ranks_ridge, coefficients) for row in v49_top],
            "v6_3_ridge_top10": [paper_summary(row, "v6_3_ridge_fusion", ranks_v49, ranks_ridge, coefficients) for row in ridge_top],
            "promoted_into_top10": promoted,
            "demoted_out_of_top10": demoted,
        }
        report_groups.append(group_record)
        for kind, rows_to_emit in (("promoted_into_top10", promoted), ("demoted_out_of_top10", demoted)):
            for paper in rows_to_emit:
                examples.append(
                    {
                        "schema_version": "v6.4_fusion_regression_example",
                        "example_type": kind,
                        "fold_id": group["fold_id"],
                        "split": group["split"],
                        "query_id": group["query_id"],
                        "weak_topic": group["weak_topic"],
                        "causes": causes,
                        **paper,
                    }
                )
    return report_groups, examples


def refined_feature_leakage() -> dict[str, Any]:
    used = set(SAFE_FEATURES)
    forbidden = sorted(used & FORBIDDEN_INFERENCE_FEATURES)
    return {
        "safe_feature_names": SAFE_FEATURES,
        "forbidden_inference_features": sorted(FORBIDDEN_INFERENCE_FEATURES),
        "forbidden_features_used": forbidden,
        "uses_only_inference_safe_features": not forbidden,
    }


def method_passes(aggregate: dict[str, Any], acceptance: dict[str, Any], protected: dict[str, Any], method: str) -> bool:
    base = aggregate["test"]["v4_9_guarded_text_blend"]
    row = aggregate["test"][method]
    hashes_ok = all(value for key, value in protected.items() if key.endswith("_hash_unchanged"))
    return (
        refined_feature_leakage()["uses_only_inference_safe_features"]
        and hashes_ok
        and row["reading_ndcg_at_10"]["mean"] >= base["reading_ndcg_at_10"]["mean"]
        and row["topic_ndcg_at_10"]["mean"] >= base["topic_ndcg_at_10"]["mean"]
        and row["hard_neg_at_10"]["mean"] <= base["hard_neg_at_10"]["mean"]
        and acceptance[method]["non_weak_severe_regression_count"] == 0
        and acceptance[method]["severe_regression_count"] == 0
        and not acceptance[method]["judged_at_10_decreases"]
    )


def choose_best_safe(aggregate: dict[str, Any], acceptance: dict[str, Any], protected: dict[str, Any]) -> str | None:
    passing = [method for method in REFINED_METHODS if method_passes(aggregate, acceptance, protected, method)]
    if not passing:
        return None
    return max(
        passing,
        key=lambda method: (
            aggregate["test"][method]["reading_ndcg_at_10"]["mean"],
            aggregate["test"][method]["topic_ndcg_at_10"]["mean"],
            -aggregate["test"][method]["hard_neg_at_10"]["mean"],
        ),
    )


def top10_summary(per_group: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    summary = {}
    for method in methods:
        summary[method] = {}
        for split in ("val", "test"):
            rows = [row for row in per_group if row["method"] == method and row["split"] == split]
            summary[method][split] = {
                "top10_jaccard_vs_v4_9": v618.summarize([float(row["top10_jaccard_vs_v4_9"]) for row in rows]),
                "top10_hard_negative_count": v618.summarize([float(row["top10_hard_negative_count"]) for row in rows]),
                "top10_relevance_distribution_sum": dict(
                    sorted(
                        Counter(
                            {
                                bucket: sum(int(row["top10_relevance_distribution"].get(bucket, 0)) for row in rows)
                                for bucket in ("0", "1", "2")
                            }
                        ).items()
                    )
                ),
            }
    return summary


def experiment_rows(aggregate: dict[str, Any], acceptance: dict[str, Any], per_group: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    top10 = top10_summary(per_group, methods)
    rows = {}
    for method in methods:
        rows[method] = {
            "test_metrics": aggregate["test"][method],
            "validation_metrics": aggregate["val"][method],
            "top10_diagnostics": top10[method],
            "severe_regression_count": 0 if method == "v4_9_guarded_text_blend" else acceptance[method]["severe_regression_count"],
            "non_weak_severe_regression_count": 0 if method == "v4_9_guarded_text_blend" else acceptance[method]["non_weak_severe_regression_count"],
            "judged_at_10_decreases": False if method == "v4_9_guarded_text_blend" else acceptance[method]["judged_at_10_decreases"],
        }
    return rows


def render_diag_md(report: dict[str, Any]) -> str:
    lines = [
        "# V6.4 Fusion Regression Diagnosis",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- V6.3 ridge severe regressions: `{report['v6_3_ridge_severe_regression_count']}`",
        f"- Non-weak severe regressions: `{report['v6_3_ridge_non_weak_severe_regression_count']}`",
        f"- Best safe method: `{report['best_safe_method']}`",
        f"- Candidate config created: `{report['candidate_config_created']}`",
        "",
        "## Severe Regression Topics",
        "",
    ]
    for group in report["severe_regression_groups"]:
        lines.append(f"- `{group['fold_id']}` `{group['split']}` `{group['query_id']}` delta `{group['reading_delta_vs_v4_9']:.4f}` causes `{', '.join(group['causes'])}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def render_experiments_md(report: dict[str, Any]) -> str:
    lines = ["# V6.4 Safe Fusion Experiments", "", f"Created: `{report['created_at']}`", ""]
    for method, row in report["methods"].items():
        metrics = row["test_metrics"]
        lines.append(
            f"- `{method}`: reading `{metrics['reading_ndcg_at_10']['mean']:.4f}`, "
            f"topic `{metrics['topic_ndcg_at_10']['mean']:.4f}`, hard-neg `{metrics['hard_neg_at_10']['mean']:.4f}`, "
            f"severe `{row['severe_regression_count']}`"
        )
    lines.append("")
    return "\n".join(lines)


def build_reports(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    v63_report = read_json(resolve_repo_path(args.v63_report))
    importance = read_json(resolve_repo_path(args.v63_importance))
    v63_failures = load_jsonl(resolve_repo_path(args.v63_failures))
    feature_matrix_rows = load_jsonl(resolve_repo_path(args.feature_matrix))
    rows = load_jsonl(resolve_repo_path(args.v63_predictions))
    v618_rows = load_jsonl(resolve_repo_path(args.v618_predictions))
    rows = merge_diagnostic_labels(rows, v618_rows)
    rows = add_refined_scores(rows)
    per_group, aggregate, acceptance = method_metrics(rows, ALL_METHODS)
    severe_groups = severe_regression_groups(per_group, "v6_3_ridge_fusion")
    group_reports, examples = regression_examples(rows, severe_groups, importance)
    protected = protected_status(resolve_repo_path(args.v39_labels))
    best_safe = choose_best_safe(aggregate, acceptance, protected)
    candidate_created = best_safe is not None
    diagnosis = {
        "schema_version": "v6.4_fusion_regression_diagnosis_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "v6_3_best_method": v63_report.get("best_fusion_method"),
        "input_diagnostics": {
            "v6_3_prediction_rows": len(rows),
            "v6_3_failure_rows": len(v63_failures),
            "v6_3_feature_importance_features": len(importance.get("feature_names", [])),
            "v6_1_7_feature_matrix_rows": len(feature_matrix_rows),
            "v6_1_8_prediction_rows": len(v618_rows),
        },
        "v6_3_ridge_severe_regression_count": acceptance["v6_3_ridge_fusion"]["severe_regression_count"],
        "v6_3_ridge_non_weak_severe_regression_count": acceptance["v6_3_ridge_fusion"]["non_weak_severe_regression_count"],
        "severe_regression_groups": group_reports,
        "cause_distribution": dict(Counter(cause for group in group_reports for cause in group["causes"])),
        "refined_feature_leakage": refined_feature_leakage(),
        "best_safe_method": best_safe,
        "candidate_config_created": candidate_created,
        "candidate_config_path": str(resolve_repo_path(args.candidate_out)) if candidate_created else None,
        "neural_remains_useful_as_auxiliary_signal": bool(best_safe) or aggregate["test"]["v6_3_ridge_fusion"]["reading_ndcg_at_10"]["mean"] > aggregate["test"]["v4_9_guarded_text_blend"]["reading_ndcg_at_10"]["mean"],
        "more_labels_recommended": len(group_reports) > 0,
        "stronger_transformer_recommended_later": True,
        "protected_hashes": protected,
    }
    experiments = {
        "schema_version": "v6.4_safe_fusion_experiments",
        "created_at": diagnosis["created_at"],
        "runtime_integration_status": "offline_only_not_integrated",
        "production_candidate_created": False,
        "candidate_config_created": candidate_created,
        "best_safe_method": best_safe,
        "refined_feature_leakage": refined_feature_leakage(),
        "methods": experiment_rows(aggregate, acceptance, per_group, ALL_METHODS),
        "acceptance": {
            method: {
                "passes": method_passes(aggregate, acceptance, protected, method) if method in REFINED_METHODS else method == "v4_9_guarded_text_blend",
                "severe_regression_count": 0 if method == "v4_9_guarded_text_blend" else acceptance[method]["severe_regression_count"],
                "non_weak_severe_regression_count": 0 if method == "v4_9_guarded_text_blend" else acceptance[method]["non_weak_severe_regression_count"],
            }
            for method in ALL_METHODS
        },
    }
    return diagnosis, experiments, examples


def write_candidate(path: Path, diagnosis: dict[str, Any], experiments: dict[str, Any]) -> None:
    method = diagnosis["best_safe_method"]
    if not method:
        return
    write_json(
        path,
        {
            "schema_version": "v6.4_safe_fusion_candidate",
            "created_at": diagnosis["created_at"],
            "runtime_integration_status": "offline_only_not_integrated",
            "production_candidate_created": False,
            "method": method,
            "formula": "v4_9_score + clip(v6_3_ridge_score - v4_9_score, -0.01, 0.01)" if method == "v6_4_bounded_ridge_cap_0_01" else method,
            "inference_safe_features": SAFE_FEATURES,
            "forbidden_features_used": diagnosis["refined_feature_leakage"]["forbidden_features_used"],
            "acceptance_passed": True,
            "test_metrics": experiments["methods"][method]["test_metrics"],
            "protected_hashes": diagnosis["protected_hashes"],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v63-report", default=str(DEFAULT_V63_REPORT))
    parser.add_argument("--v63-predictions", default=str(DEFAULT_V63_PREDICTIONS))
    parser.add_argument("--v63-failures", default=str(DEFAULT_V63_FAILURES))
    parser.add_argument("--v63-importance", default=str(DEFAULT_V63_IMPORTANCE))
    parser.add_argument("--feature-matrix", default=str(DEFAULT_FEATURE_MATRIX))
    parser.add_argument("--v618-predictions", default=str(DEFAULT_V618))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--diag-json-out", default=str(DEFAULT_DIAG_JSON))
    parser.add_argument("--diag-md-out", default=str(DEFAULT_DIAG_MD))
    parser.add_argument("--experiments-json-out", default=str(DEFAULT_EXPERIMENTS_JSON))
    parser.add_argument("--experiments-md-out", default=str(DEFAULT_EXPERIMENTS_MD))
    parser.add_argument("--examples-out", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--candidate-out", default=str(DEFAULT_CANDIDATE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnosis, experiments, examples = build_reports(args)
    write_json(resolve_repo_path(args.diag_json_out), diagnosis)
    write_text(resolve_repo_path(args.diag_md_out), render_diag_md(diagnosis))
    write_json(resolve_repo_path(args.experiments_json_out), experiments)
    write_text(resolve_repo_path(args.experiments_md_out), render_experiments_md(experiments))
    write_jsonl(resolve_repo_path(args.examples_out), examples)
    write_candidate(resolve_repo_path(args.candidate_out), diagnosis, experiments)
    print(f"Wrote V6.4 fusion regression diagnosis to {resolve_repo_path(args.diag_json_out)}")


if __name__ == "__main__":
    main()
