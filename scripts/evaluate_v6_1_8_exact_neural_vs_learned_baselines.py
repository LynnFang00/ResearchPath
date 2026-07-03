import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import statistics
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from evaluate_v6_1_3_exact_learned_baselines import load_jsonl, write_jsonl  # noqa: E402
import train_v6_1_cross_encoder_reranker as v61  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_NEURAL = REPO_ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_predictions.jsonl"
DEFAULT_LEARNED = REPO_ROOT / "data" / "eval" / "results" / "v6_1_7_learned_baseline_predictions.jsonl"
DEFAULT_FEATURE_MATRIX = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_1_7_learned_ranker_feature_matrix.jsonl"

DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_baselines_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_baselines_report.md"
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_exact_neural_vs_learned_predictions.jsonl"
DEFAULT_FAILURES = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_neural_vs_v4_9_failure_examples.jsonl"
DEFAULT_BLEND_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_neural_blend_probe_report.json"
DEFAULT_BLEND_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_8_neural_blend_probe_report.md"

NEURAL_METHOD = "neural_cross_encoder_v6_1"
METHODS = [NEURAL_METHOD, "bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"]
PROBES = [
    "probe_v4_9_90_neural_10",
    "probe_v4_9_80_neural_20",
    "probe_v4_9_neural_hard_negative_guardrail",
    "probe_v4_9_neural_when_v49_confidence_low",
]
ALL_METHODS = METHODS + PROBES
METRICS = ["reading_ndcg_at_10", "topic_ndcg_at_10", "mrr_at_10", "recall_at_10", "hard_neg_at_10", "judged_at_10", "unjudged_at_10"]
WEAK_TOPICS = {
    "v2_ai_for_scientific_discovery",
    "v2_causal_representation_learning",
    "v2_large_language_model_agents",
    "v2_llm_evaluation",
    "v2_retrieval_augmented_generation",
    "v2_transformer_architecture",
}


def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["fold_id"]), str(row["split"]), str(row["query_id"]), int(row["paper_id"]))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if abs(high - low) < 1e-12:
        return {paper_id: 0.5 for paper_id in values}
    return {paper_id: (score - low) / (high - low) for paper_id, score in values.items()}


def feature_matrix_diagnostics(feature_matrix_path: Path) -> dict[str, Any]:
    rows = load_jsonl(feature_matrix_path)
    keys = {key(row) for row in rows}
    return {
        "feature_matrix_rows": len(rows),
        "feature_matrix_unique_keys": len(keys),
        "feature_matrix_duplicate_keys": len(rows) - len(keys),
        "feature_matrix_path": str(feature_matrix_path),
    }


def join_predictions(examples_path: Path, neural_path: Path, learned_path: Path, feature_matrix_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples_by_pair = {(str(row["query_id"]), int(row["paper_id"])): row for row in load_jsonl(examples_path)}
    learned_by_key = {key(row): row for row in load_jsonl(learned_path)}
    feature_keys = {key(row) for row in load_jsonl(feature_matrix_path)}
    rows = []
    missing_learned = []
    missing_feature_matrix = []
    for neural in load_jsonl(neural_path):
        row_key = key(neural)
        learned = learned_by_key.get(row_key)
        if learned is None:
            missing_learned.append(row_key)
            continue
        if row_key not in feature_keys:
            missing_feature_matrix.append(row_key)
        example = examples_by_pair[(str(neural["query_id"]), int(neural["paper_id"]))]
        scores = neural["scores"]
        joined = {
            "schema_version": "v6.1.8_exact_neural_vs_learned_prediction",
            "fold_id": neural["fold_id"],
            "split": neural["split"],
            "query_id": neural["query_id"],
            "query": example["query"],
            "paper_id": int(neural["paper_id"]),
            "title": neural["title"],
            "reading_value_score": float(neural["reading_value_score"]),
            "topic_match_score": float(neural["topic_match_score"]),
            "relevance_bucket": int(neural["relevance_bucket"]),
            "hard_negative": bool(neural["hard_negative"]),
            "generic_survey": bool(example.get("generic_survey")),
            "section_label": example.get("section_label"),
            "difficulty_label": example.get("difficulty_label"),
            "scores": {
                NEURAL_METHOD: float(scores["v6_1_2_cross_encoder"]),
                "bm25": float(scores["bm25"]) if scores.get("bm25") is not None else -1.0,
                "v3_3_ltr": float(learned["v3_3_score"]),
                "v4_1_blend": float(learned["v4_1_score"]),
                "v4_9_guarded_text_blend": float(learned["v4_9_score"]),
            },
            "v4_9_details": learned.get("v4_9_details"),
        }
        rows.append(joined)
    by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["fold_id"], row["split"], row["query_id"])].append(row)
    for group_rows in by_group.values():
        neural_norm = normalize({row["paper_id"]: row["scores"][NEURAL_METHOD] for row in group_rows})
        v49_norm = normalize({row["paper_id"]: row["scores"]["v4_9_guarded_text_blend"] for row in group_rows})
        for row in group_rows:
            neural = neural_norm[row["paper_id"]]
            v49 = v49_norm[row["paper_id"]]
            row["scores"]["probe_v4_9_90_neural_10"] = 0.9 * v49 + 0.1 * neural
            row["scores"]["probe_v4_9_80_neural_20"] = 0.8 * v49 + 0.2 * neural
            row["scores"]["probe_v4_9_neural_hard_negative_guardrail"] = v49 - (0.15 * neural if neural < 0.30 else 0.0)
            confidence = float((row.get("v4_9_details") or {}).get("v4_9_max_ranker_confidence") or v49)
            row["scores"]["probe_v4_9_neural_when_v49_confidence_low"] = v49 if confidence >= 0.85 else 0.85 * v49 + 0.15 * neural
    rows.sort(key=key)
    diagnostics = {
        "neural_rows": sum(1 for _ in load_jsonl(neural_path)),
        "learned_rows": len(learned_by_key),
        **feature_matrix_diagnostics(feature_matrix_path),
        "joined_rows": len(rows),
        "missing_learned_rows": len(missing_learned),
        "missing_feature_matrix_rows": len(missing_feature_matrix),
        "rows_silently_dropped": 0,
    }
    return rows, diagnostics


def score_field(method: str) -> str:
    return f"score__{method}"


def rows_with_score(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = dict(row)
        copied[score_field(method)] = float(row["scores"][method])
        output.append(copied)
    return output


def top10(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (float(row["scores"][method]), -int(row["paper_id"])), reverse=True)[:10]


def top10_diagnostics(rows: list[dict[str, Any]], method: str, v49_top10_ids: set[int]) -> dict[str, Any]:
    ranked = top10(rows, method)
    ids = {int(row["paper_id"]) for row in ranked}
    overlap = len(ids & v49_top10_ids)
    union = len(ids | v49_top10_ids)
    return {
        "top10_relevance_distribution": dict(sorted(Counter(str(row["relevance_bucket"]) for row in ranked).items())),
        "top10_hard_negative_count": sum(1 for row in ranked if row["hard_negative"]),
        "top10_jaccard_vs_v4_9": overlap / union if union else 1.0,
    }


def evaluate_group(rows: list[dict[str, Any]], method: str, v49_top10_ids: set[int]) -> dict[str, Any]:
    scored = rows_with_score(rows, method)
    metrics = v61.ranking_metrics(scored, score_field(method))["aggregate"]
    return metrics | top10_diagnostics(rows, method, v49_top10_ids)


def evaluate_all(rows: list[dict[str, Any]], methods: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_group = []
    for fold_id in sorted({row["fold_id"] for row in rows}):
        for split in ("val", "test"):
            split_rows = [row for row in rows if row["fold_id"] == fold_id and row["split"] == split]
            for topic in sorted({row["query_id"] for row in split_rows}):
                topic_rows = [row for row in split_rows if row["query_id"] == topic]
                v49_ids = {int(row["paper_id"]) for row in top10(topic_rows, "v4_9_guarded_text_blend")}
                for method in methods:
                    per_group.append(
                        {
                            "schema_version": "v6.1.8_per_group_metric",
                            "fold_id": fold_id,
                            "split": split,
                            "query_id": topic,
                            "method": method,
                            **evaluate_group(topic_rows, method, v49_ids),
                        }
                    )
    aggregate: dict[str, Any] = {split: {} for split in ("val", "test")}
    for split in ("val", "test"):
        for method in methods:
            aggregate[split][method] = {
                metric: summarize([row[metric] for row in per_group if row["split"] == split and row["method"] == method])
                for metric in METRICS
            }
    return per_group, aggregate


def deltas(aggregate: dict[str, Any], methods: list[str], baseline: str = "v4_9_guarded_text_blend") -> dict[str, Any]:
    output = {}
    for split in ("val", "test"):
        output[split] = {}
        for method in methods:
            if method == baseline:
                continue
            output[split][method] = {}
            for metric in METRICS:
                method_mean = aggregate[split][method][metric]["mean"]
                base_mean = aggregate[split][baseline][metric]["mean"]
                output[split][method][metric] = None if method_mean is None or base_mean is None else method_mean - base_mean
    return output


def acceptance_diagnostics(per_group: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    by_key = {(row["fold_id"], row["split"], row["query_id"], row["method"]): row for row in per_group}
    output = {}
    for method in methods:
        if method == "v4_9_guarded_text_blend":
            continue
        topic_rows = []
        severe = 0
        severe_nonweak = 0
        for row in per_group:
            if row["method"] != method:
                continue
            base = by_key[(row["fold_id"], row["split"], row["query_id"], "v4_9_guarded_text_blend")]
            reading_delta = row["reading_ndcg_at_10"] - base["reading_ndcg_at_10"]
            if reading_delta <= -0.05:
                severe += 1
                if row["query_id"] not in WEAK_TOPICS:
                    severe_nonweak += 1
            topic_rows.append(
                {
                    "fold_id": row["fold_id"],
                    "split": row["split"],
                    "query_id": row["query_id"],
                    "weak_topic": row["query_id"] in WEAK_TOPICS,
                    "reading_delta_vs_v4_9": reading_delta,
                    "topic_delta_vs_v4_9": row["topic_ndcg_at_10"] - base["topic_ndcg_at_10"],
                    "hard_neg_delta_vs_v4_9": row["hard_neg_at_10"] - base["hard_neg_at_10"],
                    "judged_delta_vs_v4_9": row["judged_at_10"] - base["judged_at_10"],
                }
            )
        test_rows = [row for row in topic_rows if row["split"] == "test"]
        output[method] = {
            "reading_delta_vs_v4_9_test_mean": summarize([row["reading_delta_vs_v4_9"] for row in test_rows]),
            "topic_delta_vs_v4_9_test_mean": summarize([row["topic_delta_vs_v4_9"] for row in test_rows]),
            "hard_neg_delta_vs_v4_9_test_mean": summarize([row["hard_neg_delta_vs_v4_9"] for row in test_rows]),
            "severe_regression_count": severe,
            "non_weak_severe_regression_count": severe_nonweak,
            "judged_at_10_decreases": any(row["judged_delta_vs_v4_9"] < 0 for row in topic_rows),
            "per_topic_deltas": topic_rows,
        }
    return output


def behavior_diagnosis(per_group: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["fold_id"], row["split"], row["query_id"], row["method"]): row for row in per_group}
    beats = []
    loses = []
    fewer_hard = []
    more_hard = []
    for row in per_group:
        if row["method"] != NEURAL_METHOD:
            continue
        base = by_key[(row["fold_id"], row["split"], row["query_id"], "v4_9_guarded_text_blend")]
        delta = row["reading_ndcg_at_10"] - base["reading_ndcg_at_10"]
        item = {"fold_id": row["fold_id"], "split": row["split"], "query_id": row["query_id"], "reading_delta_vs_v4_9": delta}
        if delta > 0.02:
            beats.append(item)
        if delta <= -0.05:
            loses.append(item)
        hard_delta = row["hard_neg_at_10"] - base["hard_neg_at_10"]
        if hard_delta < 0:
            fewer_hard.append(item | {"hard_neg_delta_vs_v4_9": hard_delta})
        if hard_delta > 0:
            more_hard.append(item | {"hard_neg_delta_vs_v4_9": hard_delta})
    return {
        "topics_where_neural_beats_v4_9": beats,
        "topics_where_neural_loses_badly_to_v4_9": loses,
        "topics_where_neural_has_fewer_hard_negatives_than_v4_9": fewer_hard,
        "topics_where_neural_promotes_more_hard_negatives_than_v4_9": more_hard,
        "validation_test_variance": {
            "validation_neural_reading": summarize([row["reading_ndcg_at_10"] for row in per_group if row["method"] == NEURAL_METHOD and row["split"] == "val"]),
            "test_neural_reading": summarize([row["reading_ndcg_at_10"] for row in per_group if row["method"] == NEURAL_METHOD and row["split"] == "test"]),
            "validation_v4_9_reading": summarize([row["reading_ndcg_at_10"] for row in per_group if row["method"] == "v4_9_guarded_text_blend" and row["split"] == "val"]),
            "test_v4_9_reading": summarize([row["reading_ndcg_at_10"] for row in per_group if row["method"] == "v4_9_guarded_text_blend" and row["split"] == "test"]),
        },
    }


def failure_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for fold_id in sorted({row["fold_id"] for row in rows}):
        for split in ("val", "test"):
            split_rows = [row for row in rows if row["fold_id"] == fold_id and row["split"] == split]
            for topic in sorted({row["query_id"] for row in split_rows}):
                topic_rows = [row for row in split_rows if row["query_id"] == topic]
                neural_top = top10(topic_rows, NEURAL_METHOD)
                v49_top = top10(topic_rows, "v4_9_guarded_text_blend")
                v49_ids = {row["paper_id"] for row in v49_top}
                for rank, row in enumerate(neural_top, start=1):
                    if row["hard_negative"] or row["relevance_bucket"] == 0:
                        failures.append(example_row(row, rank, "neural_promoted_negative_or_hard_negative"))
                    if row["generic_survey"] and row["paper_id"] not in v49_ids:
                        failures.append(example_row(row, rank, "neural_promoted_generic_survey_vs_v4_9"))
                    if row["section_label"] == "application" and row["paper_id"] not in v49_ids:
                        failures.append(example_row(row, rank, "neural_promoted_application_only_vs_v4_9"))
                neural_ids = {row["paper_id"] for row in neural_top}
                for rank, row in enumerate(v49_top, start=1):
                    if row["relevance_bucket"] == 2 and row["paper_id"] not in neural_ids:
                        failures.append(example_row(row, rank, "neural_demoted_good_v4_9_top10_paper"))
    return failures[:300]


def example_row(row: dict[str, Any], rank: int, failure_type: str) -> dict[str, Any]:
    return {
        "schema_version": "v6.1.8_neural_vs_v4_9_failure_example",
        "failure_type": failure_type,
        "fold_id": row["fold_id"],
        "split": row["split"],
        "query_id": row["query_id"],
        "paper_id": int(row["paper_id"]),
        "title": row["title"],
        "rank": rank,
        "neural_score": float(row["scores"][NEURAL_METHOD]),
        "v4_9_score": float(row["scores"]["v4_9_guarded_text_blend"]),
        "reading_value_score": float(row["reading_value_score"]),
        "topic_match_score": float(row["topic_match_score"]),
        "relevance_bucket": int(row["relevance_bucket"]),
        "hard_negative": bool(row["hard_negative"]),
        "generic_survey": bool(row["generic_survey"]),
        "section_label": row.get("section_label"),
    }


def recommendation(report: dict[str, Any], blend_report: dict[str, Any]) -> str:
    neural_diag = report["acceptance_diagnostics"][NEURAL_METHOD]
    neural_read_delta = neural_diag["reading_delta_vs_v4_9_test_mean"]["mean"]
    neural_hard_delta = neural_diag["hard_neg_delta_vs_v4_9_test_mean"]["mean"]
    best_probe = blend_report["best_probe"]
    if neural_read_delta is not None and neural_read_delta >= 0.02 and neural_diag["severe_regression_count"] == 0:
        return "proceed_to_v6_2_pairwise_training"
    if best_probe and best_probe["reading_delta_vs_v4_9_test_mean"] >= 0.005 and best_probe["severe_regression_count"] == 0:
        return "proceed_to_v6_2_neural_plus_v4_9_blend"
    if neural_hard_delta is not None and neural_hard_delta < 0 and neural_diag["reading_delta_vs_v4_9_test_mean"]["mean"] >= -0.02:
        return "proceed_to_v6_2_neural_hard_negative_guardrail"
    return "neural_is_not_competitive_keep_as_dl_baseline"


def blend_probe_report(per_group: list[dict[str, Any]], acceptance: dict[str, Any]) -> dict[str, Any]:
    probe_rows = {}
    for method in PROBES:
        diag = acceptance[method]
        probe_rows[method] = {
            "reading_delta_vs_v4_9_test_mean": diag["reading_delta_vs_v4_9_test_mean"]["mean"],
            "topic_delta_vs_v4_9_test_mean": diag["topic_delta_vs_v4_9_test_mean"]["mean"],
            "hard_neg_delta_vs_v4_9_test_mean": diag["hard_neg_delta_vs_v4_9_test_mean"]["mean"],
            "severe_regression_count": diag["severe_regression_count"],
            "non_weak_severe_regression_count": diag["non_weak_severe_regression_count"],
            "judged_at_10_decreases": diag["judged_at_10_decreases"],
        }
    candidates = [
        {"method": method, **row}
        for method, row in probe_rows.items()
        if row["reading_delta_vs_v4_9_test_mean"] is not None
    ]
    highest_reading = max(candidates, key=lambda row: (row["reading_delta_vs_v4_9_test_mean"], -row["severe_regression_count"])) if candidates else None
    safe_candidates = [
        row
        for row in candidates
        if row["severe_regression_count"] == 0
        and row["non_weak_severe_regression_count"] == 0
        and not row["judged_at_10_decreases"]
        and row["hard_neg_delta_vs_v4_9_test_mean"] is not None
        and row["hard_neg_delta_vs_v4_9_test_mean"] <= 0
    ]
    safest = max(safe_candidates, key=lambda row: row["reading_delta_vs_v4_9_test_mean"]) if safe_candidates else None
    best = safest or highest_reading
    return {
        "schema_version": "v6.1.8_neural_blend_probe_report",
        "created_at": datetime.now(UTC).isoformat(),
        "offline_only": True,
        "runtime_integration_status": "offline_only_not_integrated",
        "production_candidate_created": False,
        "probe_methods": probe_rows,
        "highest_reading_probe": highest_reading,
        "safest_probe": safest,
        "best_probe": best,
    }


def render_markdown(report: dict[str, Any]) -> str:
    agg = report["aggregate"]["test"]
    lines = [
        "# V6.1.8 Exact Neural vs Learned Baselines",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Joined rows: `{report['join_diagnostics']['joined_rows']}`",
        f"- Rows silently dropped: `{report['join_diagnostics']['rows_silently_dropped']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "## Test Reading NDCG@10",
        "",
    ]
    for method in METHODS:
        metric = agg[method]["reading_ndcg_at_10"]
        lines.append(f"- `{method}`: `{metric['mean']:.4f}`")
    lines.extend(["", "## Test Hard-neg@10", ""])
    for method in METHODS:
        metric = agg[method]["hard_neg_at_10"]
        lines.append(f"- `{method}`: `{metric['mean']:.4f}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def render_blend_markdown(report: dict[str, Any]) -> str:
    lines = ["# V6.1.8 Neural Blend Probes", "", f"Created: `{report['created_at']}`", ""]
    for method, row in report["probe_methods"].items():
        lines.append(
            f"- `{method}`: reading delta `{row['reading_delta_vs_v4_9_test_mean']:+.4f}`, "
            f"hard-neg delta `{row['hard_neg_delta_vs_v4_9_test_mean']:+.4f}`, severe `{row['severe_regression_count']}`"
        )
    lines.append("")
    lines.append(f"Highest-reading probe: `{(report['highest_reading_probe'] or {}).get('method')}`")
    lines.append(f"Safest no-regression probe: `{(report['safest_probe'] or {}).get('method')}`")
    lines.append(f"Best probe: `{(report['best_probe'] or {}).get('method')}`")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, join_diag = join_predictions(
        resolve_repo_path(args.examples),
        resolve_repo_path(args.neural_predictions),
        resolve_repo_path(args.learned_predictions),
        resolve_repo_path(args.feature_matrix),
    )
    per_group, aggregate = evaluate_all(rows, ALL_METHODS)
    delta_report = deltas(aggregate, ALL_METHODS)
    acceptance = acceptance_diagnostics(per_group, ALL_METHODS)
    behavior = behavior_diagnosis(per_group)
    failures = failure_examples(rows)
    blend = blend_probe_report(per_group, acceptance)
    report = {
        "schema_version": "v6.1.8_exact_neural_vs_learned_baselines_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_trained_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "production_candidate_created": False,
        "methods": METHODS,
        "blend_probe_methods": PROBES,
        "join_diagnostics": join_diag,
        "aggregate": aggregate,
        "deltas_vs_v4_9": delta_report,
        "acceptance_diagnostics": acceptance,
        "behavior_diagnosis": behavior,
        "recommendation": None,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }
    report["recommendation"] = recommendation(report, blend)
    return report, blend, rows, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--neural-predictions", default=str(DEFAULT_NEURAL))
    parser.add_argument("--learned-predictions", default=str(DEFAULT_LEARNED))
    parser.add_argument("--feature-matrix", default=str(DEFAULT_FEATURE_MATRIX))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--predictions-out", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--failures-out", default=str(DEFAULT_FAILURES))
    parser.add_argument("--blend-json-out", default=str(DEFAULT_BLEND_JSON))
    parser.add_argument("--blend-md-out", default=str(DEFAULT_BLEND_MD))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, blend, predictions, failures = build_report(args)
    write_jsonl(resolve_repo_path(args.predictions_out), predictions)
    write_jsonl(resolve_repo_path(args.failures_out), failures)
    write_json(resolve_repo_path(args.blend_json_out), blend)
    write_text(resolve_repo_path(args.blend_md_out), render_blend_markdown(blend))
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V6.1.8 exact comparison report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
