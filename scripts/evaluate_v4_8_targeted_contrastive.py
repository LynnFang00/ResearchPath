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

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from diagnose_v4_5_text_blend_regressions import TEXT_05, TEXT_10, filtered_weighted_blend  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import file_sha256  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
    evaluate_rankings,
    load_jsonl,
    resolve_repo_path,
    write_json,
    write_text,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
)
from evaluate_v4_0_production_pool_2400 import V33_METHOD, labels_with_source  # noqa: E402
from evaluate_v4_1_calibrated_blends import DEFAULT_WEIGHTED_REPORT, DEFAULT_V41_MODEL_DIR  # noqa: E402
from evaluate_v4_3_text_reranker_production_pool import DEFAULT_V43_MODEL_DIR, V41_BLEND_METHOD  # noqa: E402
from evaluate_v4_4_text_blend_sensitivity import DEFAULT_V43_TRAINING_REPORT, add_sensitivity_blends, score_context  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402
from validate_v4_6_guarded_text_blend import GUARDRAIL_METHOD  # noqa: E402


DEFAULT_V48_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v4_8_targeted_contrastive.jsonl"
DEFAULT_V47_CANDIDATES = REPO_ROOT / "data" / "eval" / "results" / "v4_7_targeted_labeling_candidates.jsonl"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_8_targeted_contrastive_reeval_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_8_targeted_contrastive_reeval_report.md"

CONTRASTIVE_TOPIC = "v2_contrastive_learning"
METHODS = [V33_METHOD, V41_BLEND_METHOD, TEXT_05, TEXT_10, GUARDRAIL_METHOD]
EXPECTED_V48_ROWS = 25


def load_all_base_labels(paths_by_source: dict[str, Path]) -> list[dict[str, Any]]:
    labels = []
    for source, path in paths_by_source.items():
        labels.extend(labels_with_source(path, source))
    return labels


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["query_id"]), int(row["paper_id"])


def score_value(row: dict[str, Any], name: str) -> float:
    return float((row.get("relevance_label") or {}).get(name) or 0.0)


def difficulty_scores(label: str) -> dict[str, float]:
    levels = {key: 0.0 for key in ["beginner", "intermediate", "advanced", "expert"]}
    for part in str(label or "").split("+"):
        if part in levels:
            levels[part] = 1.0
    return levels


def normalize_v48_label(row: dict[str, Any]) -> dict[str, Any]:
    difficulty = difficulty_scores(str(row.get("difficulty_label") or ""))
    primary_role = "negative" if row.get("hard_negative") else str(row.get("section_label") or "uncertain")
    return {
        "schema_version": "v4.8_targeted_contrastive_manual_label",
        "query_id": str(row["query_id"]),
        "query": "contrastive learning",
        "paper_id": int(row["paper_id"]),
        "title": row.get("title") or "",
        "topic_match_score": score_value(row, "topic_match_score"),
        "reading_value_score": score_value(row, "reading_value_score"),
        "beginner_fit_score": difficulty["beginner"],
        "intermediate_fit_score": difficulty["intermediate"],
        "advanced_fit_score": difficulty["advanced"],
        "expert_fit_score": difficulty["expert"],
        "intent_scores": {},
        "primary_role": primary_role,
        "secondary_roles": [] if primary_role == "negative" else [primary_role],
        "duplicate_status": "none",
        "duplicate_of_paper_id": None,
        "evidence_level": "title_abstract",
        "full_text_available": False,
        "label_confidence": "medium",
        "notes": row.get("label_notes") or "",
        "judged_source": "v4_8",
        "hard_negative": bool(row.get("hard_negative")),
        "positive": score_value(row, "reading_value_score") >= 0.5 and score_value(row, "topic_match_score") >= 0.5,
        "v4_8_original": row,
    }


def combine_with_v48_overrides(base_labels: list[dict[str, Any]], v48_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    override_keys = {row_key(row) for row in v48_rows}
    base_keys = {row_key(row) for row in base_labels}
    normalized = [normalize_v48_label(row) for row in v48_rows]
    combined = [row for row in base_labels if row_key(row) not in override_keys] + normalized
    duplicate_counts = Counter(row_key(row) for row in combined)
    return combined, {
        "base_label_count": len(base_labels),
        "v4_8_label_count": len(v48_rows),
        "effective_label_count": len(combined),
        "overlap_with_base_labels": len(override_keys & base_keys),
        "v4_8_duplicate_query_paper_rows": [
            {"query_id": query_id, "paper_id": paper_id, "count": count}
            for (query_id, paper_id), count in sorted(Counter(row_key(row) for row in v48_rows).items())
            if count > 1
        ],
        "combined_duplicate_query_paper_rows": [
            {"query_id": query_id, "paper_id": paper_id, "count": count}
            for (query_id, paper_id), count in sorted(duplicate_counts.items())
            if count > 1
        ],
    }


def v48_preflight(v48_rows: list[dict[str, Any]], v47_rows: list[dict[str, Any]]) -> dict[str, Any]:
    v48_keys = {row_key(row) for row in v48_rows}
    v47_keys = {row_key(row) for row in v47_rows}
    buckets = Counter(int((row.get("relevance_label") or {}).get("bucket")) for row in v48_rows)
    return {
        "valid_row_count": len(v48_rows) == EXPECTED_V48_ROWS,
        "row_count": len(v48_rows),
        "expected_row_count": EXPECTED_V48_ROWS,
        "all_query_ids_contrastive": all(str(row.get("query_id")) == CONTRASTIVE_TOPIC for row in v48_rows),
        "duplicate_query_paper_rows": [
            {"query_id": query_id, "paper_id": paper_id, "count": count}
            for (query_id, paper_id), count in sorted(Counter(row_key(row) for row in v48_rows).items())
            if count > 1
        ],
        "all_v4_8_rows_in_v4_7_candidates": v48_keys <= v47_keys,
        "v4_8_rows_not_in_v4_7_candidates": [
            {"query_id": query_id, "paper_id": paper_id}
            for query_id, paper_id in sorted(v48_keys - v47_keys)
        ],
        "bucket_distribution": dict(sorted(buckets.items())),
        "hard_negative_count": sum(1 for row in v48_rows if row.get("hard_negative")),
        "average_reading_value_score": sum(score_value(row, "reading_value_score") for row in v48_rows) / len(v48_rows),
        "average_topic_match_score": sum(score_value(row, "topic_match_score") for row in v48_rows) / len(v48_rows),
    }


def add_v46_guardrail(method_scores: dict[str, dict[str, dict[int, float]]], best_v41_method: str, best_v43_method: str) -> None:
    method_scores[GUARDRAIL_METHOD] = filtered_weighted_blend(
        method_scores,
        v41_method=best_v41_method,
        v43_method=best_v43_method,
        text_weight=0.10,
        v33_weight=0.675,
        v41_weight=0.225,
        mode="text_only_when_rankers_not_confident",
    )


def deltas(after: dict[str, Any], before: dict[str, Any]) -> dict[str, dict[str, float]]:
    output = {}
    for method in METHODS:
        left = after["method_averages"][method]
        right = before["method_averages"][method]
        output[method] = {
            "reading_value_ndcg_delta": left["reading_value_ndcg"] - right["reading_value_ndcg"],
            "topic_match_ndcg_delta": left["topic_match_ndcg"] - right["topic_match_ndcg"],
            "hard_negative_rate_at_10_delta": left["hard_negative_rate_at_10"] - right["hard_negative_rate_at_10"],
            "judged_at_10_delta": left["judged_at_10"] - right["judged_at_10"],
            "unjudged_at_10_delta": left["unjudged_at_10"] - right["unjudged_at_10"],
        }
    return output


def topic_rows(evaluation: dict[str, Any], topic: str) -> dict[str, dict[str, Any]]:
    return {
        method: next(row for row in evaluation["per_topic"][method] if row["query_id"] == topic)
        for method in METHODS
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    context = score_context(args)
    method_scores = add_sensitivity_blends(context["method_scores"], context["best_v41_method"], context["best_v43_method"])
    add_v46_guardrail(method_scores, context["best_v41_method"], context["best_v43_method"])
    scoped_scores = {method: method_scores[method] for method in METHODS}

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
    combined_labels, combine_summary = combine_with_v48_overrides(base_labels, v48_rows)
    before = evaluate_rankings(candidate_rows_by_q=context["candidate_rows_by_q"], labels=base_labels, method_scores=scoped_scores, k=K)
    after = evaluate_rankings(candidate_rows_by_q=context["candidate_rows_by_q"], labels=combined_labels, method_scores=scoped_scores, k=K)
    protected = protected_status(resolve_repo_path(args.v39_labels))
    v48_hash = file_sha256(resolve_repo_path(args.v48_labels))
    return {
        "schema_version": "v4.8_targeted_contrastive_reeval_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "v4_8_label_file": str(resolve_repo_path(args.v48_labels)),
        "v4_8_label_file_sha256": v48_hash,
        "preflight": v48_preflight(v48_rows, v47_rows),
        "label_combination": combine_summary,
        "components": {
            "v3_3_method": V33_METHOD,
            "v4_1_weighted_method": context["best_v41_method"],
            "v4_1_blend_method": V41_BLEND_METHOD,
            "v4_3_text_method": context["best_v43_method"],
            "v4_6_guardrail_method": GUARDRAIL_METHOD,
        },
        "before_v4_8": before,
        "after_v4_8": after,
        "aggregate_deltas_after_minus_before": deltas(after, before),
        "contrastive_topic_before": topic_rows(before, CONTRASTIVE_TOPIC),
        "contrastive_topic_after": topic_rows(after, CONTRASTIVE_TOPIC),
        "contrastive_topic_deltas_after_minus_before": {
            method: {
                "reading_value_ndcg_delta": topic_rows(after, CONTRASTIVE_TOPIC)[method]["reading_value_ndcg"] - topic_rows(before, CONTRASTIVE_TOPIC)[method]["reading_value_ndcg"],
                "topic_match_ndcg_delta": topic_rows(after, CONTRASTIVE_TOPIC)[method]["topic_match_ndcg"] - topic_rows(before, CONTRASTIVE_TOPIC)[method]["topic_match_ndcg"],
                "hard_negative_rate_at_10_delta": topic_rows(after, CONTRASTIVE_TOPIC)[method]["hard_negative_rate_at_10"] - topic_rows(before, CONTRASTIVE_TOPIC)[method]["hard_negative_rate_at_10"],
            }
            for method in METHODS
        },
        "protected_hashes": protected,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.8 Targeted Contrastive Re-Evaluation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- V4.8 rows: `{report['preflight']['row_count']}`",
        f"- Effective judged rows: `{report['label_combination']['effective_label_count']}`",
        f"- Overlap with base labels overridden: `{report['label_combination']['overlap_with_base_labels']}`",
        f"- Bucket distribution: `{report['preflight']['bucket_distribution']}`",
        f"- Hard negatives: `{report['preflight']['hard_negative_count']}`",
        "",
        "## After V4.8 Aggregate Metrics",
        "",
        "| Method | Reading | Topic | Judged | Unjudged | Hard-neg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["after_v4_8"]["method_averages"][method]
        lines.append(
            f"| `{method}` | `{row['reading_value_ndcg']:.3f}` | `{row['topic_match_ndcg']:.3f}` | "
            f"`{row['judged_at_10']:.3f}` | `{row['unjudged_at_10']:.3f}` | `{row['hard_negative_rate_at_10']:.3f}` |"
        )
    lines.extend(["", "## Contrastive Topic Deltas After V4.8", ""])
    for method, row in report["contrastive_topic_deltas_after_minus_before"].items():
        lines.append(f"- `{method}`: reading `{row['reading_value_ndcg_delta']:.3f}`, topic `{row['topic_match_ndcg_delta']:.3f}`, hard-neg `{row['hard_negative_rate_at_10_delta']:.3f}`")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v48-labels", default=str(DEFAULT_V48_LABELS))
    parser.add_argument("--v47-candidates", default=str(DEFAULT_V47_CANDIDATES))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V4.8 targeted contrastive re-evaluation report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
