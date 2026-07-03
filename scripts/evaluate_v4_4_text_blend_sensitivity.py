import argparse
from collections import defaultdict
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

from build_v4_3_text_reranker_dataset import (  # noqa: E402
    DEFAULT_DATASET_JSONL as DEFAULT_V43_DATASET,
    WEAK_TOPICS,
    protected_status,
)
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
    build_method_scores,
    evaluate_rankings,
    labels_by_query,
    load_json,
    load_jsonl,
    production_candidate_rows,
    resolve_repo_path,
    score_rows,
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
from evaluate_v4_0_production_pool_2400 import (  # noqa: E402
    V33_METHOD,
    labels_with_source,
)
from evaluate_v4_1_calibrated_blends import (  # noqa: E402
    DEFAULT_WEIGHTED_REPORT,
    DEFAULT_V41_MODEL_DIR,
    best_v41_strategy,
    method_name,
    score_v41_artifact,
    weighted_blend,
)
from evaluate_v4_3_text_reranker_production_pool import (  # noqa: E402
    DEFAULT_V43_MODEL_DIR,
    V41_BLEND_METHOD,
    choose_best_v43_method,
    load_v43_models,
    score_v43_model,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_4_text_blend_sensitivity_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_4_text_blend_sensitivity_report.md"
DEFAULT_V43_TRAINING_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_training_report.json"

BLEND_SPECS = [
    {"name": "text_weight_0_00_v41_blend", "text_weight": 0.00, "weights": {V33_METHOD: 0.75, "__v41__": 0.25}},
    {"name": "text_weight_0_05", "text_weight": 0.05, "weights": {V33_METHOD: 0.70, "__v41__": 0.25, "__v43__": 0.05}},
    {"name": "text_weight_0_10", "text_weight": 0.10, "weights": {V33_METHOD: 0.675, "__v41__": 0.225, "__v43__": 0.10}},
    {"name": "text_weight_0_15", "text_weight": 0.15, "weights": {V33_METHOD: 0.65, "__v41__": 0.20, "__v43__": 0.15}},
    {"name": "text_weight_0_20", "text_weight": 0.20, "weights": {V33_METHOD: 0.60, "__v41__": 0.20, "__v43__": 0.20}},
    {"name": "text_weight_0_30", "text_weight": 0.30, "weights": {V33_METHOD: 0.50, "__v41__": 0.20, "__v43__": 0.30}},
]
ACCEPTANCE_TOLERANCE = 1e-9


def load_all_labels(paths_by_source: dict[str, Path]) -> list[dict[str, Any]]:
    labels = []
    for source, path in paths_by_source.items():
        labels.extend(labels_with_source(path, source))
    return labels


def score_context(args: argparse.Namespace) -> dict[str, Any]:
    labels = load_all_labels(
        {
            "v2_1": resolve_repo_path(args.v21_labels),
            "v2_5": resolve_repo_path(args.v25_labels),
            "v3_2": resolve_repo_path(args.v32_labels),
            "v3_5": resolve_repo_path(args.v35_labels),
            "v3_9": resolve_repo_path(args.v39_labels),
        }
    )
    packet_rows = load_jsonl(resolve_repo_path(args.v21_packet))
    candidate_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=packet_rows, top_k=args.top_k)
    scored = score_rows(
        candidate_rows_by_q=candidate_rows_by_q,
        context_rows_by_q=candidate_rows_by_q,
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
    )
    method_scores, model_diagnostics = build_method_scores(
        production_rows_by_q=candidate_rows_by_q,
        scored=scored,
        model_dir=resolve_repo_path(args.v33_model_dir),
    )
    best_strategy = best_v41_strategy(resolve_repo_path(args.weighted_report))
    best_v41_method = method_name(best_strategy)
    method_scores[best_v41_method] = score_v41_artifact(
        model_dir=resolve_repo_path(args.v41_model_dir),
        strategy=best_strategy,
        production_rows_by_q=candidate_rows_by_q,
        scored=scored,
    )
    v43_models = load_v43_models(resolve_repo_path(args.v43_model_dir))
    v43_training_report = load_json(resolve_repo_path(args.v43_training_report))
    best_v43_method = choose_best_v43_method(v43_training_report, v43_models)
    for model in v43_models:
        score_name = model["metadata"]["score_column"]
        if score_name == best_v43_method:
            method_scores[score_name] = score_v43_model(model_payload=model, production_rows_by_q=candidate_rows_by_q, scored=scored)
            model_diagnostics[score_name] = model["metadata"]
    method_scores[V41_BLEND_METHOD] = weighted_blend(method_scores, {V33_METHOD: 0.75, best_v41_method: 0.25})
    return {
        "labels": labels,
        "candidate_rows_by_q": candidate_rows_by_q,
        "scored": scored,
        "method_scores": method_scores,
        "model_diagnostics": model_diagnostics,
        "production_diagnostics": production_diagnostics,
        "best_v41_method": best_v41_method,
        "best_v41_strategy": best_strategy,
        "best_v43_method": best_v43_method,
    }


def add_sensitivity_blends(method_scores: dict[str, dict[str, dict[int, float]]], best_v41_method: str, best_v43_method: str) -> dict[str, dict[str, dict[int, float]]]:
    output = dict(method_scores)
    for spec in BLEND_SPECS:
        weights = {
            best_v41_method if method == "__v41__" else best_v43_method if method == "__v43__" else method: weight
            for method, weight in spec["weights"].items()
        }
        output[spec["name"]] = weighted_blend(output, weights)
    return output


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = [
        "reading_value_ndcg",
        "topic_match_ndcg",
        "mrr_at_10",
        "positive_recall_at_10",
        "judged_at_10",
        "unjudged_at_10",
        "hard_negative_rate_at_10",
    ]
    return {field: sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0 for field in fields}


def weak_topic_metrics(evaluation: dict[str, Any], methods: list[str]) -> dict[str, dict[str, float]]:
    output = {}
    for method in methods:
        rows = [row for row in evaluation["per_topic"][method] if row["query_id"] in WEAK_TOPICS]
        output[method] = aggregate_metric_rows(rows)
    return output


def is_positive(label: dict[str, Any]) -> bool:
    return float(label.get("reading_value_score") or 0.0) >= 0.5 and float(label.get("topic_match_score") or 0.0) >= 0.5


def is_hard_negative(label: dict[str, Any]) -> bool:
    return str(label.get("primary_role")) == "negative" or (
        float(label.get("reading_value_score") or 0.0) <= 0.0 and float(label.get("topic_match_score") or 0.0) <= 0.0
    )


def load_disagreement_keys(path: Path) -> set[tuple[str, int]]:
    keys = set()
    for row in load_jsonl(path):
        if row.get("disagreement_kind"):
            keys.add((str(row["query_id"]), int(row["paper_id"])))
    return keys


def disagreement_pair_accuracy(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    disagreement_keys: set[tuple[str, int]],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    label_lookup = labels_by_query(labels)
    output = {}
    for method in methods:
        total_pairs = 0
        correct_pairs = 0
        scoped_rows = 0
        for query_id, rows in candidate_rows_by_q.items():
            positives = []
            hard_negatives = []
            for row in rows:
                paper_id = int(row["paper_id"])
                if (query_id, paper_id) not in disagreement_keys:
                    continue
                label = label_lookup.get(query_id, {}).get(paper_id)
                if not label:
                    continue
                scoped_rows += 1
                score = method_scores[method][query_id].get(paper_id)
                if score is None:
                    continue
                if is_positive(label):
                    positives.append(float(score))
                if is_hard_negative(label):
                    hard_negatives.append(float(score))
            for positive_score in positives:
                for negative_score in hard_negatives:
                    total_pairs += 1
                    if positive_score > negative_score:
                        correct_pairs += 1
        output[method] = {
            "scope_row_count": scoped_rows,
            "pair_count": total_pairs,
            "pair_accuracy": correct_pairs / total_pairs if total_pairs else None,
        }
    return output


def hard_negative_suppression(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    label_lookup = labels_by_query(labels)
    hard_negative_keys = {
        (query_id, paper_id)
        for query_id, query_labels in label_lookup.items()
        for paper_id, label in query_labels.items()
        if is_hard_negative(label)
    }
    output = {}
    for method in methods:
        candidate_hard_negatives = 0
        promoted_hard_negatives = 0
        for query_id, rows in candidate_rows_by_q.items():
            ranked = [
                paper_id
                for paper_id, _ in sorted(method_scores[method][query_id].items(), key=lambda item: (item[1], -item[0]), reverse=True)
            ]
            top10 = set(ranked[:K])
            for row in rows:
                key = (query_id, int(row["paper_id"]))
                if key in hard_negative_keys:
                    candidate_hard_negatives += 1
                    if key[1] in top10:
                        promoted_hard_negatives += 1
        output[method] = {
            "candidate_hard_negative_count": candidate_hard_negatives,
            "top10_hard_negative_count": promoted_hard_negatives,
            "suppression_accuracy": 1.0 - (promoted_hard_negatives / candidate_hard_negatives) if candidate_hard_negatives else None,
        }
    return output


def row_for_method(method: str, averages: dict[str, dict[str, float]], baseline: dict[str, float], weak: dict[str, dict[str, float]]) -> dict[str, Any]:
    row = averages[method]
    return {
        "method": method,
        "reading_value_ndcg": row["reading_value_ndcg"],
        "topic_match_ndcg": row["topic_match_ndcg"],
        "mrr_at_10": row["mrr_at_10"],
        "positive_recall_at_10": row["positive_recall_at_10"],
        "judged_at_10": row["judged_at_10"],
        "unjudged_at_10": row["unjudged_at_10"],
        "hard_negative_rate_at_10": row["hard_negative_rate_at_10"],
        "reading_delta_vs_v4_1_blend": row["reading_value_ndcg"] - baseline["reading_value_ndcg"],
        "topic_delta_vs_v4_1_blend": row["topic_match_ndcg"] - baseline["topic_match_ndcg"],
        "hard_negative_delta_vs_v4_1_blend": row["hard_negative_rate_at_10"] - baseline["hard_negative_rate_at_10"],
        "weak_topic_reading_ndcg": weak[method]["reading_value_ndcg"],
        "weak_topic_topic_ndcg": weak[method]["topic_match_ndcg"],
        "weak_topic_hard_negative_rate_at_10": weak[method]["hard_negative_rate_at_10"],
    }


def choose_best(sensitivity_rows: list[dict[str, Any]], baseline: dict[str, float]) -> dict[str, Any]:
    acceptable = [
        row
        for row in sensitivity_rows
        if row["reading_value_ndcg"] >= baseline["reading_value_ndcg"] - ACCEPTANCE_TOLERANCE
        and row["topic_match_ndcg"] >= baseline["topic_match_ndcg"] - ACCEPTANCE_TOLERANCE
        and row["hard_negative_rate_at_10"] <= baseline["hard_negative_rate_at_10"] + ACCEPTANCE_TOLERANCE
        and row["judged_at_10"] >= 9.75
        and row["unjudged_at_10"] <= 0.25
    ]
    candidates = acceptable or sensitivity_rows
    best = max(
        candidates,
        key=lambda row: (
            row["reading_value_ndcg"],
            row["topic_match_ndcg"],
            -row["hard_negative_rate_at_10"],
            -row["unjudged_at_10"],
        ),
    )
    return {"best": best, "acceptable_count": len(acceptable), "acceptable_methods": [row["method"] for row in acceptable]}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    context = score_context(args)
    method_scores = add_sensitivity_blends(context["method_scores"], context["best_v41_method"], context["best_v43_method"])
    blend_methods = [spec["name"] for spec in BLEND_SPECS]
    methods = [V33_METHOD, V41_BLEND_METHOD, context["best_v43_method"], *blend_methods]
    evaluation = evaluate_rankings(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores={method: method_scores[method] for method in methods},
        k=K,
    )
    averages = evaluation["method_averages"]
    weak = weak_topic_metrics(evaluation, methods)
    baseline = averages[V41_BLEND_METHOD]
    sensitivity_rows = [row_for_method(method, averages, baseline, weak) | {"text_weight": spec["text_weight"], "weights": spec["weights"]} for spec, method in zip(BLEND_SPECS, blend_methods, strict=True)]
    selection = choose_best(sensitivity_rows, baseline)
    disagreement_keys = load_disagreement_keys(resolve_repo_path(args.v43_dataset))
    disagreement_accuracy = disagreement_pair_accuracy(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        disagreement_keys=disagreement_keys,
        methods=blend_methods,
    )
    hard_negative = hard_negative_suppression(
        candidate_rows_by_q=context["candidate_rows_by_q"],
        labels=context["labels"],
        method_scores=method_scores,
        methods=blend_methods,
    )
    return {
        "schema_version": "v4.4_text_blend_sensitivity_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "existing_models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "candidate_pool_definition": {
            "top_k_per_method": args.top_k,
            "union": ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"],
            "candidate_pool_sizes_by_topic": {query_id: len(rows) for query_id, rows in context["candidate_rows_by_q"].items()},
        },
        "components": {
            "v3_3_method": V33_METHOD,
            "v4_1_weighted_method": context["best_v41_method"],
            "v4_1_blend_method": V41_BLEND_METHOD,
            "v4_3_text_method": context["best_v43_method"],
        },
        "baseline_v4_1_blend": averages[V41_BLEND_METHOD],
        "sensitivity": sensitivity_rows,
        "selection": selection,
        "weak_topic_metrics": {method: weak[method] for method in blend_methods},
        "disagreement_pool_accuracy": disagreement_accuracy,
        "hard_negative_suppression_accuracy": hard_negative,
        "evaluation": evaluation,
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.4 Text Blend Sensitivity Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Selection",
        "",
        f"- Best blend: `{report['selection']['best']['method']}`",
        f"- Acceptable blends: `{report['selection']['acceptable_count']}`",
        "",
        "## Sensitivity",
        "",
        "| Method | Text Weight | Reading | Topic | MRR | Recall | Judged | Unjudged | Hard-neg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["sensitivity"]:
        lines.append(
            f"| `{row['method']}` | `{row['text_weight']:.2f}` | `{row['reading_value_ndcg']:.3f}` | "
            f"`{row['topic_match_ndcg']:.3f}` | `{row['mrr_at_10']:.3f}` | `{row['positive_recall_at_10']:.3f}` | "
            f"`{row['judged_at_10']:.3f}` | `{row['unjudged_at_10']:.3f}` | `{row['hard_negative_rate_at_10']:.3f}` |"
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
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report))
    print(f"Wrote V4.4 text blend sensitivity report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
