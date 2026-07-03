import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
    build_method_scores,
    evaluate_rankings,
    load_json,
    load_jsonl,
    production_candidate_rows,
    resolve_repo_path,
    score_rows,
    top10_overlap,
    write_json,
    write_text,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402
from train_v3_0_ltr_models import numeric_feature_value  # noqa: E402


DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_0_ltr_2400"
DEFAULT_TRAINING_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_0_ltr_2400_report.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_0_production_pool_2400_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_0_production_pool_2400_report.md"
V33_METHOD = "v3_3_random_forest_no_v27_compact"


def labels_with_source(path: Path, source: str) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        copied = dict(row)
        copied["judged_source"] = source
        rows.append(copied)
    return rows


def load_all_labels(
    v21_labels: Path,
    v25_labels: Path,
    v32_labels: Path,
    v35_labels: Path,
    v39_labels: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_source = {
        "v2_1": labels_with_source(v21_labels, "v2_1"),
        "v2_5": labels_with_source(v25_labels, "v2_5"),
        "v3_2": labels_with_source(v32_labels, "v3_2"),
        "v3_5": labels_with_source(v35_labels, "v3_5"),
        "v3_9": labels_with_source(v39_labels, "v3_9"),
    }
    return [row for rows in by_source.values() for row in rows], {source: len(rows) for source, rows in by_source.items()}


def load_model(model_dir: Path, model_name: str) -> dict[str, Any]:
    metadata_path = model_dir / f"{model_name}.metadata.json"
    artifact_path = model_dir / f"{model_name}.joblib"
    return {
        "model_name": model_name,
        "metadata": load_json(metadata_path),
        "estimator": joblib.load(artifact_path),
    }


def score_v4_model(
    *,
    model_payload: dict[str, Any],
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    scored: dict[str, dict[int, dict[str, float]]],
) -> dict[str, dict[int, float]]:
    features = list(model_payload["metadata"]["feature_columns"])
    estimator = model_payload["estimator"]
    output: dict[str, dict[int, float]] = {}
    for query_id, rows in sorted(production_rows_by_q.items()):
        query_scores = {}
        for row in rows:
            paper_id = int(row["paper_id"])
            merged = dict(row)
            merged.update(scored[query_id][paper_id])
            x_values = np.array([[numeric_feature_value(merged, feature) for feature in features]], dtype=float)
            if model_payload["metadata"]["model_type"] == "pointwise_logistic_regression":
                query_scores[paper_id] = float(estimator.predict_proba(x_values)[:, 1][0])
            elif model_payload["metadata"]["model_type"] == "pairwise_logistic_regression":
                query_scores[paper_id] = float(estimator.decision_function(x_values)[0])
            else:
                query_scores[paper_id] = float(estimator.predict(x_values)[0])
        output[query_id] = query_scores
    return output


def best_with_v27_model(training_report: dict[str, Any]) -> str | None:
    aggregates = training_report["leave_topic_out"]["aggregate_mean_median"]
    with_v27 = [
        spec["name"]
        for spec in training_report["model_specs"]
        if spec["uses_v2_7_feature"]
    ]
    if not with_v27:
        return None
    return max(
        with_v27,
        key=lambda name: (
            aggregates[f"v4_0_{name}_score"]["mean_reading_value_ndcg"],
            aggregates[f"v4_0_{name}_score"]["mean_topic_match_ndcg"],
        ),
    )


def duplicate_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in Counter((str(row["query_id"]), int(row["paper_id"])) for row in rows).items()
        if count > 1
    ]


def weak_topic_deltas(evaluation: dict[str, Any], left: str, right: str) -> list[dict[str, Any]]:
    left_rows = {row["query_id"]: row for row in evaluation["per_topic"][left]}
    right_rows = {row["query_id"]: row for row in evaluation["per_topic"][right]}
    return [
        {
            "query_id": query_id,
            "reading_delta": left_rows[query_id]["reading_value_ndcg"] - right_rows[query_id]["reading_value_ndcg"],
            "topic_delta": left_rows[query_id]["topic_match_ndcg"] - right_rows[query_id]["topic_match_ndcg"],
            "hard_negative_delta": left_rows[query_id]["hard_negative_rate_at_10"] - right_rows[query_id]["hard_negative_rate_at_10"],
        }
        for query_id in sorted(set(left_rows) & set(right_rows))
    ]


def build_report(
    *,
    v21_packet_path: Path,
    old_model_path: Path,
    v26_model_path: Path,
    v33_model_dir: Path,
    v40_model_dir: Path,
    v40_training_report_path: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v35_labels_path: Path,
    v39_labels_path: Path,
    selected_240_path: Path,
    top_k: int,
) -> dict[str, Any]:
    labels, labels_by_source = load_all_labels(v21_labels_path, v25_labels_path, v32_labels_path, v35_labels_path, v39_labels_path)
    v21_packet_rows = load_jsonl(v21_packet_path)
    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=v21_packet_rows, top_k=top_k)
    scored = score_rows(candidate_rows_by_q=production_rows_by_q, context_rows_by_q=production_rows_by_q, old_model_path=old_model_path, v26_model_path=v26_model_path)
    method_scores, model_diagnostics = build_method_scores(production_rows_by_q=production_rows_by_q, scored=scored, model_dir=v33_model_dir)
    training_report = load_json(v40_training_report_path)
    best_no_v27 = training_report["best_no_v27_model_by_loto_reading_ndcg"]["model_name"]
    best_with_v27 = best_with_v27_model(training_report)
    for model_name in [best_no_v27] + ([best_with_v27] if best_with_v27 else []):
        payload = load_model(v40_model_dir, model_name)
        method_scores[f"v4_0_{model_name}"] = score_v4_model(model_payload=payload, production_rows_by_q=production_rows_by_q, scored=scored)
        model_diagnostics[f"v4_0_{model_name}"] = {
            "metadata_path": str(v40_model_dir / f"{model_name}.metadata.json"),
            "artifact_path": str(v40_model_dir / f"{model_name}.joblib"),
            "feature_columns": payload["metadata"]["feature_columns"],
            "uses_v2_7_feature": payload["metadata"]["uses_v2_7_feature"],
        }
    evaluation = evaluate_rankings(candidate_rows_by_q=production_rows_by_q, labels=labels, method_scores=method_scores, k=K)
    v4_method = f"v4_0_{best_no_v27}"
    v4_with_v27_method = f"v4_0_{best_with_v27}" if best_with_v27 else None
    averages = evaluation["method_averages"]
    comparison = {
        "best_v4_no_v27_method": v4_method,
        "best_v4_with_v27_method": v4_with_v27_method,
        "v4_vs_v3_3_reading_delta": averages[v4_method]["reading_value_ndcg"] - averages[V33_METHOD]["reading_value_ndcg"],
        "v4_vs_v3_3_topic_delta": averages[v4_method]["topic_match_ndcg"] - averages[V33_METHOD]["topic_match_ndcg"],
        "v4_vs_v2_7_reading_delta": averages[v4_method]["reading_value_ndcg"] - averages["v2_7"]["reading_value_ndcg"],
        "v4_vs_v2_7_topic_delta": averages[v4_method]["topic_match_ndcg"] - averages["v2_7"]["topic_match_ndcg"],
        "v4_vs_v3_3_hard_negative_delta": averages[v4_method]["hard_negative_rate_at_10"] - averages[V33_METHOD]["hard_negative_rate_at_10"],
        "v4_improves_over_v3_3_reading": averages[v4_method]["reading_value_ndcg"] > averages[V33_METHOD]["reading_value_ndcg"],
        "v4_improves_over_v3_3_topic": averages[v4_method]["topic_match_ndcg"] > averages[V33_METHOD]["topic_match_ndcg"],
        "recommend_backend_opt_in_replacement_later": False,
        "recommendation_reason": "Require review of V4.0 production metrics before replacing the already-validated V3.3 opt-in path.",
    }
    if v4_with_v27_method:
        comparison["with_v27_reading_delta_vs_no_v27"] = averages[v4_with_v27_method]["reading_value_ndcg"] - averages[v4_method]["reading_value_ndcg"]
        comparison["with_v27_topic_delta_vs_no_v27"] = averages[v4_with_v27_method]["topic_match_ndcg"] - averages[v4_method]["topic_match_ndcg"]
    return {
        "schema_version": "v4.0_production_pool_2400_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "candidate_pool_definition": {
            "top_k_per_method": top_k,
            "union": ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"],
            "candidate_pool_sizes_by_topic": {query_id: len(rows) for query_id, rows in production_rows_by_q.items()},
            "duplicate_candidate_rows": [],
        },
        "dataset": {
            "label_count": len(labels),
            "labels_by_source": labels_by_source,
            "duplicate_query_paper_rows": duplicate_keys(labels),
        },
        "production_diagnostics": production_diagnostics,
        "model_diagnostics": model_diagnostics,
        "evaluation": evaluation,
        "comparison": comparison,
        "weak_topic_wins_losses_vs_v3_3": weak_topic_deltas(evaluation, v4_method, V33_METHOD),
        "top10_overlap": {
            "v4_no_v27_vs_v3_3": top10_overlap(evaluation["per_topic"], v4_method, V33_METHOD),
            "v4_no_v27_vs_v2_7": top10_overlap(evaluation["per_topic"], v4_method, "v2_7"),
        },
        "protected_hashes": protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path, v35_labels_path, selected_240_path),
    }


def method_table(averages: dict[str, dict[str, float]], methods: list[str]) -> list[str]:
    lines = [
        "| method | reading | topic | judged@10 | unjudged@10 | hard-neg | recall | MRR | role cov | path cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['reading_value_ndcg']:.3f} | {row['topic_match_ndcg']:.3f} | {row['judged_at_10']:.2f} | "
            f"{row['unjudged_at_10']:.2f} | {row['hard_negative_rate_at_10']:.3f} | {row['positive_recall_at_10']:.3f} | "
            f"{row['mrr_at_10']:.3f} | {row['role_coverage_at_10']:.3f} | {row['path_coverage_at_10']:.3f} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    methods = ["hybrid", "v2_6", "v2_7", V33_METHOD, comparison["best_v4_no_v27_method"]]
    if comparison.get("best_v4_with_v27_method"):
        methods.append(comparison["best_v4_with_v27_method"])
    lines = [
        "# V4.0 Production Pool 2400 Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Label count: `{report['dataset']['label_count']}`",
        f"- Best V4.0 no-V2.7 method: `{comparison['best_v4_no_v27_method']}`",
        f"- V4.0 vs V3.3 reading delta: `{comparison['v4_vs_v3_3_reading_delta']:+.3f}`",
        f"- V4.0 vs V3.3 topic delta: `{comparison['v4_vs_v3_3_topic_delta']:+.3f}`",
        f"- V4.0 vs V2.7 reading delta: `{comparison['v4_vs_v2_7_reading_delta']:+.3f}`",
        f"- Runtime integration: `{report['runtime_integration_status']}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(method_table(report["evaluation"]["method_averages"], methods))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4.0 models on the current production-style 50K candidate pool.")
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v40-model-dir", default=str(DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v40-training-report", default=str(DEFAULT_TRAINING_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        v21_packet_path=resolve_repo_path(args.v21_packet),
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
        v33_model_dir=resolve_repo_path(args.v33_model_dir),
        v40_model_dir=resolve_repo_path(args.v40_model_dir),
        v40_training_report_path=resolve_repo_path(args.v40_training_report),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        v35_labels_path=resolve_repo_path(args.v35_labels),
        v39_labels_path=resolve_repo_path(args.v39_labels),
        selected_240_path=resolve_repo_path(args.selected_240),
        top_k=args.top_k,
    )
    report["command"] = " ".join(sys.argv)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    print("V4.0 production-pool evaluation complete")
    print(f"Best V4 no-V2.7: {report['comparison']['best_v4_no_v27_method']}")
    print(f"V4 vs V3.3 reading delta: {report['comparison']['v4_vs_v3_3_reading_delta']:+.3f}")
    print(f"Report: {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
