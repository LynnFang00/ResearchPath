import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from scipy import sparse


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
from evaluate_v4_0_production_pool_2400 import (  # noqa: E402
    DEFAULT_MODEL_DIR as DEFAULT_V40_MODEL_DIR,
    V33_METHOD,
    labels_with_source,
    load_model,
    score_v4_model,
    weak_topic_deltas,
)
from evaluate_v4_1_calibrated_blends import (  # noqa: E402
    DEFAULT_WEIGHTED_REPORT,
    DEFAULT_V41_MODEL_DIR,
    V40_METHOD,
    best_v41_strategy,
    method_name,
    score_v41_artifact,
    weighted_blend,
)
from train_v3_0_ltr_models import numeric_feature_value  # noqa: E402
from train_v4_1_weighted_ltr_models_2400 import FEATURE_COLUMNS  # noqa: E402
from train_v4_3_text_reranker import DEFAULT_MODEL_DIR as DEFAULT_V43_MODEL_DIR  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_production_pool_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_production_pool_report.md"
V41_BLEND_METHOD = "blend_0_75_v33_0_25_v41"
V43_BLEND_SPECS = [
    {
        "name": "blend_0_70_v33_0_20_v41_0_10_v43_text",
        "weights": {V33_METHOD: 0.70, "__v41__": 0.20, "__v43__": 0.10},
    },
    {
        "name": "blend_0_70_v41_blend_0_30_v43_text",
        "weights": {V41_BLEND_METHOD: 0.70, "__v43__": 0.30},
    },
]


def load_all_labels(paths_by_source: dict[str, Path]) -> list[dict[str, Any]]:
    labels = []
    for source, path in paths_by_source.items():
        labels.extend(labels_with_source(path, source))
    return labels


def text_input(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Query: {row.get('query') or row.get('query_id')}",
            f"Title: {row.get('title') or ''}",
            f"Abstract: {row.get('abstract') or ''}",
        ]
    )


def numeric_matrix(rows: list[dict[str, Any]], scored_by_paper: dict[int, dict[str, float]], features: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        paper_id = int(row["paper_id"])
        merged = dict(row)
        merged.update(scored_by_paper[paper_id])
        matrix.append([numeric_feature_value(merged, feature) for feature in features])
    return np.array(matrix, dtype=float)


def score_v43_model(
    *,
    model_payload: dict[str, Any],
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    scored: dict[str, dict[int, dict[str, float]]],
) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {}
    for query_id, rows in sorted(production_rows_by_q.items()):
        texts = [text_input(row) for row in rows]
        x_text = model_payload["vectorizer"].transform(texts)
        if model_payload["uses_ranker_features"]:
            x_numeric = sparse.csr_matrix(model_payload["scaler"].transform(numeric_matrix(rows, scored[query_id], model_payload["numeric_ranker_features"])))
            x_values = sparse.hstack([x_text, x_numeric], format="csr")
        else:
            x_values = x_text
        estimator = model_payload["estimator"]
        if model_payload["model_type"] == "tfidf_logistic_positive":
            predictions = estimator.predict_proba(x_values)[:, 1]
        else:
            predictions = estimator.predict(x_values)
        output[query_id] = {
            int(row["paper_id"]): float(score)
            for row, score in zip(rows, predictions, strict=True)
        }
    return output


def load_v43_models(model_dir: Path) -> list[dict[str, Any]]:
    index = load_json(model_dir / "index.json")
    models = []
    for row in index["models"]:
        payload = joblib.load(row["artifact_path"])
        metadata = load_json(Path(row["metadata_path"]))
        payload["metadata"] = metadata
        models.append(payload)
    return models


def add_v43_blends(method_scores: dict[str, dict[str, dict[int, float]]], best_v41_method: str, best_v43_method: str) -> dict[str, dict[str, dict[int, float]]]:
    output = dict(method_scores)
    output[V41_BLEND_METHOD] = weighted_blend(output, {V33_METHOD: 0.75, best_v41_method: 0.25})
    for spec in V43_BLEND_SPECS:
        weights = {}
        for method, weight in spec["weights"].items():
            if method == "__v41__":
                weights[best_v41_method] = weight
            elif method == "__v43__":
                weights[best_v43_method] = weight
            else:
                weights[method] = weight
        output[spec["name"]] = weighted_blend(output, weights)
    return output


def choose_best_v43_method(training_report: dict[str, Any], models: list[dict[str, Any]]) -> str:
    selected = training_report.get("best_model_by_loto_reading_ndcg", {}).get("score_column")
    available = {model["metadata"]["score_column"] for model in models}
    if selected in available:
        return str(selected)
    return sorted(available)[0]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
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
    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=packet_rows, top_k=args.top_k)
    scored = score_rows(
        candidate_rows_by_q=production_rows_by_q,
        context_rows_by_q=production_rows_by_q,
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
    )
    method_scores, model_diagnostics = build_method_scores(
        production_rows_by_q=production_rows_by_q,
        scored=scored,
        model_dir=resolve_repo_path(args.v33_model_dir),
    )
    v40_payload = load_model(resolve_repo_path(args.v40_model_dir), "random_forest_no_v27_compact")
    method_scores[V40_METHOD] = score_v4_model(model_payload=v40_payload, production_rows_by_q=production_rows_by_q, scored=scored)
    best_strategy = best_v41_strategy(resolve_repo_path(args.weighted_report))
    best_v41_method = method_name(best_strategy)
    method_scores[best_v41_method] = score_v41_artifact(
        model_dir=resolve_repo_path(args.v41_model_dir),
        strategy=best_strategy,
        production_rows_by_q=production_rows_by_q,
        scored=scored,
    )

    v43_models = load_v43_models(resolve_repo_path(args.v43_model_dir))
    for model in v43_models:
        score_name = model["metadata"]["score_column"]
        method_scores[score_name] = score_v43_model(model_payload=model, production_rows_by_q=production_rows_by_q, scored=scored)
        model_diagnostics[score_name] = {
            "metadata": model["metadata"],
            "uses_ranker_features": bool(model["uses_ranker_features"]),
        }
    training_report = load_json(resolve_repo_path(args.v43_training_report))
    best_v43_method = choose_best_v43_method(training_report, v43_models)
    method_scores = add_v43_blends(method_scores, best_v41_method, best_v43_method)
    evaluation = evaluate_rankings(candidate_rows_by_q=production_rows_by_q, labels=labels, method_scores=method_scores, k=K)
    averages = evaluation["method_averages"]
    comparisons = {
        "best_v43_method": best_v43_method,
        "best_v41_weighted_method": best_v41_method,
        "v4_1_blend_method": V41_BLEND_METHOD,
        "best_v43_vs_v3_3_reading_delta": averages[best_v43_method]["reading_value_ndcg"] - averages[V33_METHOD]["reading_value_ndcg"],
        "best_v43_vs_v3_3_topic_delta": averages[best_v43_method]["topic_match_ndcg"] - averages[V33_METHOD]["topic_match_ndcg"],
        "best_v43_vs_v4_1_blend_reading_delta": averages[best_v43_method]["reading_value_ndcg"] - averages[V41_BLEND_METHOD]["reading_value_ndcg"],
        "best_v43_vs_v4_1_blend_topic_delta": averages[best_v43_method]["topic_match_ndcg"] - averages[V41_BLEND_METHOD]["topic_match_ndcg"],
        "best_v43_reduces_hard_negatives_vs_v3_3": averages[best_v43_method]["hard_negative_rate_at_10"] <= averages[V33_METHOD]["hard_negative_rate_at_10"],
        "recommend_runtime_integration": False,
        "recommendation_reason": "V4.3 is an offline text-reranker experiment; keep default runtime and existing opt-in methods unchanged until results are reviewed.",
    }
    return {
        "schema_version": "v4.3_text_reranker_production_pool_report",
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
            "candidate_pool_sizes_by_topic": {query_id: len(rows) for query_id, rows in production_rows_by_q.items()},
        },
        "dataset": {
            "label_count": len(labels),
            "labels_by_source": {
                "v2_1": sum(1 for row in labels if row.get("judged_source") == "v2_1"),
                "v2_5": sum(1 for row in labels if row.get("judged_source") == "v2_5"),
                "v3_2": sum(1 for row in labels if row.get("judged_source") == "v3_2"),
                "v3_5": sum(1 for row in labels if row.get("judged_source") == "v3_5"),
                "v3_9": sum(1 for row in labels if row.get("judged_source") == "v3_9"),
            },
        },
        "production_diagnostics": production_diagnostics,
        "model_diagnostics": model_diagnostics,
        "evaluation": evaluation,
        "comparisons": comparisons,
        "weak_topic_deltas": {
            "best_v43_vs_v3_3": weak_topic_deltas(evaluation, best_v43_method, V33_METHOD),
            "best_v43_vs_v4_1_blend": weak_topic_deltas(evaluation, best_v43_method, V41_BLEND_METHOD),
        },
        "top10_overlap": {
            "best_v43_vs_v3_3": top10_overlap(evaluation["per_topic"], best_v43_method, V33_METHOD),
            "best_v43_vs_v4_1_blend": top10_overlap(evaluation["per_topic"], best_v43_method, V41_BLEND_METHOD),
        },
        "protected_hashes": protected_hashes(
            resolve_repo_path(args.v21_labels),
            resolve_repo_path(args.v25_labels),
            resolve_repo_path(args.v32_labels),
            resolve_repo_path(args.v35_labels),
            resolve_repo_path(args.selected_240),
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    averages = report["evaluation"]["method_averages"]
    key_methods = [
        "v2_7",
        V33_METHOD,
        V40_METHOD,
        report["comparisons"]["best_v41_weighted_method"],
        V41_BLEND_METHOD,
        report["comparisons"]["best_v43_method"],
        *[spec["name"] for spec in V43_BLEND_SPECS],
    ]
    lines = [
        "# V4.3 Text Reranker Production-Pool Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Best V4.3 text method: `{report['comparisons']['best_v43_method']}`",
        "- Runtime integration status: `offline_only_not_integrated`",
        "- Runtime/default/frontend behavior changed: `False`",
        "",
        "## Metrics",
        "",
        "| Method | Reading NDCG@10 | Topic NDCG@10 | Judged@10 | Unjudged@10 | Hard-neg@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in key_methods:
        if method not in averages:
            continue
        row = averages[method]
        lines.append(
            f"| `{method}` | `{row['reading_value_ndcg']:.3f}` | `{row['topic_match_ndcg']:.3f}` | "
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
    parser.add_argument("--v40-model-dir", default=str(DEFAULT_V40_MODEL_DIR))
    parser.add_argument("--v41-model-dir", default=str(DEFAULT_V41_MODEL_DIR))
    parser.add_argument("--weighted-report", default=str(DEFAULT_WEIGHTED_REPORT))
    parser.add_argument("--v43-model-dir", default=str(DEFAULT_V43_MODEL_DIR))
    parser.add_argument("--v43-training-report", default=str(REPO_ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_training_report.json"))
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
    print(f"Wrote V4.3 production-pool report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
