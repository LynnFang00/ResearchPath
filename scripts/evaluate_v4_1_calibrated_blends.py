import argparse
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
from train_v3_0_ltr_models import numeric_feature_value  # noqa: E402
from train_v4_1_weighted_ltr_models_2400 import (  # noqa: E402
    DEFAULT_JSON_OUT as DEFAULT_WEIGHTED_REPORT,
    DEFAULT_MODEL_DIR as DEFAULT_V41_MODEL_DIR,
    FEATURE_COLUMNS,
    method_name,
)
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


V40_METHOD = "v4_0_random_forest_no_v27_compact"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_calibrated_blend_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_calibrated_blend_report.md"
DEFAULT_CONFIG_OUT = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"


BLEND_SPECS = [
    {"name": "blend_0_75_v33_0_25_v40", "weights": {V33_METHOD: 0.75, V40_METHOD: 0.25}},
    {"name": "blend_0_50_v33_0_50_v40", "weights": {V33_METHOD: 0.50, V40_METHOD: 0.50}},
    {"name": "blend_0_75_v33_0_25_v41", "weights": {V33_METHOD: 0.75, "__v41__": 0.25}},
    {"name": "blend_0_60_v33_0_30_v41_0_10_hybrid", "weights": {V33_METHOD: 0.60, "__v41__": 0.30, "hybrid": 0.10}},
    {"name": "blend_0_70_v33_0_20_v40_0_10_v26", "weights": {V33_METHOD: 0.70, V40_METHOD: 0.20, "v2_6": 0.10}},
    {"name": "hard_negative_guardrail_v33_with_v40_demotion", "guardrail": "v33_with_v40_low_score_demotion"},
]


def load_all_labels(paths_by_source: dict[str, Path]) -> list[dict[str, Any]]:
    labels = []
    for source, path in paths_by_source.items():
        labels.extend(labels_with_source(path, source))
    return labels


def best_v41_strategy(weighted_report_path: Path) -> str:
    report = load_json(weighted_report_path)
    selected = report.get("selection", {}).get("selected")
    if selected:
        return str(selected["strategy"])
    return str(report["selection"]["ranked_candidates"][0]["strategy"])


def score_v41_artifact(
    *,
    model_dir: Path,
    strategy: str,
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    scored: dict[str, dict[int, dict[str, float]]],
) -> dict[str, dict[int, float]]:
    estimator = joblib.load(model_dir / f"{strategy}.joblib")
    output: dict[str, dict[int, float]] = {}
    for query_id, rows in sorted(production_rows_by_q.items()):
        paper_ids = [int(row["paper_id"]) for row in rows]
        feature_rows = []
        for row in rows:
            paper_id = int(row["paper_id"])
            merged = dict(row)
            merged.update(scored[query_id][paper_id])
            feature_rows.append([numeric_feature_value(merged, feature) for feature in FEATURE_COLUMNS])
        predictions = estimator.predict(np.array(feature_rows, dtype=float))
        output[query_id] = {paper_id: float(score) for paper_id, score in zip(paper_ids, predictions, strict=True)}
    return output


def normalize_query_scores(scores: dict[int, float]) -> dict[int, float]:
    values = list(scores.values())
    if not values:
        return {}
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return {paper_id: 0.5 for paper_id in scores}
    return {paper_id: (score - low) / (high - low) for paper_id, score in scores.items()}


def weighted_blend(method_scores: dict[str, dict[str, dict[int, float]]], weights: dict[str, float]) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {}
    query_ids = sorted(next(iter(method_scores.values())).keys())
    for query_id in query_ids:
        normalized = {method: normalize_query_scores(method_scores[method][query_id]) for method in weights}
        paper_ids = set().union(*(set(scores) for scores in normalized.values()))
        output[query_id] = {
            paper_id: sum(weight * normalized[method].get(paper_id, 0.0) for method, weight in weights.items())
            for paper_id in paper_ids
        }
    return output


def guardrail_blend(method_scores: dict[str, dict[str, dict[int, float]]]) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {}
    for query_id, v33_scores in method_scores[V33_METHOD].items():
        v33_norm = normalize_query_scores(v33_scores)
        v40_norm = normalize_query_scores(method_scores[V40_METHOD][query_id])
        output[query_id] = {}
        for paper_id, v33_score in v33_norm.items():
            demotion = 0.20 if v33_score >= 0.70 and v40_norm.get(paper_id, 0.0) <= 0.25 else 0.0
            output[query_id][paper_id] = v33_score - demotion
    return output


def add_blends(method_scores: dict[str, dict[str, dict[int, float]]], best_v41_method: str) -> dict[str, dict[str, dict[int, float]]]:
    output = dict(method_scores)
    for spec in BLEND_SPECS:
        if spec.get("guardrail"):
            output[spec["name"]] = guardrail_blend(output)
            continue
        weights = {best_v41_method if method == "__v41__" else method: weight for method, weight in spec["weights"].items()}
        output[spec["name"]] = weighted_blend(output, weights)
    return output


def choose_best_blend(evaluation: dict[str, Any]) -> dict[str, Any]:
    averages = evaluation["method_averages"]
    v33 = averages[V33_METHOD]
    rows = []
    for spec in BLEND_SPECS:
        method = spec["name"]
        row = averages[method]
        deltas = weak_topic_deltas(evaluation, method, V33_METHOD)
        severe_losses = [delta for delta in deltas if delta["reading_delta"] < -0.10]
        eligible = (
            row["reading_value_ndcg"] >= v33["reading_value_ndcg"] - 0.005
            and row["hard_negative_rate_at_10"] <= v33["hard_negative_rate_at_10"]
            and row["judged_at_10"] >= 9.75
            and row["unjudged_at_10"] <= 0.25
            and len(severe_losses) <= 2
        )
        rows.append({"method": method, "eligible_for_replacement": eligible, "reading_delta_vs_v3_3": row["reading_value_ndcg"] - v33["reading_value_ndcg"], "topic_delta_vs_v3_3": row["topic_match_ndcg"] - v33["topic_match_ndcg"], "hard_negative_delta_vs_v3_3": row["hard_negative_rate_at_10"] - v33["hard_negative_rate_at_10"], "severe_weak_topic_losses": severe_losses})
    ranked = sorted(rows, key=lambda item: (item["eligible_for_replacement"], item["reading_delta_vs_v3_3"], item["topic_delta_vs_v3_3"], -len(item["severe_weak_topic_losses"])), reverse=True)
    return {"selected": ranked[0] if ranked and ranked[0]["eligible_for_replacement"] else None, "ranked_blends": ranked}


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    labels = load_all_labels({"v2_1": resolve_repo_path(args.v21_labels), "v2_5": resolve_repo_path(args.v25_labels), "v3_2": resolve_repo_path(args.v32_labels), "v3_5": resolve_repo_path(args.v35_labels), "v3_9": resolve_repo_path(args.v39_labels)})
    packet_rows = load_jsonl(resolve_repo_path(args.v21_packet))
    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=packet_rows, top_k=args.top_k)
    scored = score_rows(candidate_rows_by_q=production_rows_by_q, context_rows_by_q=production_rows_by_q, old_model_path=resolve_repo_path(args.old_model), v26_model_path=resolve_repo_path(args.v26_model))
    method_scores, model_diagnostics = build_method_scores(production_rows_by_q=production_rows_by_q, scored=scored, model_dir=resolve_repo_path(args.v33_model_dir))
    v40_payload = load_model(resolve_repo_path(args.v40_model_dir), "random_forest_no_v27_compact")
    method_scores[V40_METHOD] = score_v4_model(model_payload=v40_payload, production_rows_by_q=production_rows_by_q, scored=scored)
    best_strategy = best_v41_strategy(resolve_repo_path(args.weighted_report))
    best_v41_method = method_name(best_strategy)
    method_scores[best_v41_method] = score_v41_artifact(model_dir=resolve_repo_path(args.v41_model_dir), strategy=best_strategy, production_rows_by_q=production_rows_by_q, scored=scored)
    model_diagnostics[V40_METHOD] = {"metadata_path": str(resolve_repo_path(args.v40_model_dir) / "random_forest_no_v27_compact.metadata.json")}
    model_diagnostics[best_v41_method] = {"metadata_path": str(resolve_repo_path(args.v41_model_dir) / f"{best_strategy}.metadata.json"), "feature_columns": FEATURE_COLUMNS}
    method_scores = add_blends(method_scores, best_v41_method)
    evaluation = evaluate_rankings(candidate_rows_by_q=production_rows_by_q, labels=labels, method_scores=method_scores, k=K)
    selection = choose_best_blend(evaluation)
    configured = selection["selected"] or selection["ranked_blends"][0]
    configured_method = configured["method"]
    configured_spec = next(spec for spec in BLEND_SPECS if spec["name"] == configured_method)
    config = {
        "schema_version": "v4.1_calibrated_blend_config",
        "created_at": datetime.now(UTC).isoformat(),
        "method": configured_method,
        "eligible_for_replacement": bool(configured.get("eligible_for_replacement")),
        "spec": configured_spec,
        "best_v41_method": best_v41_method,
        "feature_order": FEATURE_COLUMNS,
        "runtime_integration_status": "offline_only_not_integrated",
    }
    report = {
        "schema_version": "v4.1_calibrated_blend_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "best_v41_weighted_strategy": best_strategy,
        "best_v41_weighted_method": best_v41_method,
        "blend_specs": BLEND_SPECS,
        "production_diagnostics": production_diagnostics,
        "model_diagnostics": model_diagnostics,
        "evaluation": evaluation,
        "selection": selection,
        "protected_hashes": protected_hashes(resolve_repo_path(args.v21_labels), resolve_repo_path(args.v25_labels), resolve_repo_path(args.v32_labels), resolve_repo_path(args.v35_labels), resolve_repo_path(args.selected_240)),
    }
    return report, config


def metric_table(averages: dict[str, dict[str, float]], methods: list[str]) -> list[str]:
    lines = ["| method | reading | topic | judged@10 | unjudged@10 | hard-neg | MRR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        row = averages[method]
        lines.append(f"| `{method}` | {row['reading_value_ndcg']:.3f} | {row['topic_match_ndcg']:.3f} | {row['judged_at_10']:.2f} | {row['unjudged_at_10']:.2f} | {row['hard_negative_rate_at_10']:.3f} | {row['mrr_at_10']:.3f} |")
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    averages = report["evaluation"]["method_averages"]
    methods = ["hybrid", "v2_6", "v2_7", V33_METHOD, V40_METHOD, report["best_v41_weighted_method"]] + [spec["name"] for spec in BLEND_SPECS]
    selected = report["selection"]["selected"]
    lines = [
        "# V4.1 Calibrated Blend Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Best V4.1 weighted method: `{report['best_v41_weighted_method']}`",
        f"- Replacement-eligible blend: `{selected['method'] if selected else None}`",
        f"- Runtime integration: `{report['runtime_integration_status']}`",
        "",
        "## Production-Pool Metrics",
        "",
    ]
    lines.extend(metric_table(averages, methods))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4.1 calibrated score-level blends.")
    parser.add_argument("--weighted-report", default=str(DEFAULT_WEIGHTED_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--v41-model-dir", default=str(DEFAULT_V41_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v40-model-dir", default=str(DEFAULT_V40_MODEL_DIR.relative_to(REPO_ROOT)))
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
    report, config = build_report(args)
    report["command"] = " ".join(sys.argv)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    write_json(resolve_repo_path(args.config_out), config)
    print("V4.1 calibrated blend evaluation complete")
    print(f"Replacement-eligible blend: {report['selection']['selected']['method'] if report['selection']['selected'] else None}")
    print(f"Best ranked blend: {report['selection']['ranked_blends'][0]['method']}")
    print(f"Report: {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
