import argparse
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.db.session import SessionLocal  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.services.ltr_model_service import load_v4_1_blend_model  # noqa: E402
from app.services.recommendation_service import (  # noqa: E402
    build_v4_1_blend_retriever,
    normalize_method,
    recommend_from_query,
)
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import DEFAULT_V21_PACKET, load_jsonl, production_candidate_rows  # noqa: E402
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes,
)


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_2_runtime_v4_1_blend_smoke_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_2_runtime_v4_1_blend_smoke_report.md"
DEFAULT_V33_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
DEFAULT_V41_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"
DEFAULT_BLEND_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"
REPRESENTATIVE_QUERIES = [
    ("v2_graph_neural_networks", "graph neural networks"),
    ("v2_large_language_model_agents", "large language model agents"),
    ("v2_retrieval_augmented_generation", "retrieval augmented generation"),
]


def finite_scores(rows: list[Any]) -> bool:
    return all(math.isfinite(float(row.score)) for row in rows)


def formula_checks(debug: dict[str, Any], *, tolerance: float) -> dict[str, Any]:
    failures = []
    max_abs_delta = 0.0
    for paper_id, values in debug.get("scores", {}).items():
        expected = 0.75 * float(values["v3_3_score"]) + 0.25 * float(values["v4_1_weighted_score"])
        observed = float(values["v4_1_blend_score"])
        delta = abs(expected - observed)
        max_abs_delta = max(max_abs_delta, delta)
        if delta > tolerance:
            failures.append({"paper_id": int(paper_id), "expected": expected, "observed": observed, "delta": delta})
    return {
        "checked_count": len(debug.get("scores", {})),
        "max_abs_delta": max_abs_delta,
        "within_tolerance": not failures,
        "failures": failures[:5],
    }


def run_smoke(*, fixed_query_id: str, fixed_query: str, k: int, top_k: int, tolerance: float) -> dict[str, Any]:
    blend_model = load_v4_1_blend_model(
        v33_model_dir=DEFAULT_V33_MODEL_DIR,
        v41_model_dir=DEFAULT_V41_MODEL_DIR,
        blend_config_path=DEFAULT_BLEND_CONFIG,
    )
    with SessionLocal() as db:
        paper_count = int(db.query(Paper).count())
        default_results = recommend_from_query(db=db, query=fixed_query, k=3)
        opt_in_results = recommend_from_query(db=db, query=fixed_query, k=3, method="v4_1_blend")
        papers = list(db.query(Paper).order_by(Paper.id).all())
        retriever = build_v4_1_blend_retriever(papers)
        query_reports = []
        fixed_debug: dict[str, Any] | None = None
        first_run_for_determinism: list[int] | None = None
        second_run_for_determinism: list[int] | None = None
        for query_id, query in REPRESENTATIVE_QUERIES:
            debug = retriever.score_query(query=query, top_k=top_k)
            results_first = retriever.search(query=query, k=k)
            results_second = retriever.search(query=query, k=k)
            if query_id == fixed_query_id:
                fixed_debug = debug
                first_run_for_determinism = [result.document_id for result in results_first]
                second_run_for_determinism = [result.document_id for result in results_second]
            query_reports.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "candidate_pool_size": len(debug["candidate_pool_ids"]),
                    "top_ids": [result.document_id for result in results_first],
                    "top_scores": [float(result.score) for result in results_first],
                    "finite_scores": finite_scores(results_first),
                    "method_name": retriever.method_name,
                    "returned_sorted_desc": [float(result.score) for result in results_first]
                    == sorted([float(result.score) for result in results_first], reverse=True),
                    "formula": formula_checks(debug, tolerance=tolerance),
                }
            )

    production_rows_by_q, _diagnostics = production_candidate_rows(packet_rows=load_jsonl(DEFAULT_V21_PACKET), top_k=top_k)
    offline_ids = {int(row["paper_id"]) for row in production_rows_by_q[fixed_query_id]}
    live_ids = set(int(paper_id) for paper_id in (fixed_debug or {}).get("candidate_pool_ids", []))
    model_diagnostics = (fixed_debug or {}).get("model", {})
    report = {
        "schema_version": "v4.2_runtime_v4_1_blend_smoke_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "database_paper_count": paper_count,
        "database_count_expected_around": 50424,
        "database_count_within_expected_window": 49000 <= paper_count <= 51000,
        "default_method_expected": "bm25",
        "default_method_observed": default_results[0].method if default_results else None,
        "default_behavior_unchanged": bool(default_results and all(row.method == "bm25" for row in default_results)),
        "opt_in_method": "v4_1_blend",
        "opt_in_method_observed": opt_in_results[0].method if opt_in_results else None,
        "opt_in_method_returns_v4_1_blend": bool(opt_in_results and all(row.method == "v4_1_blend" for row in opt_in_results)),
        "normalized_method": normalize_method("v4_1_blend"),
        "artifact": {
            "blend_config_path": str(DEFAULT_BLEND_CONFIG),
            "blend_name": blend_model.blend_name,
            "v3_3_weight": blend_model.v33_weight,
            "v4_1_weight": blend_model.v41_weight,
            "v3_3_model_name": blend_model.v33_model.model_name,
            "v4_1_weighted_model_name": blend_model.v41_model.model_name,
            "v3_3_feature_columns": blend_model.v33_model.feature_columns,
            "v4_1_feature_columns": blend_model.v41_model.feature_columns,
            "feature_orders_match": blend_model.v33_model.feature_columns == blend_model.v41_model.feature_columns,
        },
        "representative_queries": query_reports,
        "fixed_query_parity": {
            "query_id": fixed_query_id,
            "query": fixed_query,
            "offline_candidate_count": len(offline_ids),
            "live_candidate_count": len(live_ids),
            "overlap_count": len(offline_ids & live_ids),
            "overlap_jaccard": len(offline_ids & live_ids) / len(offline_ids | live_ids) if offline_ids or live_ids else 0.0,
            "feature_columns_match_artifacts": bool(
                model_diagnostics.get("feature_columns") == blend_model.v41_model.feature_columns
                and blend_model.v33_model.feature_columns == blend_model.v41_model.feature_columns
            ),
            "deterministic_top_ids": first_run_for_determinism == second_run_for_determinism,
            "first_run_top_ids": first_run_for_determinism or [],
            "second_run_top_ids": second_run_for_determinism or [],
        },
        "protected_hashes": protected_hashes(DEFAULT_V21_LABELS, DEFAULT_V25_LABELS, DEFAULT_V32_LABELS, DEFAULT_V35_LABELS, DEFAULT_SELECTED_240),
    }
    status_checks = [
        report["database_count_within_expected_window"],
        report["default_behavior_unchanged"],
        report["opt_in_method_returns_v4_1_blend"],
        report["artifact"]["feature_orders_match"],
        all(row["finite_scores"] for row in query_reports),
        all(row["returned_sorted_desc"] for row in query_reports),
        all(row["formula"]["within_tolerance"] for row in query_reports),
        report["fixed_query_parity"]["feature_columns_match_artifacts"],
        report["fixed_query_parity"]["deterministic_top_ids"],
        all(report["protected_hashes"].values()),
    ]
    report["overall_status"] = "pass" if all(status_checks) else "fail"
    return report


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V4.2 Runtime V4.1 Blend Smoke/Parity Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Database paper count: `{report['database_paper_count']}`",
        f"- Default behavior unchanged: `{report['default_behavior_unchanged']}`",
        f"- Opt-in method observed: `{report['opt_in_method_observed']}`",
        f"- Blend: `{report['artifact']['blend_name']}`",
        f"- Formula: `0.75 * V3.3 + 0.25 * V4.1 weighted`",
        f"- Feature orders match: `{report['artifact']['feature_orders_match']}`",
        "",
        "## Fixed Query Parity",
        "",
        f"- Query: `{report['fixed_query_parity']['query']}`",
        f"- Offline candidate count: `{report['fixed_query_parity']['offline_candidate_count']}`",
        f"- Live candidate count: `{report['fixed_query_parity']['live_candidate_count']}`",
        f"- Overlap count: `{report['fixed_query_parity']['overlap_count']}`",
        f"- Overlap Jaccard: `{report['fixed_query_parity']['overlap_jaccard']:.3f}`",
        f"- Feature columns match artifacts: `{report['fixed_query_parity']['feature_columns_match_artifacts']}`",
        f"- Deterministic top IDs: `{report['fixed_query_parity']['deterministic_top_ids']}`",
        "",
        "## Representative Queries",
        "",
        "| query | pool | top IDs | finite | sorted | formula ok |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in report["representative_queries"]:
        lines.append(
            f"| `{row['query']}` | {row['candidate_pool_size']} | `{row['top_ids']}` | "
            f"`{row['finite_scores']}` | `{row['returned_sorted_desc']}` | `{row['formula']['within_tolerance']}` |"
        )
    lines.extend(
        [
            "",
            "## Protected Hashes",
            "",
            f"- V2.1 labels unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
            f"- V3.2 labels unchanged: `{report['protected_hashes']['v3_2_labels_hash_unchanged']}`",
            f"- V3.5 labels unchanged: `{report['protected_hashes']['v3_5_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4.2 live backend smoke/parity checks for method=v4_1_blend.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--fixed-query-id", default="v2_graph_neural_networks")
    parser.add_argument("--fixed-query", default="graph neural networks")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_smoke(fixed_query_id=args.fixed_query_id, fixed_query=args.fixed_query, k=args.k, top_k=args.top_k, tolerance=args.tolerance)
    json_out = Path(args.json_out)
    if not json_out.is_absolute():
        json_out = REPO_ROOT / json_out
    md_out = Path(args.md_out)
    if not md_out.is_absolute():
        md_out = REPO_ROOT / md_out
    report["command"] = " ".join(sys.argv)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    print(f"V4.2 smoke status: {report['overall_status']}")
    print(f"Default unchanged: {report['default_behavior_unchanged']}")
    print(f"Opt-in observed: {report['opt_in_method_observed']}")
    print(f"Fixed-query overlap Jaccard: {report['fixed_query_parity']['overlap_jaccard']:.3f}")
    print(f"Blend formula ok: {all(row['formula']['within_tolerance'] for row in report['representative_queries'])}")
    print(f"Report: {json_out}")


if __name__ == "__main__":
    main()
