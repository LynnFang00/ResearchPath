import argparse
from collections import Counter
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
from app.services.ltr_model_service import RAW_SCORE_FEATURES, load_v3_3_ltr_model  # noqa: E402
from app.services.recommendation_service import (  # noqa: E402
    build_v3_3_ltr_retriever,
    normalize_method,
    recommend_from_query,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V21_PACKET,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes,
)
from train_v2_6_production_aware_learned_hybrid import production_candidate_rows  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import load_jsonl, write_json, write_text  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_7b_runtime_v3_3_smoke_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_7b_runtime_v3_3_smoke_report.md"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
REPRESENTATIVE_QUERIES = [
    ("v2_transformer_architecture", "transformer architecture"),
    ("v2_retrieval_augmented_generation", "retrieval augmented generation"),
    ("v2_graph_neural_networks", "graph neural networks"),
    ("v2_large_language_model_agents", "large language model agents"),
    ("v2_ai_for_scientific_discovery", "ai for scientific discovery"),
]


def finite_scores(rows: list[Any]) -> bool:
    return all(math.isfinite(float(row.score)) for row in rows)


def run_smoke(*, fixed_query_id: str, fixed_query: str, k: int, top_k: int) -> dict[str, Any]:
    model = load_v3_3_ltr_model(DEFAULT_MODEL_DIR)
    with SessionLocal() as db:
        paper_count = int(db.query(Paper).count())
        default_results = recommend_from_query(db=db, query=fixed_query, k=3)
        opt_in_results = recommend_from_query(db=db, query=fixed_query, k=3, method="v3_3_ltr")
        retriever = build_v3_3_ltr_retriever(list(db.query(Paper).order_by(Paper.id).all()))
        query_reports = []
        first_run_for_determinism: list[int] | None = None
        second_run_for_determinism: list[int] | None = None
        fixed_debug: dict[str, Any] | None = None
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
                }
            )

    production_rows_by_q, _diagnostics = production_candidate_rows(packet_rows=load_jsonl(DEFAULT_V21_PACKET), top_k=top_k)
    offline_ids = {int(row["paper_id"]) for row in production_rows_by_q[fixed_query_id]}
    live_ids = set(int(paper_id) for paper_id in (fixed_debug or {}).get("candidate_pool_ids", []))
    feature_columns = model.feature_columns
    report = {
        "schema_version": "v3.7b_runtime_v3_3_smoke_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "database_paper_count": paper_count,
        "default_method_expected": "bm25",
        "default_method_observed": default_results[0].method if default_results else None,
        "default_behavior_unchanged": bool(default_results and all(row.method == "bm25" for row in default_results)),
        "opt_in_method": "v3_3_ltr",
        "opt_in_method_observed": opt_in_results[0].method if opt_in_results else None,
        "opt_in_method_returns_v3_3_ltr": bool(opt_in_results and all(row.method == "v3_3_ltr" for row in opt_in_results)),
        "normalized_method": normalize_method("v3_3_ltr"),
        "artifact": {
            "model_name": model.model_name,
            "score_column": model.score_column,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "uses_v2_7_feature": "v2_7_score" in feature_columns or model.metadata.get("uses_v2_7_feature") is True,
            "raw_score_feature_intersection": sorted(RAW_SCORE_FEATURES & set(feature_columns)),
        },
        "representative_queries": query_reports,
        "fixed_query_parity": {
            "query_id": fixed_query_id,
            "query": fixed_query,
            "offline_candidate_count": len(offline_ids),
            "live_candidate_count": len(live_ids),
            "overlap_count": len(offline_ids & live_ids),
            "overlap_jaccard": len(offline_ids & live_ids) / len(offline_ids | live_ids) if offline_ids or live_ids else 0.0,
            "feature_columns_match_artifact": bool(
                fixed_debug
                and fixed_debug.get("model", {}).get("feature_columns") == feature_columns
            ),
            "deterministic_top_ids": first_run_for_determinism == second_run_for_determinism,
            "first_run_top_ids": first_run_for_determinism or [],
            "second_run_top_ids": second_run_for_determinism or [],
        },
        "overall_status": "pass",
        "protected_hashes": protected_hashes(DEFAULT_V21_LABELS, DEFAULT_V25_LABELS, DEFAULT_V32_LABELS, DEFAULT_V35_LABELS, DEFAULT_SELECTED_240),
    }
    status_checks = [
        report["default_behavior_unchanged"],
        report["opt_in_method_returns_v3_3_ltr"],
        not report["artifact"]["uses_v2_7_feature"],
        not report["artifact"]["raw_score_feature_intersection"],
        all(row["finite_scores"] for row in query_reports),
        all(row["returned_sorted_desc"] for row in query_reports),
        report["fixed_query_parity"]["feature_columns_match_artifact"],
        report["fixed_query_parity"]["deterministic_top_ids"],
        all(report["protected_hashes"].values()),
    ]
    report["overall_status"] = "pass" if all(status_checks) else "fail"
    return report


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3.7b Runtime V3.3 Smoke/Parity Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Database paper count: `{report['database_paper_count']}`",
        f"- Default behavior unchanged: `{report['default_behavior_unchanged']}`",
        f"- Opt-in method: `{report['opt_in_method']}`",
        f"- Opt-in method observed: `{report['opt_in_method_observed']}`",
        f"- Artifact model: `{report['artifact']['model_name']}`",
        f"- Feature count: `{report['artifact']['feature_count']}`",
        f"- Uses V2.7 feature: `{report['artifact']['uses_v2_7_feature']}`",
        f"- Raw score feature intersection: `{report['artifact']['raw_score_feature_intersection']}`",
        "",
        "## Fixed Query Parity",
        "",
        f"- Query: `{report['fixed_query_parity']['query']}`",
        f"- Offline candidate count: `{report['fixed_query_parity']['offline_candidate_count']}`",
        f"- Live candidate count: `{report['fixed_query_parity']['live_candidate_count']}`",
        f"- Overlap count: `{report['fixed_query_parity']['overlap_count']}`",
        f"- Overlap Jaccard: `{report['fixed_query_parity']['overlap_jaccard']:.3f}`",
        f"- Feature columns match artifact: `{report['fixed_query_parity']['feature_columns_match_artifact']}`",
        f"- Deterministic top IDs: `{report['fixed_query_parity']['deterministic_top_ids']}`",
        "",
        "## Representative Queries",
        "",
        "| query | pool | top IDs | finite | sorted |",
        "|---|---:|---|---:|---:|",
    ]
    for row in report["representative_queries"]:
        lines.append(
            f"| `{row['query']}` | {row['candidate_pool_size']} | `{row['top_ids']}` | "
            f"`{row['finite_scores']}` | `{row['returned_sorted_desc']}` |"
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
    parser = argparse.ArgumentParser(description="Run V3.7b live backend smoke/parity checks for method=v3_3_ltr.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--fixed-query-id", default="v2_graph_neural_networks")
    parser.add_argument("--fixed-query", default="graph neural networks")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_smoke(fixed_query_id=args.fixed_query_id, fixed_query=args.fixed_query, k=args.k, top_k=args.top_k)
    json_out = Path(args.json_out)
    if not json_out.is_absolute():
        json_out = REPO_ROOT / json_out
    md_out = Path(args.md_out)
    if not md_out.is_absolute():
        md_out = REPO_ROOT / md_out
    report["command"] = " ".join(sys.argv)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    print(f"V3.7b smoke status: {report['overall_status']}")
    print(f"Default unchanged: {report['default_behavior_unchanged']}")
    print(f"Opt-in observed: {report['opt_in_method_observed']}")
    print(f"Fixed-query overlap Jaccard: {report['fixed_query_parity']['overlap_jaccard']:.3f}")
    print(f"Report: {json_out}")


if __name__ == "__main__":
    main()
