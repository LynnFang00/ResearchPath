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

from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_LABELS,
    DEFAULT_V21_PACKET,
    DEFAULT_V25_LABELS,
    DEFAULT_V26_MODEL,
    DEFAULT_V32_LABELS,
    DEFAULT_V33_MODEL_DIR,
    K,
    build_method_scores,
    evaluate_rankings,
    is_hard_negative,
    is_positive,
    labels_with_source,
    load_json,
    load_jsonl,
    production_candidate_rows,
    resolve_repo_path,
    score_rows,
    top10_overlap,
    write_json,
    write_text,
)
from validate_v3_5_unjudged_audit_labels import (  # noqa: E402
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    DEFAULT_SELECTED_240,
    file_sha256,
)


DEFAULT_V35_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_5_unjudged_audit.jsonl"
DEFAULT_V35_CANDIDATES = REPO_ROOT / "data" / "eval" / "labeling" / "v3_5_unjudged_top10_audit_candidates.jsonl"
DEFAULT_V34_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_production_pool_report.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_6_revealed_v3_5_labels_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_6_revealed_v3_5_labels_report.md"

EXPECTED_V35_LABELS_SHA256 = "55A593C3CFA3EC3148FCB9AA38F0081607065D2E506D7C057C9804A71B05113B"
V33_METHOD = "v3_3_random_forest_no_v27_compact"
METHODS_FOR_TABLE = [
    "v3_3_random_forest_no_v27_compact",
    "v3_3_hist_gradient_boosting_no_v27_compact",
    "v2_7",
    "v2_6",
    "old_v2_2b",
    "hybrid",
    "embedding",
    "faiss_embedding",
]
METRIC_FIELDS = [
    "reading_value_ndcg",
    "topic_match_ndcg",
    "judged_at_10",
    "unjudged_at_10",
    "hard_negative_rate_at_10",
    "positive_recall_at_10",
    "mrr_at_10",
    "role_coverage_at_10",
    "path_coverage_at_10",
]


def label_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["query_id"]), int(row["paper_id"]))


def label_bucket(row: dict[str, Any]) -> str:
    if is_positive(row):
        return "positive"
    if is_hard_negative(row):
        return "negative"
    return "partial"


def load_labels_with_sources(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v35_labels_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_source = {
        "v2_1": labels_with_source(v21_labels_path, "v2_1"),
        "v2_5": labels_with_source(v25_labels_path, "v2_5"),
        "v3_2": labels_with_source(v32_labels_path, "v3_2"),
        "v3_5": labels_with_source(v35_labels_path, "v3_5"),
    }
    combined = by_source["v2_1"] + by_source["v2_5"] + by_source["v3_2"] + by_source["v3_5"]
    return combined, by_source


def duplicate_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(label_key(row) for row in rows)
    return [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in sorted(counts.items())
        if count > 1
    ]


def v35_label_distribution(v35_labels: list[dict[str, Any]], v35_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_by_key = {label_key(row): row for row in v35_candidates}
    rows = []
    for label in v35_labels:
        key = label_key(label)
        candidate = candidate_by_key.get(key, {})
        reasons = list(candidate.get("diagnostic_reasons", []))
        rows.append(
            {
                "query_id": key[0],
                "paper_id": key[1],
                "title": label.get("title"),
                "bucket": label_bucket(label),
                "reading_value_score": label.get("reading_value_score"),
                "topic_match_score": label.get("topic_match_score"),
                "primary_role": label.get("primary_role"),
                "diagnostic_reasons": reasons,
                "is_v3_3_high_scoring_unjudged_top10": "v3_3_high_scoring_unjudged_top10" in reasons,
            }
        )
    high_top10 = [row for row in rows if row["is_v3_3_high_scoring_unjudged_top10"]]
    counts = Counter(row["bucket"] for row in rows)
    high_counts = Counter(row["bucket"] for row in high_top10)
    return {
        "label_count": len(rows),
        "count_by_bucket": {bucket: counts.get(bucket, 0) for bucket in ["positive", "partial", "negative"]},
        "count_by_topic": dict(Counter(row["query_id"] for row in rows)),
        "v3_3_high_scoring_unjudged_top10_count": len(high_top10),
        "v3_3_high_scoring_unjudged_top10_by_bucket": {
            bucket: high_counts.get(bucket, 0) for bucket in ["positive", "partial", "negative"]
        },
        "mostly_hidden_positives": high_counts.get("positive", 0) > (high_counts.get("partial", 0) + high_counts.get("negative", 0)),
        "rows": rows,
    }


def method_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, float]]:
    before_avg = before["evaluation"]["method_averages"]
    after_avg = after["evaluation"]["method_averages"]
    output: dict[str, dict[str, float]] = {}
    for method in sorted(set(before_avg) & set(after_avg)):
        output[method] = {
            field: float(after_avg[method][field]) - float(before_avg[method][field])
            for field in METRIC_FIELDS
        }
    return output


def per_topic_reveal_effects(before: dict[str, Any], after: dict[str, Any], method: str) -> list[dict[str, Any]]:
    before_rows = {row["query_id"]: row for row in before["evaluation"]["per_topic"][method]}
    after_rows = {row["query_id"]: row for row in after["evaluation"]["per_topic"][method]}
    rows = []
    for query_id in sorted(set(before_rows) & set(after_rows)):
        before_row = before_rows[query_id]
        after_row = after_rows[query_id]
        rows.append(
            {
                "query_id": query_id,
                "reading_value_ndcg_delta": after_row["reading_value_ndcg"] - before_row["reading_value_ndcg"],
                "topic_match_ndcg_delta": after_row["topic_match_ndcg"] - before_row["topic_match_ndcg"],
                "judged_at_10_delta": after_row["judged_at_10"] - before_row["judged_at_10"],
                "unjudged_at_10_delta": after_row["unjudged_at_10"] - before_row["unjudged_at_10"],
                "after_reading_value_ndcg": after_row["reading_value_ndcg"],
                "after_topic_match_ndcg": after_row["topic_match_ndcg"],
                "after_judged_at_10": after_row["judged_at_10"],
                "after_unjudged_at_10": after_row["unjudged_at_10"],
            }
        )
    return sorted(rows, key=lambda row: (row["judged_at_10_delta"], row["reading_value_ndcg_delta"]), reverse=True)


def protected_hashes(v21_labels: Path, v25_labels: Path, v32_labels: Path, v35_labels: Path, selected_240: Path) -> dict[str, Any]:
    v21_hash = file_sha256(v21_labels)
    v25_hash = file_sha256(v25_labels)
    v32_hash = file_sha256(v32_labels)
    v35_hash = file_sha256(v35_labels)
    selected_hash = file_sha256(selected_240)
    return {
        "v2_1_labels_sha256": v21_hash,
        "v2_1_labels_hash_unchanged": v21_hash == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": v25_hash,
        "v2_5_labels_hash_unchanged": v25_hash == EXPECTED_V25_LABELS_SHA256,
        "v3_2_labels_sha256": v32_hash,
        "v3_2_labels_hash_unchanged": v32_hash == EXPECTED_V32_LABELS_SHA256,
        "v3_5_labels_sha256": v35_hash,
        "v3_5_labels_hash_unchanged": v35_hash == EXPECTED_V35_LABELS_SHA256,
        "selected_240_sha256": selected_hash,
        "selected_240_hash_unchanged": selected_hash == EXPECTED_SELECTED_240_SHA256,
    }


def build_report(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v35_labels_path: Path,
    v35_candidates_path: Path,
    v21_packet_path: Path,
    old_model_path: Path,
    v26_model_path: Path,
    v33_model_dir: Path,
    v34_report_path: Path,
    selected_240_path: Path,
    top_k: int,
) -> dict[str, Any]:
    combined_labels, labels_by_source = load_labels_with_sources(
        v21_labels_path=v21_labels_path,
        v25_labels_path=v25_labels_path,
        v32_labels_path=v32_labels_path,
        v35_labels_path=v35_labels_path,
    )
    v35_candidates = load_jsonl(v35_candidates_path)
    v21_packet_rows = load_jsonl(v21_packet_path)
    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=v21_packet_rows, top_k=top_k)
    scored = score_rows(
        candidate_rows_by_q=production_rows_by_q,
        context_rows_by_q=production_rows_by_q,
        old_model_path=old_model_path,
        v26_model_path=v26_model_path,
    )
    method_scores, model_diagnostics = build_method_scores(
        production_rows_by_q=production_rows_by_q,
        scored=scored,
        model_dir=v33_model_dir,
    )
    evaluation = evaluate_rankings(
        candidate_rows_by_q=production_rows_by_q,
        labels=combined_labels,
        method_scores=method_scores,
        k=K,
    )
    v34_report = load_json(v34_report_path)
    avg = evaluation["method_averages"]
    candidate_pool_sizes = {query_id: len(rows) for query_id, rows in production_rows_by_q.items()}
    candidate_pool_duplicate_rows = []
    for query_id, rows in sorted(production_rows_by_q.items()):
        counts = Counter(int(row["paper_id"]) for row in rows)
        candidate_pool_duplicate_rows.extend(
            {"query_id": query_id, "paper_id": paper_id, "count": count}
            for paper_id, count in sorted(counts.items())
            if count > 1
        )
    report_shell = {"evaluation": evaluation}
    deltas = method_deltas(v34_report, report_shell)
    v35_distribution = v35_label_distribution(labels_by_source["v3_5"], v35_candidates)
    v33_after = avg[V33_METHOD]
    v27_after = avg["v2_7"]
    v26_after = avg["v2_6"]
    return {
        "schema_version": "v3.6_revealed_v3_5_labels_production_pool_evaluation",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_changed_by_this_script": False,
        "runtime_integration_status": "offline_only_not_integrated",
        "candidate_pool_definition": {
            "union": ["top_50_bm25", "top_50_tfidf", "top_50_embedding_exact_matrix", "top_50_faiss_embedding", "top_50_current_hybrid"],
            "deduplicate_by": "paper_id",
            "top_k_per_method": top_k,
            "candidate_pool_sizes_by_topic": candidate_pool_sizes,
            "mean_candidate_pool_size": sum(candidate_pool_sizes.values()) / len(candidate_pool_sizes),
            "duplicate_candidate_rows": candidate_pool_duplicate_rows,
        },
        "dataset": {
            "label_count": len(combined_labels),
            "labels_by_source": {source: len(rows) for source, rows in labels_by_source.items()},
            "topic_count": len({str(row["query_id"]) for row in combined_labels}),
            "duplicate_query_paper_rows": duplicate_keys(combined_labels),
        },
        "v3_5_label_distribution": v35_distribution,
        "production_diagnostics": production_diagnostics,
        "model_diagnostics": model_diagnostics,
        "evaluation": evaluation,
        "before_after": {
            "before_report": str(v34_report_path),
            "after_report": str(DEFAULT_JSON_OUT),
            "method_metric_deltas": deltas,
            "v3_3_unjudged_at_10_delta": deltas[V33_METHOD]["unjudged_at_10"],
            "v3_3_judged_at_10_delta": deltas[V33_METHOD]["judged_at_10"],
            "v3_3_reading_ndcg_delta": deltas[V33_METHOD]["reading_value_ndcg"],
            "v3_3_topic_ndcg_delta": deltas[V33_METHOD]["topic_match_ndcg"],
            "topics_benefiting_most_from_revealed_v3_5_labels": per_topic_reveal_effects(v34_report, report_shell, V33_METHOD)[:10],
        },
        "top10_overlap": {
            "v3_3_vs_v2_7": top10_overlap(evaluation["per_topic"], V33_METHOD, "v2_7"),
            "v3_3_vs_v2_6": top10_overlap(evaluation["per_topic"], V33_METHOD, "v2_6"),
        },
        "comparison": {
            "v3_3_method": V33_METHOD,
            "v3_3_beats_v2_7_reading_ndcg": v33_after["reading_value_ndcg"] > v27_after["reading_value_ndcg"],
            "v3_3_beats_v2_7_topic_ndcg": v33_after["topic_match_ndcg"] > v27_after["topic_match_ndcg"],
            "v3_3_beats_v2_6_reading_ndcg": v33_after["reading_value_ndcg"] > v26_after["reading_value_ndcg"],
            "v3_3_beats_v2_6_topic_ndcg": v33_after["topic_match_ndcg"] > v26_after["topic_match_ndcg"],
            "reading_delta_vs_v2_7": v33_after["reading_value_ndcg"] - v27_after["reading_value_ndcg"],
            "topic_delta_vs_v2_7": v33_after["topic_match_ndcg"] - v27_after["topic_match_ndcg"],
            "reading_delta_vs_v2_6": v33_after["reading_value_ndcg"] - v26_after["reading_value_ndcg"],
            "topic_delta_vs_v2_6": v33_after["topic_match_ndcg"] - v26_after["topic_match_ndcg"],
            "unjudged_delta_vs_v2_7": v33_after["unjudged_at_10"] - v27_after["unjudged_at_10"],
            "hard_negative_rate_delta_vs_v2_7": v33_after["hard_negative_rate_at_10"] - v27_after["hard_negative_rate_at_10"],
            "v3_5_unjudged_promotions_were_mostly_hidden_positives": v35_distribution["mostly_hidden_positives"],
            "safe_to_present_as_offline_production_pool_relevance_upgrade": (
                v33_after["reading_value_ndcg"] > v27_after["reading_value_ndcg"]
                and v33_after["topic_match_ndcg"] > v27_after["topic_match_ndcg"]
                and v35_distribution["mostly_hidden_positives"]
            ),
            "safe_for_opt_in_runtime_integration": False,
            "runtime_integration_blocker": "Still blocked on backend parity and live scoring work; V3.6 is offline revealed-label evaluation only.",
        },
        "protected_hashes": protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path, v35_labels_path, selected_240_path),
    }


def method_table(averages: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | reading NDCG@10 | topic NDCG@10 | judged@10 | unjudged@10 | hard-neg rate@10 | recall@10 | MRR@10 | role cov | path cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS_FOR_TABLE:
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['reading_value_ndcg']:.3f} | {row['topic_match_ndcg']:.3f} | "
            f"{row['judged_at_10']:.2f} | {row['unjudged_at_10']:.2f} | {row['hard_negative_rate_at_10']:.3f} | "
            f"{row['positive_recall_at_10']:.3f} | {row['mrr_at_10']:.3f} | {row['role_coverage_at_10']:.3f} | "
            f"{row['path_coverage_at_10']:.3f} |"
        )
    return lines


def delta_table(deltas: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | reading delta | topic delta | judged@10 delta | unjudged@10 delta | hard-neg delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS_FOR_TABLE:
        row = deltas[method]
        lines.append(
            f"| `{method}` | {row['reading_value_ndcg']:+.3f} | {row['topic_match_ndcg']:+.3f} | "
            f"{row['judged_at_10']:+.2f} | {row['unjudged_at_10']:+.2f} | {row['hard_negative_rate_at_10']:+.3f} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    avg = report["evaluation"]["method_averages"]
    deltas = report["before_after"]["method_metric_deltas"]
    comparison = report["comparison"]
    distribution = report["v3_5_label_distribution"]
    high_counts = distribution["v3_3_high_scoring_unjudged_top10_by_bucket"]
    lines = [
        "# V3.6 Revealed V3.5 Label Production-Pool Evaluation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "V3.6 reruns the frozen V3.3 production-pool comparison after adding the manually supplied V3.5 audit labels. It does not retrain models or change runtime defaults.",
        "",
        "## Dataset",
        "",
        f"- Judgments evaluated: `{report['dataset']['label_count']}`",
        f"- Labels by source: `{report['dataset']['labels_by_source']}`",
        f"- Duplicate query-paper rows: `{len(report['dataset']['duplicate_query_paper_rows'])}`",
        f"- Candidate pool duplicate rows: `{len(report['candidate_pool_definition']['duplicate_candidate_rows'])}`",
        "",
        "## V3.5 Label Distribution",
        "",
        f"- Overall: `{distribution['count_by_bucket']}`",
        f"- V3.3 high-scoring unjudged top-10: `{high_counts}`",
        f"- Mostly hidden positives: `{distribution['mostly_hidden_positives']}`",
        "",
        "## V3.6 Metrics",
        "",
    ]
    lines.extend(method_table(avg))
    lines.extend(
        [
            "",
            "## V3.4 to V3.6 Deltas",
            "",
        ]
    )
    lines.extend(delta_table(deltas))
    lines.extend(
        [
            "",
            "## V3.3 vs References",
            "",
            f"- Reading NDCG delta vs V2.7: `{comparison['reading_delta_vs_v2_7']:+.3f}`",
            f"- Topic NDCG delta vs V2.7: `{comparison['topic_delta_vs_v2_7']:+.3f}`",
            f"- Reading NDCG delta vs V2.6: `{comparison['reading_delta_vs_v2_6']:+.3f}`",
            f"- Topic NDCG delta vs V2.6: `{comparison['topic_delta_vs_v2_6']:+.3f}`",
            f"- Unjudged@10 delta vs V2.7: `{comparison['unjudged_delta_vs_v2_7']:+.2f}`",
            f"- V3.3/V2.7 mean top-10 overlap: `{report['top10_overlap']['v3_3_vs_v2_7']['mean_overlap_count']:.2f}`",
            f"- V3.3/V2.7 mean top-10 Jaccard: `{report['top10_overlap']['v3_3_vs_v2_7']['mean_jaccard']:.3f}`",
            f"- Offline production-pool relevance upgrade: `{comparison['safe_to_present_as_offline_production_pool_relevance_upgrade']}`",
            f"- Safe for opt-in runtime integration: `{comparison['safe_for_opt_in_runtime_integration']}`",
            f"- Runtime blocker: {comparison['runtime_integration_blocker']}",
            "",
            "## Topics Benefiting Most",
            "",
            "| topic | reading delta | topic delta | judged@10 delta | unjudged@10 delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["before_after"]["topics_benefiting_most_from_revealed_v3_5_labels"][:10]:
        lines.append(
            f"| `{row['query_id']}` | {row['reading_value_ndcg_delta']:+.3f} | {row['topic_match_ndcg_delta']:+.3f} | "
            f"{row['judged_at_10_delta']:+.0f} | {row['unjudged_at_10_delta']:+.0f} |"
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
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production-pool metrics after revealing V3.5 labels.")
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-candidates", default=str(DEFAULT_V35_CANDIDATES.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v34-report", default=str(DEFAULT_V34_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    report = build_report(
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        v35_labels_path=resolve_repo_path(args.v35_labels),
        v35_candidates_path=resolve_repo_path(args.v35_candidates),
        v21_packet_path=resolve_repo_path(args.v21_packet),
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
        v33_model_dir=resolve_repo_path(args.v33_model_dir),
        v34_report_path=resolve_repo_path(args.v34_report),
        selected_240_path=resolve_repo_path(args.selected_240),
        top_k=args.top_k,
    )
    report["command"] = " ".join(sys.argv)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    comparison = report["comparison"]
    print("V3.6 revealed-label production-pool evaluation complete")
    print(f"Judgments evaluated: {report['dataset']['label_count']}")
    print(f"V3.3 reading delta vs V2.7: {comparison['reading_delta_vs_v2_7']:+.3f}")
    print(f"V3.3 topic delta vs V2.7: {comparison['topic_delta_vs_v2_7']:+.3f}")
    print(f"V3.3 unjudged@10 delta from V3.4: {report['before_after']['v3_3_unjudged_at_10_delta']:+.2f}")
    print(f"V3.5 high-scoring top10 buckets: {report['v3_5_label_distribution']['v3_3_high_scoring_unjudged_top10_by_bucket']}")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")


if __name__ == "__main__":
    main()
