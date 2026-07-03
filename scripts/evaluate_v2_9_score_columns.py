import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    file_sha256,
    load_json,
    resolve_repo_path,
    write_json,
    write_text,
)
from evaluate_v2_7_score_blends import EXPECTED_V25_LABELS_SHA256  # noqa: E402
from train_v2_6_production_aware_learned_hybrid import EXPECTED_V21_LABELS_SHA256  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v2_9_ltr_dataset.jsonl"
DEFAULT_SPLITS = REPO_ROOT / "data" / "eval" / "training" / "v2_9_splits.json"
DEFAULT_V27_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v2_7_score_blend_report.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_9_score_column_baselines.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_9_score_column_baselines.md"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"

DEFAULT_SCORE_COLUMNS = [
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
    "v2_7_score",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def ndcg_at_k(ranked_rows: list[dict[str, Any]], field: str, k: int) -> float:
    gains = [float(row.get(field) or 0.0) for row in ranked_rows[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(sorted(gains_for_field(ranked_rows, field), reverse=True)[:k]))
    return dcg / ideal if ideal > 0 else 0.0


def gains_for_field(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row.get(field) or 0.0) for row in rows]


def mrr_at_k(ranked_rows: list[dict[str, Any]], k: int) -> float:
    for index, row in enumerate(ranked_rows[:k], start=1):
        if row.get("positive"):
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_rows: list[dict[str, Any]], k: int) -> float:
    positives = sum(1 for row in ranked_rows if row.get("positive"))
    if positives == 0:
        return 0.0
    retrieved = sum(1 for row in ranked_rows[:k] if row.get("positive"))
    return retrieved / positives


def duplicate_penalty_at_k(ranked_rows: list[dict[str, Any]], k: int) -> float:
    penalties = {"none": 0.0, "uncertain": 0.25, "near_duplicate": 0.5, "exact_duplicate": 1.0}
    top = ranked_rows[:k]
    return sum(penalties.get(str(row.get("duplicate_status", "none")), 0.0) for row in top) / max(len(top), 1)


def rank_rows(rows: list[dict[str, Any]], score_column: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (float(row.get(score_column) or 0.0), -int(row["paper_id"])), reverse=True)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [
        "reading_value_ndcg",
        "topic_match_ndcg",
        "mrr_at_10",
        "positive_recall_at_10",
        "judged_at_10",
        "unjudged_at_10",
        "hard_negative_promotion_rate_at_10",
        "duplicate_penalty_at_10",
    ]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def evaluate_score_columns(
    rows: list[dict[str, Any]],
    *,
    score_columns: list[str],
    k: int,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source["combined"].extend(rows)
    for row in rows:
        by_source[str(row["judged_source"])].append(row)

    scopes: dict[str, Any] = {}
    for scope_name, scope_rows in sorted(by_source.items()):
        rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scope_rows:
            rows_by_topic[str(row["query_id"])].append(row)
        per_method: dict[str, list[dict[str, Any]]] = {}
        for score_column in score_columns:
            topic_metrics: list[dict[str, Any]] = []
            for query_id, topic_rows in sorted(rows_by_topic.items()):
                ranked = rank_rows(topic_rows, score_column)
                top = ranked[:k]
                topic_metrics.append(
                    {
                        "query_id": query_id,
                        "method": score_column,
                        "retrieved_ids_at_k": [int(row["paper_id"]) for row in top],
                        "candidate_pool_size": len(topic_rows),
                        "judged_at_10": len(top),
                        "unjudged_at_10": 0.0,
                        "reading_value_ndcg": ndcg_at_k(ranked, "reading_value_score", k),
                        "topic_match_ndcg": ndcg_at_k(ranked, "topic_match_score", k),
                        "mrr_at_10": mrr_at_k(ranked, k),
                        "positive_recall_at_10": recall_at_k(ranked, k),
                        "hard_negative_promotion_rate_at_10": (
                            sum(1 for row in top if row.get("hard_negative")) / max(len(top), 1)
                        ),
                        "duplicate_penalty_at_10": duplicate_penalty_at_k(ranked, k),
                    }
                )
            per_method[score_column] = topic_metrics
        scopes[scope_name] = {
            "row_count": len(scope_rows),
            "topic_count": len(rows_by_topic),
            "method_averages": {method: aggregate(metrics) for method, metrics in per_method.items()},
            "per_topic": per_method,
        }
    return scopes


def duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((str(row["query_id"]), int(row["paper_id"])) for row in rows)
    return [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in sorted(counts.items())
        if count > 1
    ]


def validate_splits(splits: dict[str, Any]) -> dict[str, Any]:
    split = splits["train_dev_test"]
    train = set(split["train_topics"])
    dev = set(split["dev_topics"])
    test = set(split["test_topics"])
    loto_ok = all(
        not (set(fold["train_topics"]) & set(fold["test_topics"]) or set(fold["train_topics"]) & set(fold["dev_topics"]))
        for fold in splits["leave_topic_out"]
    )
    return {
        "train_dev_test_no_topic_leakage": not (train & dev or train & test or dev & test),
        "leave_topic_out_no_topic_leakage": loto_ok,
    }


def compare_v27_metrics(scopes: dict[str, Any], v27_report: dict[str, Any], *, tolerance: float) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    scope_map = {"v2_1": "v2_1_packet", "v2_5": "v2_5_hard_negative"}
    selected = "blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding"
    for dataset_scope, report_scope in scope_map.items():
        observed = scopes[dataset_scope]["method_averages"]["v2_7_score"]
        expected = v27_report["scopes"][report_scope]["method_averages"][selected]
        for metric, expected_metric in [
            ("reading_value_ndcg", "reading_value_ndcg"),
            ("topic_match_ndcg", "topic_match_ndcg"),
        ]:
            delta = float(observed[metric]) - float(expected[expected_metric])
            comparisons.append(
                {
                    "dataset_scope": dataset_scope,
                    "v2_7_report_scope": report_scope,
                    "metric": metric,
                    "observed": observed[metric],
                    "expected": expected[expected_metric],
                    "delta": delta,
                    "within_tolerance": abs(delta) <= tolerance,
                }
            )
    return {
        "tolerance": tolerance,
        "all_within_tolerance": all(row["within_tolerance"] for row in comparisons),
        "comparisons": comparisons,
        "note": (
            "This checks V2.1 and V2.5 judged-candidate scopes, where the V2.9 dataset has the same "
            "candidate rows as V2.7. Expanded production-pool metrics require unjudged production candidates "
            "and are not reproduced from this judged-only training table."
        ),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2.9 Score Column Baselines",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "This report evaluates score columns in the V2.9 judged learning-to-rank table. No labels were created or modified.",
        "",
        "## Dataset",
        "",
        f"- Rows: `{report['dataset']['row_count']}`",
        f"- Topics: `{report['dataset']['topic_count']}`",
        f"- Duplicate query-paper rows: `{len(report['validation']['duplicate_query_paper_rows'])}`",
        f"- Train/dev/test topic leakage: `{not report['validation']['split_validation']['train_dev_test_no_topic_leakage']}`",
        "",
        "## Baselines",
        "",
        "| scope | score column | reading NDCG@10 | topic NDCG@10 | MRR@10 | positive Recall@10 | hard-neg rate@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope_name in ["v2_1", "v2_5", "combined"]:
        scope = report["scopes"][scope_name]
        for method, avg in sorted(
            scope["method_averages"].items(),
            key=lambda item: item[1]["reading_value_ndcg"],
            reverse=True,
        ):
            lines.append(
                f"| `{scope_name}` | `{method}` | {avg['reading_value_ndcg']:.3f} | "
                f"{avg['topic_match_ndcg']:.3f} | {avg['mrr_at_10']:.3f} | "
                f"{avg['positive_recall_at_10']:.3f} | {avg['hard_negative_promotion_rate_at_10']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## V2.7 Reproduction Check",
            "",
            f"- All checked V2.7 judged-scope metrics within tolerance: `{report['v2_7_reproduction_check']['all_within_tolerance']}`",
            f"- Tolerance: `{report['v2_7_reproduction_check']['tolerance']}`",
            "",
            "| scope | metric | observed | expected | delta | within tolerance |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["v2_7_reproduction_check"]["comparisons"]:
        lines.append(
            f"| `{row['dataset_scope']}` | `{row['metric']}` | {row['observed']:.6f} | "
            f"{row['expected']:.6f} | {row['delta']:+.6f} | `{row['within_tolerance']}` |"
        )
    lines.extend(
        [
            "",
            "## Hash Checks",
            "",
            f"- V2.1 labels unchanged: `{report['hashes']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels unchanged: `{report['hashes']['v2_5_labels_hash_unchanged']}`",
            f"- selected 240 unchanged: `{report['hashes']['selected_240_hash_unchanged']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate score columns in the V2.9 LTR dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--v27-report", default=str(DEFAULT_V27_REPORT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS))
    parser.add_argument("--score-columns", nargs="*", default=DEFAULT_SCORE_COLUMNS)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--v27-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    splits_path = resolve_repo_path(args.splits)
    v27_report_path = resolve_repo_path(args.v27_report)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    rows = load_jsonl(dataset_path)
    splits = load_json(splits_path)
    v27_report = load_json(v27_report_path)
    missing_columns = [
        score_column
        for score_column in args.score_columns
        if any(score_column not in row for row in rows)
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing requested score columns: {sorted(set(missing_columns))}")
    split_validation = validate_splits(splits)
    duplicates = duplicate_rows(rows)
    scopes = evaluate_score_columns(rows, score_columns=args.score_columns, k=args.eval_k)
    reproduction_check = compare_v27_metrics(scopes, v27_report, tolerance=args.v27_tolerance)
    report = {
        "schema_version": "v2.9_score_column_baselines",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "command": " ".join(sys.argv),
        "dataset": {
            "path": str(dataset_path),
            "row_count": len(rows),
            "topic_count": len({str(row["query_id"]) for row in rows}),
            "rows_by_judged_source": dict(Counter(str(row["judged_source"]) for row in rows)),
            "judged_only": True,
            "unjudged_metric_note": "unjudged@10 is zero for this judged-only training table.",
        },
        "score_columns": args.score_columns,
        "scopes": scopes,
        "validation": {
            "duplicate_query_paper_rows": duplicates,
            "no_duplicate_query_paper_rows": not duplicates,
            "split_validation": split_validation,
        },
        "v2_7_reproduction_check": reproduction_check,
        "hashes": {
            "v2_1_labels_sha256": file_sha256(resolve_repo_path(args.v21_labels)),
            "v2_1_labels_hash_unchanged": file_sha256(resolve_repo_path(args.v21_labels)) == EXPECTED_V21_LABELS_SHA256,
            "v2_5_labels_sha256": file_sha256(resolve_repo_path(args.v25_labels)),
            "v2_5_labels_hash_unchanged": file_sha256(resolve_repo_path(args.v25_labels)) == EXPECTED_V25_LABELS_SHA256,
            "selected_240_sha256": file_sha256(SELECTED_240),
            "selected_240_hash_unchanged": file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256,
        },
    }
    if duplicates:
        raise ValueError(f"Duplicate query-paper rows found: {duplicates[:5]}")
    if not split_validation["train_dev_test_no_topic_leakage"] or not split_validation["leave_topic_out_no_topic_leakage"]:
        raise ValueError("Topic split leakage detected.")
    if not reproduction_check["all_within_tolerance"]:
        raise ValueError("V2.7 score-column reproduction check failed.")
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    print("V2.9 score-column baseline evaluation complete")
    print(f"Dataset rows: {len(rows)}")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")
    print(f"V2.7 reproduction within tolerance: {reproduction_check['all_within_tolerance']}")


if __name__ == "__main__":
    main()
