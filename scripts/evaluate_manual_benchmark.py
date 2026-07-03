import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_fulltext_v1.jsonl"
DEFAULT_POOL = REPO_ROOT / "data" / "eval" / "manual_label_pool_v1.jsonl"
DEFAULT_JSON_OUTPUT = REPO_ROOT / "data" / "eval" / "results" / "manual_benchmark_method_comparison.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "data" / "eval" / "results" / "manual_benchmark_method_comparison.md"
DEFAULT_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid", "learned_hybrid"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File was not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def labels_by_query(labels: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for label in labels:
        grouped[str(label["query_id"])][int(label["paper_id"])] = label
    return dict(grouped)


def pool_by_query(pool_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in pool_rows:
        key = (str(row["query_id"]), int(row["paper_id"]))
        if key in seen:
            continue
        seen.add(key)
        grouped[key[0]].append(row)
    return dict(grouped)


def available_methods(pool_rows: list[dict[str, Any]], requested_methods: list[str]) -> tuple[list[str], dict[str, str]]:
    rank_methods: set[str] = set()
    for row in pool_rows:
        ranks = row.get("retrieval_ranks_by_method") or {}
        rank_methods.update(str(method) for method in ranks)

    methods: list[str] = []
    skipped: dict[str, str] = {}
    for method in requested_methods:
        if method in rank_methods:
            methods.append(method)
        else:
            skipped[method] = "no materialized ranks in candidate pool"
    return methods, skipped


def ranked_pool_rows(query_rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for original_index, row in enumerate(query_rows):
        ranks = row.get("retrieval_ranks_by_method") or {}
        if method not in ranks:
            continue
        ranked.append((int(ranks[method]), original_index, row))
    return [row for _, _, row in sorted(ranked, key=lambda item: (item[0], item[1]))]


def evaluate_method_for_query(
    *,
    method: str,
    query_id: str,
    query_rows: list[dict[str, Any]],
    query_labels: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    retrieved_rows = ranked_pool_rows(query_rows, method)
    retrieved_ids = [int(row["paper_id"]) for row in retrieved_rows]
    relevant_ids = {paper_id for paper_id, label in query_labels.items() if int(label["relevance"]) >= 2}
    recommend_ids = {paper_id for paper_id, label in query_labels.items() if bool(label["would_recommend"])}
    duplicate_ids = {paper_id for paper_id, label in query_labels.items() if bool(label["duplicate"])}
    labels_relevance = {paper_id: int(label["relevance"]) for paper_id, label in query_labels.items()}

    return {
        "query_id": query_id,
        "query": str(next(iter(query_labels.values()))["query"]) if query_labels else "",
        "method": method,
        "retrieved_ids_at_10": retrieved_ids[:10],
        "judged_retrieved_at_10": sum(1 for paper_id in retrieved_ids[:10] if paper_id in query_labels),
        "relevant_label_count": len(relevant_ids),
        "would_recommend_label_count": len(recommend_ids),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "precision_at_5": precision_at_k(retrieved_ids, recommend_ids, 5),
        "precision_at_10": precision_at_k(retrieved_ids, recommend_ids, 10),
        "ndcg_at_5": ndcg_at_k(retrieved_ids, labels_relevance, 5),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, labels_relevance, 10),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "would_recommend_count_at_10": sum(1 for paper_id in retrieved_ids[:10] if paper_id in recommend_ids),
        "duplicate_count_at_10": sum(1 for paper_id in retrieved_ids[:10] if paper_id in duplicate_ids),
    }


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return sum(1 for paper_id in retrieved_ids[:k] if paper_id in relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], positive_ids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for paper_id in retrieved_ids[:k] if paper_id in positive_ids) / k


def ndcg_at_k(retrieved_ids: list[int], relevance_by_id: dict[int, int], k: int) -> float:
    gains = [relevance_by_id.get(paper_id, 0) for paper_id in retrieved_ids[:k]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for index, paper_id in enumerate(retrieved_ids, start=1):
        if paper_id in relevant_ids:
            return 1.0 / index
    return 0.0


def mean_metric(rows: list[dict[str, Any]], field_name: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[field_name]) for row in rows) / len(rows)


def label_summary(labels: list[dict[str, Any]]) -> dict[str, Any]:
    labels_per_query = Counter(str(label["query_id"]) for label in labels)
    return {
        "total_labels": len(labels),
        "labels_per_query": dict(sorted(labels_per_query.items())),
        "relevance_distribution": dict(sorted(Counter(int(label["relevance"]) for label in labels).items())),
        "would_recommend_count": sum(1 for label in labels if bool(label["would_recommend"])),
        "duplicate_count": sum(1 for label in labels if bool(label["duplicate"])),
    }


def build_report(
    *,
    labels: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    requested_methods: list[str],
) -> dict[str, Any]:
    label_lookup = labels_by_query(labels)
    pool_lookup = pool_by_query(pool_rows)
    methods, skipped_methods = available_methods(pool_rows, requested_methods)
    query_ids = sorted(label_lookup)

    per_query_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id in query_ids:
        per_query_by_method[query_id] = {}
        for method in methods:
            row = evaluate_method_for_query(
                method=method,
                query_id=query_id,
                query_rows=pool_lookup.get(query_id, []),
                query_labels=label_lookup[query_id],
            )
            per_query_by_method[query_id][method] = row
            method_rows[method].append(row)

    metric_fields = [
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "precision_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "mrr",
        "would_recommend_count_at_10",
        "duplicate_count_at_10",
        "judged_retrieved_at_10",
    ]
    methods_payload = {
        method: {
            "averages": {field: mean_metric(rows, field) for field in metric_fields},
            "queries": rows,
        }
        for method, rows in method_rows.items()
    }
    best_method = max(methods, key=lambda method: methods_payload[method]["averages"]["ndcg_at_10"]) if methods else None

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "label_source": str(DEFAULT_LABELS.relative_to(REPO_ROOT)),
        "candidate_pool": str(DEFAULT_POOL.relative_to(REPO_ROOT)),
        "evaluation_scope": "candidate_pool_ranked_judged_benchmark",
        "caveats": [
            "Small benchmark: 80 total labels across 8 topics.",
            "Candidate-pool evaluation treats unjudged retrieved candidates as zero relevance and not recommended.",
            "One intentional duplicate label is retained for the GNN survey duplicate row.",
            "learned_hybrid is skipped unless materialized ranks already exist; no training is run by this script.",
        ],
        "label_summary": label_summary(labels),
        "methods_evaluated": methods,
        "methods_skipped": skipped_methods,
        "best_method_by_ndcg_at_10": best_method,
        "methods": methods_payload,
        "per_query": per_query_by_method,
        "hybrid_comparisons": hybrid_comparisons(per_query_by_method, methods_payload),
    }


def hybrid_comparisons(
    per_query_by_method: dict[str, dict[str, dict[str, Any]]],
    methods_payload: dict[str, Any],
) -> dict[str, Any]:
    if "hybrid" not in methods_payload:
        return {}
    comparisons: dict[str, Any] = {}
    for baseline in ("bm25", "embedding"):
        if baseline not in methods_payload:
            continue
        per_query: list[dict[str, Any]] = []
        wins = losses = ties = 0
        for query_id, rows_by_method in per_query_by_method.items():
            hybrid_value = rows_by_method["hybrid"]["ndcg_at_10"]
            baseline_value = rows_by_method[baseline]["ndcg_at_10"]
            delta = hybrid_value - baseline_value
            if delta > 1e-12:
                wins += 1
                outcome = "beats"
            elif delta < -1e-12:
                losses += 1
                outcome = "loses"
            else:
                ties += 1
                outcome = "ties"
            per_query.append(
                {
                    "query_id": query_id,
                    "hybrid_ndcg_at_10": hybrid_value,
                    f"{baseline}_ndcg_at_10": baseline_value,
                    "delta": delta,
                    "outcome": outcome,
                }
            )
        comparisons[baseline] = {
            "overall_delta_ndcg_at_10": methods_payload["hybrid"]["averages"]["ndcg_at_10"]
            - methods_payload[baseline]["averages"]["ndcg_at_10"],
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "per_query": per_query,
        }
    return comparisons


def write_report(report: dict[str, Any], json_output: Path, md_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(markdown_report(report), encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Manual Full-Text Benchmark Method Comparison")
    lines.append("")
    lines.append(f"Created: `{report['created_at']}`")
    lines.append("")
    lines.append("## Label Summary")
    summary = report["label_summary"]
    lines.append("")
    lines.append(f"- Total labels: `{summary['total_labels']}`")
    lines.append(f"- Duplicate count: `{summary['duplicate_count']}`")
    lines.append(f"- Relevance distribution: `{summary['relevance_distribution']}`")
    lines.append("")
    lines.append("| query_id | labels |")
    lines.append("|---|---:|")
    for query_id, count in summary["labels_per_query"].items():
        lines.append(f"| `{query_id}` | {count} |")

    lines.append("")
    lines.append("## Method Comparison")
    lines.extend(method_table(report))

    lines.append("")
    lines.append("## Per-Topic Breakdown")
    lines.extend(per_topic_table(report))

    lines.append("")
    lines.append("## Best Method")
    best = report["best_method_by_ndcg_at_10"]
    if best:
        value = report["methods"][best]["averages"]["ndcg_at_10"]
        lines.append(f"Best method by mean NDCG@10: `{best}` ({value:.3f}).")
    else:
        lines.append("No methods were evaluated.")

    lines.append("")
    lines.append("## Hybrid Comparisons")
    lines.extend(hybrid_comparison_section(report))

    lines.append("")
    lines.append("## Caveats")
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    if report["methods_skipped"]:
        lines.append(f"- Skipped methods: `{report['methods_skipped']}`")
    lines.append("")
    return "\n".join(lines)


def method_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| method | R@5 | R@10 | P@5 | P@10 | NDCG@5 | NDCG@10 | MRR | rec@10 | dup@10 | judged@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in sorted(
        report["methods"].items(),
        key=lambda item: item[1]["averages"]["ndcg_at_10"],
        reverse=True,
    ):
        avg = payload["averages"]
        lines.append(
            f"| `{method}` | {avg['recall_at_5']:.3f} | {avg['recall_at_10']:.3f} | "
            f"{avg['precision_at_5']:.3f} | {avg['precision_at_10']:.3f} | "
            f"{avg['ndcg_at_5']:.3f} | {avg['ndcg_at_10']:.3f} | {avg['mrr']:.3f} | "
            f"{avg['would_recommend_count_at_10']:.2f} | {avg['duplicate_count_at_10']:.2f} | "
            f"{avg['judged_retrieved_at_10']:.2f} |"
        )
    return lines


def per_topic_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| query_id | method | R@10 | P@10 | NDCG@10 | MRR | rec@10 | dup@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for query_id, rows_by_method in sorted(report["per_query"].items()):
        for method, row in sorted(rows_by_method.items(), key=lambda item: item[1]["ndcg_at_10"], reverse=True):
            lines.append(
                f"| `{query_id}` | `{method}` | {row['recall_at_10']:.3f} | {row['precision_at_10']:.3f} | "
                f"{row['ndcg_at_10']:.3f} | {row['mrr']:.3f} | {row['would_recommend_count_at_10']} | "
                f"{row['duplicate_count_at_10']} |"
            )
    return lines


def hybrid_comparison_section(report: dict[str, Any]) -> list[str]:
    comparisons = report.get("hybrid_comparisons") or {}
    if not comparisons:
        return ["Hybrid comparison was unavailable."]
    lines: list[str] = []
    for baseline, payload in comparisons.items():
        lines.append(
            f"- Hybrid vs `{baseline}`: delta mean NDCG@10 = {payload['overall_delta_ndcg_at_10']:.3f}; "
            f"wins={payload['wins']}, losses={payload['losses']}, ties={payload['ties']}."
        )
        details = ", ".join(
            f"{row['query_id']} {row['delta']:+.3f}" for row in payload["per_query"] if abs(row["delta"]) > 1e-12
        )
        if details:
            lines.append(f"  Per-topic non-ties: {details}.")
    return lines


def print_report_tables(report: dict[str, Any]) -> None:
    print("\nManual Full-Text Benchmark")
    print(f"Labels: {report['label_summary']['total_labels']}")
    print(f"Labels per query: {report['label_summary']['labels_per_query']}")
    print(f"Best method by NDCG@10: {report['best_method_by_ndcg_at_10']}")
    print("\nMethod Comparison")
    for line in method_table(report):
        print(line)
    print("\nHybrid Comparisons")
    for line in hybrid_comparison_section(report):
        print(line)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate-pool methods against the manual full-text benchmark.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUTPUT))
    parser.add_argument("--method", action="append", default=None, help="Method to evaluate; may be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_methods = args.method if args.method else DEFAULT_METHODS
    labels = load_jsonl(resolve_repo_path(args.labels))
    pool_rows = load_jsonl(resolve_repo_path(args.pool))
    report = build_report(labels=labels, pool_rows=pool_rows, requested_methods=requested_methods)
    json_output = resolve_repo_path(args.json_out)
    md_output = resolve_repo_path(args.md_out)
    write_report(report, json_output, md_output)
    print_report_tables(report)
    print(f"\nJSON: {json_output}")
    print(f"Markdown: {md_output}")


if __name__ == "__main__":
    main()
