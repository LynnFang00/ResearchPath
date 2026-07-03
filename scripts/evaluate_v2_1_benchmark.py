import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_JSON_OUTPUT = REPO_ROOT / "data" / "eval" / "results" / "v2_1_benchmark_method_comparison.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "data" / "eval" / "results" / "v2_1_benchmark_method_comparison.md"
DEFAULT_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid", "learned_hybrid"]

AUDIENCE_SCORE_FIELDS = {
    "beginner": "beginner_fit_score",
    "intermediate": "intermediate_fit_score",
    "advanced": "advanced_fit_score",
    "expert": "expert_fit_score",
}
INTENT_SCORE_FIELDS = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
PATH_ROLES = ["background", "foundational", "core_methods", "recent_frontier"]
ROLE_VALUES_FOR_COVERAGE = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
DUPLICATE_PENALTIES = {"none": 0.0, "uncertain": 0.25, "near_duplicate": 0.5, "exact_duplicate": 1.0}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File was not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def labels_by_query(labels: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for label in labels:
        grouped[str(label["query_id"])][int(label["paper_id"])] = label
    return dict(grouped)


def packet_by_query(packet_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in packet_rows:
        key = (str(row["query_id"]), int(row["paper_id"]))
        if key in seen:
            continue
        seen.add(key)
        grouped[key[0]].append(row)
    return dict(grouped)


def available_methods(packet_rows: list[dict[str, Any]], requested_methods: list[str]) -> tuple[list[str], dict[str, str]]:
    rank_methods: set[str] = set()
    for row in packet_rows:
        ranks = row.get("retrieval_ranks_by_method") or {}
        rank_methods.update(str(method) for method in ranks)

    methods: list[str] = []
    skipped: dict[str, str] = {}
    for method in requested_methods:
        if method in rank_methods:
            methods.append(method)
        else:
            skipped[method] = "no materialized ranks in packet"
    return methods, skipped


def ranked_packet_rows(query_rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
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
    k: int,
) -> dict[str, Any]:
    retrieved_rows = ranked_packet_rows(query_rows, method)
    retrieved_ids = [int(row["paper_id"]) for row in retrieved_rows]

    score_maps = {
        "topic_match_ndcg": score_map(query_labels, "topic_match_score"),
        "reading_value_ndcg": score_map(query_labels, "reading_value_score"),
    }
    for audience, field_name in AUDIENCE_SCORE_FIELDS.items():
        score_maps[f"{audience}_ndcg"] = score_map(query_labels, field_name)
    for intent in INTENT_SCORE_FIELDS:
        score_maps[f"intent_{intent}_ndcg"] = intent_score_map(query_labels, intent)

    ndcg_metrics = {
        metric_name: ndcg_at_k(retrieved_ids, relevance_by_id, k)
        for metric_name, relevance_by_id in score_maps.items()
    }
    role_coverage = role_coverage_at_k(retrieved_ids, query_labels, k)
    path_coverage = path_level_coverage_at_k(retrieved_ids, query_labels, k)
    duplicate_penalty = duplicate_penalty_at_k(retrieved_ids, query_labels, k)

    return {
        "query_id": query_id,
        "method": method,
        "retrieved_ids_at_k": retrieved_ids[:k],
        "judged_retrieved_at_k": sum(1 for paper_id in retrieved_ids[:k] if paper_id in query_labels),
        **ndcg_metrics,
        "role_coverage_at_k": role_coverage["coverage"],
        "roles_covered_at_k": role_coverage["covered_roles"],
        "roles_available": role_coverage["available_roles"],
        "duplicate_penalty_at_k": duplicate_penalty,
        "path_level_coverage_at_k": path_coverage["coverage"],
        "path_roles_covered_at_k": path_coverage["covered_roles"],
    }


def score_map(labels: dict[int, dict[str, Any]], field_name: str) -> dict[int, float]:
    return {paper_id: clamp_score(label.get(field_name)) for paper_id, label in labels.items()}


def intent_score_map(labels: dict[int, dict[str, Any]], intent: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    for paper_id, label in labels.items():
        intent_scores = label.get("intent_scores")
        scores[paper_id] = clamp_score(intent_scores.get(intent) if isinstance(intent_scores, dict) else 0.0)
    return scores


def ndcg_at_k(retrieved_ids: list[int], relevance_by_id: dict[int, float], k: int) -> float:
    gains = [relevance_by_id.get(paper_id, 0.0) for paper_id in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / ideal if ideal > 0 else 0.0


def role_coverage_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> dict[str, Any]:
    available = topic_roles(labels.values(), threshold=0.3)
    covered = topic_roles((labels[paper_id] for paper_id in retrieved_ids[:k] if paper_id in labels), threshold=0.3)
    denominator = len(available) if available else len(ROLE_VALUES_FOR_COVERAGE)
    return {
        "coverage": len(covered & available) / denominator if denominator else 0.0,
        "covered_roles": sorted(covered),
        "available_roles": sorted(available),
    }


def topic_roles(rows: Any, threshold: float) -> set[str]:
    roles: set[str] = set()
    for row in rows:
        if clamp_score(row.get("reading_value_score")) < threshold:
            continue
        primary_role = row.get("primary_role")
        if primary_role in ROLE_VALUES_FOR_COVERAGE:
            roles.add(str(primary_role))
        secondary_roles = row.get("secondary_roles")
        if isinstance(secondary_roles, list):
            roles.update(str(role) for role in secondary_roles if role in ROLE_VALUES_FOR_COVERAGE)
    return roles


def path_level_coverage_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> dict[str, Any]:
    covered: set[str] = set()
    for paper_id in retrieved_ids[:k]:
        label = labels.get(paper_id)
        if not label or clamp_score(label.get("reading_value_score")) < 0.5:
            continue
        roles = [label.get("primary_role")]
        secondary_roles = label.get("secondary_roles")
        if isinstance(secondary_roles, list):
            roles.extend(secondary_roles)
        covered.update(str(role) for role in roles if role in PATH_ROLES)
    return {
        "coverage": len(covered) / len(PATH_ROLES),
        "covered_roles": sorted(covered),
    }


def duplicate_penalty_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> float:
    if k <= 0:
        return 0.0
    penalties = [
        DUPLICATE_PENALTIES.get(str(labels.get(paper_id, {}).get("duplicate_status", "none")), 0.0)
        for paper_id in retrieved_ids[:k]
    ]
    return sum(penalties) / k


def clamp_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def mean_metric(rows: list[dict[str, Any]], field_name: str) -> float:
    return sum(float(row[field_name]) for row in rows) / len(rows) if rows else 0.0


def build_report(
    *,
    labels: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    requested_methods: list[str],
    k: int,
    labels_path: Path,
    packet_path: Path,
) -> dict[str, Any]:
    label_lookup = labels_by_query(labels)
    packet_lookup = packet_by_query(packet_rows)
    methods, skipped_methods = available_methods(packet_rows, requested_methods)
    query_ids = sorted(label_lookup)

    per_query_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id in query_ids:
        per_query_by_method[query_id] = {}
        for method in methods:
            row = evaluate_method_for_query(
                method=method,
                query_id=query_id,
                query_rows=packet_lookup.get(query_id, []),
                query_labels=label_lookup[query_id],
                k=k,
            )
            per_query_by_method[query_id][method] = row
            method_rows[method].append(row)

    metric_fields = [
        "topic_match_ndcg",
        "reading_value_ndcg",
        "beginner_ndcg",
        "intermediate_ndcg",
        "advanced_ndcg",
        "expert_ndcg",
        "role_coverage_at_k",
        "duplicate_penalty_at_k",
        "path_level_coverage_at_k",
        "judged_retrieved_at_k",
    ] + [f"intent_{intent}_ndcg" for intent in INTENT_SCORE_FIELDS]
    methods_payload = {
        method: {
            "averages": {field: mean_metric(rows, field) for field in metric_fields},
            "queries": rows,
        }
        for method, rows in method_rows.items()
    }
    best_method = (
        max(methods, key=lambda method: methods_payload[method]["averages"]["reading_value_ndcg"])
        if methods
        else None
    )

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2.1",
        "label_source": str(labels_path),
        "packet_source": str(packet_path),
        "evaluation_scope": "v2_1_candidate_packet_ranked_judged_benchmark",
        "k": k,
        "labels_created_by_this_script": False,
        "models_trained_by_this_script": False,
        "label_summary": label_summary(labels),
        "methods_evaluated": methods,
        "methods_skipped": skipped_methods,
        "best_method_by_reading_value_ndcg": best_method,
        "methods": methods_payload,
        "per_query": per_query_by_method,
        "caveats": [
            "Candidate-packet evaluation treats unjudged retrieved candidates as zero gain.",
            "Continuous V2.1 scores are judged metrics, not exhaustive corpus relevance.",
            "Duplicate penalty is reported separately and not subtracted from NDCG in this scaffold.",
            "Path-level coverage is measured over background/foundational/core_methods/recent_frontier roles.",
        ],
    }


def label_summary(labels: list[dict[str, Any]]) -> dict[str, Any]:
    labels_by_topic = defaultdict(int)
    for label in labels:
        labels_by_topic[str(label["query_id"])] += 1
    return {
        "total_labels": len(labels),
        "labels_by_topic": dict(sorted(labels_by_topic.items())),
    }


def write_report(report: dict[str, Any], json_output: Path, md_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(markdown_report(report), encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Benchmark Method Comparison",
        "",
        f"Created: `{report['created_at']}`",
        f"Labels: `{report['label_summary']['total_labels']}`",
        f"Cutoff K: `{report['k']}`",
        "",
        "## Method Comparison",
        "",
        "| method | topic NDCG | reading NDCG | beginner | intermediate | advanced | expert | role coverage | path coverage | dup penalty | judged@K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in sorted(
        report["methods"].items(),
        key=lambda item: item[1]["averages"]["reading_value_ndcg"],
        reverse=True,
    ):
        avg = payload["averages"]
        lines.append(
            f"| `{method}` | {avg['topic_match_ndcg']:.3f} | {avg['reading_value_ndcg']:.3f} | "
            f"{avg['beginner_ndcg']:.3f} | {avg['intermediate_ndcg']:.3f} | "
            f"{avg['advanced_ndcg']:.3f} | {avg['expert_ndcg']:.3f} | "
            f"{avg['role_coverage_at_k']:.3f} | {avg['path_level_coverage_at_k']:.3f} | "
            f"{avg['duplicate_penalty_at_k']:.3f} | {avg['judged_retrieved_at_k']:.2f} |"
        )

    lines.extend(["", "## Intent-Specific NDCG", ""])
    lines.append("| method | " + " | ".join(INTENT_SCORE_FIELDS) + " |")
    lines.append("|---|" + "|".join("---:" for _ in INTENT_SCORE_FIELDS) + "|")
    for method, payload in sorted(report["methods"].items()):
        avg = payload["averages"]
        values = " | ".join(f"{avg[f'intent_{intent}_ndcg']:.3f}" for intent in INTENT_SCORE_FIELDS)
        lines.append(f"| `{method}` | {values} |")

    lines.extend(["", "## Per-Topic Reading Value NDCG", ""])
    lines.append("| query_id | " + " | ".join(report["methods_evaluated"]) + " |")
    lines.append("|---|" + "|".join("---:" for _ in report["methods_evaluated"]) + "|")
    for query_id, rows_by_method in sorted(report["per_query"].items()):
        values = " | ".join(
            f"{rows_by_method[method]['reading_value_ndcg']:.3f}" if method in rows_by_method else ""
            for method in report["methods_evaluated"]
        )
        lines.append(f"| `{query_id}` | {values} |")

    lines.extend(["", "## Caveats", ""])
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    if report["methods_skipped"]:
        lines.append(f"- Skipped methods: `{report['methods_skipped']}`")
    lines.append("")
    return "\n".join(lines)


def print_report(report: dict[str, Any]) -> None:
    print("\nV2.1 Benchmark")
    print(f"Labels: {report['label_summary']['total_labels']}")
    print(f"Best method by reading-value NDCG: {report['best_method_by_reading_value_ndcg']}")
    for method, payload in sorted(
        report["methods"].items(),
        key=lambda item: item[1]["averages"]["reading_value_ndcg"],
        reverse=True,
    ):
        avg = payload["averages"]
        print(
            f"- {method}: topic_ndcg={avg['topic_match_ndcg']:.3f}, "
            f"reading_ndcg={avg['reading_value_ndcg']:.3f}, "
            f"path_coverage={avg['path_level_coverage_at_k']:.3f}, "
            f"dup_penalty={avg['duplicate_penalty_at_k']:.3f}"
        )


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2.1 continuous-score labels against packet ranks.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUTPUT))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--method", action="append", default=None, help="Method to evaluate; may be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_methods = args.method if args.method else DEFAULT_METHODS
    labels_path = resolve_repo_path(args.labels)
    packet_path = resolve_repo_path(args.packet)
    labels = load_jsonl(labels_path)
    packet_rows = load_jsonl(packet_path)
    report = build_report(
        labels=labels,
        packet_rows=packet_rows,
        requested_methods=requested_methods,
        k=args.k,
        labels_path=labels_path,
        packet_path=packet_path,
    )
    write_report(report, resolve_repo_path(args.json_out), resolve_repo_path(args.md_out))
    print_report(report)
    print(f"\nJSON: {resolve_repo_path(args.json_out)}")
    print(f"Markdown: {resolve_repo_path(args.md_out)}")


if __name__ == "__main__":
    main()
