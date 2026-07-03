import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = REPO_ROOT / "data" / "eval" / "results" / "v2_1_benchmark_method_comparison.json"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_1_learned_ranker_milestone.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_1_learned_ranker_milestone.md"

AUDIENCE_METRICS = ["beginner_ndcg", "intermediate_ndcg", "advanced_ndcg", "expert_ndcg"]
INTENT_METRICS = [
    "intent_background_ndcg",
    "intent_foundational_ndcg",
    "intent_core_methods_ndcg",
    "intent_recent_frontier_ndcg",
    "intent_evaluation_benchmark_ndcg",
    "intent_application_ndcg",
]
METHOD_ORDER = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
RANK_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def method_queries(eval_report: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, payload in eval_report.get("methods", {}).items():
        for row in payload.get("queries", []):
            by_query[str(row["query_id"])][method] = row
    return dict(by_query)


def best_method(rows_by_method: dict[str, dict[str, Any]], metric: str) -> tuple[str, float]:
    candidates = [
        (method, float(rows_by_method[method].get(metric, 0.0)))
        for method in METHOD_ORDER
        if method in rows_by_method
    ]
    return max(candidates, key=lambda item: (item[1], -METHOD_ORDER.index(item[0])))


def build_per_topic_winners(eval_report: dict[str, Any]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for query_id, rows_by_method in sorted(method_queries(eval_report).items()):
        topic_method, topic_score = best_method(rows_by_method, "topic_match_ndcg")
        reading_method, reading_score = best_method(rows_by_method, "reading_value_ndcg")
        beginner_method, beginner_score = best_method(rows_by_method, "beginner_ndcg")
        winners.append(
            {
                "query_id": query_id,
                "topic_match_winner": topic_method,
                "topic_match_ndcg": topic_score,
                "reading_value_winner": reading_method,
                "reading_value_ndcg": reading_score,
                "beginner_winner": beginner_method,
                "beginner_ndcg": beginner_score,
            }
        )
    return winners


def build_hybrid_vs_bm25(eval_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id, rows_by_method in sorted(method_queries(eval_report).items()):
        if "hybrid" not in rows_by_method or "bm25" not in rows_by_method:
            continue
        hybrid = rows_by_method["hybrid"]
        bm25 = rows_by_method["bm25"]
        rows.append(
            {
                "query_id": query_id,
                "topic_delta": float(hybrid["topic_match_ndcg"]) - float(bm25["topic_match_ndcg"]),
                "reading_delta": float(hybrid["reading_value_ndcg"]) - float(bm25["reading_value_ndcg"]),
                "beginner_delta": float(hybrid["beginner_ndcg"]) - float(bm25["beginner_ndcg"]),
                "expert_delta": float(hybrid["expert_ndcg"]) - float(bm25["expert_ndcg"]),
                "path_coverage_delta": float(hybrid["path_level_coverage_at_k"])
                - float(bm25["path_level_coverage_at_k"]),
            }
        )
    return rows


def worst_topics(eval_report: dict[str, Any], metric: str, *, method: str = "hybrid", limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id, rows_by_method in method_queries(eval_report).items():
        if method in rows_by_method:
            rows.append({"query_id": query_id, "method": method, metric: float(rows_by_method[method][metric])})
    return sorted(rows, key=lambda row: row[metric])[:limit]


def intent_winners(eval_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in INTENT_METRICS:
        intent = metric.removeprefix("intent_").removesuffix("_ndcg")
        candidates = []
        for method, payload in eval_report.get("methods", {}).items():
            average = payload.get("averages", {})
            if metric in average:
                candidates.append((method, float(average[metric])))
        winner, score = max(candidates, key=lambda item: item[1])
        rows.append({"intent": intent, "winner": winner, "ndcg": score})
    return rows


def collect_feature_inventory(packet_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rank_methods = Counter()
    score_methods = Counter()
    selection_reasons = Counter()
    sources = Counter()
    evidence = Counter()
    likely_coverage = Counter()
    has_field = Counter()
    citation_values: list[float] = []
    year_values: list[int] = []

    for row in packet_rows:
        for method in (row.get("retrieval_ranks_by_method") or {}):
            rank_methods[str(method)] += 1
        for method in (row.get("retrieval_scores_by_method") or {}):
            score_methods[str(method)] += 1
        for reason in row.get("selection_reasons") or []:
            selection_reasons[str(reason)] += 1
        for source in row.get("sources_provenance") or []:
            sources[str(source)] += 1
        evidence[evidence_level(row.get("evidence_availability"))] += 1
        likely_coverage[coverage_value(row.get("likely_coverage"))] += 1
        for field_name in [
            "title",
            "abstract",
            "venue",
            "authors",
            "citation_count",
            "year",
            "source_url",
            "pdf_url",
            "duplicate_title_cluster",
            "source_specific_metadata",
            "identifiers",
        ]:
            if row.get(field_name) not in (None, "", [], {}):
                has_field[field_name] += 1
        if isinstance(row.get("citation_count"), (int, float)):
            citation_values.append(float(row["citation_count"]))
        if isinstance(row.get("year"), int):
            year_values.append(int(row["year"]))

    duplicate_status = Counter(str(row.get("duplicate_status") or "missing") for row in label_rows)
    primary_roles = Counter(str(row.get("primary_role") or "missing") for row in label_rows)

    return {
        "packet_rows": len(packet_rows),
        "rank_methods_available": dict(sorted(rank_methods.items())),
        "score_methods_available": dict(sorted(score_methods.items())),
        "selection_reasons": dict(selection_reasons.most_common()),
        "sources_provenance": dict(sources.most_common()),
        "evidence_availability": dict(sorted(evidence.items())),
        "likely_coverage": dict(sorted(likely_coverage.items())),
        "nonempty_packet_fields": dict(sorted(has_field.items())),
        "citation_count_summary": summarize_numeric(citation_values),
        "year_summary": summarize_numeric(year_values),
        "label_duplicate_status_distribution": dict(sorted(duplicate_status.items())),
        "label_primary_role_distribution": dict(primary_roles.most_common()),
    }


def evidence_level(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("level") or "missing")
    return str(value or "missing")


def coverage_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "missing")
    return str(value or "missing")


def summarize_numeric(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def target_score_proposal() -> dict[str, Any]:
    return {
        "name": "v2_1_beginner_path_gain",
        "formula": (
            "clamp01(0.40*reading_value_score + 0.25*topic_match_score + "
            "0.15*beginner_fit_score + 0.05*intermediate_fit_score + "
            "0.05*intent_path_score + 0.05*intent_application_or_benchmark_score - "
            "0.05*duplicate_penalty)"
        ),
        "components": {
            "intent_path_score": "max(intent_scores.background, foundational, core_methods, recent_frontier)",
            "intent_application_or_benchmark_score": "max(intent_scores.evaluation_benchmark, application)",
            "duplicate_penalty": "0.0 none, 0.25 uncertain, 0.5 near_duplicate, 1.0 exact_duplicate",
        },
        "rationale": [
            "Use reading_value_score as the main target because the product goal is a useful reading path, not only topical retrieval.",
            "Keep topic_match_score heavily weighted so the reranker does not learn to over-promote readable but off-topic papers.",
            "Give beginner fit explicit weight while preserving some intermediate value for papers that help readers progress.",
            "Include intent terms so path-shaping roles are learnable without turning role coverage into an unavailable per-row label.",
            "Treat duplicate penalty as a small demotion; continue reporting duplicate penalties separately in evaluation.",
        ],
    }


def feature_plan() -> list[dict[str, str]]:
    return [
        {"group": "retrieval_rank", "features": "rank, reciprocal rank, missing-rank flag for bm25/tfidf/embedding/faiss_embedding/hybrid"},
        {"group": "retrieval_score", "features": "raw score, per-query z-score, min-max normalized score, missing-score flag by method"},
        {"group": "rank_ensemble", "features": "best rank, mean reciprocal rank, number of methods retrieving candidate, lexical-vs-vector rank gap"},
        {"group": "text_match", "features": "query-title token overlap, query-abstract token overlap, phrase hit flags, title length, abstract length"},
        {"group": "metadata", "features": "publication year, age, log1p(citation_count), venue present, author count"},
        {"group": "source_quality", "features": "sources_provenance one-hot/count, arxiv/doi/openalex identifier flags, source_url/pdf_url flags"},
        {"group": "packet_heuristics", "features": "selection_reasons one-hot, likely_coverage one-hot as heuristic-only input, anchor_category_hint flag"},
        {"group": "evidence", "features": "evidence_availability one-hot, fulltext_sections present, abstract missing/short flags"},
        {"group": "dedup_heuristics", "features": "duplicate_title_cluster present/size and normalized-title collision indicators, not label duplicate_status"},
        {"group": "domain_metadata", "features": "arXiv categories, OpenAlex concepts/topics, keywords when present"},
    ]


def training_strategy() -> list[str]:
    return [
        "Do not hold out individual papers at random; split or cross-validate by query_id so validation measures generalization to unseen topics.",
        "Use leave-one-topic-out or 4-fold grouped cross-validation over the 16 topics, with 12 topics train and 4 topics validation per fold.",
        "Train a very small model first: Ridge/ElasticNet or LightGBM/XGBoost with shallow trees and strong regularization. Avoid neural rerankers at this label count.",
        "Optimize the combined target as a regression gain, then evaluate NDCG@10 for topic_match, reading_value, audience-fit, intent-fit, duplicate penalty, and path coverage.",
        "Select model/hyperparameters by mean validation reading_value NDCG@10, using topic_match NDCG@10 and beginner NDCG@10 as guardrails.",
        "Use repeated grouped CV or bootstrap over topics for confidence intervals; with 16 topics, report fold variance rather than a single split.",
        "Keep a frozen final test slice only after the first feature/model choice is stable, for example 12 train topics, 2 validation topics, 2 final test topics.",
        "Prevent leakage: labels, primary_role, duplicate_status, and intent_scores can define targets/metrics, but should not be inference features.",
        "Calibrate against baselines by requiring learned_hybrid to beat hybrid on reading_value NDCG@10 and avoid losing more than 0.01 to bm25 on topic_match NDCG@10.",
    ]


def build_payload(eval_report: dict[str, Any], packet_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "source_evaluation": str(DEFAULT_EVAL),
        "labels_created_by_this_script": False,
        "models_trained_by_this_script": False,
        "per_topic_winners": build_per_topic_winners(eval_report),
        "hybrid_vs_bm25_deltas": build_hybrid_vs_bm25(eval_report),
        "worst_topics_by_hybrid_reading_value_ndcg": worst_topics(eval_report, "reading_value_ndcg"),
        "worst_topics_by_hybrid_beginner_ndcg": worst_topics(eval_report, "beginner_ndcg"),
        "intent_specific_winners": intent_winners(eval_report),
        "target_score_proposal": target_score_proposal(),
        "feature_inventory": collect_feature_inventory(packet_rows, label_rows),
        "available_feature_plan": feature_plan(),
        "train_validation_strategy": training_strategy(),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# V2.1 Learned-Ranker Milestone Prep",
        "",
        f"Created: `{payload['created_at']}`",
        "",
        "No labels were created or changed. No models were trained.",
        "",
        "## Per-Topic Winners",
        "",
        "| query_id | topic winner | topic NDCG@10 | reading winner | reading NDCG@10 | beginner winner | beginner NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_topic_winners"]:
        lines.append(
            f"| `{row['query_id']}` | `{row['topic_match_winner']}` | {row['topic_match_ndcg']:.3f} | "
            f"`{row['reading_value_winner']}` | {row['reading_value_ndcg']:.3f} | "
            f"`{row['beginner_winner']}` | {row['beginner_ndcg']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Hybrid vs BM25 Deltas",
            "",
            "Positive means hybrid is better than BM25.",
            "",
            "| query_id | topic delta | reading delta | beginner delta | expert delta | path coverage delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(payload["hybrid_vs_bm25_deltas"], key=lambda item: item["reading_delta"]):
        lines.append(
            f"| `{row['query_id']}` | {row['topic_delta']:+.3f} | {row['reading_delta']:+.3f} | "
            f"{row['beginner_delta']:+.3f} | {row['expert_delta']:+.3f} | {row['path_coverage_delta']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Worst Topics by Hybrid Reading-Value NDCG@10",
            "",
            "| query_id | method | reading NDCG@10 |",
            "|---|---:|---:|",
        ]
    )
    for row in payload["worst_topics_by_hybrid_reading_value_ndcg"]:
        lines.append(f"| `{row['query_id']}` | `{row['method']}` | {row['reading_value_ndcg']:.3f} |")

    lines.extend(
        [
            "",
            "## Worst Topics by Hybrid Beginner NDCG@10",
            "",
            "| query_id | method | beginner NDCG@10 |",
            "|---|---:|---:|",
        ]
    )
    for row in payload["worst_topics_by_hybrid_beginner_ndcg"]:
        lines.append(f"| `{row['query_id']}` | `{row['method']}` | {row['beginner_ndcg']:.3f} |")

    lines.extend(
        [
            "",
            "## Intent-Specific Winners",
            "",
            "| intent | winner | NDCG@10 |",
            "|---|---:|---:|",
        ]
    )
    for row in payload["intent_specific_winners"]:
        lines.append(f"| `{row['intent']}` | `{row['winner']}` | {row['ndcg']:.3f} |")

    target = payload["target_score_proposal"]
    lines.extend(
        [
            "",
            "## Proposed Combined Target Score",
            "",
            f"Target name: `{target['name']}`",
            "",
            "```text",
            target["formula"],
            "```",
            "",
        ]
    )
    for note in target["rationale"]:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Lightweight learned_hybrid Feature Set",
            "",
            "| feature group | available features |",
            "|---|---|",
        ]
    )
    for row in payload["available_feature_plan"]:
        lines.append(f"| `{row['group']}` | {row['features']} |")

    inventory = payload["feature_inventory"]
    lines.extend(
        [
            "",
            "## Feature Inventory Snapshot",
            "",
            f"- Packet rows: `{inventory['packet_rows']}`",
            f"- Rank methods available: `{json.dumps(inventory['rank_methods_available'], sort_keys=True)}`",
            f"- Score methods available: `{json.dumps(inventory['score_methods_available'], sort_keys=True)}`",
            f"- Evidence availability: `{json.dumps(inventory['evidence_availability'], sort_keys=True)}`",
            f"- Likely coverage heuristic: `{json.dumps(inventory['likely_coverage'], sort_keys=True)}`",
            f"- Non-empty fields: `{json.dumps(inventory['nonempty_packet_fields'], sort_keys=True)}`",
            f"- Citation count summary: `{json.dumps(inventory['citation_count_summary'], sort_keys=True)}`",
            f"- Year summary: `{json.dumps(inventory['year_summary'], sort_keys=True)}`",
            "",
            "## Train/Validation Strategy for 240 Labels",
            "",
        ]
    )
    for item in payload["train_validation_strategy"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Milestone Recommendation",
            "",
            "- Build a feature-extraction script that emits one row per `(query_id, paper_id)` from the V2.1 packet.",
            "- Add an evaluation-only `learned_hybrid` rank materialization path before any training run.",
            "- Start with grouped cross-validation and a strongly regularized lightweight model.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare V2.1 learned-ranker milestone analysis.")
    parser.add_argument("--eval", default=str(DEFAULT_EVAL), help="V2.1 benchmark evaluation JSON.")
    parser.add_argument("--packet", default=str(DEFAULT_PACKET), help="V2.1 labeling packet JSONL.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="V2.1 manual labels JSONL.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="Output JSON report.")
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT), help="Output Markdown report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_report = load_json(Path(args.eval))
    packet_rows = load_jsonl(Path(args.packet))
    label_rows = load_jsonl(Path(args.labels))
    payload = build_payload(eval_report, packet_rows, label_rows)
    write_json(Path(args.json_out), payload)
    write_text(Path(args.md_out), render_markdown(payload))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    print("No labels were created or changed. No models were trained.")


if __name__ == "__main__":
    main()
