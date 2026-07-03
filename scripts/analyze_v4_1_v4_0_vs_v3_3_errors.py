import argparse
from collections import Counter, defaultdict
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
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
    build_method_scores,
    evaluate_rankings,
    load_jsonl,
    production_candidate_rows,
    rank_from_scores,
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
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


V40_METHOD = "v4_0_random_forest_no_v27_compact"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_v4_0_vs_v3_3_error_analysis.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_v4_0_vs_v3_3_error_analysis.md"
DEFAULT_EXAMPLES_OUT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_v4_0_vs_v3_3_disagreement_examples.jsonl"
FEATURES_FOR_EXAMPLES = [
    "bm25_score",
    "bm25_rank",
    "bm25_reciprocal_rank",
    "tfidf_score",
    "embedding_score",
    "faiss_embedding_score",
    "hybrid_score",
    "old_v2_2b_score",
    "v2_6_score",
    "year",
    "paper_age_years",
    "citation_count_log1p",
]


def load_all_labels(paths_by_source: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in paths_by_source.items():
        rows.extend(labels_with_source(path, source))
    return rows


def labels_by_key(labels: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["query_id"]), int(row["paper_id"])): row for row in labels}


def is_positive(label: dict[str, Any] | None) -> bool:
    return bool(label) and float(label.get("reading_value_score") or 0.0) >= 0.5 and float(label.get("topic_match_score") or 0.0) >= 0.5


def is_high_reading(label: dict[str, Any] | None) -> bool:
    return bool(label) and float(label.get("reading_value_score") or 0.0) >= 1.0


def is_hard_negative(label: dict[str, Any] | None) -> bool:
    return bool(label) and (
        str(label.get("primary_role")) == "negative"
        or (float(label.get("reading_value_score") or 0.0) <= 0.0 and float(label.get("topic_match_score") or 0.0) <= 0.0)
    )


def rank_maps(method_scores: dict[str, dict[str, dict[int, float]]], methods: list[str]) -> dict[str, dict[str, dict[int, int]]]:
    output: dict[str, dict[str, dict[int, int]]] = defaultdict(dict)
    for method in methods:
        for query_id, scores in method_scores[method].items():
            output[method][query_id] = {paper_id: rank for rank, paper_id in enumerate(rank_from_scores(scores), start=1)}
    return output


def row_lookup(rows_by_q: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(query_id, int(row["paper_id"])): row for query_id, rows in rows_by_q.items() for row in rows}


def feature_values(query_id: str, paper_id: int, rows_lookup: dict[tuple[str, int], dict[str, Any]], scored: dict[str, dict[int, dict[str, float]]]) -> dict[str, float | None]:
    row = dict(rows_lookup.get((query_id, paper_id), {}))
    row.update(scored.get(query_id, {}).get(paper_id, {}))
    values: dict[str, float | None] = {}
    for feature in FEATURES_FOR_EXAMPLES:
        value = row.get(feature)
        try:
            values[feature] = float(value) if value is not None else None
        except (TypeError, ValueError):
            values[feature] = None
    return values


def example(
    *,
    kind: str,
    query_id: str,
    paper_id: int,
    label_lookup: dict[tuple[str, int], dict[str, Any]],
    ranks: dict[str, dict[str, dict[int, int]]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    rows_lookup: dict[tuple[str, int], dict[str, Any]],
    scored: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    label = label_lookup.get((query_id, paper_id), {})
    row = rows_lookup.get((query_id, paper_id), {})
    return {
        "kind": kind,
        "query_id": query_id,
        "paper_id": paper_id,
        "title": label.get("title") or row.get("title"),
        "judged_source": label.get("judged_source"),
        "primary_role": label.get("primary_role"),
        "reading_value_score": label.get("reading_value_score"),
        "topic_match_score": label.get("topic_match_score"),
        "label_confidence": label.get("label_confidence"),
        "v3_3_rank": ranks[V33_METHOD].get(query_id, {}).get(paper_id),
        "v4_0_rank": ranks[V40_METHOD].get(query_id, {}).get(paper_id),
        "v3_3_score": method_scores[V33_METHOD].get(query_id, {}).get(paper_id),
        "v4_0_score": method_scores[V40_METHOD].get(query_id, {}).get(paper_id),
        "feature_values": feature_values(query_id, paper_id, rows_lookup, scored),
    }


def disagreement_examples(
    *,
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    label_lookup: dict[tuple[str, int], dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    scored: dict[str, dict[int, dict[str, float]]],
) -> dict[str, list[dict[str, Any]]]:
    rows_lookup = row_lookup(production_rows_by_q)
    ranks = rank_maps(method_scores, [V33_METHOD, V40_METHOD])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id, rows in production_rows_by_q.items():
        for row in rows:
            paper_id = int(row["paper_id"])
            label = label_lookup.get((query_id, paper_id))
            v33_rank = ranks[V33_METHOD].get(query_id, {}).get(paper_id, 999999)
            v40_rank = ranks[V40_METHOD].get(query_id, {}).get(paper_id, 999999)
            if is_high_reading(label) and v33_rank <= 10 and v40_rank > v33_rank + 4:
                groups["v3_3_high_reading_above_v4_0"].append(
                    example(kind="v3_3_high_reading_above_v4_0", query_id=query_id, paper_id=paper_id, label_lookup=label_lookup, ranks=ranks, method_scores=method_scores, rows_lookup=rows_lookup, scored=scored)
                )
            if is_hard_negative(label) and v33_rank <= 10 and v40_rank > 10:
                groups["v4_0_fixes_v3_3_hard_negative"].append(
                    example(kind="v4_0_fixes_v3_3_hard_negative", query_id=query_id, paper_id=paper_id, label_lookup=label_lookup, ranks=ranks, method_scores=method_scores, rows_lookup=rows_lookup, scored=scored)
                )
            if v40_rank <= 10 and v33_rank > 10:
                groups["v4_0_promoted_v3_3_demoted"].append(
                    example(kind="v4_0_promoted_v3_3_demoted", query_id=query_id, paper_id=paper_id, label_lookup=label_lookup, ranks=ranks, method_scores=method_scores, rows_lookup=rows_lookup, scored=scored)
                )
            if v33_rank <= 10 and v40_rank > 10:
                groups["v3_3_promoted_v4_0_demoted"].append(
                    example(kind="v3_3_promoted_v4_0_demoted", query_id=query_id, paper_id=paper_id, label_lookup=label_lookup, ranks=ranks, method_scores=method_scores, rows_lookup=rows_lookup, scored=scored)
                )
    for key, values in groups.items():
        values.sort(key=lambda item: ((item.get("reading_value_score") or 0.0), (item.get("v4_0_rank") or 999999) - (item.get("v3_3_rank") or 999999)), reverse=True)
        groups[key] = values[:25]
    return dict(groups)


def top_source_distribution(evaluation: dict[str, Any], label_lookup: dict[tuple[str, int], dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    output = {}
    for method in methods:
        counter: Counter[str] = Counter()
        positive_counter: Counter[str] = Counter()
        for topic_row in evaluation["per_topic"][method]:
            query_id = str(topic_row["query_id"])
            for paper_id in topic_row["retrieved_ids_at_k"]:
                label = label_lookup.get((query_id, int(paper_id)))
                source = str(label.get("judged_source")) if label else "unjudged"
                counter[source] += 1
                if is_positive(label):
                    positive_counter[source] += 1
        output[method] = {"top10_by_source": dict(counter), "positive_top10_by_source": dict(positive_counter)}
    return output


def lexical_reliance(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        return {"example_count": 0}
    lexical_ranked = sum(1 for row in examples if (row["feature_values"].get("bm25_rank") or 999999) <= 10)
    semantic_ranked = sum(1 for row in examples if (row["feature_values"].get("embedding_score") or 0.0) >= (row["feature_values"].get("bm25_score") or 0.0))
    return {
        "example_count": len(examples),
        "bm25_rank_top10_fraction": lexical_ranked / len(examples),
        "embedding_score_gte_bm25_score_fraction": semantic_ranked / len(examples),
        "interpretation": "High BM25 top-10 fraction among V4.0-only promotions is evidence of lexical over-reliance.",
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = load_all_labels(
        {
            "v2_1": resolve_repo_path(args.v21_labels),
            "v2_5": resolve_repo_path(args.v25_labels),
            "v3_2": resolve_repo_path(args.v32_labels),
            "v3_5": resolve_repo_path(args.v35_labels),
            "v3_9": resolve_repo_path(args.v39_labels),
        }
    )
    label_lookup = labels_by_key(labels)
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
    model_diagnostics[V40_METHOD] = {
        "metadata_path": str(resolve_repo_path(args.v40_model_dir) / "random_forest_no_v27_compact.metadata.json"),
        "feature_columns": v40_payload["metadata"]["feature_columns"],
        "feature_importances": v40_payload["metadata"].get("estimator_info", {}).get("feature_importances", []),
    }
    evaluation = evaluate_rankings(candidate_rows_by_q=production_rows_by_q, labels=labels, method_scores=method_scores, k=K)
    examples_by_kind = disagreement_examples(production_rows_by_q=production_rows_by_q, label_lookup=label_lookup, method_scores=method_scores, scored=scored)
    flat_examples = [row for rows in examples_by_kind.values() for row in rows]
    deltas = weak_topic_deltas(evaluation, V40_METHOD, V33_METHOD)
    v40_only = examples_by_kind.get("v4_0_promoted_v3_3_demoted", [])
    v33_only = examples_by_kind.get("v3_3_promoted_v4_0_demoted", [])
    report = {
        "schema_version": "v4.1_v4_0_vs_v3_3_error_analysis",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "runtime_integration_status": "offline_diagnostics_only",
        "methods": {"v3_3": V33_METHOD, "v4_0": V40_METHOD},
        "dataset": {"label_count": len(labels), "labels_by_source": dict(Counter(str(row["judged_source"]) for row in labels))},
        "production_diagnostics": production_diagnostics,
        "model_diagnostics": model_diagnostics,
        "evaluation": evaluation,
        "per_topic_v4_0_minus_v3_3": deltas,
        "summary": {
            "v4_0_reading_delta_vs_v3_3": evaluation["method_averages"][V40_METHOD]["reading_value_ndcg"] - evaluation["method_averages"][V33_METHOD]["reading_value_ndcg"],
            "v4_0_topic_delta_vs_v3_3": evaluation["method_averages"][V40_METHOD]["topic_match_ndcg"] - evaluation["method_averages"][V33_METHOD]["topic_match_ndcg"],
            "v4_0_hard_negative_delta_vs_v3_3": evaluation["method_averages"][V40_METHOD]["hard_negative_rate_at_10"] - evaluation["method_averages"][V33_METHOD]["hard_negative_rate_at_10"],
            "topics_v4_0_wins_reading": [row["query_id"] for row in deltas if row["reading_delta"] > 0],
            "topics_v4_0_loses_reading": [row["query_id"] for row in deltas if row["reading_delta"] < 0],
            "v4_0_only_promotions": len(v40_only),
            "v3_3_only_promotions": len(v33_only),
            "v4_0_fixed_v3_3_hard_negatives": len(examples_by_kind.get("v4_0_fixes_v3_3_hard_negative", [])),
        },
        "label_source_distribution_among_top10": top_source_distribution(evaluation, label_lookup, [V33_METHOD, V40_METHOD]),
        "lexical_reliance_diagnostics": lexical_reliance(v40_only),
        "semantic_hidden_positive_diagnostics": {
            "v3_3_only_positive_count": sum(1 for row in v33_only if is_positive(label_lookup.get((row["query_id"], int(row["paper_id"]))))),
            "v4_0_only_positive_count": sum(1 for row in v40_only if is_positive(label_lookup.get((row["query_id"], int(row["paper_id"]))))),
            "interpretation": "If V3.3-only positives exceed V4.0-only positives, V4.0 is under-ranking semantic or hidden-positive papers that V3.3 still surfaces.",
        },
        "v3_9_conservatism_diagnostics": {
            "v4_0_top10_v3_9_count": top_source_distribution(evaluation, label_lookup, [V40_METHOD])[V40_METHOD]["top10_by_source"].get("v3_9", 0),
            "v3_3_top10_v3_9_count": top_source_distribution(evaluation, label_lookup, [V33_METHOD])[V33_METHOD]["top10_by_source"].get("v3_9", 0),
            "interpretation": "V4.0 was trained with many V3.9 active-learning negatives; conservative behavior is indicated when it demotes V3.3 positives while improving hard negatives.",
        },
        "examples_by_kind": examples_by_kind,
        "protected_hashes": protected_hashes(
            resolve_repo_path(args.v21_labels),
            resolve_repo_path(args.v25_labels),
            resolve_repo_path(args.v32_labels),
            resolve_repo_path(args.v35_labels),
            resolve_repo_path(args.selected_240),
        ),
    }
    return report, flat_examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# V4.1 V4.0 vs V3.3 Error Analysis",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- V4.0 reading delta vs V3.3: `{summary['v4_0_reading_delta_vs_v3_3']:+.3f}`",
        f"- V4.0 topic delta vs V3.3: `{summary['v4_0_topic_delta_vs_v3_3']:+.3f}`",
        f"- V4.0 hard-negative delta vs V3.3: `{summary['v4_0_hard_negative_delta_vs_v3_3']:+.3f}`",
        f"- V4.0 reading wins: `{len(summary['topics_v4_0_wins_reading'])}` topics",
        f"- V4.0 reading losses: `{len(summary['topics_v4_0_loses_reading'])}` topics",
        f"- V4.0 fixed V3.3 hard negatives: `{summary['v4_0_fixed_v3_3_hard_negatives']}` examples",
        "",
        "## Per-Topic Deltas",
        "",
        "| topic | reading delta | topic delta | hard-neg delta |",
        "|---|---:|---:|---:|",
    ]
    for row in report["per_topic_v4_0_minus_v3_3"]:
        lines.append(f"| `{row['query_id']}` | {row['reading_delta']:+.3f} | {row['topic_delta']:+.3f} | {row['hard_negative_delta']:+.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- V4.0 improves the hard-negative rate but gives back reading/topic NDCG on most topics.",
            "- V4.0-only promotions have the lexical-reliance diagnostics shown in the JSON report.",
            "- V3.3-only positive counts in the JSON report indicate whether V4.0 is demoting semantic hidden positives.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze V4.0 RF no-V2.7 errors against V3.3 RF no-V2.7.")
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
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--examples-out", default=str(DEFAULT_EXAMPLES_OUT.relative_to(REPO_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, examples = build_report(args)
    report["command"] = " ".join(sys.argv)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), build_markdown(report))
    write_jsonl(resolve_repo_path(args.examples_out), examples)
    print("V4.1 V4.0 vs V3.3 error analysis complete")
    print(f"V4.0 vs V3.3 reading delta: {report['summary']['v4_0_reading_delta_vs_v3_3']:+.3f}")
    print(f"V4.0 fixed hard-negative examples: {report['summary']['v4_0_fixed_v3_3_hard_negatives']}")
    print(f"Report: {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
