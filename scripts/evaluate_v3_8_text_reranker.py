import argparse
from collections import Counter
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

from evaluate_v2_4_true_production_candidate_pool import load_jsonl, write_json, write_text  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import (  # noqa: E402
    DEFAULT_OLD_MODEL,
    DEFAULT_V21_PACKET,
    DEFAULT_V26_MODEL,
    DEFAULT_V33_MODEL_DIR,
    K,
    build_method_scores,
    evaluate_rankings,
    rank_from_scores,
    score_rows,
    top10_overlap,
)
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    load_labels_with_sources,
    protected_hashes,
)
from train_v2_6_production_aware_learned_hybrid import production_candidate_rows  # noqa: E402


DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_8_text_reranker_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_8_text_reranker_report.md"
DEFAULT_CACHE_OUT = REPO_ROOT / "data" / "eval" / "cache" / "v3_8_text_reranker_scores.jsonl"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
V33_METHOD = "v3_3_random_forest_no_v27_compact"
WEAK_TOPICS = {
    "v2_ai_for_scientific_discovery",
    "v2_causal_representation_learning",
    "v2_large_language_model_agents",
    "v2_llm_evaluation",
    "v2_retrieval_augmented_generation",
    "v2_transformer_architecture",
}
TABLE_METHODS = [
    "v3_8_cross_encoder",
    "v3_8_v3_3_ce_blend_0_8_0_2",
    "v3_8_v2_7_ce_blend_0_8_0_2",
    V33_METHOD,
    "v2_7",
    "v2_6",
    "hybrid",
]


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high <= low:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def text_for_row(row: dict[str, Any]) -> str:
    return f"Title: {row.get('title') or ''}\nAbstract: {row.get('abstract') or ''}"


def load_score_cache(path: Path, model_name: str) -> dict[tuple[str, int], float]:
    if not path.exists():
        return {}
    output: dict[tuple[str, int], float] = {}
    for row in load_jsonl(path):
        if row.get("model_name") != model_name:
            continue
        output[(str(row["query_id"]), int(row["paper_id"]))] = float(row["score"])
    return output


def append_score_cache(path: Path, model_name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "schema_version": "v3.8_text_reranker_score",
                        "created_at": datetime.now(UTC).isoformat(),
                        "model_name": model_name,
                        "query_id": row["query_id"],
                        "query": row["query"],
                        "paper_id": int(row["paper_id"]),
                        "score": float(row["score"]),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )


def try_load_cross_encoder(model_name: str) -> tuple[Any | None, dict[str, Any]]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        return None, {
            "available": False,
            "stage": "import",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "recommended_option": "Install sentence-transformers with CrossEncoder support before rerunning V3.8.",
        }
    try:
        return CrossEncoder(model_name), {"available": True, "model_name": model_name}
    except Exception as exc:
        return None, {
            "available": False,
            "stage": "load_model",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_name": model_name,
            "recommended_option": "Use a locally available reranker or allow loading cross-encoder/ms-marco-MiniLM-L-6-v2.",
        }


def score_cross_encoder(
    *,
    model: Any,
    model_name: str,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    cache_path: Path,
    batch_size: int,
) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    cached = load_score_cache(cache_path, model_name)
    missing_pairs: list[tuple[str, str]] = []
    missing_rows: list[dict[str, Any]] = []
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        for row in rows:
            key = (query_id, int(row["paper_id"]))
            if key in cached:
                continue
            missing_rows.append({"query_id": query_id, "query": row["query"], "paper_id": int(row["paper_id"])})
            missing_pairs.append((str(row["query"]), text_for_row(row)))
    if missing_pairs:
        scores = model.predict(missing_pairs, batch_size=batch_size, show_progress_bar=True)
        cache_rows = []
        for row, score in zip(missing_rows, scores, strict=True):
            row["score"] = float(score)
            cached[(str(row["query_id"]), int(row["paper_id"]))] = float(score)
            cache_rows.append(row)
        append_score_cache(cache_path, model_name, cache_rows)
    output: dict[str, dict[int, float]] = {}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        output[query_id] = {
            int(row["paper_id"]): float(cached[(query_id, int(row["paper_id"]))])
            for row in rows
        }
    return output, {
        "cache_path": str(cache_path),
        "cached_scores_before": len(cached) - len(missing_rows),
        "computed_scores": len(missing_rows),
        "total_scores_used": sum(len(rows) for rows in output.values()),
    }


def normalized_scores_by_query(scores_by_q: dict[str, dict[int, float]]) -> dict[str, dict[int, float]]:
    return {query_id: minmax(scores) for query_id, scores in scores_by_q.items()}


def blend_scores(
    base_scores: dict[str, dict[int, float]],
    ce_scores_norm: dict[str, dict[int, float]],
    *,
    base_weight: float = 0.8,
    ce_weight: float = 0.2,
) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {}
    for query_id, scores in base_scores.items():
        output[query_id] = {
            paper_id: base_weight * float(score) + ce_weight * float(ce_scores_norm.get(query_id, {}).get(paper_id, 0.0))
            for paper_id, score in scores.items()
        }
    return output


def per_topic_delta(evaluation: dict[str, Any], left: str, right: str) -> list[dict[str, Any]]:
    left_rows = {row["query_id"]: row for row in evaluation["per_topic"][left]}
    right_rows = {row["query_id"]: row for row in evaluation["per_topic"][right]}
    rows = []
    for query_id in sorted(set(left_rows) & set(right_rows)):
        rows.append(
            {
                "query_id": query_id,
                "weak_topic": query_id in WEAK_TOPICS,
                "reading_delta": left_rows[query_id]["reading_value_ndcg"] - right_rows[query_id]["reading_value_ndcg"],
                "topic_delta": left_rows[query_id]["topic_match_ndcg"] - right_rows[query_id]["topic_match_ndcg"],
                "left_reading_ndcg": left_rows[query_id]["reading_value_ndcg"],
                "right_reading_ndcg": right_rows[query_id]["reading_value_ndcg"],
            }
        )
    return sorted(rows, key=lambda row: row["reading_delta"])


def build_examples(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    left: str,
    right: str,
) -> dict[str, list[dict[str, Any]]]:
    label_lookup = {(str(row["query_id"]), int(row["paper_id"])): row for row in labels}
    row_lookup = {(query_id, int(row["paper_id"])): row for query_id, rows in candidate_rows_by_q.items() for row in rows}
    fixes: list[dict[str, Any]] = []
    hurts: list[dict[str, Any]] = []
    for query_id in sorted(candidate_rows_by_q):
        left_top = rank_from_scores(method_scores[left][query_id])[:10]
        right_top = rank_from_scores(method_scores[right][query_id])[:10]
        for paper_id in left_top:
            if paper_id in right_top:
                continue
            label = label_lookup.get((query_id, paper_id))
            if label and float(label.get("reading_value_score") or 0.0) >= 0.5 and float(label.get("topic_match_score") or 0.0) >= 0.5:
                row = row_lookup.get((query_id, paper_id), {})
                fixes.append({"query_id": query_id, "paper_id": paper_id, "title": row.get("title"), "reason": f"{left}_adds_positive_missed_by_{right}"})
            if label and str(label.get("primary_role")) == "negative":
                row = row_lookup.get((query_id, paper_id), {})
                hurts.append({"query_id": query_id, "paper_id": paper_id, "title": row.get("title"), "reason": f"{left}_adds_known_negative"})
        for paper_id in right_top:
            if paper_id in left_top:
                continue
            label = label_lookup.get((query_id, paper_id))
            if label and float(label.get("reading_value_score") or 0.0) >= 0.5 and float(label.get("topic_match_score") or 0.0) >= 0.5:
                row = row_lookup.get((query_id, paper_id), {})
                hurts.append({"query_id": query_id, "paper_id": paper_id, "title": row.get("title"), "reason": f"{left}_drops_{right}_positive"})
    return {"fixes": fixes[:30], "hurts": hurts[:30]}


def unavailable_report(reason: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "v3.8_text_reranker_report",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "not_run",
        "cross_encoder": reason,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "protected_hashes": protected_hashes(DEFAULT_V21_LABELS, DEFAULT_V25_LABELS, DEFAULT_V32_LABELS, DEFAULT_V35_LABELS, DEFAULT_SELECTED_240),
        "command": " ".join(sys.argv),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    model, model_status = try_load_cross_encoder(args.cross_encoder_model)
    if model is None:
        return unavailable_report(model_status, args)
    labels, labels_by_source = load_labels_with_sources(
        v21_labels_path=DEFAULT_V21_LABELS,
        v25_labels_path=DEFAULT_V25_LABELS,
        v32_labels_path=DEFAULT_V32_LABELS,
        v35_labels_path=DEFAULT_V35_LABELS,
    )
    candidate_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=load_jsonl(DEFAULT_V21_PACKET), top_k=args.top_k)
    scored = score_rows(
        candidate_rows_by_q=candidate_rows_by_q,
        context_rows_by_q=candidate_rows_by_q,
        old_model_path=DEFAULT_OLD_MODEL,
        v26_model_path=DEFAULT_V26_MODEL,
    )
    method_scores, model_diagnostics = build_method_scores(
        production_rows_by_q=candidate_rows_by_q,
        scored=scored,
        model_dir=DEFAULT_V33_MODEL_DIR,
    )
    ce_raw_scores, cache_diagnostics = score_cross_encoder(
        model=model,
        model_name=args.cross_encoder_model,
        candidate_rows_by_q=candidate_rows_by_q,
        cache_path=Path(args.cache_out) if Path(args.cache_out).is_absolute() else REPO_ROOT / args.cache_out,
        batch_size=args.batch_size,
    )
    ce_norm_scores = normalized_scores_by_query(ce_raw_scores)
    method_scores["v3_8_cross_encoder"] = ce_norm_scores
    method_scores["v3_8_v3_3_ce_blend_0_8_0_2"] = blend_scores(method_scores[V33_METHOD], ce_norm_scores)
    method_scores["v3_8_v2_7_ce_blend_0_8_0_2"] = blend_scores(method_scores["v2_7"], ce_norm_scores)
    evaluation = evaluate_rankings(candidate_rows_by_q=candidate_rows_by_q, labels=labels, method_scores=method_scores, k=K)
    avg = evaluation["method_averages"]
    best_method = max(TABLE_METHODS, key=lambda method: avg[method]["reading_value_ndcg"])
    examples = build_examples(
        candidate_rows_by_q=candidate_rows_by_q,
        labels=labels,
        method_scores=method_scores,
        left="v3_8_cross_encoder",
        right=V33_METHOD,
    )
    weak_deltas = [row for row in per_topic_delta(evaluation, "v3_8_cross_encoder", V33_METHOD) if row["weak_topic"]]
    report = {
        "schema_version": "v3.8_text_reranker_report",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "complete",
        "cross_encoder": {"available": True, "model_name": args.cross_encoder_model},
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "dataset": {"label_count": len(labels), "labels_by_source": {key: len(value) for key, value in labels_by_source.items()}},
        "candidate_pool": {
            "top_k_per_method": args.top_k,
            "topics": len(candidate_rows_by_q),
            "candidate_count": sum(len(rows) for rows in candidate_rows_by_q.values()),
        },
        "cache": cache_diagnostics,
        "production_diagnostics": production_diagnostics,
        "model_diagnostics": model_diagnostics,
        "evaluation": evaluation,
        "top10_overlap": {
            "cross_encoder_vs_v3_3": top10_overlap(evaluation["per_topic"], "v3_8_cross_encoder", V33_METHOD),
            "cross_encoder_blend_vs_v3_3": top10_overlap(evaluation["per_topic"], "v3_8_v3_3_ce_blend_0_8_0_2", V33_METHOD),
        },
        "comparison": {
            "best_method_by_reading_ndcg": best_method,
            "cross_encoder_reading_delta_vs_v3_3": avg["v3_8_cross_encoder"]["reading_value_ndcg"] - avg[V33_METHOD]["reading_value_ndcg"],
            "cross_encoder_topic_delta_vs_v3_3": avg["v3_8_cross_encoder"]["topic_match_ndcg"] - avg[V33_METHOD]["topic_match_ndcg"],
            "blend_reading_delta_vs_v3_3": avg["v3_8_v3_3_ce_blend_0_8_0_2"]["reading_value_ndcg"] - avg[V33_METHOD]["reading_value_ndcg"],
            "blend_topic_delta_vs_v3_3": avg["v3_8_v3_3_ce_blend_0_8_0_2"]["topic_match_ndcg"] - avg[V33_METHOD]["topic_match_ndcg"],
            "cross_encoder_reading_delta_vs_v2_7": avg["v3_8_cross_encoder"]["reading_value_ndcg"] - avg["v2_7"]["reading_value_ndcg"],
            "cross_encoder_topic_delta_vs_v2_7": avg["v3_8_cross_encoder"]["topic_match_ndcg"] - avg["v2_7"]["topic_match_ndcg"],
            "text_understanding_improves_over_v3_3": avg["v3_8_cross_encoder"]["reading_value_ndcg"] > avg[V33_METHOD]["reading_value_ndcg"],
            "blend_improves_over_v3_3": avg["v3_8_v3_3_ce_blend_0_8_0_2"]["reading_value_ndcg"] > avg[V33_METHOD]["reading_value_ndcg"],
            "recommend_v3_9_cross_encoder_score_feature": avg["v3_8_v3_3_ce_blend_0_8_0_2"]["reading_value_ndcg"] > avg[V33_METHOD]["reading_value_ndcg"],
        },
        "weak_topic_analysis": {
            "weak_topics": sorted(WEAK_TOPICS),
            "cross_encoder_vs_v3_3": weak_deltas,
            "wins": [row for row in weak_deltas if row["reading_delta"] > 0],
            "losses": [row for row in weak_deltas if row["reading_delta"] < 0],
        },
        "examples": examples,
        "protected_hashes": protected_hashes(DEFAULT_V21_LABELS, DEFAULT_V25_LABELS, DEFAULT_V32_LABELS, DEFAULT_V35_LABELS, DEFAULT_SELECTED_240),
        "command": " ".join(sys.argv),
    }
    return report


def method_table(averages: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | reading NDCG@10 | topic NDCG@10 | judged@10 | unjudged@10 | hard-neg rate@10 | recall@10 | MRR@10 | role cov | path cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in TABLE_METHODS:
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['reading_value_ndcg']:.3f} | {row['topic_match_ndcg']:.3f} | "
            f"{row['judged_at_10']:.2f} | {row['unjudged_at_10']:.2f} | {row['hard_negative_rate_at_10']:.3f} | "
            f"{row['positive_recall_at_10']:.3f} | {row['mrr_at_10']:.3f} | {row['role_coverage_at_10']:.3f} | "
            f"{row['path_coverage_at_10']:.3f} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3.8 Text Reranker Evaluation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Status: `{report['status']}`",
    ]
    if report["status"] != "complete":
        lines.extend(
            [
                f"- Cross-encoder available: `{report['cross_encoder'].get('available')}`",
                f"- Error: `{report['cross_encoder'].get('error_type')}: {report['cross_encoder'].get('error')}`",
                f"- Recommended option: {report['cross_encoder'].get('recommended_option')}",
            ]
        )
        return "\n".join(lines) + "\n"
    avg = report["evaluation"]["method_averages"]
    comparison = report["comparison"]
    lines.extend(
        [
            f"- Cross-encoder model: `{report['cross_encoder']['model_name']}`",
            f"- Candidate rows scored: `{report['candidate_pool']['candidate_count']}`",
            f"- Cached scores reused/computed: `{report['cache']['total_scores_used'] - report['cache']['computed_scores']}` / `{report['cache']['computed_scores']}`",
            "",
            "## Metrics",
            "",
        ]
    )
    lines.extend(method_table(avg))
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- Cross-encoder reading delta vs V3.3: `{comparison['cross_encoder_reading_delta_vs_v3_3']:+.3f}`",
            f"- Cross-encoder topic delta vs V3.3: `{comparison['cross_encoder_topic_delta_vs_v3_3']:+.3f}`",
            f"- V3.3 + cross-encoder blend reading delta vs V3.3: `{comparison['blend_reading_delta_vs_v3_3']:+.3f}`",
            f"- V3.3 + cross-encoder blend topic delta vs V3.3: `{comparison['blend_topic_delta_vs_v3_3']:+.3f}`",
            f"- Cross-encoder reading delta vs V2.7: `{comparison['cross_encoder_reading_delta_vs_v2_7']:+.3f}`",
            f"- Text understanding improves over V3.3: `{comparison['text_understanding_improves_over_v3_3']}`",
            f"- Recommend V3.9 cross-encoder feature: `{comparison['recommend_v3_9_cross_encoder_score_feature']}`",
            "",
            "## Weak Topics",
            "",
            "| topic | reading delta CE vs V3.3 | topic delta CE vs V3.3 |",
            "|---|---:|---:|",
        ]
    )
    for row in report["weak_topic_analysis"]["cross_encoder_vs_v3_3"]:
        lines.append(f"| `{row['query_id']}` | {row['reading_delta']:+.3f} | {row['topic_delta']:+.3f} |")
    lines.extend(
        [
            "",
            "## Examples",
            "",
            f"- Cross-encoder fixes over V3.3: `{len(report['examples']['fixes'])}`",
            f"- Cross-encoder hurts vs V3.3: `{len(report['examples']['hurts'])}`",
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
    parser = argparse.ArgumentParser(description="Evaluate a zero-shot text reranker over the production candidate pool.")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--cache-out", default=str(DEFAULT_CACHE_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    json_out = Path(args.json_out)
    if not json_out.is_absolute():
        json_out = REPO_ROOT / json_out
    md_out = Path(args.md_out)
    if not md_out.is_absolute():
        md_out = REPO_ROOT / md_out
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    print(f"V3.8 status: {report['status']}")
    if report["status"] == "complete":
        print(f"Cross-encoder delta vs V3.3: {report['comparison']['cross_encoder_reading_delta_vs_v3_3']:+.3f}")
        print(f"Blend delta vs V3.3: {report['comparison']['blend_reading_delta_vs_v3_3']:+.3f}")
    else:
        print(f"Cross-encoder unavailable: {report['cross_encoder']}")
    print(f"Report: {json_out}")


if __name__ == "__main__":
    main()
