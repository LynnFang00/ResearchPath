import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    EXPECTED_SELECTED_240_SHA256,
    PATH_ROLES,
    ROLE_VALUES_FOR_COVERAGE,
    SELECTED_240,
    clamp_score,
    duplicate_penalty_at_k,
    file_sha256,
    labels_by_query,
    load_json,
    load_jsonl,
    ndcg_at_k,
    path_level_coverage_at_k,
    resolve_repo_path,
    role_coverage_at_k,
    score_map,
    write_json,
    write_text,
)
from evaluate_v2_7_score_blends import (  # noqa: E402
    EXPECTED_V25_LABELS_SHA256,
    fixed_and_penalty_scores,
    score_rows,
)
from train_v2_6_production_aware_learned_hybrid import (  # noqa: E402
    EXPECTED_V21_LABELS_SHA256,
    flatten_rows,
    production_candidate_rows,
    rows_by_query,
)
from train_v3_0_ltr_models import numeric_feature_value  # noqa: E402


DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V32_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
DEFAULT_V21_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_OLD_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_V26_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"
DEFAULT_V33_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800"
DEFAULT_V33_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v3_3_ltr_800_training_report.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_production_pool_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_production_pool_report.md"
DEFAULT_EXAMPLES_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_error_examples.jsonl"

BASELINE_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
REFERENCE_METHODS = [
    "hybrid",
    "embedding",
    "faiss_embedding",
    "old_v2_2b",
    "v2_6",
    "v2_7",
    "v3_3_random_forest_no_v27_compact",
    "v3_3_hist_gradient_boosting_no_v27_compact",
]
K = 10
SELECTED_V27_METHOD = "blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["query_id"]), int(row["paper_id"]))


def labels_with_source(path: Path, source: str) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        copied = dict(row)
        copied["judged_source"] = source
        copied["hard_negative"] = is_hard_negative(copied)
        copied["positive"] = is_positive(copied)
        rows.append(copied)
    return rows


def is_positive(label: dict[str, Any]) -> bool:
    return clamp_score(label.get("reading_value_score")) >= 0.5 and clamp_score(label.get("topic_match_score")) >= 0.5


def is_hard_negative(label: dict[str, Any]) -> bool:
    return str(label.get("primary_role")) == "negative" or (
        clamp_score(label.get("reading_value_score")) <= 0.0 and clamp_score(label.get("topic_match_score")) <= 0.0
    )


def mrr_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> float:
    for rank, paper_id in enumerate(retrieved_ids[:k], start=1):
        label = labels.get(paper_id)
        if label and is_positive(label):
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> float:
    positives = {paper_id for paper_id, label in labels.items() if is_positive(label)}
    if not positives:
        return 0.0
    retrieved = set(retrieved_ids[:k]) & positives
    return len(retrieved) / len(positives)


def hard_negative_rate_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for paper_id in top if paper_id in labels and is_hard_negative(labels[paper_id])) / len(top)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = [
        "reading_value_ndcg",
        "topic_match_ndcg",
        "judged_at_10",
        "unjudged_at_10",
        "hard_negative_rate_at_10",
        "positive_recall_at_10",
        "mrr_at_10",
        "role_coverage_at_10",
        "path_coverage_at_10",
        "duplicate_penalty_at_10",
        "candidate_pool_size",
        "judged_candidates_in_pool",
    ]
    return {field: sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0 for field in fields}


def rank_from_scores(scores: dict[int, float]) -> list[int]:
    return [paper_id for paper_id, _ in sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True)]


def evaluate_rankings(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    k: int,
) -> dict[str, Any]:
    label_lookup = labels_by_query(labels)
    per_method: dict[str, list[dict[str, Any]]] = {method: [] for method in method_scores}
    rankings: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for method, scores_by_q in method_scores.items():
        for query_id, scores in scores_by_q.items():
            rankings[query_id][method] = rank_from_scores(scores)

    for query_id, rows in sorted(candidate_rows_by_q.items()):
        candidate_ids = {int(row["paper_id"]) for row in rows}
        query_labels = label_lookup.get(query_id, {})
        judged_in_pool = {paper_id: label for paper_id, label in query_labels.items() if paper_id in candidate_ids}
        for method in method_scores:
            retrieved_ids = rankings.get(query_id, {}).get(method, [])
            judged_top = sum(1 for paper_id in retrieved_ids[:k] if paper_id in query_labels)
            per_method[method].append(
                {
                    "query_id": query_id,
                    "method": method,
                    "retrieved_ids_at_k": retrieved_ids[:k],
                    "candidate_pool_size": len(candidate_ids),
                    "judged_candidates_in_pool": len(judged_in_pool),
                    "judged_at_10": judged_top,
                    "unjudged_at_10": max(0, min(k, len(retrieved_ids)) - judged_top),
                    "reading_value_ndcg": ndcg_at_k(retrieved_ids, score_map(query_labels, "reading_value_score"), k),
                    "topic_match_ndcg": ndcg_at_k(retrieved_ids, score_map(query_labels, "topic_match_score"), k),
                    "mrr_at_10": mrr_at_k(retrieved_ids, query_labels, k),
                    "positive_recall_at_10": recall_at_k(retrieved_ids, query_labels, k),
                    "hard_negative_rate_at_10": hard_negative_rate_at_k(retrieved_ids, query_labels, k),
                    "role_coverage_at_10": role_coverage_at_k(retrieved_ids, query_labels, k)["coverage"],
                    "path_coverage_at_10": path_level_coverage_at_k(retrieved_ids, query_labels, k)["coverage"],
                    "duplicate_penalty_at_10": duplicate_penalty_at_k(retrieved_ids, query_labels, k),
                }
            )
    return {
        "method_averages": {method: aggregate(rows) for method, rows in per_method.items()},
        "per_topic": per_method,
    }


def load_v33_model(model_dir: Path, model_name: str) -> dict[str, Any]:
    metadata_path = model_dir / f"{model_name}.metadata.json"
    artifact_path = model_dir / f"{model_name}.joblib"
    metadata = load_json(metadata_path)
    estimator = joblib.load(artifact_path)
    return {"model_name": model_name, "metadata": metadata, "estimator": estimator}


def feature_row(packet_row: dict[str, Any], scored_values: dict[str, float]) -> dict[str, Any]:
    row = dict(packet_row)
    row.update(scored_values)
    return row


def v33_scores_for_model(
    *,
    model_payload: dict[str, Any],
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    scored: dict[str, dict[int, dict[str, float]]],
) -> dict[str, dict[int, float]]:
    estimator = model_payload["estimator"]
    features = list(model_payload["metadata"]["feature_columns"])
    output: dict[str, dict[int, float]] = {}
    for query_id, rows in sorted(production_rows_by_q.items()):
        query_scores: dict[int, float] = {}
        for row in rows:
            paper_id = int(row["paper_id"])
            values = scored[query_id][paper_id]
            merged = feature_row(row, values)
            x_values = np.array([[numeric_feature_value(merged, feature) for feature in features]], dtype=float)
            query_scores[paper_id] = float(estimator.predict(x_values)[0])
        output[query_id] = query_scores
    return output


def build_method_scores(
    *,
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    scored: dict[str, dict[int, dict[str, float]]],
    model_dir: Path,
) -> tuple[dict[str, dict[str, dict[int, float]]], dict[str, Any]]:
    v27_scores = fixed_and_penalty_scores(scored)[SELECTED_V27_METHOD]
    method_scores: dict[str, dict[str, dict[int, float]]] = {
        "hybrid": {},
        "embedding": {},
        "faiss_embedding": {},
        "old_v2_2b": {},
        "v2_6": {},
        "v2_7": v27_scores,
    }
    for query_id, rows in scored.items():
        method_scores["hybrid"][query_id] = {paper_id: values["hybrid_score"] for paper_id, values in rows.items()}
        method_scores["embedding"][query_id] = {paper_id: values["embedding_score"] for paper_id, values in rows.items()}
        method_scores["faiss_embedding"][query_id] = {paper_id: values["faiss_embedding_score"] for paper_id, values in rows.items()}
        method_scores["old_v2_2b"][query_id] = {paper_id: values["old_v2_2b_score"] for paper_id, values in rows.items()}
        method_scores["v2_6"][query_id] = {paper_id: values["v2_6_score"] for paper_id, values in rows.items()}

    model_diagnostics: dict[str, Any] = {}
    for model_name in ["random_forest_no_v27_compact", "hist_gradient_boosting_no_v27_compact"]:
        payload = load_v33_model(model_dir, model_name)
        method_name = f"v3_3_{model_name}"
        method_scores[method_name] = v33_scores_for_model(
            model_payload=payload,
            production_rows_by_q=production_rows_by_q,
            scored=scored,
        )
        model_diagnostics[method_name] = {
            "metadata_path": str(model_dir / f"{model_name}.metadata.json"),
            "artifact_path": str(model_dir / f"{model_name}.joblib"),
            "feature_columns": payload["metadata"]["feature_columns"],
            "feature_importances": payload["metadata"].get("estimator_info", {}).get("feature_importances", []),
            "raw_score_feature_intersection": payload["metadata"].get("raw_score_feature_intersection", []),
            "uses_v2_7_feature": payload["metadata"].get("uses_v2_7_feature"),
        }
    return method_scores, model_diagnostics


def top10_overlap(per_topic: dict[str, list[dict[str, Any]]], left: str, right: str) -> dict[str, Any]:
    left_rows = {row["query_id"]: row for row in per_topic[left]}
    right_rows = {row["query_id"]: row for row in per_topic[right]}
    rows = []
    for query_id in sorted(set(left_rows) & set(right_rows)):
        left_ids = set(left_rows[query_id]["retrieved_ids_at_k"])
        right_ids = set(right_rows[query_id]["retrieved_ids_at_k"])
        union = left_ids | right_ids
        rows.append(
            {
                "query_id": query_id,
                "overlap_count": len(left_ids & right_ids),
                "jaccard": len(left_ids & right_ids) / len(union) if union else 0.0,
                "left_only": sorted(left_ids - right_ids),
                "right_only": sorted(right_ids - left_ids),
            }
        )
    return {
        "left_method": left,
        "right_method": right,
        "mean_overlap_count": sum(row["overlap_count"] for row in rows) / len(rows) if rows else 0.0,
        "mean_jaccard": sum(row["jaccard"] for row in rows) / len(rows) if rows else 0.0,
        "per_topic": rows,
    }


def row_lookup(rows_by_q: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(query_id, int(row["paper_id"])): row for query_id, rows in rows_by_q.items() for row in rows}


def example_base(
    *,
    query_id: str,
    paper_id: int,
    rank: int,
    method: str,
    rows_lookup: dict[tuple[str, int], dict[str, Any]],
    label_lookup: dict[str, dict[int, dict[str, Any]]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    reason: str,
) -> dict[str, Any]:
    row = rows_lookup.get((query_id, paper_id), {})
    label = label_lookup.get(query_id, {}).get(paper_id)
    return {
        "query_id": query_id,
        "paper_id": paper_id,
        "rank": rank,
        "method": method,
        "reason": reason,
        "title": row.get("title") or (label or {}).get("title"),
        "abstract": row.get("abstract"),
        "judged": label is not None,
        "reading_value_score": (label or {}).get("reading_value_score"),
        "topic_match_score": (label or {}).get("topic_match_score"),
        "primary_role": (label or {}).get("primary_role"),
        "judged_source": (label or {}).get("judged_source"),
        "scores": {
            score_method: method_scores[score_method].get(query_id, {}).get(paper_id)
            for score_method in method_scores
            if paper_id in method_scores[score_method].get(query_id, {})
        },
    }


def build_error_examples(
    *,
    evaluation: dict[str, Any],
    production_rows_by_q: dict[str, list[dict[str, Any]]],
    labels: list[dict[str, Any]],
    method_scores: dict[str, dict[str, dict[int, float]]],
    v33_method: str,
    examples_out: Path,
) -> dict[str, Any]:
    label_lookup = labels_by_query(labels)
    rows_lookup = row_lookup(production_rows_by_q)
    per_topic = evaluation["per_topic"]
    v33_rows = {row["query_id"]: row for row in per_topic[v33_method]}
    v27_rows = {row["query_id"]: row for row in per_topic["v2_7"]}
    v26_rows = {row["query_id"]: row for row in per_topic["v2_6"]}
    examples: list[dict[str, Any]] = []
    losses_v27: list[dict[str, Any]] = []
    losses_v26: list[dict[str, Any]] = []
    for query_id in sorted(v33_rows):
        v33 = v33_rows[query_id]
        v27 = v27_rows[query_id]
        v26 = v26_rows[query_id]
        delta_v27 = v33["reading_value_ndcg"] - v27["reading_value_ndcg"]
        delta_v26 = v33["reading_value_ndcg"] - v26["reading_value_ndcg"]
        if delta_v27 < 0:
            losses_v27.append({"query_id": query_id, "reading_delta": delta_v27, "v3_3": v33["reading_value_ndcg"], "v2_7": v27["reading_value_ndcg"]})
        if delta_v26 < 0:
            losses_v26.append({"query_id": query_id, "reading_delta": delta_v26, "v3_3": v33["reading_value_ndcg"], "v2_6": v26["reading_value_ndcg"]})
        v33_top = v33["retrieved_ids_at_k"]
        v27_top = v27["retrieved_ids_at_k"]
        for rank, paper_id in enumerate(v33_top, start=1):
            label = label_lookup.get(query_id, {}).get(paper_id)
            if paper_id not in v27_top:
                examples.append(example_base(query_id=query_id, paper_id=paper_id, rank=rank, method=v33_method, rows_lookup=rows_lookup, label_lookup=label_lookup, method_scores=method_scores, reason="v3_3_promoted_not_v2_7"))
            if label is not None and is_hard_negative(label):
                examples.append(example_base(query_id=query_id, paper_id=paper_id, rank=rank, method=v33_method, rows_lookup=rows_lookup, label_lookup=label_lookup, method_scores=method_scores, reason="v3_3_known_hard_negative_top10"))
            if label is None:
                examples.append(example_base(query_id=query_id, paper_id=paper_id, rank=rank, method=v33_method, rows_lookup=rows_lookup, label_lookup=label_lookup, method_scores=method_scores, reason="v3_3_high_scoring_unjudged_top10"))
        for rank, paper_id in enumerate(v27_top, start=1):
            if paper_id not in v33_top:
                examples.append(example_base(query_id=query_id, paper_id=paper_id, rank=rank, method="v2_7", rows_lookup=rows_lookup, label_lookup=label_lookup, method_scores=method_scores, reason="v2_7_promoted_not_v3_3"))
        v33_rank_all = rank_from_scores(method_scores[v33_method][query_id])
        v33_rank_lookup = {paper_id: rank for rank, paper_id in enumerate(v33_rank_all, start=1)}
        for paper_id, label in label_lookup.get(query_id, {}).items():
            if is_positive(label) and paper_id in method_scores[v33_method].get(query_id, {}) and v33_rank_lookup.get(paper_id, 999999) > 10:
                examples.append(example_base(query_id=query_id, paper_id=paper_id, rank=v33_rank_lookup[paper_id], method=v33_method, rows_lookup=rows_lookup, label_lookup=label_lookup, method_scores=method_scores, reason="v3_3_judged_positive_below_top10"))
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for example in examples:
        key = (example["query_id"], int(example["paper_id"]))
        observation = {
            "method": example["method"],
            "reason": example["reason"],
            "rank": example["rank"],
        }
        if key not in grouped:
            grouped[key] = {
                **example,
                "reasons": [example["reason"]],
                "observations": [observation],
            }
            continue
        grouped[key]["observations"].append(observation)
        if example["reason"] not in grouped[key]["reasons"]:
            grouped[key]["reasons"].append(example["reason"])
        grouped[key]["reason"] = ";".join(sorted(grouped[key]["reasons"]))
        if example["judged"] and not grouped[key].get("judged"):
            grouped[key].update({field: example.get(field) for field in [
                "judged",
                "reading_value_score",
                "topic_match_score",
                "primary_role",
                "judged_source",
            ]})
    deduped = sorted(grouped.values(), key=lambda row: (str(row["query_id"]), int(row["paper_id"])))
    reason_counter: Counter[str] = Counter()
    for row in deduped:
        for reason in row["reasons"]:
            reason_counter[reason] += 1
    query_paper_counts = Counter((row["query_id"], int(row["paper_id"])) for row in deduped)
    duplicate_query_paper_rows = [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in sorted(query_paper_counts.items())
        if count > 1
    ]
    for row in deduped:
        if len(row["observations"]) == 1:
            continue
        row["method"] = "multiple"
        row["rank"] = min(int(observation["rank"]) for observation in row["observations"])
    write_jsonl(examples_out, deduped)
    return {
        "examples_path": str(examples_out),
        "example_count": len(deduped),
        "raw_observation_count": len(examples),
        "duplicate_query_paper_rows": duplicate_query_paper_rows,
        "topics_where_v3_3_loses_to_v2_7": sorted(losses_v27, key=lambda row: row["reading_delta"]),
        "topics_where_v3_3_loses_to_v2_6": sorted(losses_v26, key=lambda row: row["reading_delta"]),
        "reason_counts": dict(reason_counter),
        "sample_examples": deduped[:25],
    }


def fixed_test_stdev_audit(v33_report: dict[str, Any]) -> dict[str, Any]:
    test_scope = v33_report["train_dev_test"]["scopes"]["test"]["evaluation"]["combined"]
    method_averages = test_scope["method_averages"]
    per_topic = test_scope["per_topic"]
    computed: dict[str, float] = {}
    for method, rows in per_topic.items():
        values = [float(row["reading_value_ndcg"]) for row in rows]
        computed[method] = statistics.pstdev(values) if len(values) > 1 else 0.0
    nonzero = {method: value for method, value in computed.items() if value > 1e-12}
    return {
        "v3_3_markdown_showed_zero_fixed_test_stdev": True,
        "is_reporting_bug": bool(nonzero),
        "reason": (
            "The V3.3 markdown rendered fixed-test method_averages as if they were a single-row mean table. "
            "That transformation does not carry per-topic values, so the displayed reading stdev column was 0.000. "
            "The underlying JSON still contains per-topic fixed-test metrics."
        ),
        "fixed_test_topic_count": len(next(iter(per_topic.values()))) if per_topic else 0,
        "computed_fixed_test_reading_stdev_by_method": computed,
        "method_average_keys_sample": sorted(next(iter(method_averages.values())).keys()) if method_averages else [],
        "report_fix_applied": False,
        "report_fix_note": "V3.4 records the audit without overwriting the completed V3.3 training report.",
    }


def protected_hashes(v21_labels: Path, v25_labels: Path, v32_labels: Path) -> dict[str, Any]:
    v21_hash = file_sha256(v21_labels)
    v25_hash = file_sha256(v25_labels)
    selected_hash = file_sha256(SELECTED_240)
    return {
        "v2_1_labels_sha256": v21_hash,
        "v2_1_labels_hash_unchanged": v21_hash == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": v25_hash,
        "v2_5_labels_hash_unchanged": v25_hash == EXPECTED_V25_LABELS_SHA256,
        "v3_2_labels_sha256": file_sha256(v32_labels),
        "selected_240_sha256": selected_hash,
        "selected_240_hash_unchanged": selected_hash == EXPECTED_SELECTED_240_SHA256,
    }


def build_report(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v21_packet_path: Path,
    old_model_path: Path,
    v26_model_path: Path,
    v33_model_dir: Path,
    v33_report_path: Path,
    examples_out: Path,
    top_k: int,
) -> dict[str, Any]:
    v21_labels = labels_with_source(v21_labels_path, "v2_1")
    v25_labels = labels_with_source(v25_labels_path, "v2_5")
    v32_labels = labels_with_source(v32_labels_path, "v3_2")
    combined_labels = v21_labels + v25_labels + v32_labels
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
    v33_method = "v3_3_random_forest_no_v27_compact"
    error_analysis = build_error_examples(
        evaluation=evaluation,
        production_rows_by_q=production_rows_by_q,
        labels=combined_labels,
        method_scores=method_scores,
        v33_method=v33_method,
        examples_out=examples_out,
    )
    v33_report = load_json(v33_report_path)
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
    report = {
        "schema_version": "v3.4_v3_3_production_pool_evaluation",
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
            "labels_by_source": dict(Counter(row["judged_source"] for row in combined_labels)),
            "topic_count": len({str(row["query_id"]) for row in combined_labels}),
        },
        "production_diagnostics": production_diagnostics,
        "model_diagnostics": model_diagnostics,
        "evaluation": evaluation,
        "top10_overlap": {
            "v3_3_vs_v2_7": top10_overlap(evaluation["per_topic"], v33_method, "v2_7"),
            "v3_3_vs_v2_6": top10_overlap(evaluation["per_topic"], v33_method, "v2_6"),
        },
        "comparison": {
            "v3_3_method": v33_method,
            "v3_3_beats_v2_7_reading_ndcg": avg[v33_method]["reading_value_ndcg"] > avg["v2_7"]["reading_value_ndcg"],
            "v3_3_beats_v2_7_topic_ndcg": avg[v33_method]["topic_match_ndcg"] > avg["v2_7"]["topic_match_ndcg"],
            "v3_3_beats_v2_6_reading_ndcg": avg[v33_method]["reading_value_ndcg"] > avg["v2_6"]["reading_value_ndcg"],
            "v3_3_beats_v2_6_topic_ndcg": avg[v33_method]["topic_match_ndcg"] > avg["v2_6"]["topic_match_ndcg"],
            "reading_delta_vs_v2_7": avg[v33_method]["reading_value_ndcg"] - avg["v2_7"]["reading_value_ndcg"],
            "topic_delta_vs_v2_7": avg[v33_method]["topic_match_ndcg"] - avg["v2_7"]["topic_match_ndcg"],
            "unjudged_delta_vs_v2_7": avg[v33_method]["unjudged_at_10"] - avg["v2_7"]["unjudged_at_10"],
            "hard_negative_rate_delta_vs_v2_7": avg[v33_method]["hard_negative_rate_at_10"] - avg["v2_7"]["hard_negative_rate_at_10"],
            "safe_to_present_as_offline_production_pool_upgrade": (
                avg[v33_method]["reading_value_ndcg"] > avg["v2_7"]["reading_value_ndcg"]
                and avg[v33_method]["topic_match_ndcg"] > avg["v2_7"]["topic_match_ndcg"]
                and avg[v33_method]["unjudged_at_10"] <= avg["v2_7"]["unjudged_at_10"]
            ),
            "safe_for_opt_in_runtime_integration": False,
            "runtime_integration_blocker": "Offline production-pool evaluation only; backend parity and live runtime scoring have not been implemented for V3.3.",
        },
        "error_analysis": error_analysis,
        "fixed_test_stdev_audit": fixed_test_stdev_audit(v33_report),
        "protected_hashes": protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path),
    }
    return report


def method_table(averages: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | reading NDCG@10 | topic NDCG@10 | judged@10 | unjudged@10 | hard-neg rate@10 | recall@10 | MRR@10 | role cov | path cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in sorted(averages.items(), key=lambda item: item[1]["reading_value_ndcg"], reverse=True):
        lines.append(
            f"| `{method}` | {row['reading_value_ndcg']:.3f} | {row['topic_match_ndcg']:.3f} | "
            f"{row['judged_at_10']:.2f} | {row['unjudged_at_10']:.2f} | {row['hard_negative_rate_at_10']:.3f} | "
            f"{row['positive_recall_at_10']:.3f} | {row['mrr_at_10']:.3f} | {row['role_coverage_at_10']:.3f} | "
            f"{row['path_coverage_at_10']:.3f} |"
        )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    avg = report["evaluation"]["method_averages"]
    comparison = report["comparison"]
    errors = report["error_analysis"]
    stdev = report["fixed_test_stdev_audit"]
    lines = [
        "# V3.4 V3.3 Production-Pool Evaluation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "V3.4 evaluates the frozen V3.3 model on the expanded production-style shared candidate pool. It does not modify labels, retrain models, or change runtime defaults.",
        "",
        "## Candidate Pool",
        "",
        f"- Definition: `{report['candidate_pool_definition']['union']}`",
        f"- Deduplicate by: `{report['candidate_pool_definition']['deduplicate_by']}`",
        f"- Mean candidate pool size: `{report['candidate_pool_definition']['mean_candidate_pool_size']:.1f}`",
        f"- Labels evaluated: `{report['dataset']['label_count']}`",
        f"- Labels by source: `{report['dataset']['labels_by_source']}`",
        "",
        "## Production-Pool Metrics",
        "",
    ]
    lines.extend(method_table(avg))
    lines.extend(
        [
            "",
            "## V3.3 vs V2.7",
            "",
            f"- V3.3 method: `{comparison['v3_3_method']}`",
            f"- Reading NDCG delta vs V2.7: `{comparison['reading_delta_vs_v2_7']:+.3f}`",
            f"- Topic NDCG delta vs V2.7: `{comparison['topic_delta_vs_v2_7']:+.3f}`",
            f"- Unjudged@10 delta vs V2.7: `{comparison['unjudged_delta_vs_v2_7']:+.2f}`",
            f"- Hard-negative rate delta vs V2.7: `{comparison['hard_negative_rate_delta_vs_v2_7']:+.3f}`",
            f"- Safe as offline production-pool upgrade: `{comparison['safe_to_present_as_offline_production_pool_upgrade']}`",
            f"- Safe for opt-in runtime integration: `{comparison['safe_for_opt_in_runtime_integration']}`",
            f"- Runtime blocker: {comparison['runtime_integration_blocker']}",
            "",
            "## Error Analysis",
            "",
            f"- Topics where V3.3 loses to V2.7: `{len(errors['topics_where_v3_3_loses_to_v2_7'])}`",
            f"- Topics where V3.3 loses to V2.6: `{len(errors['topics_where_v3_3_loses_to_v2_6'])}`",
            f"- Error examples exported: `{errors['example_count']}`",
            f"- Example reason counts: `{errors['reason_counts']}`",
            f"- V3.3/V2.7 mean top-10 overlap: `{report['top10_overlap']['v3_3_vs_v2_7']['mean_overlap_count']:.2f}`",
            f"- V3.3/V2.7 mean top-10 Jaccard: `{report['top10_overlap']['v3_3_vs_v2_7']['mean_jaccard']:.3f}`",
            "",
            "## Feature Importance",
            "",
        ]
    )
    rf_importance = report["model_diagnostics"]["v3_3_random_forest_no_v27_compact"]["feature_importances"]
    for row in sorted(rf_importance, key=lambda item: item["importance"], reverse=True)[:10]:
        lines.append(f"- `{row['feature']}`: `{row['importance']:.4f}`")
    lines.extend(
        [
            "",
            "## Fixed-Test Stdev Audit",
            "",
            f"- Reporting bug found: `{stdev['is_reporting_bug']}`",
            f"- Fix applied to V3.3 report: `{stdev['report_fix_applied']}`",
            f"- Reason: {stdev['reason']}",
            "",
            "## Protected Hashes",
            "",
            f"- V2.1 labels unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3.3 LTR models on the V2.7-style production pool.")
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-model-dir", default=str(DEFAULT_V33_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-report", default=str(DEFAULT_V33_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--examples-out", default=str(DEFAULT_EXAMPLES_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    examples_out = resolve_repo_path(args.examples_out)
    report = build_report(
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        v21_packet_path=resolve_repo_path(args.v21_packet),
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
        v33_model_dir=resolve_repo_path(args.v33_model_dir),
        v33_report_path=resolve_repo_path(args.v33_report),
        examples_out=examples_out,
        top_k=args.top_k,
    )
    report["command"] = " ".join(sys.argv)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    comparison = report["comparison"]
    print("V3.4 V3.3 production-pool evaluation complete")
    print(f"V3.3 reading delta vs V2.7: {comparison['reading_delta_vs_v2_7']:+.3f}")
    print(f"V3.3 topic delta vs V2.7: {comparison['topic_delta_vs_v2_7']:+.3f}")
    print(f"Safe offline production-pool upgrade: {comparison['safe_to_present_as_offline_production_pool_upgrade']}")
    print(f"Safe opt-in runtime integration: {comparison['safe_for_opt_in_runtime_integration']}")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")
    print(f"Examples: {examples_out}")


if __name__ == "__main__":
    main()
