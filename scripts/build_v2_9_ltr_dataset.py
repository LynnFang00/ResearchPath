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

from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    BASELINE_METHODS,
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    file_sha256,
    load_jsonl,
    resolve_repo_path,
    target_gain,
    write_json,
)
from evaluate_v2_7_score_blends import (  # noqa: E402
    EXPECTED_V25_LABELS_SHA256,
    FIXED_BLEND_WEIGHTS,
    fixed_and_penalty_scores,
    label_is_hidden_positive,
    label_is_negative,
    score_rows,
)
from train_v2_6_production_aware_learned_hybrid import (  # noqa: E402
    EXPECTED_V21_LABELS_SHA256,
    flatten_rows,
    production_candidate_rows,
    rows_by_key,
    rows_by_query,
)


DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V21_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_V25_PACKET = REPO_ROOT / "data" / "eval" / "v2_5_hard_negative_labeling_packet.jsonl"
DEFAULT_OLD_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_V26_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"
DEFAULT_DATASET_OUT = REPO_ROOT / "data" / "eval" / "training" / "v2_9_ltr_dataset.jsonl"
DEFAULT_SPLITS_OUT = REPO_ROOT / "data" / "eval" / "training" / "v2_9_splits.json"

SELECTED_V27_METHOD = "blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_labels_with_source(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in [("v2_1", v21_labels_path), ("v2_5", v25_labels_path)]:
        for label in load_jsonl(path):
            copied = dict(label)
            copied["judged_source"] = source
            rows.append(copied)
    return rows


def labels_by_key(labels: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["query_id"]), int(row["paper_id"])): row for row in labels}


def validate_no_duplicate_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((str(row["query_id"]), int(row["paper_id"])) for row in rows)
    return [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in sorted(counts.items())
        if count > 1
    ]


def positive_flag(label: dict[str, Any]) -> bool:
    return float(label.get("reading_value_score") or 0.0) >= 0.5 and float(label.get("topic_match_score") or 0.0) >= 0.5


def extract_source_pool_info(row: dict[str, Any], *, judged_source: str) -> dict[str, Any]:
    metadata = row.get("source_specific_metadata") or {}
    return {
        "source_pool": "v2_1_labeling_packet" if judged_source == "v2_1" else "v2_5_hard_negative_packet",
        "selection_bucket": row.get("selection_bucket"),
        "selection_reasons": row.get("selection_reasons") or [],
        "why_selected": row.get("why_selected"),
        "candidate_source": metadata.get("candidate_source") or [],
        "source_methods": metadata.get("source_methods") or [],
        "appears_in_n_methods": metadata.get("appears_in_n_methods"),
        "best_rank_hint": metadata.get("best_rank"),
        "mean_rank_hint": metadata.get("mean_rank"),
    }


def build_dataset_row(
    *,
    label: dict[str, Any],
    packet_row: dict[str, Any],
    scored_values: dict[str, float],
) -> dict[str, Any]:
    judged_source = str(label["judged_source"])
    paper_id = int(label["paper_id"])
    query_id = str(label["query_id"])
    retrieval_scores = packet_row.get("retrieval_scores_by_method") or {}
    retrieval_ranks = packet_row.get("retrieval_ranks_by_method") or {}
    row: dict[str, Any] = {
        "schema_version": "v2.9_ltr_dataset_row",
        "query_id": query_id,
        "query": label.get("query") or packet_row.get("query"),
        "topic": label.get("query") or packet_row.get("query"),
        "paper_id": paper_id,
        "title": label.get("title") or packet_row.get("title"),
        "abstract": packet_row.get("abstract"),
        "year": packet_row.get("year"),
        "authors": packet_row.get("authors") or [],
        "venue": packet_row.get("venue"),
        "citation_count": packet_row.get("citation_count"),
        "identifiers": packet_row.get("identifiers") or {},
        "source_url": packet_row.get("source_url"),
        "pdf_url": packet_row.get("pdf_url"),
        "sources_provenance": packet_row.get("sources_provenance") or [],
        "judged_source": judged_source,
        "relevance_label": target_gain(label),
        "topic_match_score": label.get("topic_match_score"),
        "reading_value_score": label.get("reading_value_score"),
        "beginner_fit_score": label.get("beginner_fit_score"),
        "intermediate_fit_score": label.get("intermediate_fit_score"),
        "advanced_fit_score": label.get("advanced_fit_score"),
        "expert_fit_score": label.get("expert_fit_score"),
        "difficulty_labels": {
            "beginner": label.get("beginner_fit_score"),
            "intermediate": label.get("intermediate_fit_score"),
            "advanced": label.get("advanced_fit_score"),
            "expert": label.get("expert_fit_score"),
        },
        "intent_scores": label.get("intent_scores") or {},
        "primary_role": label.get("primary_role"),
        "secondary_roles": label.get("secondary_roles") or [],
        "duplicate_status": label.get("duplicate_status"),
        "duplicate_of_paper_id": label.get("duplicate_of_paper_id"),
        "evidence_level": label.get("evidence_level") or (packet_row.get("evidence_availability") or {}).get("level"),
        "full_text_available": label.get("full_text_available")
        if label.get("full_text_available") is not None
        else (packet_row.get("evidence_availability") or {}).get("fulltext_available"),
        "label_confidence": label.get("label_confidence"),
        "label_notes": label.get("notes"),
        "hard_negative": judged_source == "v2_5" and label_is_negative(label),
        "hidden_positive": judged_source == "v2_5" and label_is_hidden_positive(label),
        "positive": positive_flag(label),
        "source_pool_info": extract_source_pool_info(packet_row, judged_source=judged_source),
    }
    for method in BASELINE_METHODS:
        row[f"{method}_score"] = float(scored_values.get(f"{method}_score", 0.0))
        row[f"{method}_raw_score"] = float(scored_values.get(f"{method}_raw_score", retrieval_scores.get(method, 0.0)))
        row[f"{method}_rank"] = int(scored_values.get(f"{method}_rank", retrieval_ranks.get(method, 999999)))
        row[f"{method}_reciprocal_rank"] = float(scored_values.get(f"{method}_reciprocal_rank", 0.0))
    row["old_v2_2b_score"] = float(scored_values["old_v2_2b_score"])
    row["v2_6_score"] = float(scored_values["v2_6_score"])
    row["v2_7_score"] = float(
        fixed_and_penalty_scores({query_id: {paper_id: scored_values}})[SELECTED_V27_METHOD][query_id][paper_id]
    )
    row["old_minus_v26"] = float(scored_values.get("old_minus_v26", 0.0))
    row["old_minus_v26_positive"] = float(scored_values.get("old_minus_v26_positive", 0.0))
    row["v26_minus_old_positive"] = float(scored_values.get("v26_minus_old_positive", 0.0))
    return row


def build_topic_splits(topics: list[str]) -> dict[str, Any]:
    topics = sorted(topics)
    dev_topics = topics[10:13]
    test_topics = topics[13:16]
    train_topics = [topic for topic in topics if topic not in set(dev_topics + test_topics)]
    leave_topic_out = [
        {
            "fold_id": f"loto_{index:02d}",
            "test_topics": [topic],
            "dev_topics": [],
            "train_topics": [other for other in topics if other != topic],
        }
        for index, topic in enumerate(topics, start=1)
    ]
    return {
        "schema_version": "v2.9_topic_splits",
        "created_at": datetime.now(UTC).isoformat(),
        "split_strategy": "topic_grouped_no_query_id_leakage",
        "topics": topics,
        "train_dev_test": {
            "train_topics": train_topics,
            "dev_topics": dev_topics,
            "test_topics": test_topics,
        },
        "leave_topic_out": leave_topic_out,
    }


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
        "train_topic_count": len(train),
        "dev_topic_count": len(dev),
        "test_topic_count": len(test),
        "leave_topic_out_fold_count": len(splits["leave_topic_out"]),
    }


def build_dataset(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v21_packet_path: Path,
    v25_packet_path: Path,
    old_model_path: Path,
    v26_model_path: Path,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v21_labels = load_jsonl(v21_labels_path)
    v25_labels = load_jsonl(v25_labels_path)
    labels = load_labels_with_source(v21_labels_path=v21_labels_path, v25_labels_path=v25_labels_path)
    duplicate_labels = validate_no_duplicate_keys(labels)
    if duplicate_labels:
        raise ValueError(f"Duplicate query_id/paper_id labels found: {duplicate_labels[:5]}")

    v21_packet_rows = load_jsonl(v21_packet_path)
    v25_packet_rows = load_jsonl(v25_packet_path)
    v21_rows_by_q = rows_by_query(v21_packet_rows)
    v25_rows_by_q = rows_by_query(v25_packet_rows)
    production_rows_by_q, production_diagnostics = production_candidate_rows(packet_rows=v21_packet_rows, top_k=top_k)

    v21_scores = score_rows(
        candidate_rows_by_q=v21_rows_by_q,
        context_rows_by_q=v21_rows_by_q,
        old_model_path=old_model_path,
        v26_model_path=v26_model_path,
    )
    v25_scores = score_rows(
        candidate_rows_by_q=v25_rows_by_q,
        context_rows_by_q=production_rows_by_q,
        old_model_path=old_model_path,
        v26_model_path=v26_model_path,
    )

    label_lookup = labels_by_key(labels)
    v21_packet_lookup = rows_by_key(v21_packet_rows)
    v25_packet_lookup = rows_by_key(v25_packet_rows)
    rows: list[dict[str, Any]] = []
    missing_packet_rows: list[dict[str, Any]] = []
    missing_score_rows: list[dict[str, Any]] = []
    for key, label in sorted(label_lookup.items()):
        query_id, paper_id = key
        if label["judged_source"] == "v2_1":
            packet_row = v21_packet_lookup.get(key)
            scored_values = v21_scores.get(query_id, {}).get(paper_id)
        else:
            packet_row = v25_packet_lookup.get(key)
            scored_values = v25_scores.get(query_id, {}).get(paper_id)
        if packet_row is None:
            missing_packet_rows.append({"query_id": query_id, "paper_id": paper_id, "judged_source": label["judged_source"]})
            continue
        if scored_values is None:
            missing_score_rows.append({"query_id": query_id, "paper_id": paper_id, "judged_source": label["judged_source"]})
            continue
        rows.append(build_dataset_row(label=label, packet_row=packet_row, scored_values=scored_values))

    topics = sorted({str(row["query_id"]) for row in rows})
    diagnostics = {
        "schema_version": "v2.9_ltr_dataset_summary",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "row_count": len(rows),
        "expected_row_count": len(v21_labels) + len(v25_labels),
        "rows_by_judged_source": dict(Counter(row["judged_source"] for row in rows)),
        "topic_count": len(topics),
        "topics": topics,
        "duplicate_query_paper_rows": validate_no_duplicate_keys(rows),
        "missing_packet_rows": missing_packet_rows,
        "missing_score_rows": missing_score_rows,
        "score_columns": [
            "bm25_score",
            "tfidf_score",
            "embedding_score",
            "faiss_embedding_score",
            "hybrid_score",
            "old_v2_2b_score",
            "v2_6_score",
            "v2_7_score",
        ],
        "v2_7_formula": {
            "method": SELECTED_V27_METHOD,
            "weights": FIXED_BLEND_WEIGHTS[SELECTED_V27_METHOD],
        },
        "hashes": {
            "v2_1_labels_sha256": file_sha256(v21_labels_path),
            "v2_1_labels_hash_unchanged": file_sha256(v21_labels_path) == EXPECTED_V21_LABELS_SHA256,
            "v2_5_labels_sha256": file_sha256(v25_labels_path),
            "v2_5_labels_hash_unchanged": file_sha256(v25_labels_path) == EXPECTED_V25_LABELS_SHA256,
            "selected_240_sha256": file_sha256(SELECTED_240),
            "selected_240_hash_unchanged": file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256,
        },
        "production_diagnostics": production_diagnostics,
    }
    return rows, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V2.9 supervised learning-to-rank dataset.")
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET))
    parser.add_argument("--v25-packet", default=str(DEFAULT_V25_PACKET))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL))
    parser.add_argument("--dataset-out", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--splits-out", default=str(DEFAULT_SPLITS_OUT))
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_out = resolve_repo_path(args.dataset_out)
    splits_out = resolve_repo_path(args.splits_out)
    rows, diagnostics = build_dataset(
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v21_packet_path=resolve_repo_path(args.v21_packet),
        v25_packet_path=resolve_repo_path(args.v25_packet),
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
        top_k=args.top_k,
    )
    splits = build_topic_splits(diagnostics["topics"])
    split_validation = validate_splits(splits)
    splits["validation"] = split_validation
    diagnostics["split_validation"] = split_validation
    diagnostics["outputs"] = {"dataset": str(dataset_out), "splits": str(splits_out)}
    if diagnostics["row_count"] != diagnostics["expected_row_count"]:
        raise ValueError(f"Expected {diagnostics['expected_row_count']} rows, wrote {diagnostics['row_count']}.")
    if diagnostics["duplicate_query_paper_rows"]:
        raise ValueError("Dataset contains duplicate query_id/paper_id rows.")
    if not split_validation["train_dev_test_no_topic_leakage"] or not split_validation["leave_topic_out_no_topic_leakage"]:
        raise ValueError("Topic split leakage detected.")
    write_jsonl(dataset_out, rows)
    write_json(splits_out, splits)
    print("V2.9 LTR dataset build complete")
    print(f"Rows: {diagnostics['row_count']}")
    print(f"Dataset: {dataset_out}")
    print(f"Splits: {splits_out}")
    print(f"Protected hashes unchanged: {diagnostics['hashes']}")


if __name__ == "__main__":
    main()
