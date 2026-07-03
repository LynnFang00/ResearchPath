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

from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    file_sha256,
    resolve_repo_path,
    target_gain,
    write_json,
)
from evaluate_v2_7_score_blends import EXPECTED_V25_LABELS_SHA256  # noqa: E402
from train_v2_6_production_aware_learned_hybrid import EXPECTED_V21_LABELS_SHA256  # noqa: E402


DEFAULT_V29_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v2_9_ltr_dataset.jsonl"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V32_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
DEFAULT_V32_CANDIDATES = REPO_ROOT / "data" / "eval" / "labeling" / "v3_2_800_active_learning_candidates.jsonl"
DEFAULT_DATASET_OUT = REPO_ROOT / "data" / "eval" / "training" / "v3_3_ltr_dataset_800.jsonl"
DEFAULT_SPLITS_OUT = REPO_ROOT / "data" / "eval" / "training" / "v3_3_splits_800.json"

BASELINE_METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
EXPECTED_ROW_COUNT = 800
EXPECTED_TOPIC_COUNT = 16
EXPECTED_ROWS_PER_TOPIC = 50


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["query_id"]), int(row["paper_id"]))


def rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {row_key(row): row for row in rows}


def duplicate_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row_key(row) for row in rows)
    return [
        {"query_id": query_id, "paper_id": paper_id, "count": count}
        for (query_id, paper_id), count in sorted(counts.items())
        if count > 1
    ]


def positive_flag(label: dict[str, Any]) -> bool:
    return float(label.get("reading_value_score") or 0.0) >= 0.5 and float(label.get("topic_match_score") or 0.0) >= 0.5


def reciprocal_rank(rank: int) -> float:
    return 1.0 / rank if 0 < rank < 999999 else 1.000001000001e-06


def v32_score(candidate: dict[str, Any], key: str) -> float:
    scores = candidate.get("scores") or {}
    return float(scores.get(key, 0.0) or 0.0)


def v32_rank(candidate: dict[str, Any], method: str) -> int:
    retrieval_ranks = candidate.get("retrieval_ranks") or {}
    ranks = candidate.get("ranks") or {}
    return int(retrieval_ranks.get(method, ranks.get(method, 999999)) or 999999)


def build_v32_row(label: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    paper_id = int(label["paper_id"])
    query_id = str(label["query_id"])
    row: dict[str, Any] = {
        "schema_version": "v3.3_ltr_dataset_800_row",
        "query_id": query_id,
        "query": label.get("query") or candidate.get("query"),
        "topic": label.get("query") or candidate.get("topic") or candidate.get("query"),
        "paper_id": paper_id,
        "title": label.get("title") or candidate.get("title"),
        "abstract": candidate.get("abstract"),
        "year": candidate.get("year"),
        "authors": candidate.get("authors") or [],
        "venue": candidate.get("venue"),
        "citation_count": candidate.get("citation_count"),
        "identifiers": candidate.get("identifiers") or {},
        "source_url": candidate.get("source_url"),
        "pdf_url": candidate.get("pdf_url"),
        "sources_provenance": candidate.get("sources_provenance") or [],
        "judged_source": "v3_2",
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
        "evidence_level": label.get("evidence_level"),
        "full_text_available": label.get("full_text_available"),
        "label_confidence": label.get("label_confidence"),
        "label_notes": label.get("notes"),
        "hard_negative": str(label.get("primary_role")) == "negative",
        "hidden_positive": False,
        "positive": positive_flag(label),
        "source_pool_info": {
            "source_pool": "v3_2_800_active_learning_packet",
            "selection_score": candidate.get("selection_score"),
            "rank_disagreement": candidate.get("rank_disagreement"),
            "score_disagreement": candidate.get("score_disagreement"),
            "source_methods": candidate.get("source_methods") or [],
            "why_selected": candidate.get("why_selected") or [],
            "v3_1_carryover": candidate.get("v3_1_carryover"),
        },
        "v3_2_raw_scores_available": False,
    }
    for method in BASELINE_METHODS:
        rank = v32_rank(candidate, method)
        row[f"{method}_score"] = v32_score(candidate, method)
        row[f"{method}_raw_score"] = 0.0
        row[f"{method}_rank"] = rank
        row[f"{method}_reciprocal_rank"] = reciprocal_rank(rank)
    row["old_v2_2b_score"] = v32_score(candidate, "old_v2_2b")
    row["v2_6_score"] = v32_score(candidate, "v2_6")
    row["v2_7_score"] = v32_score(candidate, "v2_7")
    row["old_minus_v26"] = row["old_v2_2b_score"] - row["v2_6_score"]
    row["old_minus_v26_positive"] = max(row["old_minus_v26"], 0.0)
    row["v26_minus_old_positive"] = max(-row["old_minus_v26"], 0.0)
    return row


def normalize_existing_row(row: dict[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    copied["schema_version"] = "v3.3_ltr_dataset_800_row"
    copied["v3_2_raw_scores_available"] = True
    return copied


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
        "schema_version": "v3.3_topic_splits_800",
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


def build_dataset(
    *,
    v29_dataset_path: Path,
    v32_labels_path: Path,
    v32_candidates_path: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v29_rows = [normalize_existing_row(row) for row in load_jsonl(v29_dataset_path)]
    v32_labels = load_jsonl(v32_labels_path)
    v32_candidates = load_jsonl(v32_candidates_path)
    candidate_lookup = rows_by_key(v32_candidates)
    missing_candidates: list[dict[str, Any]] = []
    v32_rows: list[dict[str, Any]] = []
    for label in sorted(v32_labels, key=lambda row: row_key(row)):
        candidate = candidate_lookup.get(row_key(label))
        if candidate is None:
            missing_candidates.append({"query_id": str(label["query_id"]), "paper_id": int(label["paper_id"])})
            continue
        v32_rows.append(build_v32_row(label, candidate))

    rows = sorted(v29_rows + v32_rows, key=lambda row: (str(row["query_id"]), int(row["paper_id"])))
    topics = sorted({str(row["query_id"]) for row in rows})
    topic_counts = Counter(str(row["query_id"]) for row in rows)
    duplicate_rows = duplicate_keys(rows)
    diagnostics = {
        "schema_version": "v3.3_ltr_dataset_800_summary",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "row_count": len(rows),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "rows_by_judged_source": dict(Counter(str(row["judged_source"]) for row in rows)),
        "topic_count": len(topics),
        "expected_topic_count": EXPECTED_TOPIC_COUNT,
        "rows_per_topic": dict(sorted(topic_counts.items())),
        "expected_rows_per_topic": EXPECTED_ROWS_PER_TOPIC,
        "all_topics_have_50_rows": all(count == EXPECTED_ROWS_PER_TOPIC for count in topic_counts.values()),
        "duplicate_query_paper_rows": duplicate_rows,
        "missing_v3_2_candidates": missing_candidates,
        "feature_columns": {
            "baseline_score_columns": [
                "bm25_score",
                "tfidf_score",
                "embedding_score",
                "faiss_embedding_score",
                "hybrid_score",
                "old_v2_2b_score",
                "v2_6_score",
                "v2_7_score",
            ],
            "rank_columns": [f"{method}_rank" for method in BASELINE_METHODS],
            "reciprocal_rank_columns": [f"{method}_reciprocal_rank" for method in BASELINE_METHODS],
            "raw_score_columns_present_but_not_used_in_main_v3_3_variants": [f"{method}_raw_score" for method in BASELINE_METHODS],
        },
        "hashes": protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path),
    }
    return rows, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V3.3 800-row supervised LTR dataset.")
    parser.add_argument("--v29-dataset", default=str(DEFAULT_V29_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-candidates", default=str(DEFAULT_V32_CANDIDATES.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--dataset-out", default=str(DEFAULT_DATASET_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--splits-out", default=str(DEFAULT_SPLITS_OUT.relative_to(REPO_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_out = resolve_repo_path(args.dataset_out)
    splits_out = resolve_repo_path(args.splits_out)
    rows, diagnostics = build_dataset(
        v29_dataset_path=resolve_repo_path(args.v29_dataset),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        v32_candidates_path=resolve_repo_path(args.v32_candidates),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
    )
    splits = build_topic_splits(diagnostics["rows_per_topic"].keys())
    split_validation = validate_splits(splits)
    splits["validation"] = split_validation
    diagnostics["split_validation"] = split_validation
    diagnostics["outputs"] = {"dataset": str(dataset_out), "splits": str(splits_out)}
    if diagnostics["row_count"] != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} rows, got {diagnostics['row_count']}.")
    if diagnostics["topic_count"] != EXPECTED_TOPIC_COUNT or not diagnostics["all_topics_have_50_rows"]:
        raise ValueError("Expected 16 topics with exactly 50 rows each.")
    if diagnostics["duplicate_query_paper_rows"]:
        raise ValueError("Dataset contains duplicate query_id/paper_id rows.")
    if diagnostics["missing_v3_2_candidates"]:
        raise ValueError("Some V3.2 labels do not have candidate packet rows.")
    if not split_validation["train_dev_test_no_topic_leakage"] or not split_validation["leave_topic_out_no_topic_leakage"]:
        raise ValueError("Topic split leakage detected.")
    if not all(
        [
            diagnostics["hashes"]["v2_1_labels_hash_unchanged"],
            diagnostics["hashes"]["v2_5_labels_hash_unchanged"],
            diagnostics["hashes"]["selected_240_hash_unchanged"],
        ]
    ):
        raise ValueError("Protected V2.1/V2.5/selected-240 hash changed.")
    write_jsonl(dataset_out, rows)
    write_json(splits_out, splits)
    print("V3.3 800-row LTR dataset build complete")
    print(f"Rows: {diagnostics['row_count']}")
    print(f"Rows by source: {diagnostics['rows_by_judged_source']}")
    print(f"Dataset: {dataset_out}")
    print(f"Splits: {splits_out}")
    print(f"Protected hashes unchanged: {diagnostics['hashes']}")


if __name__ == "__main__":
    main()
