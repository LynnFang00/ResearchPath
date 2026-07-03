import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.validate_v3_9_semantic_expansion_labels import (  # noqa: E402
    DEFAULT_LABELS as V39_LABELS,
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    EXPECTED_V35_LABELS_SHA256,
    file_sha256,
)


LABELS = ROOT / "data" / "eval" / "neural_reranker" / "manual_labels_v6_0_2_targeted_neural.jsonl"
EXAMPLES = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
TRAIN = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_train.jsonl"
VAL = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_val.jsonl"
TEST = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_test.jsonl"
PAIRWISE_TRAIN = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_pairwise_train.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_0_2_neural_reranker_dataset_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_0_2_imported_label_file_shape() -> None:
    rows = load_jsonl(LABELS)
    keys = [(row["query_id"], int(row["paper_id"])) for row in rows]
    assert len(rows) == 80
    assert len(keys) == len(set(keys))
    assert {row["schema_version"] for row in rows} == {"v6.0.2_targeted_neural_suggested_label"}
    assert {int(row["relevance_label"]["bucket"]) for row in rows} <= {0, 1, 2}
    assert sum(1 for row in rows if row["hard_negative"]) == 28
    assert all(row["do_not_modify_official_labels"] is True for row in rows)


def test_v6_0_2_report_schema_preflight_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.0.2_neural_reranker_dataset_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False
    assert report["v6_0_outputs_overwritten"] is False
    assert report["v6_0_1_outputs_overwritten"] is False
    assert report["preflight"]["row_count"] == 80
    assert report["preflight"]["duplicate_query_paper_rows"] == []
    assert report["preflight"]["rows_not_in_v6_0_1_recommendations"] == []
    assert report["preflight"]["overlap_with_v6_0_examples"] == []


def test_v6_0_2_examples_and_split_counts() -> None:
    report = load_report()
    rows = load_jsonl(EXAMPLES)
    split_rows = {"train": load_jsonl(TRAIN), "val": load_jsonl(VAL), "test": load_jsonl(TEST)}
    assert len(rows) == report["dataset"]["example_count"] == 2480
    assert sum(1 for row in rows if row["label_source"] == "v6_0_2") == 80
    assert report["overall_relevance_distribution"] == {"0": 590, "1": 704, "2": 1186}
    assert report["overall_hard_negative_count"] == 564
    assert {split: len(split_rows[split]) for split in split_rows} == {"train": 1820, "val": 332, "test": 328}
    assert report["split_diagnostics"]["val"]["hard_negative_count"] == 87
    assert report["split_diagnostics"]["test"]["hard_negative_count"] == 84


def test_v6_0_2_grouped_split_topics_remain_disjoint() -> None:
    split_rows = {"train": load_jsonl(TRAIN), "val": load_jsonl(VAL), "test": load_jsonl(TEST)}
    split_query_ids = {split: {row["query_id"] for row in rows} for split, rows in split_rows.items()}
    assert split_query_ids["train"].isdisjoint(split_query_ids["val"])
    assert split_query_ids["train"].isdisjoint(split_query_ids["test"])
    assert split_query_ids["val"].isdisjoint(split_query_ids["test"])
    for split, rows in split_rows.items():
        assert all(row["query_grouped_split"] == split for row in rows)
        assert all(row["v6_0_2_grouped_split"] == split for row in rows)
        assert [(row["query_id"], row["paper_id"]) for row in rows] == sorted(
            (row["query_id"], row["paper_id"]) for row in rows
        )


def test_v6_0_2_pairwise_examples_are_valid() -> None:
    examples = {}
    for path in (TRAIN, VAL, TEST):
        for row in load_jsonl(path):
            examples[(row["query_id"], row["paper_id"])] = row
    pairs = load_jsonl(PAIRWISE_TRAIN)
    report = load_report()
    assert len(pairs) == report["dataset"]["pairwise_train_count"] == 7200
    for pair in pairs[:500]:
        preferred = examples[(pair["query_id"], pair["preferred_paper_id"])]
        rejected = examples[(pair["query_id"], pair["rejected_paper_id"])]
        assert preferred["relevance_bucket"] >= rejected["relevance_bucket"]
        assert preferred["reading_value_score"] > rejected["reading_value_score"] or preferred["relevance_bucket"] > rejected["relevance_bucket"]
        assert pair["hard_negative_pair"] == rejected["hard_negative"]


def test_v6_0_2_protected_hashes_are_unchanged() -> None:
    report = load_report()
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
    assert file_sha256(DEFAULT_V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(DEFAULT_V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(DEFAULT_V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == report["protected_hashes"]["v3_9_labels_sha256"]


def test_v6_0_2_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_0_2_neural_reranker.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "neural_reranker_service.py").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_0_2_neural_reranker_candidate.json").exists()
