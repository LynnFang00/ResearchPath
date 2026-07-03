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


TRAIN = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_1_train.jsonl"
VAL = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_1_val.jsonl"
TEST = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_1_test.jsonl"
PAIRWISE_TRAIN = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_1_pairwise_train.jsonl"
RECOMMENDATIONS = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_1_targeted_labeling_recommendations.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_0_1_neural_split_balance_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_0_1_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.0.1_neural_split_balance_report"
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


def test_v6_0_1_grouped_splits_have_no_query_overlap_and_are_deterministic() -> None:
    report = load_report()
    split_rows = {"train": load_jsonl(TRAIN), "val": load_jsonl(VAL), "test": load_jsonl(TEST)}
    split_query_ids = {split: {row["query_id"] for row in rows} for split, rows in split_rows.items()}

    assert len(split_rows["train"]) == 1800
    assert len(split_rows["val"]) == 300
    assert len(split_rows["test"]) == 300
    assert split_query_ids["train"].isdisjoint(split_query_ids["val"])
    assert split_query_ids["train"].isdisjoint(split_query_ids["test"])
    assert split_query_ids["val"].isdisjoint(split_query_ids["test"])
    assert report["split_strategy"]["balanced_split_topics"]["val"] == [
        "v2_ai_for_scientific_discovery",
        "v2_multimodal_learning",
    ]
    assert report["split_strategy"]["balanced_split_topics"]["test"] == [
        "v2_retrieval_augmented_generation",
        "v2_robot_learning",
    ]
    for split, rows in split_rows.items():
        assert all(row["v6_0_1_grouped_split"] == split for row in rows)
        assert [(row["query_id"], row["paper_id"]) for row in rows] == sorted(
            (row["query_id"], row["paper_id"]) for row in rows
        )


def test_v6_0_1_hard_negative_balance_improves_validation_and_keeps_test_meaningful() -> None:
    report = load_report()
    original = report["split_comparison"]["original"]
    balanced = report["split_comparison"]["balanced"]
    assert report["grouped_balance_improved"] is True
    assert balanced["val"]["hard_negative_count"] == 72
    assert balanced["test"]["hard_negative_count"] == 71
    assert balanced["val"]["hard_negative_count"] > original["val"]["hard_negative_count"]
    assert balanced["test"]["hard_negative_count"] >= original["test"]["hard_negative_count"]
    for split in ("val", "test"):
        assert all(balanced[split]["relevance_distribution"].get(str(bucket), 0) > 0 for bucket in (0, 1, 2))


def test_v6_0_1_pairwise_examples_are_valid() -> None:
    examples = {}
    for path in (TRAIN, VAL, TEST):
        for row in load_jsonl(path):
            examples[(row["query_id"], row["paper_id"])] = row
    pairs = load_jsonl(PAIRWISE_TRAIN)
    report = load_report()
    assert len(pairs) == report["pairwise_train_count"] == 7200
    for pair in pairs[:500]:
        preferred = examples[(pair["query_id"], pair["preferred_paper_id"])]
        rejected = examples[(pair["query_id"], pair["rejected_paper_id"])]
        assert preferred["relevance_bucket"] >= rejected["relevance_bucket"]
        assert preferred["reading_value_score"] > rejected["reading_value_score"] or preferred["relevance_bucket"] > rejected["relevance_bucket"]
        assert pair["hard_negative_pair"] == rejected["hard_negative"]


def test_v6_0_1_targeted_labeling_recommendations_are_expanded_and_actionable() -> None:
    rows = load_jsonl(RECOMMENDATIONS)
    report = load_report()
    required = {
        "query_id",
        "query",
        "paper_id",
        "title",
        "abstract",
        "abstract_word_count",
        "year",
        "venue",
        "source",
        "current_label",
        "candidate_rankings",
        "reason_selected",
        "suggested_labeling_focus",
    }
    assert len(rows) == report["targeted_labeling_recommendation_count"] == 80
    assert all(required <= set(row) for row in rows)
    assert all(row["reason_selected"] for row in rows)
    assert all(row["suggested_labeling_focus"] for row in rows)
    assert all(row["do_not_modify_official_labels"] is True for row in rows)
    assert {"v2_graph_neural_networks", "v2_multimodal_learning", "v2_recommendation_systems", "v2_robot_learning"} <= {
        row["query_id"] for row in rows
    }


def test_v6_0_1_topic_sufficiency_audit_marks_weak_topics() -> None:
    report = load_report()
    assert report["more_labels_recommended_before_v6_1_training"] is True
    assert report["topics_needing_labels_most"] == [
        "v2_graph_neural_networks",
        "v2_multimodal_learning",
        "v2_recommendation_systems",
        "v2_robot_learning",
    ]
    for topic in report["topics_needing_labels_most"]:
        assert report["topic_sufficiency_audit"][topic]["weak_for_neural_training_or_evaluation"] is True
        assert report["topic_sufficiency_audit"][topic]["weakness_reasons"]


def test_v6_0_1_protected_hashes_are_unchanged() -> None:
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


def test_v6_0_1_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_0_1_neural_reranker.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "neural_reranker_service.py").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_0_1_neural_reranker_candidate.json").exists()
