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


EXAMPLES = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_neural_reranker_examples.jsonl"
TRAIN = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_train.jsonl"
VAL = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_val.jsonl"
TEST = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_test.jsonl"
PAIRWISE_TRAIN = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_pairwise_train.jsonl"
RECOMMENDATIONS = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_targeted_labeling_recommendations.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_0_neural_reranker_dataset_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_0_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.0_neural_reranker_dataset_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False


def test_v6_0_examples_have_required_neural_fields() -> None:
    rows = load_jsonl(EXAMPLES)
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
        "relevance_bucket",
        "classification_target",
        "reading_value_score",
        "topic_match_score",
        "regression_target_reading",
        "regression_target_topic",
        "section_label",
        "difficulty_label",
        "hard_negative",
        "generic_survey",
        "label_source",
        "query_grouped_split",
        "within_topic_debug_split",
        "baseline_rankings",
    }
    assert len(rows) == 2400
    assert len(rows) == report["dataset"]["example_count"]
    assert all(required <= set(row) for row in rows[:25])
    assert {row["relevance_bucket"] for row in rows} <= {0, 1, 2}
    assert all(row["classification_target"] == row["relevance_bucket"] for row in rows)
    assert all(0.0 <= row["reading_value_score"] <= 1.0 for row in rows)
    assert all(0.0 <= row["topic_match_score"] <= 1.0 for row in rows)
    assert all(row["abstract_word_count"] == len(str(row["abstract"]).split()) for row in rows[:100])


def test_v6_0_v4_8_overrides_are_included_without_changing_total_count() -> None:
    rows = load_jsonl(EXAMPLES)
    report = load_report()
    override_rows = [row for row in rows if row["v4_8_override"]]
    assert report["dataset"]["v4_8_overrides_included"] is True
    assert report["dataset"]["v4_8_override_count"] == 25
    assert report["dataset"]["v4_8_override_behavior"]["base_label_count"] == 2400
    assert report["dataset"]["v4_8_override_behavior"]["effective_label_count"] == 2400
    assert len(override_rows) == 25
    assert {row["query_id"] for row in override_rows} == {"v2_contrastive_learning"}


def test_v6_0_grouped_splits_are_query_disjoint_and_deterministic() -> None:
    split_rows = {"train": load_jsonl(TRAIN), "val": load_jsonl(VAL), "test": load_jsonl(TEST)}
    report = load_report()
    split_query_ids = {split: {row["query_id"] for row in rows} for split, rows in split_rows.items()}

    assert len(split_rows["train"]) == 1800
    assert len(split_rows["val"]) == 300
    assert len(split_rows["test"]) == 300
    assert split_query_ids["train"].isdisjoint(split_query_ids["val"])
    assert split_query_ids["train"].isdisjoint(split_query_ids["test"])
    assert split_query_ids["val"].isdisjoint(split_query_ids["test"])
    for split, rows in split_rows.items():
        assert split_query_ids[split] == set(report["split_diagnostics"][split]["query_ids"])
        assert all(row["query_grouped_split"] == split for row in rows)
        assert [(row["query_id"], row["paper_id"]) for row in rows] == sorted(
            (row["query_id"], row["paper_id"]) for row in rows
        )


def test_v6_0_hard_negatives_and_relevance_distributions_match_report() -> None:
    rows = load_jsonl(EXAMPLES)
    report = load_report()
    relevance = {str(bucket): sum(1 for row in rows if row["relevance_bucket"] == bucket) for bucket in (0, 1, 2)}
    assert relevance == report["overall_relevance_distribution"]
    assert sum(1 for row in rows if row["hard_negative"]) == report["overall_hard_negative_count"] == 536
    assert sum(1 for row in rows if row["short_or_missing_abstract"]) > 0


def test_v6_0_pairwise_examples_have_valid_preference_direction() -> None:
    examples = {(row["query_id"], row["paper_id"]): row for row in load_jsonl(EXAMPLES)}
    pairs = load_jsonl(PAIRWISE_TRAIN)
    report = load_report()
    assert len(pairs) == report["dataset"]["pairwise_train_count"] == 7200

    for pair in pairs[:500]:
        preferred = examples[(pair["query_id"], pair["preferred_paper_id"])]
        rejected = examples[(pair["query_id"], pair["rejected_paper_id"])]
        assert pair["preferred_paper_id"] != pair["rejected_paper_id"]
        assert preferred["relevance_bucket"] >= rejected["relevance_bucket"]
        assert preferred["reading_value_score"] > rejected["reading_value_score"] or preferred["relevance_bucket"] > rejected["relevance_bucket"]
        assert pair["hard_negative_pair"] == rejected["hard_negative"]
        assert pair["pair_type"] in {
            "positive_over_negative",
            "positive_over_hard_negative",
            "strong_positive_over_partial",
        }


def test_v6_0_label_sufficiency_and_recommendations_are_reported() -> None:
    report = load_report()
    recommendations = load_jsonl(RECOMMENDATIONS)
    sufficiency = report["label_sufficiency"]
    assert sufficiency["recommend_more_labels_before_v6_1_training"] is True
    assert "split:val" in sufficiency["insufficient_scopes"]
    assert "topic:v2_multimodal_learning" in sufficiency["insufficient_scopes"]
    assert len(recommendations) == report["targeted_labeling_recommendation_count"] == 4
    assert all(row["schema_version"] == "v6.0_targeted_labeling_recommendation" for row in recommendations)
    assert all(row["do_not_modify_official_labels"] is True for row in recommendations)


def test_v6_0_protected_hashes_are_unchanged() -> None:
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


def test_v6_0_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_0_neural_reranker.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "neural_reranker_service.py").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_0_neural_reranker_candidate.json").exists()
