import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_v6_1_2_neural_cross_validation import (  # noqa: E402
    METHODS,
    NEURAL_METHOD,
    build_folds,
    evaluate_method,
    load_jsonl,
)
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


EXAMPLES = ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cross_validation_report.json"
PREDICTIONS = ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_predictions.jsonl"
FAILURES = ROOT / "data" / "eval" / "results" / "v6_1_2_neural_cv_failure_examples.jsonl"
BASELINE_REPORT = ROOT / "data" / "eval" / "results" / "v6_1_2_exact_baseline_scoring_report.json"


EXPECTED_FOLDS = [
    {
        "fold_id": "fold_1",
        "val_topics": ["v2_llm_evaluation", "v2_self_supervised_vision"],
        "test_topics": ["v2_efficient_transformers", "v2_graph_recommendation"],
    },
    {
        "fold_id": "fold_2",
        "val_topics": ["v2_causal_representation_learning", "v2_transformer_architecture"],
        "test_topics": ["v2_bayesian_optimization", "v2_contrastive_learning"],
    },
    {
        "fold_id": "fold_3",
        "val_topics": ["v2_graph_neural_networks", "v2_large_language_model_agents"],
        "test_topics": ["v2_retrieval_augmented_generation", "v2_robot_learning"],
    },
    {
        "fold_id": "fold_4",
        "val_topics": ["v2_ai_for_scientific_discovery", "v2_multimodal_learning"],
        "test_topics": ["v2_diffusion_image_generation", "v2_recommendation_systems"],
    },
]


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v6_1_2_report_schema_and_offline_flags() -> None:
    report = load_report()
    assert report["schema_version"] == "v6.1.2_neural_cross_validation_report"
    assert report["runtime_integration_status"] == "offline_only_not_integrated"
    assert report["default_ranker_status"] == "unchanged_bm25"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["protected_labels_modified_by_this_script"] is False
    assert report["selected_240_modified_by_this_script"] is False
    assert report["existing_models_retrained_by_this_script"] is False
    assert report["production_candidate_trained"] is False
    assert report["large_checkpoints_saved"] is False
    assert report["model_type"] in {"cross_encoder_transformer", "fallback_neural_mlp"}
    assert report["dataset"] == {"example_count": 2480, "topic_count": 16}


def test_v6_1_2_grouped_folds_are_deterministic_and_topic_disjoint() -> None:
    examples = load_jsonl(EXAMPLES)
    report = load_report()
    generated = build_folds(examples)
    all_topics = {row["query_id"] for row in examples}

    assert [{key: fold[key] for key in ("fold_id", "val_topics", "test_topics")} for fold in generated] == EXPECTED_FOLDS
    assert [{key: fold[key] for key in ("fold_id", "val_topics", "test_topics")} for fold in report["folds"]] == EXPECTED_FOLDS

    for fold in report["folds"]:
        train_topics = set(fold["train_topics"])
        val_topics = set(fold["val_topics"])
        test_topics = set(fold["test_topics"])
        assert len(train_topics) == 12
        assert len(val_topics) == 2
        assert len(test_topics) == 2
        assert train_topics.isdisjoint(val_topics)
        assert train_topics.isdisjoint(test_topics)
        assert val_topics.isdisjoint(test_topics)
        assert train_topics | val_topics | test_topics == all_topics


def test_v6_1_2_metrics_and_baseline_coverage_are_reported() -> None:
    report = load_report()
    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    assert report["baseline_scoring"] == baseline_report
    assert report["aggregate"]["validation"][NEURAL_METHOD]["reading_ndcg_at_10"]["count"] == 4
    assert report["aggregate"]["test"][NEURAL_METHOD]["reading_ndcg_at_10"]["count"] == 4
    assert report["aggregate"]["test"]["bm25"]["reading_ndcg_at_10"]["count"] == 4
    assert report["aggregate"]["test"]["v4_9_guarded_text_blend"]["reading_ndcg_at_10"]["count"] == 0
    assert report["baseline_scoring"]["runtime_scoring_invoked"] is False
    assert report["baseline_scoring"]["coverage_summary"]["test"]["bm25"]["mean"] > 0.9
    for method in ("v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"):
        assert report["baseline_scoring"]["coverage_summary"]["test"][method]["mean"] == 0.0
        assert "not comparable" in report["baseline_scoring"]["comparability"][method]


def test_v6_1_2_metrics_computation_smoke() -> None:
    rows = [
        {
            "query_id": "q",
            f"{NEURAL_METHOD}_score_for_eval": 0.9,
            "reading_value_score": 1.0,
            "topic_match_score": 1.0,
            "relevance_bucket": 2,
            "hard_negative": False,
        },
        {
            "query_id": "q",
            f"{NEURAL_METHOD}_score_for_eval": 0.1,
            "reading_value_score": 0.0,
            "topic_match_score": 0.0,
            "relevance_bucket": 0,
            "hard_negative": True,
        },
    ]
    metrics = evaluate_method(rows, NEURAL_METHOD)
    assert metrics["coverage"]["coverage"] == 1.0
    assert metrics["metrics"]["reading_ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["metrics"]["hard_neg_at_10"] == pytest.approx(0.1)


def test_v6_1_2_prediction_and_failure_schema() -> None:
    predictions = load_jsonl(PREDICTIONS)
    failures = load_jsonl(FAILURES)
    required_prediction = {
        "schema_version",
        "fold_id",
        "split",
        "query_id",
        "paper_id",
        "title",
        "scores",
        "reading_value_score",
        "topic_match_score",
        "relevance_bucket",
        "hard_negative",
    }
    assert len(predictions) == 2480
    assert all(required_prediction <= set(row) for row in predictions[:25])
    assert all(row["schema_version"] == "v6.1.2_neural_cv_prediction" for row in predictions)
    assert all(row["scores"][NEURAL_METHOD] is not None for row in predictions)
    assert {row["split"] for row in predictions} == {"val", "test"}
    assert failures
    assert all(row["schema_version"] == "v6.1.2_neural_cv_failure_example" for row in failures)


def test_v6_1_2_instability_and_recommendation_are_explicit() -> None:
    report = load_report()
    diagnosis = report["instability_diagnosis"]
    assert "v2_multimodal_learning" in diagnosis["failing_topics"]
    assert diagnosis["original_v6_1_validation_topics_unusually_difficult"] is True
    assert diagnosis["original_v6_1_test_topics_unusually_easy"] is True
    assert report["recommendation"]["recommendation"] == "improve_baseline_scoring_first"
    assert report["aggregate"]["delta_vs_v4_9_guarded_text_blend"]["available"] is False
    assert "unavailable" in report["aggregate"]["delta_vs_v4_9_guarded_text_blend"]["reason"]


def test_v6_1_2_methods_constant_matches_reported_methods() -> None:
    assert METHODS == [NEURAL_METHOD, "bm25", "v3_3_ltr", "v4_1_blend", "v4_9_guarded_text_blend"]
    report = load_report()
    for split in ("validation", "test"):
        assert set(report["aggregate"][split]) == set(METHODS)


def test_v6_1_2_protected_hashes_are_unchanged() -> None:
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


def test_v6_1_2_does_not_create_runtime_or_frontend_config() -> None:
    assert not (ROOT / "backend" / "app" / "services" / "retrievers" / "v6_1_2_neural_cross_validation.py").exists()
    assert not (ROOT / "backend" / "app" / "services" / "v6_1_2_neural_ranker.py").exists()
    assert not (ROOT / "frontend" / "src" / "v6_1_2_neural_cross_validation.ts").exists()
    assert not (ROOT / "data" / "processed" / "models" / "v6_1_2_neural_cross_validation_candidate.json").exists()
