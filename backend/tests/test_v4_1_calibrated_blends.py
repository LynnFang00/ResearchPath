import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_calibrated_blend_report.json"
CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score_blend(config: dict, scores: dict[str, float]) -> float:
    spec = config["spec"]
    if spec.get("guardrail") == "v33_with_v40_low_score_demotion":
        demotion = 0.20 if scores["v3_3_random_forest_no_v27_compact"] >= 0.70 and scores["v4_0_random_forest_no_v27_compact"] <= 0.25 else 0.0
        return scores["v3_3_random_forest_no_v27_compact"] - demotion
    total = 0.0
    for method, weight in spec["weights"].items():
        resolved = config["best_v41_method"] if method == "__v41__" else method
        total += float(weight) * scores[resolved]
    return total


def test_v4_1_calibrated_blend_report_and_config() -> None:
    report = load_json(REPORT)
    assert report["schema_version"] == "v4.1_calibrated_blend_report"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert "v3_3_random_forest_no_v27_compact" in report["evaluation"]["method_averages"]
    assert "v4_0_random_forest_no_v27_compact" in report["evaluation"]["method_averages"]
    assert report["best_v41_weighted_method"] in report["evaluation"]["method_averages"]
    assert report["selection"]["ranked_blends"]
    assert all(value is True for key, value in report["protected_hashes"].items() if key.endswith("_unchanged"))

    config = load_json(CONFIG)
    assert config["schema_version"] == "v4.1_calibrated_blend_config"
    assert config["runtime_integration_status"] == "offline_only_not_integrated"
    assert config["feature_order"]


def test_v4_1_blend_config_scores_deterministically() -> None:
    config = load_json(CONFIG)
    scores = {
        "v3_3_random_forest_no_v27_compact": 0.8,
        "v4_0_random_forest_no_v27_compact": 0.2,
        "v4_1_uniform_all_labels": 0.6,
        config["best_v41_method"]: 0.6,
        "hybrid": 0.4,
        "v2_6": 0.5,
    }
    first = score_blend(config, scores)
    second = score_blend(config, scores)
    assert isinstance(first, float)
    assert first == second
