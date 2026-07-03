import json
from pathlib import Path

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_1_weighted_ltr_2400_report.json"
MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v4_1_weighted_report_and_model_metadata() -> None:
    report = load_json(REPORT)
    assert report["schema_version"] == "v4.1_weighted_ltr_2400_report"
    assert report["dataset"]["row_count"] == 2400
    assert report["model_family"] == "random_forest_no_v27_compact"
    assert report["runtime_defaults_changed_by_this_script"] is False
    assert report["frontend_defaults_changed_by_this_script"] is False
    assert report["labels_modified_by_this_script"] is False
    assert report["corpus_expanded_by_this_script"] is False
    assert len(report["weighting_strategies"]) >= 10
    assert all(value is True for key, value in report["protected_hashes"].items() if key.endswith("_unchanged"))
    assert "v3_3_random_forest_no_v27_compact" in report["production_pool"]["evaluation"]["method_averages"]
    assert "v4_0_random_forest_no_v27_compact" in report["production_pool"]["evaluation"]["method_averages"]

    index = load_json(MODEL_DIR / "index.json")
    strategy = index["best_candidate_strategy"]
    metadata = load_json(Path(index["artifacts"][strategy]["metadata_path"]))
    assert metadata["feature_columns"] == report["feature_columns"]
    assert metadata["uses_v2_7_feature"] is False
    assert metadata["weight_summary"]["strategy"] == strategy


def test_v4_1_weighted_model_scores_are_finite() -> None:
    index = load_json(MODEL_DIR / "index.json")
    strategy = index["best_candidate_strategy"]
    metadata = load_json(Path(index["artifacts"][strategy]["metadata_path"]))
    estimator = joblib.load(Path(index["artifacts"][strategy]["artifact_path"]))
    x_values = np.zeros((3, len(metadata["feature_columns"])), dtype=float)
    scores = estimator.predict(x_values)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
