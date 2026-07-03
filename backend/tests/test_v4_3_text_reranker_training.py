import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_training_report.json"
MODEL_DIR = ROOT / "data" / "processed" / "models" / "v4_3_text_reranker"


def test_v4_3_training_report_and_artifacts() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["dataset"]["row_count"] == 2400
    assert report["dataset"]["validation_valid"] is True
    assert report["best_model_by_loto_reading_ndcg"]["score_column"].startswith("v4_3_")
    assert len(report["artifacts"]) == 3
    for artifact in report["artifacts"]:
        artifact_path = Path(artifact["artifact_path"])
        metadata_path = Path(artifact["metadata_path"])
        assert artifact_path.exists()
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["excludes_v2_7_feature"] is True
        assert metadata["excludes_raw_score_features"] is True
        payload = joblib.load(artifact_path)
        assert payload["model_name"] == artifact["model_name"]


def test_v4_3_model_index() -> None:
    index = json.loads((MODEL_DIR / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "v4.3_text_reranker_model_index"
    assert {row["model_name"] for row in index["models"]} == {
        "text_only_ridge",
        "text_positive_logistic",
        "text_plus_ranker_features_ridge",
    }
