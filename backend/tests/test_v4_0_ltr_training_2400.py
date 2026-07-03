import json
from pathlib import Path
import sys

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v4_0_ltr_dataset_2400 import DEFAULT_DATASET_JSONL, load_jsonl  # noqa: E402
from scripts.train_v3_0_ltr_models import numeric_feature_value  # noqa: E402


MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v4_0_ltr_2400"
REPORT = REPO_ROOT / "data" / "eval" / "results" / "v4_0_ltr_2400_report.json"


def test_v4_0_training_report_and_index_exist() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    index = json.loads((MODEL_DIR / "index.json").read_text(encoding="utf-8"))

    assert report["dataset"]["row_count"] == 2400
    assert report["best_no_v27_model_by_loto_reading_ndcg"]["model_name"]
    assert index["trained_on_all_v4_0_2400_rows"] is True
    assert index["runtime_integration_status"] == "offline_only_not_integrated"


def test_v4_0_saved_best_model_loads_and_scores_sample_rows() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    model_name = report["best_no_v27_model_by_loto_reading_ndcg"]["model_name"]
    metadata = json.loads((MODEL_DIR / f"{model_name}.metadata.json").read_text(encoding="utf-8"))
    estimator = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    sample_rows = load_jsonl(DEFAULT_DATASET_JSONL)[:5]

    assert metadata["feature_columns"]
    assert metadata["uses_v2_7_feature"] is False
    matrix = np.array(
        [[numeric_feature_value(row, feature) for feature in metadata["feature_columns"]] for row in sample_rows],
        dtype=float,
    )
    scores = estimator.predict(matrix)

    assert len(scores) == 5
    assert np.isfinite(scores).all()
