import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_biencoder import load_training_rows, rows_to_pairs, rows_to_triplets  # noqa: E402


def test_load_training_rows_validates_required_fields(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_text": "query",
                "positive_text": "positive",
                "negative_texts": ["negative"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_training_rows(path)

    assert rows[0]["query_text"] == "query"


def test_rows_to_triplets_expands_negatives_deterministically():
    rows = [
        {
            "query_text": "query",
            "positive_text": "positive",
            "negative_texts": ["negative 1", "negative 2"],
        }
    ]

    triplets = rows_to_triplets(rows, seed=1)

    assert sorted(triplets) == [
        ("query", "positive", "negative 1"),
        ("query", "positive", "negative 2"),
    ]
    assert rows_to_triplets(rows, max_triplets=1, seed=1) == triplets[:1]


def test_rows_to_pairs_uses_query_positive_only():
    rows = [
        {
            "query_text": "query",
            "positive_text": "positive",
            "negative_texts": ["negative 1", "negative 2"],
        }
    ]

    assert rows_to_pairs(rows, seed=1) == [("query", "positive")]
