import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_reranker_dataset import (  # noqa: E402
    build_report,
    load_biencoder_rows,
    rows_to_reranker_examples,
)


def test_load_biencoder_rows_validates_negative_alignment(tmp_path):
    path = tmp_path / "bi_encoder_train.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_paper_id": 1,
                "positive_paper_id": 2,
                "negative_paper_ids": [3],
                "split": "train",
                "query_text": "query",
                "positive_text": "positive",
                "negative_texts": ["negative"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_biencoder_rows(path)

    assert rows[0]["query_paper_id"] == 1


def test_rows_to_reranker_examples_keeps_positive_over_conflicting_negative():
    rows = [
        {
            "query_paper_id": 1,
            "positive_paper_id": 2,
            "negative_paper_ids": [2, 3],
            "split": "train",
            "label_source": "citation_graph",
            "negative_source": "bm25",
            "query_text": "query",
            "positive_text": "positive",
            "negative_texts": ["should not become negative", "negative"],
        }
    ]

    examples = rows_to_reranker_examples(rows, seed=1)
    labels_by_candidate = {example.candidate_paper_id: example.label for example in examples}

    assert labels_by_candidate == {2: 1, 3: 0}


def test_reranker_report_counts_labels():
    examples = rows_to_reranker_examples(
        [
            {
                "query_paper_id": 1,
                "positive_paper_id": 2,
                "negative_paper_ids": [3, 4],
                "split": "train",
                "query_text": "query",
                "positive_text": "positive",
                "negative_texts": ["negative 1", "negative 2"],
            }
        ]
    )

    report = build_report(examples=examples, output_paths={"train": "train.jsonl"})

    assert report["positive_count"] == 1
    assert report["negative_count"] == 2
    assert report["split_counts"]["train"] == 3
