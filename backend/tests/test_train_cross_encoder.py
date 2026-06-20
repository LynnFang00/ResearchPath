from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_cross_encoder import (  # noqa: E402
    label_counts,
    rows_to_cross_encoder_examples,
    sample_balanced_rows,
)


def test_sample_balanced_rows_respects_label_balance():
    rows = [
        {"query_text": f"q{i}", "candidate_text": f"p{i}", "label": 1}
        for i in range(4)
    ] + [
        {"query_text": f"q{i}", "candidate_text": f"n{i}", "label": 0}
        for i in range(10)
    ]

    sampled = sample_balanced_rows(rows, max_examples=6, positive_fraction=0.5, seed=1)

    assert len(sampled) == 6
    assert label_counts(sampled) == {"positive": 3, "negative": 3}


def test_rows_to_cross_encoder_examples_returns_text_pair_and_float_label():
    rows = [{"query_text": "query", "candidate_text": "candidate", "label": 1}]

    examples = rows_to_cross_encoder_examples(rows)

    assert examples == [("query", "candidate", 1.0)]
