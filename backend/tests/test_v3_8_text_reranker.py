from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_v3_8_text_reranker import (  # noqa: E402
    blend_scores,
    build_examples,
    minmax,
    normalized_scores_by_query,
    text_for_row,
)


def test_v3_8_minmax_handles_constant_and_ordered_scores() -> None:
    assert minmax({1: 3.0, 2: 3.0}) == {1: 0.0, 2: 0.0}
    assert minmax({1: 2.0, 2: 4.0}) == {1: 0.0, 2: 1.0}


def test_v3_8_text_for_row_uses_query_title_abstract_candidate_text() -> None:
    text = text_for_row({"title": "A Title", "abstract": "An abstract."})

    assert "Title: A Title" in text
    assert "Abstract: An abstract." in text


def test_v3_8_blend_scores_uses_normalized_cross_encoder_scores() -> None:
    ce_norm = normalized_scores_by_query({"q": {1: 0.1, 2: 0.3}})
    blended = blend_scores({"q": {1: 0.5, 2: 0.5}}, ce_norm, base_weight=0.8, ce_weight=0.2)

    assert blended["q"][1] == pytest.approx(0.4)
    assert blended["q"][2] == pytest.approx(0.6)


def test_v3_8_build_examples_classifies_fixes_and_hurts() -> None:
    candidate_rows_by_q = {
        "q": [
            {"query_id": "q", "query": "query", "paper_id": paper_id, "title": f"paper {paper_id}"}
            for paper_id in range(1, 13)
        ]
    }
    labels = [
        {"query_id": "q", "paper_id": 1, "reading_value_score": 1.0, "topic_match_score": 1.0, "primary_role": "core_methods"},
        {"query_id": "q", "paper_id": 2, "reading_value_score": 0.0, "topic_match_score": 0.0, "primary_role": "negative"},
        {"query_id": "q", "paper_id": 3, "reading_value_score": 1.0, "topic_match_score": 1.0, "primary_role": "background"},
    ]
    method_scores = {
        "left": {"q": {paper_id: 1.0 / paper_id for paper_id in range(1, 13)}},
        "right": {
            "q": {
                3: 1.0,
                4: 0.9,
                5: 0.8,
                6: 0.7,
                7: 0.6,
                8: 0.5,
                9: 0.4,
                10: 0.3,
                11: 0.2,
                12: 0.1,
                1: 0.01,
                2: 0.0,
            }
        },
    }

    examples = build_examples(
        candidate_rows_by_q=candidate_rows_by_q,
        labels=labels,
        method_scores=method_scores,
        left="left",
        right="right",
    )

    assert any(row["paper_id"] == 1 for row in examples["fixes"])
    assert any(row["paper_id"] == 2 for row in examples["hurts"])
