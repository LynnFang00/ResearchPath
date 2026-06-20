from argparse import Namespace
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.models.citation_edge import CitationEdge  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from scripts.build_biencoder_dataset import (  # noqa: E402
    TrainingExample,
    build_positive_map,
    build_report,
    merge_negatives,
    split_examples,
    stable_split_for_id,
)


def paper(paper_id: int, title: str) -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract=f"Abstract for {title}",
        authors="ResearchPath",
        categories="cs.AI",
        citation_count=0,
    )


def test_build_positive_map_supports_bidirectional_edges():
    papers = [paper(1, "A"), paper(2, "B"), paper(3, "C")]
    edges = [
        CitationEdge(source_paper_id=1, target_paper_id=2, relationship_type="citation"),
        CitationEdge(source_paper_id=2, target_paper_id=3, relationship_type="citation"),
    ]

    positives = build_positive_map(papers=papers, edges=edges, bidirectional=True)

    assert positives[1] == {2}
    assert positives[2] == {1, 3}
    assert positives[3] == {2}


def test_stable_split_for_id_is_deterministic():
    assert stable_split_for_id(123, seed=9) == stable_split_for_id(123, seed=9)
    assert stable_split_for_id(123, seed=9) in {"train", "val", "test"}


def test_merge_negatives_deduplicates_in_priority_order():
    assert merge_negatives([1, 2], [2, 3], [4], max_negatives=3) == [1, 2, 3]


def test_split_examples_and_report_shape():
    papers = [paper(1, "Query"), paper(2, "Positive"), paper(3, "Negative")]
    edges = [CitationEdge(source_paper_id=1, target_paper_id=2, relationship_type="citation")]
    example = TrainingExample(
        query_paper_id=1,
        positive_paper_id=2,
        negative_paper_ids=[3],
        split="train",
        label_source="citation_graph",
        negative_source="bm25_random",
        query_text="query",
        positive_text="positive",
        negative_texts=["negative"],
    )

    splits = split_examples([example])
    report = build_report(
        papers=papers,
        edges=edges,
        examples=[example],
        output_paths={"train": "train.jsonl"},
        args=Namespace(seed=13),
    )

    assert len(splits["train"]) == 1
    assert report["example_count"] == 1
    assert report["negative_count"] == 1
    assert report["split_counts"]["train"] == 1
