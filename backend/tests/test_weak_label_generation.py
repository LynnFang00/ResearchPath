import sys
from pathlib import Path

from app.models.citation_edge import CitationEdge
from app.models.paper import Paper


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_weak_labels import build_query, generate_examples  # noqa: E402


def test_build_query_modes() -> None:
    paper = Paper(
        id=1,
        title="AI Agents for Science",
        abstract="Agents can plan experiments.",
        categories="Artificial Intelligence; Scientific Discovery",
    )

    assert build_query(paper, "title") == "AI Agents for Science"
    assert "Agents can plan experiments" in build_query(paper, "title_abstract")
    assert "Artificial Intelligence" in build_query(paper, "goal")


def test_generate_examples_from_citation_edges_bidirectional() -> None:
    papers = [
        Paper(id=1, title="Seed Paper", abstract="Seed abstract"),
        Paper(id=2, title="Referenced Paper", abstract="Reference abstract"),
    ]
    edges = [
        CitationEdge(
            id=1,
            source_paper_id=1,
            target_paper_id=2,
            relationship_type="reference",
            source="test",
        )
    ]

    examples = generate_examples(papers=papers, edges=edges, bidirectional=True)

    assert len(examples) == 2
    assert examples[0]["relevant_paper_ids"] == [2]
    assert examples[1]["relevant_paper_ids"] == [1]


def test_generate_examples_respects_min_relevant() -> None:
    papers = [
        Paper(id=1, title="Lonely Paper", abstract="No edges"),
        Paper(id=2, title="Other Paper", abstract="No edges"),
    ]

    examples = generate_examples(papers=papers, edges=[], min_relevant=1)

    assert examples == []
