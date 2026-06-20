import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.models.citation_edge import CitationEdge  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from sqlalchemy import select  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "evaluation_examples" / "weak_labels.jsonl"


def build_query(paper: Paper, mode: str) -> str:
    if mode == "title":
        return paper.title
    if mode == "title_abstract":
        return f"{paper.title}. {paper.abstract}"
    if mode == "goal":
        categories = paper.categories.replace(";", ",")
        category_text = f" in {categories}" if categories else ""
        return f"I want to understand {paper.title}{category_text}"
    raise ValueError(f"Unsupported query mode: {mode}")


def generate_examples(
    *,
    papers: list[Paper],
    edges: list[CitationEdge],
    query_mode: str = "title",
    bidirectional: bool = True,
    min_relevant: int = 1,
    max_relevant: int = 20,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    paper_by_id = {paper.id: paper for paper in papers}
    relevant_by_paper: dict[int, set[int]] = {paper.id: set() for paper in papers}

    for edge in edges:
        if edge.source_paper_id not in paper_by_id or edge.target_paper_id not in paper_by_id:
            continue
        relevant_by_paper[edge.source_paper_id].add(edge.target_paper_id)
        if bidirectional:
            relevant_by_paper[edge.target_paper_id].add(edge.source_paper_id)

    examples: list[dict[str, Any]] = []
    for paper in papers:
        relevant_ids = sorted(relevant_by_paper.get(paper.id, set()))
        if len(relevant_ids) < min_relevant:
            continue
        relevant_ids = relevant_ids[:max_relevant]
        examples.append(
            {
                "query_id": f"paper_{paper.id}_citation_neighbors",
                "query": build_query(paper, query_mode),
                "relevant_paper_ids": relevant_ids,
                "notes": (
                    "Weak labels generated from citation graph edges. "
                    "Labels are noisy and indicate relatedness, not human-judged relevance."
                ),
            }
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


def write_examples(examples: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")


def load_graph() -> tuple[list[Paper], list[CitationEdge]]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    with SessionLocal() as db:
        papers = list(db.scalars(select(Paper).order_by(Paper.id)).all())
        edges = list(db.scalars(select(CitationEdge).order_by(CitationEdge.id)).all())
    return papers, edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak retrieval labels from citation edges.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--query-mode", choices=["title", "title_abstract", "goal"], default="title")
    parser.add_argument("--min-relevant", type=int, default=1)
    parser.add_argument("--max-relevant", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--directed", action="store_true", help="Use only source -> target relevance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    papers, edges = load_graph()
    examples = generate_examples(
        papers=papers,
        edges=edges,
        query_mode=args.query_mode,
        bidirectional=not args.directed,
        min_relevant=args.min_relevant,
        max_relevant=args.max_relevant,
        max_examples=args.max_examples,
    )
    write_examples(examples, output_path)

    print(f"Papers: {len(papers)}")
    print(f"Citation edges: {len(edges)}")
    print(f"Weak-label examples: {len(examples)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
