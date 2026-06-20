import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.retrievers import papers_to_documents  # noqa: E402
from app.services.retrievers.embedding import EmbeddingRetriever  # noqa: E402
from scripts.evaluate_retrieval import load_papers  # noqa: E402


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "embeddings" / "all_minilm_l6_v2.npz"


def write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local frozen-transformer embedding index.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-papers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    papers = load_papers()
    if args.max_papers is not None:
        papers = papers[: args.max_papers]
    documents = papers_to_documents(papers)

    retriever = EmbeddingRetriever(model_name=args.model_name, batch_size=args.batch_size)
    retriever.fit(documents)
    retriever.save_index(output_path)
    metadata_path = write_metadata(
        output_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "model_name": args.model_name,
            "paper_count": len(papers),
            "embedding_path": str(output_path),
            "embedding_shape": list(retriever.embeddings.shape) if retriever.embeddings is not None else None,
            "notes": "Frozen transformer embeddings for ResearchPath semantic retrieval baseline.",
        },
    )

    print(f"Papers encoded: {len(papers)}")
    print(f"Embedding index: {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
