import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.retrievers.faiss import FaissRetriever  # noqa: E402


DEFAULT_EMBEDDINGS = REPO_ROOT / "data" / "processed" / "embeddings" / "all_minilm_l6_v2_5k.npz"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "faiss" / "all_minilm_l6_v2_5k.faiss"


def default_id_map_path(index_path: Path) -> Path:
    return index_path.with_suffix(".ids.npz")


def load_embedding_payload(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    required = {"document_ids", "embeddings"}
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"Embedding index is missing required arrays: {sorted(missing)}")
    return {
        "document_ids": payload["document_ids"].astype(np.int64),
        "embeddings": payload["embeddings"].astype(np.float32),
        "model_name": str(payload["model_name"]) if "model_name" in payload.files else "",
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from a ResearchPath embedding .npz file.")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--id-map", default=None)
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_path = Path(args.embeddings)
    output_path = Path(args.output)
    if not embeddings_path.is_absolute():
        embeddings_path = REPO_ROOT / embeddings_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    id_map_path = Path(args.id_map) if args.id_map else default_id_map_path(output_path)
    if not id_map_path.is_absolute():
        id_map_path = REPO_ROOT / id_map_path

    payload = load_embedding_payload(embeddings_path)
    model_name = args.model_name or payload["model_name"] or "sentence-transformers/all-MiniLM-L6-v2"
    retriever = FaissRetriever(model_name=model_name)
    retriever.build_from_embeddings(payload["document_ids"], payload["embeddings"])
    retriever.save_index(output_path, id_map_path)

    metadata_path = write_metadata(
        output_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_embeddings": str(embeddings_path),
            "faiss_index": str(output_path),
            "id_map": str(id_map_path),
            "model_name": model_name,
            "index_type": "IndexFlatIP",
            "paper_count": int(len(payload["document_ids"])),
            "dimension": int(payload["embeddings"].shape[1]),
            "notes": "FAISS exact inner-product index over normalized frozen-transformer embeddings.",
        },
    )

    print(f"FAISS index: {output_path}")
    print(f"ID map: {id_map_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Vectors indexed: {len(payload['document_ids'])}")


if __name__ == "__main__":
    main()
