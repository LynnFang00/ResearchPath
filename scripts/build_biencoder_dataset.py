import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.models.citation_edge import CitationEdge  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.services.bm25 import BM25Retriever  # noqa: E402
from app.services.retrievers import papers_to_documents  # noqa: E402
from app.services.retrievers.embedding import normalize_rows  # noqa: E402
from app.services.retrievers.faiss import load_faiss  # noqa: E402
from sqlalchemy import select  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "training"
DEFAULT_EMBEDDING_INDEX = REPO_ROOT / "data" / "processed" / "embeddings" / "all_minilm_l6_v2_5k.npz"
DEFAULT_FAISS_INDEX = REPO_ROOT / "data" / "processed" / "faiss" / "all_minilm_l6_v2_5k.faiss"
DEFAULT_FAISS_ID_MAP = REPO_ROOT / "data" / "processed" / "faiss" / "all_minilm_l6_v2_5k.ids.npz"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class TrainingExample:
    query_paper_id: int
    positive_paper_id: int
    negative_paper_ids: list[int]
    split: str
    label_source: str
    negative_source: str
    query_text: str
    positive_text: str
    negative_texts: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "query_paper_id": self.query_paper_id,
            "positive_paper_id": self.positive_paper_id,
            "negative_paper_ids": self.negative_paper_ids,
            "split": self.split,
            "label_source": self.label_source,
            "negative_source": self.negative_source,
            "query_text": self.query_text,
            "positive_text": self.positive_text,
            "negative_texts": self.negative_texts,
        }


def load_papers_and_edges() -> tuple[list[Paper], list[CitationEdge]]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    with SessionLocal() as db:
        papers = list(db.scalars(select(Paper).order_by(Paper.id)).all())
        edges = list(db.scalars(select(CitationEdge).order_by(CitationEdge.id)).all())
    return papers, edges


def build_positive_map(
    *,
    papers: list[Paper],
    edges: list[CitationEdge],
    bidirectional: bool = True,
) -> dict[int, set[int]]:
    paper_ids = {paper.id for paper in papers}
    positives: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        if edge.source_paper_id not in paper_ids or edge.target_paper_id not in paper_ids:
            continue
        if edge.source_paper_id == edge.target_paper_id:
            continue
        positives[edge.source_paper_id].add(edge.target_paper_id)
        if bidirectional:
            positives[edge.target_paper_id].add(edge.source_paper_id)
    return dict(positives)


def stable_split_for_id(
    paper_id: int,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 13,
) -> str:
    digest = hashlib.sha256(f"{seed}:{paper_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def mine_bm25_negatives(
    *,
    query_paper: Paper,
    positives: set[int],
    retriever: BM25Retriever,
    candidates_k: int,
    negatives_per_positive: int,
    query_field: str = "title",
) -> list[int]:
    exclude_ids = set(positives)
    exclude_ids.add(query_paper.id)
    query = query_paper.title if query_field == "title" else query_paper.searchable_text
    results = retriever.search(query, k=candidates_k, exclude_ids=exclude_ids)
    return [result.document_id for result in results[:negatives_per_positive]]


@dataclass
class FaissPaperIndex:
    document_ids: np.ndarray
    embeddings: np.ndarray
    index: Any
    position_by_id: dict[int, int]

    @classmethod
    def load(
        cls,
        *,
        embedding_index_path: Path,
        faiss_index_path: Path,
        faiss_id_map_path: Path,
    ) -> "FaissPaperIndex":
        embedding_payload = np.load(embedding_index_path, allow_pickle=False)
        id_payload = np.load(faiss_id_map_path, allow_pickle=False)
        embedding_document_ids = embedding_payload["document_ids"].astype(np.int64)
        faiss_document_ids = id_payload["document_ids"].astype(np.int64)
        if not np.array_equal(embedding_document_ids, faiss_document_ids):
            raise ValueError("Embedding document ids and FAISS id-map document ids do not match.")

        faiss = load_faiss()
        index = faiss.read_index(str(faiss_index_path))
        embeddings = normalize_rows(embedding_payload["embeddings"]).astype(np.float32)
        return cls(
            document_ids=faiss_document_ids,
            embeddings=embeddings,
            index=index,
            position_by_id={int(paper_id): position for position, paper_id in enumerate(faiss_document_ids)},
        )

    def search_by_paper_id(
        self,
        paper_id: int,
        *,
        k: int,
        exclude_ids: set[int],
    ) -> list[int]:
        position = self.position_by_id.get(paper_id)
        if position is None:
            return []
        search_k = min(int(self.index.ntotal), max(k + len(exclude_ids), k * 5))
        query_embedding = self.embeddings[position : position + 1]
        _, indices = self.index.search(query_embedding, search_k)
        results: list[int] = []
        for index_position in indices[0]:
            if index_position < 0:
                continue
            candidate_id = int(self.document_ids[int(index_position)])
            if candidate_id in exclude_ids:
                continue
            results.append(candidate_id)
            if len(results) >= k:
                break
        return results


def mine_faiss_negatives(
    *,
    query_paper_id: int,
    positives: set[int],
    index: FaissPaperIndex,
    candidates_k: int,
    negatives_per_positive: int,
) -> list[int]:
    exclude_ids = set(positives)
    exclude_ids.add(query_paper_id)
    return index.search_by_paper_id(
        query_paper_id,
        k=min(candidates_k, negatives_per_positive),
        exclude_ids=exclude_ids,
    )[:negatives_per_positive]


def fallback_random_negatives(
    *,
    query_paper_id: int,
    positives: set[int],
    all_paper_ids: list[int],
    negatives_per_positive: int,
    rng: random.Random,
) -> list[int]:
    excluded = set(positives)
    excluded.add(query_paper_id)
    candidates = [paper_id for paper_id in all_paper_ids if paper_id not in excluded]
    rng.shuffle(candidates)
    return candidates[:negatives_per_positive]


def merge_negatives(
    *negative_lists: list[int],
    max_negatives: int,
) -> list[int]:
    seen: set[int] = set()
    merged: list[int] = []
    for negative_list in negative_lists:
        for paper_id in negative_list:
            if paper_id in seen:
                continue
            seen.add(paper_id)
            merged.append(paper_id)
            if len(merged) >= max_negatives:
                return merged
    return merged


def build_training_examples(
    *,
    papers: list[Paper],
    positives_by_query: dict[int, set[int]],
    negatives_per_positive: int = 3,
    bm25_candidates_k: int = 50,
    bm25_query_field: str = "title",
    use_faiss: bool = False,
    embedding_index_path: Path | None = None,
    faiss_index_path: Path | None = None,
    faiss_id_map_path: Path | None = None,
    seed: int = 13,
    max_examples: int | None = None,
) -> list[TrainingExample]:
    paper_by_id = {paper.id: paper for paper in papers}
    documents = papers_to_documents(papers)
    bm25 = BM25Retriever(documents)
    faiss_paper_index: FaissPaperIndex | None = None
    if use_faiss:
        if embedding_index_path is None or faiss_index_path is None or faiss_id_map_path is None:
            raise ValueError("Embedding index, FAISS index, and id map paths are required when use_faiss=True.")
        faiss_paper_index = FaissPaperIndex.load(
            embedding_index_path=embedding_index_path,
            faiss_index_path=faiss_index_path,
            faiss_id_map_path=faiss_id_map_path,
        )

    rng = random.Random(seed)
    all_paper_ids = sorted(paper_by_id)
    examples: list[TrainingExample] = []

    for query_paper_id in sorted(positives_by_query):
        query_paper = paper_by_id.get(query_paper_id)
        if query_paper is None:
            continue
        positives = {paper_id for paper_id in positives_by_query[query_paper_id] if paper_id in paper_by_id}
        if not positives:
            continue
        split = stable_split_for_id(query_paper_id, seed=seed)

        bm25_negatives = mine_bm25_negatives(
            query_paper=query_paper,
            positives=positives,
            retriever=bm25,
            candidates_k=bm25_candidates_k,
            negatives_per_positive=negatives_per_positive,
            query_field=bm25_query_field,
        )
        faiss_negatives: list[int] = []
        if faiss_paper_index is not None:
            faiss_negatives = mine_faiss_negatives(
                query_paper_id=query_paper.id,
                positives=positives,
                index=faiss_paper_index,
                candidates_k=bm25_candidates_k,
                negatives_per_positive=negatives_per_positive,
            )
        random_negatives = fallback_random_negatives(
            query_paper_id=query_paper_id,
            positives=positives,
            all_paper_ids=all_paper_ids,
            negatives_per_positive=negatives_per_positive,
            rng=rng,
        )

        negative_ids = merge_negatives(
            bm25_negatives,
            faiss_negatives,
            random_negatives,
            max_negatives=negatives_per_positive,
        )
        if not negative_ids:
            continue
        negative_source = "bm25_faiss_random" if faiss_negatives else "bm25_random"

        for positive_paper_id in sorted(positives):
            positive_paper = paper_by_id[positive_paper_id]
            examples.append(
                TrainingExample(
                    query_paper_id=query_paper_id,
                    positive_paper_id=positive_paper_id,
                    negative_paper_ids=negative_ids,
                    split=split,
                    label_source="citation_graph",
                    negative_source=negative_source,
                    query_text=query_paper.searchable_text,
                    positive_text=positive_paper.searchable_text,
                    negative_texts=[paper_by_id[negative_id].searchable_text for negative_id in negative_ids],
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                return examples

    return examples


def split_examples(examples: list[TrainingExample]) -> dict[str, list[TrainingExample]]:
    splits: dict[str, list[TrainingExample]] = {split: [] for split in SPLIT_NAMES}
    for example in examples:
        splits[example.split].append(example)
    return splits


def write_jsonl(examples: list[TrainingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=True) + "\n")


def build_report(
    *,
    papers: list[Paper],
    edges: list[CitationEdge],
    examples: list[TrainingExample],
    output_paths: dict[str, str],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    query_ids = {example.query_paper_id for example in examples}
    positive_pairs = {(example.query_paper_id, example.positive_paper_id) for example in examples}
    negatives_count = sum(len(example.negative_paper_ids) for example in examples)
    split_counts = Counter(example.split for example in examples)
    negative_sources = Counter(example.negative_source for example in examples)
    positives_per_query = Counter(example.query_paper_id for example in examples)
    negatives_per_example = [len(example.negative_paper_ids) for example in examples]

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "paper_count": len(papers),
        "citation_edge_count": len(edges),
        "query_paper_count": len(query_ids),
        "positive_pair_count": len(positive_pairs),
        "example_count": len(examples),
        "negative_count": negatives_count,
        "average_positives_per_query": (
            sum(positives_per_query.values()) / len(positives_per_query) if positives_per_query else 0.0
        ),
        "average_negatives_per_example": (
            sum(negatives_per_example) / len(negatives_per_example) if negatives_per_example else 0.0
        ),
        "split_counts": {split: split_counts.get(split, 0) for split in SPLIT_NAMES},
        "label_sources": {"citation_graph": len(examples)},
        "negative_sources": dict(negative_sources),
        "output_paths": output_paths,
        "config": vars(args) if args is not None else {},
        "notes": (
            "Weak contrastive training data generated from citation graph positives and "
            "BM25/FAISS hard negatives. Labels are noisy and should be evaluated against baselines."
        ),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weakly supervised bi-encoder training data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--bm25-candidates-k", type=int, default=50)
    parser.add_argument("--bm25-query-field", choices=["title", "title_abstract"], default="title")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--directed", action="store_true", help="Use only directed citation edges as positives.")
    parser.add_argument("--use-faiss", action="store_true", help="Use FAISS results as additional hard negatives.")
    parser.add_argument("--embedding-index", default=str(DEFAULT_EMBEDDING_INDEX))
    parser.add_argument("--faiss-index", default=str(DEFAULT_FAISS_INDEX))
    parser.add_argument("--faiss-id-map", default=str(DEFAULT_FAISS_ID_MAP))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    embedding_index_path = Path(args.embedding_index)
    faiss_index_path = Path(args.faiss_index)
    faiss_id_map_path = Path(args.faiss_id_map)
    if not embedding_index_path.is_absolute():
        embedding_index_path = REPO_ROOT / embedding_index_path
    if not faiss_index_path.is_absolute():
        faiss_index_path = REPO_ROOT / faiss_index_path
    if not faiss_id_map_path.is_absolute():
        faiss_id_map_path = REPO_ROOT / faiss_id_map_path

    papers, edges = load_papers_and_edges()
    positives_by_query = build_positive_map(
        papers=papers,
        edges=edges,
        bidirectional=not args.directed,
    )
    examples = build_training_examples(
        papers=papers,
        positives_by_query=positives_by_query,
        negatives_per_positive=args.negatives_per_positive,
        bm25_candidates_k=args.bm25_candidates_k,
        bm25_query_field=args.bm25_query_field,
        use_faiss=args.use_faiss,
        embedding_index_path=embedding_index_path,
        faiss_index_path=faiss_index_path,
        faiss_id_map_path=faiss_id_map_path,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    splits = split_examples(examples)
    output_paths = {
        "train": str(output_dir / "bi_encoder_train.jsonl"),
        "val": str(output_dir / "bi_encoder_val.jsonl"),
        "test": str(output_dir / "bi_encoder_test.jsonl"),
        "report": str(output_dir / "bi_encoder_dataset_report.json"),
    }

    for split, split_examples_ in splits.items():
        write_jsonl(split_examples_, Path(output_paths[split]))

    report = build_report(
        papers=papers,
        edges=edges,
        examples=examples,
        output_paths=output_paths,
        args=args,
    )
    write_report(report, Path(output_paths["report"]))

    print(f"Papers: {len(papers)}")
    print(f"Citation edges: {len(edges)}")
    print(f"Examples: {len(examples)}")
    print(f"Train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Report: {output_paths['report']}")


if __name__ == "__main__":
    main()
