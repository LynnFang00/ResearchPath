import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.ml.evaluation import evaluate_ranking, time_call  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.services.bm25 import BM25Retriever  # noqa: E402
from app.services.retrievers import papers_to_documents  # noqa: E402
from app.services.retrievers.citation_recency import CitationRecencyRetriever  # noqa: E402
from app.services.retrievers.embedding import EmbeddingRetriever  # noqa: E402
from app.services.retrievers.faiss import FaissRetriever  # noqa: E402
from app.services.retrievers.tfidf import TfidfRetriever  # noqa: E402
from sqlalchemy import select  # noqa: E402


DEFAULT_EXAMPLES_PATH = REPO_ROOT / "data" / "raw" / "evaluation_examples.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "evaluations"


def load_evaluation_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation examples file was not found: {path}")

    examples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("query_id"):
            raise ValueError(f"Line {line_number}: missing query_id.")
        if not record.get("query"):
            raise ValueError(f"Line {line_number}: missing query.")
        if not isinstance(record.get("relevant_paper_ids"), list):
            raise ValueError(f"Line {line_number}: relevant_paper_ids must be a list.")
        examples.append(record)
    return examples


def load_papers() -> list[Paper]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    with SessionLocal() as db:
        return list(db.scalars(select(Paper).order_by(Paper.id)).all())


def build_retrievers(
    documents,
    *,
    embedding_index: Path | None = None,
    embedding_model: str | None = None,
    faiss_index: Path | None = None,
    faiss_id_map: Path | None = None,
):
    retrievers = [
        BM25Retriever(documents),
        TfidfRetriever(documents),
        CitationRecencyRetriever(documents),
    ]
    if embedding_index is not None:
        embedding_retriever = EmbeddingRetriever(model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
        embedding_retriever.load_index(embedding_index)
        retrievers.append(embedding_retriever)
    if faiss_index is not None:
        if faiss_id_map is None:
            raise ValueError("--faiss-id-map is required when --faiss-index is provided.")
        faiss_retriever = FaissRetriever(model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2")
        faiss_retriever.load_index(faiss_index, faiss_id_map)
        retrievers.append(faiss_retriever)
    return {retriever.method_name: retriever for retriever in retrievers}


def evaluate_retrievers(
    *,
    examples: list[dict[str, Any]],
    papers: list[Paper],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    embedding_index: Path | None = None,
    embedding_model: str | None = None,
    faiss_index: Path | None = None,
    faiss_id_map: Path | None = None,
) -> dict[str, Any]:
    documents = papers_to_documents(papers)
    retrievers = build_retrievers(
        documents,
        embedding_index=embedding_index,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        faiss_id_map=faiss_id_map,
    )
    per_method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for example in examples:
        relevant_ids = {int(item) for item in example["relevant_paper_ids"]}
        for method_name, retriever in retrievers.items():
            results, latency_ms = time_call(lambda: retriever.search(example["query"], k=10))
            retrieved_ids = [result.document_id for result in results]
            metrics_at_10 = evaluate_ranking(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=10,
                latency_ms=latency_ms,
            )
            recall_at_5 = evaluate_ranking(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=5,
            ).recall_at_k
            per_method_rows[method_name].append(
                {
                    "query_id": example["query_id"],
                    "query": example["query"],
                    "retrieved_ids": retrieved_ids,
                    "recall_at_5": recall_at_5,
                    "recall_at_10": metrics_at_10.recall_at_k,
                    "ndcg_at_10": metrics_at_10.ndcg_at_k,
                    "mrr": metrics_at_10.mrr,
                    "latency_ms": latency_ms,
                }
            )

    methods: dict[str, Any] = {}
    for method_name, rows in per_method_rows.items():
        methods[method_name] = {
            "queries": rows,
            "averages": {
                "recall_at_5": _mean(row["recall_at_5"] for row in rows),
                "recall_at_10": _mean(row["recall_at_10"] for row in rows),
                "ndcg_at_10": _mean(row["ndcg_at_10"] for row in rows),
                "mrr": _mean(row["mrr"] for row in rows),
                "latency_ms": _mean(row["latency_ms"] for row in rows),
            },
        }

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "paper_count": len(papers),
        "query_count": len(examples),
        "metrics": ["recall_at_5", "recall_at_10", "ndcg_at_10", "mrr", "latency_ms"],
        "methods": methods,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"retrieval_evaluation_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_comparison_table(report: dict[str, Any]) -> None:
    print("\nRetrieval Evaluation")
    print(f"Papers: {report['paper_count']} | Queries: {report['query_count']}")
    print("-" * 76)
    print(f"{'method':<20} {'R@5':>8} {'R@10':>8} {'NDCG@10':>10} {'MRR':>8} {'latency':>12}")
    print("-" * 76)
    for method_name, payload in report["methods"].items():
        averages = payload["averages"]
        print(
            f"{method_name:<20} "
            f"{averages['recall_at_5']:>8.3f} "
            f"{averages['recall_at_10']:>8.3f} "
            f"{averages['ndcg_at_10']:>10.3f} "
            f"{averages['mrr']:>8.3f} "
            f"{averages['latency_ms']:>9.2f} ms"
        )
    print("-" * 76)
    print(f"Saved report: {report['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResearchPath retrieval baselines.")
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES_PATH), help="JSONL evaluation examples.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON reports.")
    parser.add_argument("--embedding-index", default=None, help="Optional .npz embedding index to evaluate.")
    parser.add_argument("--embedding-model", default=None, help="Model name used for query embeddings.")
    parser.add_argument("--faiss-index", default=None, help="Optional .faiss index to evaluate.")
    parser.add_argument("--faiss-id-map", default=None, help="Optional .npz document id map for the FAISS index.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap for faster development runs.")
    args = parser.parse_args()

    examples = load_evaluation_examples(Path(args.examples))
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    papers = load_papers()
    embedding_index = Path(args.embedding_index) if args.embedding_index else None
    if embedding_index is not None and not embedding_index.is_absolute():
        embedding_index = REPO_ROOT / embedding_index
    faiss_index = Path(args.faiss_index) if args.faiss_index else None
    if faiss_index is not None and not faiss_index.is_absolute():
        faiss_index = REPO_ROOT / faiss_index
    faiss_id_map = Path(args.faiss_id_map) if args.faiss_id_map else None
    if faiss_id_map is not None and not faiss_id_map.is_absolute():
        faiss_id_map = REPO_ROOT / faiss_id_map
    report = evaluate_retrievers(
        examples=examples,
        papers=papers,
        output_dir=Path(args.output_dir),
        embedding_index=embedding_index,
        embedding_model=args.embedding_model,
        faiss_index=faiss_index,
        faiss_id_map=faiss_id_map,
    )
    print_comparison_table(report)


if __name__ == "__main__":
    main()
