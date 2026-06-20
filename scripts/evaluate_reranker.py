import argparse
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.db.schema import ensure_runtime_schema  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.ml.evaluation import evaluate_ranking  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.services.retrievers import papers_to_documents  # noqa: E402
from scripts.evaluate_retrieval import build_retrievers, load_evaluation_examples  # noqa: E402
from sqlalchemy import select  # noqa: E402


DEFAULT_EXAMPLES_PATH = REPO_ROOT / "data" / "processed" / "evaluation_examples" / "openalex_ai_ml_5k_weak_labels.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "evaluations"
DEFAULT_CROSS_ENCODER = REPO_ROOT / "data" / "processed" / "models" / "cross_encoder_minilm_smoke"
DEFAULT_BI_ENCODER_EMBEDDINGS = REPO_ROOT / "data" / "processed" / "embeddings" / "biencoder_mnrl_1k_5k.npz"
DEFAULT_BI_ENCODER_FAISS = REPO_ROOT / "data" / "processed" / "faiss" / "biencoder_mnrl_1k_5k.faiss"
DEFAULT_BI_ENCODER_ID_MAP = REPO_ROOT / "data" / "processed" / "faiss" / "biencoder_mnrl_1k_5k.ids.npz"
DEFAULT_BI_ENCODER_MODEL = REPO_ROOT / "data" / "processed" / "models" / "biencoder_all_minilm_mnrl_1k"


def load_papers() -> list[Paper]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    with SessionLocal() as db:
        return list(db.scalars(select(Paper).order_by(Paper.id)).all())


def rerank_candidate_ids(
    candidate_ids: list[int],
    scores: list[float],
    *,
    k: int,
) -> list[int]:
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate_ids and scores must have the same length.")
    ranked = sorted(
        zip(candidate_ids, scores, range(len(candidate_ids)), strict=True),
        key=lambda item: (-item[1], item[2]),
    )
    return [candidate_id for candidate_id, _, _ in ranked[:k]]


def score_query_candidates(
    *,
    query: str,
    candidate_ids: list[int],
    paper_by_id: dict[int, Paper],
    score_pairs: Callable[[list[tuple[str, str]]], list[float]],
) -> list[float]:
    pairs = [
        (query, paper_by_id[candidate_id].searchable_text)
        for candidate_id in candidate_ids
        if candidate_id in paper_by_id
    ]
    if len(pairs) != len(candidate_ids):
        missing_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in paper_by_id]
        raise ValueError(f"Candidate ids missing from paper map: {missing_ids[:5]}")
    return score_pairs(pairs)


def evaluate_reranker(
    *,
    examples: list[dict[str, Any]],
    papers: list[Paper],
    candidate_method: str,
    score_pairs: Callable[[list[tuple[str, str]]], list[float]],
    candidate_k: int = 50,
    rerank_k: int = 10,
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
    if candidate_method not in retrievers:
        raise ValueError(f"Unknown candidate method '{candidate_method}'. Available: {sorted(retrievers)}")

    paper_by_id = {paper.id: paper for paper in papers}
    candidate_retriever = retrievers[candidate_method]
    method_name = f"{candidate_method}_cross_encoder"
    rows: list[dict[str, Any]] = []

    for example in examples:
        relevant_ids = {int(item) for item in example["relevant_paper_ids"]}
        start = time.perf_counter()
        candidates = candidate_retriever.search(example["query"], k=candidate_k)
        candidate_ids = [candidate.document_id for candidate in candidates]
        scores = score_query_candidates(
            query=example["query"],
            candidate_ids=candidate_ids,
            paper_by_id=paper_by_id,
            score_pairs=score_pairs,
        )
        reranked_ids = rerank_candidate_ids(candidate_ids, scores, k=rerank_k)
        latency_ms = (time.perf_counter() - start) * 1000
        metrics_at_10 = evaluate_ranking(
            retrieved_ids=reranked_ids,
            relevant_ids=relevant_ids,
            k=10,
            latency_ms=latency_ms,
        )
        recall_at_5 = evaluate_ranking(
            retrieved_ids=reranked_ids,
            relevant_ids=relevant_ids,
            k=5,
        ).recall_at_k
        rows.append(
            {
                "query_id": example["query_id"],
                "query": example["query"],
                "candidate_method": candidate_method,
                "candidate_ids": candidate_ids,
                "reranked_ids": reranked_ids,
                "recall_at_5": recall_at_5,
                "recall_at_10": metrics_at_10.recall_at_k,
                "ndcg_at_10": metrics_at_10.ndcg_at_k,
                "mrr": metrics_at_10.mrr,
                "latency_ms": latency_ms,
            }
        )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "paper_count": len(papers),
        "query_count": len(examples),
        "candidate_k": candidate_k,
        "rerank_k": rerank_k,
        "metrics": ["recall_at_5", "recall_at_10", "ndcg_at_10", "mrr", "latency_ms"],
        "methods": {
            method_name: {
                "queries": rows,
                "averages": {
                    "recall_at_5": _mean(row["recall_at_5"] for row in rows),
                    "recall_at_10": _mean(row["recall_at_10"] for row in rows),
                    "ndcg_at_10": _mean(row["ndcg_at_10"] for row in rows),
                    "mrr": _mean(row["mrr"] for row in rows),
                    "latency_ms": _mean(row["latency_ms"] for row in rows),
                },
            }
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reranker_evaluation_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_comparison_table(report: dict[str, Any]) -> None:
    print("\nCross-Encoder Reranker Evaluation")
    print(f"Papers: {report['paper_count']} | Queries: {report['query_count']} | Candidate@{report['candidate_k']}")
    print("-" * 84)
    print(f"{'method':<32} {'R@5':>8} {'R@10':>8} {'NDCG@10':>10} {'MRR':>8} {'latency':>12}")
    print("-" * 84)
    for method_name, payload in report["methods"].items():
        averages = payload["averages"]
        print(
            f"{method_name:<32} "
            f"{averages['recall_at_5']:>8.3f} "
            f"{averages['recall_at_10']:>8.3f} "
            f"{averages['ndcg_at_10']:>10.3f} "
            f"{averages['mrr']:>8.3f} "
            f"{averages['latency_ms']:>9.2f} ms"
        )
    print("-" * 84)
    print(f"Saved report: {report['output_path']}")


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_model_name_or_path(value: str) -> str:
    path = Path(value)
    candidate = path if path.is_absolute() else REPO_ROOT / path
    if candidate.exists() or value.startswith(("data/", "data\\")):
        return str(candidate)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a cross-encoder reranker over first-stage candidates.")
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cross-encoder-model", default=str(DEFAULT_CROSS_ENCODER))
    parser.add_argument(
        "--candidate-method",
        choices=["bm25", "tfidf", "citation_recency", "embedding", "faiss_embedding"],
        default="faiss_embedding",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--embedding-index", default=str(DEFAULT_BI_ENCODER_EMBEDDINGS))
    parser.add_argument("--embedding-model", default=str(DEFAULT_BI_ENCODER_MODEL))
    parser.add_argument("--faiss-index", default=str(DEFAULT_BI_ENCODER_FAISS))
    parser.add_argument("--faiss-id-map", default=str(DEFAULT_BI_ENCODER_ID_MAP))
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    examples = load_evaluation_examples(resolve_path(args.examples) or DEFAULT_EXAMPLES_PATH)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    papers = load_papers()
    model = CrossEncoder(resolve_model_name_or_path(args.cross_encoder_model))

    def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    report = evaluate_reranker(
        examples=examples,
        papers=papers,
        candidate_method=args.candidate_method,
        score_pairs=score_pairs,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
        output_dir=resolve_path(args.output_dir) or DEFAULT_OUTPUT_DIR,
        embedding_index=resolve_path(args.embedding_index),
        embedding_model=args.embedding_model,
        faiss_index=resolve_path(args.faiss_index),
        faiss_id_map=resolve_path(args.faiss_id_map),
    )
    print_comparison_table(report)


if __name__ == "__main__":
    main()
