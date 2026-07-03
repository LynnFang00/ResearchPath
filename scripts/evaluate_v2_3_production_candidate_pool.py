import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.bm25 import BM25Retriever  # noqa: E402
from app.services.difficulty import score_difficulty  # noqa: E402
from app.services.quality_signals import compute_quality_signals, hybrid_score, normalize_scores  # noqa: E402
from app.services.retrievers.base import PaperDocument  # noqa: E402
from app.services.retrievers.tfidf import TfidfRetriever  # noqa: E402
from app.services.v2_2_learned_ranker import (  # noqa: E402
    build_feature_context,
    load_v2_2_ridge_ranker,
)


DEFAULT_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_V22B = REPO_ROOT / "data" / "eval" / "results" / "v2_2b_fair_learned_hybrid_report.json"
DEFAULT_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_RAW_CORPUS = REPO_ROOT / "data" / "raw" / "arxiv_ml_ai_50k_incremental.jsonl"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_3_production_candidate_pool_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v2_3_production_candidate_pool_report.md"
SELECTED_240 = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"
EXPECTED_SELECTED_240_SHA256 = "6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F"

METHODS = ["learned_hybrid", "hybrid", "bm25", "tfidf", "embedding", "faiss_embedding"]
AVAILABLE_RETRIEVAL_METHODS = ["bm25", "tfidf", "hybrid"]
UNAVAILABLE_RETRIEVAL_METHODS = {
    "embedding": "local 50K embedding artifact lacks a reachable paper_id-to-metadata mapping without the database",
    "faiss_embedding": "local 50K FAISS id map lacks a reachable paper_id-to-metadata mapping without the database",
}
INTENT_SCORE_FIELDS = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
AUDIENCE_SCORE_FIELDS = {
    "beginner": "beginner_fit_score",
    "intermediate": "intermediate_fit_score",
    "advanced": "advanced_fit_score",
    "expert": "expert_fit_score",
}
PATH_ROLES = ["background", "foundational", "core_methods", "recent_frontier"]
ROLE_VALUES_FOR_COVERAGE = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
DUPLICATE_PENALTIES = {"none": 0.0, "uncertain": 0.25, "near_duplicate": 0.5, "exact_duplicate": 1.0}
LEAKAGE_FIELD_NAMES = [
    "reading_value_score",
    "topic_match_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
    "intent_scores",
    "primary_role",
    "secondary_roles",
    "duplicate_status",
    "label_confidence",
    "notes",
    "selection_reasons",
    "likely_coverage",
]


@dataclass
class CorpusPaper:
    id: int
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    venue: str | None
    categories: str
    citation_count: int
    source_url: str | None
    external_id: str | None
    source: str | None
    doi: str | None
    url: str | None
    references_count: int
    influential_citation_count: int
    abstract_word_count: int
    identifiers: dict[str, Any]
    evidence_availability: Any = None
    duplicate_title_cluster: Any = None

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n\n{self.abstract}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_corpus(raw_path: Path, packet_rows: list[dict[str, Any]]) -> dict[int, CorpusPaper]:
    papers: dict[int, CorpusPaper] = {}
    for index, row in enumerate(load_jsonl(raw_path), start=1):
        paper_id = 1_000_000 + index
        papers[paper_id] = CorpusPaper(
            id=paper_id,
            title=str(row.get("title") or ""),
            abstract=str(row.get("abstract") or ""),
            authors=list(row.get("authors") or []),
            year=int(row["year"]) if isinstance(row.get("year"), int) else None,
            venue=row.get("venue"),
            categories=", ".join(str(value) for value in row.get("categories") or []),
            citation_count=int(row.get("citation_count") or 0),
            source_url=row.get("source_url") or row.get("url"),
            external_id=row.get("external_id"),
            source=row.get("source"),
            doi=row.get("doi"),
            url=row.get("url"),
            references_count=int(row.get("references_count") or 0),
            influential_citation_count=int(row.get("influential_citation_count") or 0),
            abstract_word_count=int(row.get("abstract_word_count") or 0),
            identifiers=dict(row.get("identifiers") or {}),
        )

    for row in packet_rows:
        paper_id = int(row["paper_id"])
        identifiers = dict(row.get("identifiers") or {})
        papers[paper_id] = CorpusPaper(
            id=paper_id,
            title=str(row.get("title") or ""),
            abstract=str(row.get("abstract") or ""),
            authors=list(row.get("authors") or []),
            year=int(row["year"]) if isinstance(row.get("year"), int) else None,
            venue=row.get("venue"),
            categories=", ".join(str(value) for value in row.get("source_specific_metadata", {}).get("categories") or []),
            citation_count=int(row.get("citation_count") or 0),
            source_url=row.get("source_url"),
            external_id=identifiers.get("arxiv_id") or identifiers.get("openalex_id"),
            source=(row.get("sources_provenance") or ["packet"])[0],
            doi=identifiers.get("doi"),
            url=row.get("source_url"),
            references_count=0,
            influential_citation_count=0,
            abstract_word_count=len(str(row.get("abstract") or "").split()),
            identifiers=identifiers,
            evidence_availability=row.get("evidence_availability"),
            duplicate_title_cluster=row.get("duplicate_title_cluster"),
        )
    return papers


def paper_documents(papers: dict[int, CorpusPaper]) -> list[PaperDocument]:
    return [
        PaperDocument(
            document_id=paper.id,
            text=paper.searchable_text,
            title=paper.title,
            abstract=paper.abstract,
            year=paper.year,
            citation_count=paper.citation_count,
        )
        for paper in papers.values()
    ]


def labels_by_query(labels: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for label in labels:
        grouped[str(label["query_id"])][int(label["paper_id"])] = label
    return dict(grouped)


def packet_by_query(packet_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packet_rows:
        grouped[str(row["query_id"])].append(row)
    return dict(grouped)


def build_retrieval_runs(
    *,
    papers: dict[int, CorpusPaper],
    packet_by_q: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> dict[str, dict[str, list[dict[str, float | int]]]]:
    documents = paper_documents(papers)
    bm25 = BM25Retriever(documents)
    tfidf = TfidfRetriever(documents)
    runs: dict[str, dict[str, list[dict[str, float | int]]]] = defaultdict(dict)
    for query_id, rows in sorted(packet_by_q.items()):
        query = str(rows[0]["query"])
        bm25_results = bm25.search(query, k=top_k)
        tfidf_results = tfidf.search(query, k=top_k)
        runs[query_id]["bm25"] = scored_rows(bm25_results)
        runs[query_id]["tfidf"] = scored_rows(tfidf_results)
        runs[query_id]["hybrid"] = hybrid_results(
            query=query,
            papers=papers,
            bm25_results=bm25.search(query, k=top_k * 8),
            tfidf_results=tfidf.search(query, k=top_k * 8),
            top_k=top_k,
        )
    return dict(runs)


def scored_rows(results: Any) -> list[dict[str, float | int]]:
    return [
        {"paper_id": int(item.document_id), "score": float(item.score), "rank": rank}
        for rank, item in enumerate(results, start=1)
    ]


def hybrid_results(
    *,
    query: str,
    papers: dict[int, CorpusPaper],
    bm25_results: Any,
    tfidf_results: Any,
    top_k: int,
) -> list[dict[str, float | int]]:
    bm25_raw = {int(item.document_id): float(item.score) for item in bm25_results}
    tfidf_raw = {int(item.document_id): float(item.score) for item in tfidf_results}
    bm25_scores = normalize_scores(bm25_raw)
    tfidf_scores = normalize_scores(tfidf_raw)
    candidate_ids = set(bm25_scores) | set(tfidf_scores)
    scored: list[tuple[float, int]] = []
    for paper_id in candidate_ids:
        paper = papers.get(paper_id)
        if paper is None:
            continue
        difficulty = score_difficulty(paper)
        signals = compute_quality_signals(
            query=query,
            paper=paper,
            difficulty=difficulty,
            background_level="basic_ml",
            bm25_score=bm25_scores.get(paper_id, 0.0),
            tfidf_score=tfidf_scores.get(paper_id, 0.0),
            faiss_score=0.0,
            current_year=2026,
        )
        scored.append((hybrid_score(signals), paper_id))
    return [
        {"paper_id": paper_id, "score": float(score), "rank": rank}
        for rank, (score, paper_id) in enumerate(sorted(scored, reverse=True)[:top_k], start=1)
    ]


def build_candidate_rows(
    *,
    papers: dict[int, CorpusPaper],
    packet_by_q: dict[str, list[dict[str, Any]]],
    runs: dict[str, dict[str, list[dict[str, float | int]]]],
) -> dict[str, list[dict[str, Any]]]:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for query_id, packet_rows in sorted(packet_by_q.items()):
        query = str(packet_rows[0]["query"])
        candidate_ids: set[int] = set()
        for method in AVAILABLE_RETRIEVAL_METHODS:
            candidate_ids.update(int(row["paper_id"]) for row in runs[query_id].get(method, []))
        method_maps = {
            method: {int(row["paper_id"]): row for row in rows}
            for method, rows in runs[query_id].items()
        }
        rows: list[dict[str, Any]] = []
        for paper_id in sorted(candidate_ids):
            paper = papers[paper_id]
            ranks = {
                method: int(method_maps[method][paper_id]["rank"])
                for method in AVAILABLE_RETRIEVAL_METHODS
                if paper_id in method_maps.get(method, {})
            }
            scores = {
                method: float(method_maps[method][paper_id]["score"])
                for method in AVAILABLE_RETRIEVAL_METHODS
                if paper_id in method_maps.get(method, {})
            }
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "paper_id": paper_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "venue": paper.venue,
                    "authors": paper.authors,
                    "citation_count": paper.citation_count,
                    "sources_provenance": [paper.source] if paper.source else [],
                    "identifiers": {
                        "arxiv_id": paper.identifiers.get("arxiv_id") or paper.identifiers.get("arxiv"),
                        "doi": paper.doi or paper.identifiers.get("doi"),
                        "openalex_id": paper.identifiers.get("openalex_id") or paper.identifiers.get("openalex"),
                    },
                    "source_url": paper.source_url,
                    "pdf_url": None,
                    "evidence_availability": paper.evidence_availability
                    or {"level": "title_abstract", "fulltext_available": False},
                    "duplicate_title_cluster": paper.duplicate_title_cluster,
                    "retrieval_ranks_by_method": ranks,
                    "retrieval_scores_by_method": scores,
                }
            )
        all_rows[query_id] = rows
    return all_rows


def rank_methods(
    *,
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    model_path: Path,
) -> dict[str, dict[str, list[int]]]:
    all_rows = [row for rows in candidate_rows_by_q.values() for row in rows]
    context = build_feature_context(all_rows)
    ranker = load_v2_2_ridge_ranker(model_path)
    ranked: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        ranked[query_id]["learned_hybrid"] = [
            paper_id
            for _, paper_id in sorted(
                (
                    (ranker.score_packet_row(row, context), int(row["paper_id"]))
                    for row in rows
                ),
                reverse=True,
            )
        ]
        for method in AVAILABLE_RETRIEVAL_METHODS:
            ranked[query_id][method] = [
                int(row["paper_id"])
                for row in sorted(
                    rows,
                    key=lambda row: (
                        method in row.get("retrieval_scores_by_method", {}),
                        float(row.get("retrieval_scores_by_method", {}).get(method, 0.0)),
                        1.0 / max(float(row.get("retrieval_ranks_by_method", {}).get(method, 999999)), 1.0)
                        if method in row.get("retrieval_ranks_by_method", {})
                        else 0.0,
                        -int(row["paper_id"]),
                    ),
                    reverse=True,
                )
            ]
        for method in UNAVAILABLE_RETRIEVAL_METHODS:
            ranked[query_id][method] = []
    return dict(ranked)


def evaluate_methods(
    *,
    ranked: dict[str, dict[str, list[int]]],
    candidate_rows_by_q: dict[str, list[dict[str, Any]]],
    labels_by_q: dict[str, dict[int, dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for query_id, method_rankings in ranked.items():
        candidate_ids = {int(row["paper_id"]) for row in candidate_rows_by_q[query_id]}
        query_labels = {
            paper_id: label
            for paper_id, label in labels_by_q[query_id].items()
            if paper_id in candidate_ids
        }
        for method in METHODS:
            row = evaluate_ranked_ids(
                method=method,
                query_id=query_id,
                query_labels=query_labels,
                all_query_labels=labels_by_q[query_id],
                candidate_ids=candidate_ids,
                retrieved_ids=method_rankings.get(method, []),
                k=k,
            )
            rows_by_method[method].append(row)
    return {
        "method_averages": {method: aggregate(rows) for method, rows in rows_by_method.items()},
        "per_topic": rows_by_method,
        "failure_cases": build_failure_cases(rows_by_method),
    }


def evaluate_ranked_ids(
    *,
    method: str,
    query_id: str,
    query_labels: dict[int, dict[str, Any]],
    all_query_labels: dict[int, dict[str, Any]],
    candidate_ids: set[int],
    retrieved_ids: list[int],
    k: int,
) -> dict[str, Any]:
    score_maps = {
        "topic_match_ndcg": score_map(query_labels, "topic_match_score"),
        "reading_value_ndcg": score_map(query_labels, "reading_value_score"),
    }
    for audience, field_name in AUDIENCE_SCORE_FIELDS.items():
        score_maps[f"{audience}_ndcg"] = score_map(query_labels, field_name)
    for intent in INTENT_SCORE_FIELDS:
        score_maps[f"intent_{intent}_ndcg"] = intent_score_map(query_labels, intent)
    metrics = {
        metric_name: ndcg_at_k(retrieved_ids, relevance_by_id, k)
        for metric_name, relevance_by_id in score_maps.items()
    }
    judged_top_k = sum(1 for paper_id in retrieved_ids[:k] if paper_id in query_labels)
    return {
        "query_id": query_id,
        "method": method,
        "retrieved_ids_at_k": retrieved_ids[:k],
        "candidate_pool_size": len(candidate_ids),
        "judged_candidates_in_pool": len(query_labels),
        "packet_candidate_coverage": len(query_labels) / max(len(all_query_labels), 1),
        "judged_retrieved_at_k": judged_top_k,
        "unjudged_retrieved_at_k": max(0, min(k, len(retrieved_ids)) - judged_top_k),
        "can_rank_all_candidates": len(retrieved_ids) == len(candidate_ids),
        **metrics,
        "role_coverage_at_k": role_coverage_at_k(retrieved_ids, query_labels, k)["coverage"],
        "path_level_coverage_at_k": path_level_coverage_at_k(retrieved_ids, query_labels, k)["coverage"],
        "duplicate_penalty_at_k": duplicate_penalty_at_k(retrieved_ids, query_labels, k),
    }


def score_map(labels: dict[int, dict[str, Any]], field_name: str) -> dict[int, float]:
    return {paper_id: clamp_score(label.get(field_name)) for paper_id, label in labels.items()}


def intent_score_map(labels: dict[int, dict[str, Any]], intent: str) -> dict[int, float]:
    return {paper_id: clamp_score((label.get("intent_scores") or {}).get(intent)) for paper_id, label in labels.items()}


def ndcg_at_k(retrieved_ids: list[int], relevance_by_id: dict[int, float], k: int) -> float:
    gains = [relevance_by_id.get(paper_id, 0.0) for paper_id in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / ideal if ideal > 0 else 0.0


def role_coverage_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> dict[str, Any]:
    available = topic_roles(labels.values(), threshold=0.3)
    covered = topic_roles((labels[paper_id] for paper_id in retrieved_ids[:k] if paper_id in labels), threshold=0.3)
    denominator = len(available) if available else len(ROLE_VALUES_FOR_COVERAGE)
    return {"coverage": len(covered & available) / denominator if denominator else 0.0}


def topic_roles(rows: Any, threshold: float) -> set[str]:
    roles: set[str] = set()
    for row in rows:
        if clamp_score(row.get("reading_value_score")) < threshold:
            continue
        primary_role = row.get("primary_role")
        if primary_role in ROLE_VALUES_FOR_COVERAGE:
            roles.add(str(primary_role))
        secondary_roles = row.get("secondary_roles")
        if isinstance(secondary_roles, list):
            roles.update(str(role) for role in secondary_roles if role in ROLE_VALUES_FOR_COVERAGE)
    return roles


def path_level_coverage_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> dict[str, Any]:
    covered: set[str] = set()
    for paper_id in retrieved_ids[:k]:
        label = labels.get(paper_id)
        if not label or clamp_score(label.get("reading_value_score")) < 0.5:
            continue
        roles = [label.get("primary_role")]
        secondary_roles = label.get("secondary_roles")
        if isinstance(secondary_roles, list):
            roles.extend(secondary_roles)
        covered.update(str(role) for role in roles if role in PATH_ROLES)
    return {"coverage": len(covered) / len(PATH_ROLES)}


def duplicate_penalty_at_k(retrieved_ids: list[int], labels: dict[int, dict[str, Any]], k: int) -> float:
    if k <= 0:
        return 0.0
    penalties = [
        DUPLICATE_PENALTIES.get(str(labels.get(paper_id, {}).get("duplicate_status", "none")), 0.0)
        for paper_id in retrieved_ids[:k]
    ]
    return sum(penalties) / k


def clamp_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def metric_fields() -> list[str]:
    return [
        "topic_match_ndcg",
        "reading_value_ndcg",
        "beginner_ndcg",
        "intermediate_ndcg",
        "advanced_ndcg",
        "expert_ndcg",
        "role_coverage_at_k",
        "path_level_coverage_at_k",
        "duplicate_penalty_at_k",
        "judged_retrieved_at_k",
        "unjudged_retrieved_at_k",
        "candidate_pool_size",
        "judged_candidates_in_pool",
        "packet_candidate_coverage",
    ] + [f"intent_{intent}_ndcg" for intent in INTENT_SCORE_FIELDS]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {field: sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0 for field in metric_fields()}


def build_failure_cases(rows_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, rows in rows_by_method.items():
        for row in rows:
            by_query[row["query_id"]][method] = row
    failures = {
        "learned_loses_to_hybrid_on_reading_value_ndcg": [],
        "learned_loses_to_bm25_on_topic_match_ndcg": [],
        "embedding_or_faiss_beats_learned": [],
    }
    for query_id, rows in sorted(by_query.items()):
        learned = rows.get("learned_hybrid")
        if not learned:
            continue
        hybrid = rows.get("hybrid")
        bm25 = rows.get("bm25")
        if hybrid and learned["reading_value_ndcg"] < hybrid["reading_value_ndcg"]:
            failures["learned_loses_to_hybrid_on_reading_value_ndcg"].append(
                failure_row(query_id, learned, hybrid, "reading_value_ndcg")
            )
        if bm25 and learned["topic_match_ndcg"] < bm25["topic_match_ndcg"]:
            failures["learned_loses_to_bm25_on_topic_match_ndcg"].append(
                failure_row(query_id, learned, bm25, "topic_match_ndcg")
            )
        for method in ["embedding", "faiss_embedding"]:
            row = rows.get(method)
            if row and row["reading_value_ndcg"] > learned["reading_value_ndcg"]:
                failures["embedding_or_faiss_beats_learned"].append(
                    failure_row(query_id, learned, row, "reading_value_ndcg")
                )
    return failures


def failure_row(query_id: str, learned: dict[str, Any], competitor: dict[str, Any], metric: str) -> dict[str, Any]:
    coverage_issue = learned["judged_candidates_in_pool"] < 15
    missing_scores = not learned["can_rank_all_candidates"] or not competitor["can_rank_all_candidates"]
    return {
        "query_id": query_id,
        "competitor": competitor["method"],
        "metric": metric,
        "learned": learned[metric],
        "competitor_value": competitor[metric],
        "delta": learned[metric] - competitor[metric],
        "likely_cause": "retrieval coverage" if coverage_issue else "missing scores" if missing_scores else "ranking order/model behavior",
    }


def packet_prediction_repro_check(packet_rows: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    context = build_feature_context(packet_rows)
    ranker = load_v2_2_ridge_ranker(model_path)
    scores_a = [ranker.score_packet_row(row, context) for row in packet_rows]
    scores_b = [ranker.score_packet_row(row, context) for row in packet_rows]
    diffs = [abs(a - b) for a, b in zip(scores_a, scores_b, strict=True)]
    return {
        "packet_prediction_count": len(packet_rows),
        "artifact_prediction_max_abs_diff": max(diffs, default=0.0),
        "artifact_prediction_mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "artifact_can_regenerate_packet_predictions": max(diffs, default=0.0) < 1e-12,
    }


def build_report(
    *,
    labels_path: Path,
    packet_path: Path,
    v22b_path: Path,
    model_path: Path,
    raw_corpus_path: Path,
    top_k: int,
    eval_k: int,
) -> dict[str, Any]:
    labels = load_jsonl(labels_path)
    packet_rows = load_jsonl(packet_path)
    v22b = load_json(v22b_path)
    papers = load_corpus(raw_corpus_path, packet_rows)
    packet_lookup = packet_by_query(packet_rows)
    runs = build_retrieval_runs(papers=papers, packet_by_q=packet_lookup, top_k=top_k)
    candidate_rows = build_candidate_rows(papers=papers, packet_by_q=packet_lookup, runs=runs)
    ranked = rank_methods(candidate_rows_by_q=candidate_rows, model_path=model_path)
    evaluation = evaluate_methods(
        ranked=ranked,
        candidate_rows_by_q=candidate_rows,
        labels_by_q=labels_by_query(labels),
        k=eval_k,
    )
    names = load_v2_2_ridge_ranker(model_path).feature_names
    feature_text = "\n".join(names)
    checks = {
        "grouped_cv_by_query_id": "not_applicable_no_training_or_cv_in_v2_3",
        "label_fields_used_as_features": [name for name in LEAKAGE_FIELD_NAMES if name in feature_text],
        "selection_reasons_used": "selection_reasons" in feature_text,
        "likely_coverage_used": "likely_coverage" in feature_text,
        "selected_240_sha256": file_sha256(SELECTED_240),
        "selected_240_matches_expected_hash": file_sha256(SELECTED_240) == EXPECTED_SELECTED_240_SHA256,
        "labels_sha256": file_sha256(labels_path),
    } | packet_prediction_repro_check(packet_rows, model_path)
    learned = evaluation["method_averages"]["learned_hybrid"]
    hybrid = evaluation["method_averages"]["hybrid"]
    bm25 = evaluation["method_averages"]["bm25"]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2.3",
        "evaluation_scope": "production_style_candidate_pool_fallback_50k_jsonl_plus_packet",
        "is_true_50k_production_pool_retrieval": False,
        "fallback_reason": (
            "The local database was not reachable, and embedding/FAISS id maps do not include a standalone "
            "paper_id-to-metadata mapping. The fallback builds BM25, TF-IDF, and current-hybrid over the local "
            "50K JSONL plus injected labeled packet papers."
        ),
        "candidate_pool_definition": {
            "corpus": str(raw_corpus_path),
            "injected_labeled_packet_rows": len(packet_rows),
            "corpus_size_with_packet_injection": len(papers),
            "union": ["top_50_bm25", "top_50_tfidf", "top_50_current_hybrid"],
            "deduplicate_by": "paper_id",
            "unavailable_methods": UNAVAILABLE_RETRIEVAL_METHODS,
        },
        "comparison_to_v2_2b": {
            "v2_2b_same_packet_learned_reading_value_ndcg": v22b["same_packet_reranking"]["method_averages"]["learned_hybrid"]["reading_value_ndcg"],
            "v2_2b_restricted_learned_reading_value_ndcg": v22b["restricted_candidate_evaluation"]["method_averages"]["learned_hybrid"]["reading_value_ndcg"],
            "v2_3_learned_reading_value_ndcg": learned["reading_value_ndcg"],
            "interpretation": (
                "V2.3 is stricter than same-packet evaluation because it includes many unjudged production-style "
                "candidates and reports judged coverage separately."
            ),
        },
        "method_averages": evaluation["method_averages"],
        "per_topic": evaluation["per_topic"],
        "failure_cases": evaluation["failure_cases"],
        "success_criteria": {
            "learned_vs_hybrid_reading_value_delta": learned["reading_value_ndcg"] - hybrid["reading_value_ndcg"],
            "learned_vs_bm25_topic_match_delta": learned["topic_match_ndcg"] - bm25["topic_match_ndcg"],
            "beat_hybrid_on_reading_value_ndcg": learned["reading_value_ndcg"] > hybrid["reading_value_ndcg"],
            "avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg": (
                learned["topic_match_ndcg"] - bm25["topic_match_ndcg"]
            ) >= -0.01,
        },
        "reproducibility_checks": checks,
        "safe_to_present_as_production_style_learned_reranking": False,
        "safety_note": (
            "This is safe to present as a production-style fallback experiment, not as true production-pool "
            "retrieval, because dense/FAISS full-corpus metadata mapping was unavailable without the DB."
        ),
        "runtime_backend_integration": "scaffold_only_v2_2b_feature_scorer_exists_but_full_candidate_context_integration_not_done",
    }


def markdown_report(report: dict[str, Any], *, command: str) -> str:
    lines = [
        "# V2.3 Production Candidate-Pool learned_hybrid Evaluation",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "## Candidate Pool",
        "",
        f"- True 50K production-pool retrieval: `{report['is_true_50k_production_pool_retrieval']}`",
        f"- Fallback reason: {report['fallback_reason']}",
        f"- Corpus size with packet injection: `{report['candidate_pool_definition']['corpus_size_with_packet_injection']}`",
        f"- Union: `{', '.join(report['candidate_pool_definition']['union'])}`",
        f"- Unavailable methods: `{json.dumps(report['candidate_pool_definition']['unavailable_methods'], sort_keys=True)}`",
        "",
        "## Method Comparison",
        "",
    ]
    lines.extend(method_table(report["method_averages"]))
    criteria = report["success_criteria"]
    lines.extend(
        [
            "",
            "## V2.2b Comparison",
            "",
            f"- V2.2b same-packet learned reading NDCG@10: `{report['comparison_to_v2_2b']['v2_2b_same_packet_learned_reading_value_ndcg']:.3f}`",
            f"- V2.2b restricted learned reading NDCG@10: `{report['comparison_to_v2_2b']['v2_2b_restricted_learned_reading_value_ndcg']:.3f}`",
            f"- V2.3 learned reading NDCG@10: `{report['comparison_to_v2_2b']['v2_3_learned_reading_value_ndcg']:.3f}`",
            "",
            "## Success Criteria",
            "",
            f"- learned_hybrid vs hybrid reading-value NDCG@10: `{criteria['learned_vs_hybrid_reading_value_delta']:+.3f}`",
            f"- learned_hybrid vs BM25 topic-match NDCG@10: `{criteria['learned_vs_bm25_topic_match_delta']:+.3f}`",
            f"- Beat hybrid on reading-value NDCG@10: `{criteria['beat_hybrid_on_reading_value_ndcg']}`",
            f"- Avoid losing more than 0.01 to BM25 on topic-match NDCG@10: `{criteria['avoid_losing_more_than_0_01_to_bm25_on_topic_match_ndcg']}`",
            "",
            "## Failure Cases",
            "",
            "### learned_hybrid loses to hybrid on reading_value NDCG@10",
            "",
            "| query_id | learned | competitor | delta | likely cause |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["failure_cases"]["learned_loses_to_hybrid_on_reading_value_ndcg"]:
        lines.append(
            f"| `{row['query_id']}` | {row['learned']:.3f} | {row['competitor_value']:.3f} | "
            f"{row['delta']:+.3f} | {row['likely_cause']} |"
        )
    lines.extend(
        [
            "",
            "### learned_hybrid loses to BM25 on topic_match NDCG@10",
            "",
            "| query_id | learned | competitor | delta | likely cause |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["failure_cases"]["learned_loses_to_bm25_on_topic_match_ndcg"]:
        lines.append(
            f"| `{row['query_id']}` | {row['learned']:.3f} | {row['competitor_value']:.3f} | "
            f"{row['delta']:+.3f} | {row['likely_cause']} |"
        )
    checks = report["reproducibility_checks"]
    lines.extend(
        [
            "",
            "## Reproducibility Checks",
            "",
            f"- Grouped CV by query_id: `{checks['grouped_cv_by_query_id']}`",
            f"- Label fields used as features: `{checks['label_fields_used_as_features']}`",
            f"- `selection_reasons` used: `{checks['selection_reasons_used']}`",
            f"- `likely_coverage` used: `{checks['likely_coverage_used']}`",
            f"- Selected 240 SHA256: `{checks['selected_240_sha256']}`",
            f"- Selected 240 hash matches expected: `{checks['selected_240_matches_expected_hash']}`",
            f"- Labels SHA256: `{checks['labels_sha256']}`",
            f"- Saved V2.2b packet prediction max diff: `{checks['artifact_prediction_max_abs_diff']:.12f}`",
            "",
            "## Conclusion",
            "",
            f"- Safe to present as production-style learned reranking: `{report['safe_to_present_as_production_style_learned_reranking']}`",
            f"- Runtime backend integration: `{report['runtime_backend_integration']}`",
            f"- Safety note: {report['safety_note']}",
            "",
        ]
    )
    return "\n".join(lines)


def method_table(averages: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| method | topic NDCG@10 | reading NDCG@10 | beginner | intermediate | advanced | expert | path coverage | role coverage | dup penalty | judged@10 | unjudged@10 | pool size | judged in pool | packet coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(averages, key=lambda name: averages[name]["reading_value_ndcg"], reverse=True):
        row = averages[method]
        lines.append(
            f"| `{method}` | {row['topic_match_ndcg']:.3f} | {row['reading_value_ndcg']:.3f} | "
            f"{row['beginner_ndcg']:.3f} | {row['intermediate_ndcg']:.3f} | {row['advanced_ndcg']:.3f} | "
            f"{row['expert_ndcg']:.3f} | {row['path_level_coverage_at_k']:.3f} | {row['role_coverage_at_k']:.3f} | "
            f"{row['duplicate_penalty_at_k']:.3f} | {row['judged_retrieved_at_k']:.2f} | "
            f"{row['unjudged_retrieved_at_k']:.2f} | {row['candidate_pool_size']:.2f} | "
            f"{row['judged_candidates_in_pool']:.2f} | {row['packet_candidate_coverage']:.3f} |"
        )
    return lines


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2.3 production-style shared candidate pools.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--v22b-report", default=str(DEFAULT_V22B))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--raw-corpus", default=str(DEFAULT_RAW_CORPUS))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = resolve_repo_path(args.labels)
    packet_path = resolve_repo_path(args.packet)
    v22b_path = resolve_repo_path(args.v22b_report)
    model_path = resolve_repo_path(args.model)
    raw_corpus_path = resolve_repo_path(args.raw_corpus)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    report = build_report(
        labels_path=labels_path,
        packet_path=packet_path,
        v22b_path=v22b_path,
        model_path=model_path,
        raw_corpus_path=raw_corpus_path,
        top_k=args.top_k,
        eval_k=args.eval_k,
    )
    command = (
        f".\\backend\\.venv\\Scripts\\python.exe scripts\\evaluate_v2_3_production_candidate_pool.py "
        f"--labels {args.labels} --packet {args.packet} --v22b-report {args.v22b_report} "
        f"--model {args.model} --raw-corpus {args.raw_corpus} --json-out {args.json_out} "
        f"--md-out {args.md_out} --top-k {args.top_k} --eval-k {args.eval_k}"
    )
    report["command"] = command
    write_json(json_out, report)
    write_text(md_out, markdown_report(report, command=command))
    criteria = report["success_criteria"]
    print("V2.3 production-style candidate-pool evaluation complete")
    print(f"Report JSON: {json_out}")
    print(f"Report Markdown: {md_out}")
    print(f"True 50K production retrieval: {report['is_true_50k_production_pool_retrieval']}")
    print(f"learned vs hybrid reading delta: {criteria['learned_vs_hybrid_reading_value_delta']:+.3f}")
    print(f"learned vs BM25 topic delta: {criteria['learned_vs_bm25_topic_match_delta']:+.3f}")
    print(f"safe as production-style: {report['safe_to_present_as_production_style_learned_reranking']}")


if __name__ == "__main__":
    main()
