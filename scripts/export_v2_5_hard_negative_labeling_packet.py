import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.v2_2_learned_ranker import build_feature_context, load_v2_2_ridge_ranker  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    BASELINE_METHODS,
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    DEFAULT_PACKET,
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    build_candidate_rows,
    build_documents,
    build_retrieval_runs,
    file_sha256,
    labels_by_query,
    load_db_assets,
    load_json,
    load_jsonl,
    load_vector_assets,
    packet_by_query,
    resolve_repo_path,
    write_json,
    write_text,
)


DEFAULT_V24 = REPO_ROOT / "data" / "eval" / "results" / "v2_4_true_production_candidate_pool_report.json"
DEFAULT_JSONL_OUT = REPO_ROOT / "data" / "eval" / "v2_5_hard_negative_labeling_packet.jsonl"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "data" / "eval" / "results" / "v2_5_hard_negative_packet_summary.json"
DEFAULT_SUMMARY_MD = REPO_ROOT / "data" / "eval" / "results" / "v2_5_hard_negative_packet_summary.md"
EXPECTED_LABELS_SHA256 = "0ECC4A0F638E77E76D44913281EC6431ADDF1873474C3B80A48531B9529359F7"
METHODS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid", "learned_hybrid"]
REQUIRED_FIELDS = [
    "schema_version",
    "query_id",
    "query",
    "paper_id",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "citation_count",
    "sources_provenance",
    "identifiers",
    "source_url",
    "pdf_url",
    "retrieval_ranks_by_method",
    "retrieval_scores_by_method",
    "why_selected",
    "selection_bucket",
    "v2_4_topic_context",
]
LABEL_FIELDS = {
    "topic_match_score",
    "reading_value_score",
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
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def add_learned_scores(candidate_rows_by_q: dict[str, list[dict[str, Any]]], model_path: Path) -> dict[str, dict[str, list[int]]]:
    all_rows = [row for rows in candidate_rows_by_q.values() for row in rows]
    context = build_feature_context(all_rows)
    ranker = load_v2_2_ridge_ranker(model_path)
    ranked: dict[str, dict[str, list[int]]] = {}
    for query_id, rows in sorted(candidate_rows_by_q.items()):
        learned_scores = {
            int(row["paper_id"]): float(ranker.score_packet_row(row, context))
            for row in rows
        }
        learned_ranked = [
            paper_id
            for paper_id, _ in sorted(
                learned_scores.items(),
                key=lambda item: (item[1], -item[0]),
                reverse=True,
            )
        ]
        ranked[query_id] = {"learned_hybrid": learned_ranked}
        learned_ranks = {paper_id: rank for rank, paper_id in enumerate(learned_ranked, start=1)}
        for row in rows:
            paper_id = int(row["paper_id"])
            row["retrieval_scores_by_method"]["learned_hybrid"] = learned_scores[paper_id]
            row["retrieval_ranks_by_method"]["learned_hybrid"] = learned_ranks[paper_id]
        for method in BASELINE_METHODS:
            ranked[query_id][method] = [
                int(row["paper_id"])
                for row in sorted(
                    rows,
                    key=lambda row: (
                        float(row["retrieval_scores_by_method"].get(method, 0.0)),
                        -int(row["retrieval_ranks_by_method"].get(method, 999999)),
                        -int(row["paper_id"]),
                    ),
                    reverse=True,
                )
            ]
    return ranked


def report_per_topic_by_method(v24_report: dict[str, Any], method: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["query_id"]): row
        for row in v24_report.get("per_topic", {}).get(method, [])
    }


def topic_context(v24_report: dict[str, Any], query_id: str) -> dict[str, Any]:
    learned = report_per_topic_by_method(v24_report, "learned_hybrid").get(query_id, {})
    hybrid = report_per_topic_by_method(v24_report, "hybrid").get(query_id, {})
    oracle = {
        str(row["query_id"]): row
        for row in v24_report.get("oracle_upper_bounds", {})
        .get("per_topic", {})
        .get("oracle_reading_value", [])
    }.get(query_id, {})
    return {
        "learned_hybrid_reading_value_ndcg": learned.get("reading_value_ndcg"),
        "hybrid_reading_value_ndcg": hybrid.get("reading_value_ndcg"),
        "oracle_reading_value_ndcg": oracle.get("reading_value_ndcg"),
        "learned_hybrid_unjudged_at_10": learned.get("unjudged_retrieved_at_k"),
        "learned_hybrid_judged_at_10": learned.get("judged_retrieved_at_k"),
        "packet_candidate_coverage": learned.get("packet_candidate_coverage"),
    }


def candidate_export_row(
    *,
    row: dict[str, Any],
    bucket: str,
    why_selected: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "v2.5_hard_negative_packet",
        "query_id": row["query_id"],
        "query": row["query"],
        "paper_id": int(row["paper_id"]),
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "year": row.get("year"),
        "authors": row.get("authors") or [],
        "venue": row.get("venue"),
        "citation_count": row.get("citation_count"),
        "sources_provenance": row.get("sources_provenance") or [],
        "identifiers": row.get("identifiers") or {},
        "source_url": row.get("source_url"),
        "pdf_url": row.get("pdf_url"),
        "evidence_availability": row.get("evidence_availability"),
        "duplicate_title_cluster": row.get("duplicate_title_cluster"),
        "retrieval_ranks_by_method": {
            method: int(row["retrieval_ranks_by_method"][method])
            for method in METHODS
            if method in row.get("retrieval_ranks_by_method", {})
        },
        "retrieval_scores_by_method": {
            method: float(row["retrieval_scores_by_method"][method])
            for method in METHODS
            if method in row.get("retrieval_scores_by_method", {})
        },
        "selection_bucket": bucket,
        "why_selected": why_selected,
        "v2_4_topic_context": context,
        "already_labeled_v2_1": False,
        "label_fields_intentionally_blank": True,
    }


def is_unlabeled(query_id: str, paper_id: int, labels_by_q: dict[str, dict[int, dict[str, Any]]]) -> bool:
    return paper_id not in labels_by_q.get(query_id, {})


def best_dense_rank(row: dict[str, Any]) -> int:
    ranks = row["retrieval_ranks_by_method"]
    return min(int(ranks.get("embedding", 999999)), int(ranks.get("faiss_embedding", 999999)))


def best_production_rank(row: dict[str, Any]) -> int:
    ranks = row["retrieval_ranks_by_method"]
    return min(int(ranks.get("bm25", 999999)), int(ranks.get("tfidf", 999999)), int(ranks.get("hybrid", 999999)))


def select_for_query(
    *,
    query_id: str,
    rows: list[dict[str, Any]],
    ranked: dict[str, list[int]],
    labels_by_q: dict[str, dict[int, dict[str, Any]]],
    v24_context: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {int(row["paper_id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def add_candidate(paper_id: int, bucket: str, why: str) -> bool:
        if paper_id in selected_ids:
            return False
        if not is_unlabeled(query_id, paper_id, labels_by_q):
            return False
        row = by_id.get(paper_id)
        if not row:
            return False
        selected.append(candidate_export_row(row=row, bucket=bucket, why_selected=why, context=v24_context))
        selected_ids.add(paper_id)
        return True

    learned_added = 0
    learned_ranked = ranked.get("learned_hybrid", [])
    for paper_id in learned_ranked:
        if learned_added >= 3:
            break
        row = by_id[paper_id]
        learned_rank = int(row["retrieval_ranks_by_method"]["learned_hybrid"])
        prefer_note = "top-10 learned_hybrid promotion" if learned_rank <= 10 else "learned_hybrid fallback promotion"
        if add_candidate(
            paper_id,
            "learned_promoted_unjudged",
            (
                f"{prefer_note}: learned_hybrid rank {learned_rank}; "
                f"unjudged in V2.1 for this query."
            ),
        ):
            learned_added += 1

    dense_added = 0
    dense_candidates = sorted(
        (
            row
            for row in rows
            if int(row["paper_id"]) not in selected_ids
            and is_unlabeled(query_id, int(row["paper_id"]), labels_by_q)
        ),
        key=lambda row: (
            best_dense_rank(row),
            int(row["retrieval_ranks_by_method"].get("learned_hybrid", 999999)),
            int(row["paper_id"]),
        ),
    )
    for row in dense_candidates:
        if dense_added >= 2:
            break
        paper_id = int(row["paper_id"])
        dense_rank = best_dense_rank(row)
        if add_candidate(
            paper_id,
            "dense_promoted_unjudged",
            (
                f"dense retrieval promotion: best embedding/FAISS rank {dense_rank}; "
                f"unjudged in V2.1 and not already selected."
            ),
        ):
            dense_added += 1

    disagreement_candidates = []
    for row in rows:
        paper_id = int(row["paper_id"])
        if paper_id in selected_ids or not is_unlabeled(query_id, paper_id, labels_by_q):
            continue
        learned_rank = int(row["retrieval_ranks_by_method"].get("learned_hybrid", 999999))
        production_rank = best_production_rank(row)
        gap = abs(learned_rank - production_rank)
        top_side = 1 if learned_rank <= 10 or production_rank <= 10 else 0
        cross_side = 1 if (learned_rank <= 10 and production_rank >= 30) or (production_rank <= 10 and learned_rank >= 30) else 0
        disagreement_candidates.append((cross_side, top_side, gap, -min(learned_rank, production_rank), -paper_id, row))
    if disagreement_candidates:
        _, _, gap, _, _, row = max(disagreement_candidates)
        paper_id = int(row["paper_id"])
        learned_rank = int(row["retrieval_ranks_by_method"].get("learned_hybrid", 999999))
        production_rank = best_production_rank(row)
        add_candidate(
            paper_id,
            "ranker_disagreement",
            (
                f"ranker disagreement: learned_hybrid rank {learned_rank}, "
                f"best BM25/TF-IDF/hybrid rank {production_rank}, absolute rank gap {gap}."
            ),
        )

    return selected


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V2.5 Hard-Negative Labeling Packet Export",
        "",
        "## Command",
        "",
        "```powershell",
        summary["command"],
        "```",
        "",
        "## Outputs",
        "",
    ]
    for path in summary["files_created_or_modified"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Candidates exported: {summary['candidate_count']}",
            f"- Ready for labeling: {summary['ready_for_labeling']}",
            f"- No V2.1 label duplicates: {summary['validation']['no_existing_v2_1_labels_duplicated']}",
            f"- Selected 240 hash unchanged: {summary['reproducibility_checks']['selected_240_hash_unchanged']}",
            f"- Labels hash unchanged: {summary['reproducibility_checks']['labels_hash_unchanged']}",
            "",
            "## Candidates By Topic",
            "",
            "| Query | Count |",
            "| --- | ---: |",
        ]
    )
    for query_id, count in summary["candidates_per_topic"].items():
        lines.append(f"| `{query_id}` | {count} |")
    lines.extend(["", "## Candidates By Bucket", "", "| Bucket | Count |", "| --- | ---: |"])
    for bucket, count in summary["candidates_by_selection_bucket"].items():
        lines.append(f"| `{bucket}` | {count} |")
    lines.extend(["", "## Top Learned-Promoted Examples", ""])
    for item in summary["top_learned_promoted_examples"]:
        lines.append(
            f"- `{item['query_id']}` paper `{item['paper_id']}`: "
            f"learned rank {item['learned_hybrid_rank']}, title: {item['title']}"
        )
    lines.extend(["", "## Validation", ""])
    for name, value in summary["validation"].items():
        lines.append(f"- {name}: {value}")
    lines.append("")
    return "\n".join(lines)


def validate_rows(
    *,
    rows: list[dict[str, Any]],
    labels_by_q: dict[str, dict[int, dict[str, Any]]],
    valid_query_ids: set[str],
    selected_before: str,
    selected_after: str,
    labels_before: str,
    labels_after: str,
) -> dict[str, Any]:
    seen: set[tuple[str, int]] = set()
    duplicate_pairs: list[dict[str, Any]] = []
    missing_required: list[dict[str, Any]] = []
    existing_label_duplicates: list[dict[str, Any]] = []
    invalid_query_ids: set[str] = set()
    rows_with_label_fields: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row.get("query_id"))
        paper_id = int(row.get("paper_id"))
        pair = (query_id, paper_id)
        if pair in seen:
            duplicate_pairs.append({"query_id": query_id, "paper_id": paper_id})
        seen.add(pair)
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            missing_required.append({"query_id": query_id, "paper_id": paper_id, "missing": missing})
        if paper_id in labels_by_q.get(query_id, {}):
            existing_label_duplicates.append({"query_id": query_id, "paper_id": paper_id})
        if query_id not in valid_query_ids:
            invalid_query_ids.add(query_id)
        label_fields_present = sorted(field for field in LABEL_FIELDS if field in row)
        if label_fields_present:
            rows_with_label_fields.append(
                {"query_id": query_id, "paper_id": paper_id, "label_fields_present": label_fields_present}
            )
    target_count_ok = 80 <= len(rows) <= 112
    return {
        "every_row_has_required_fields": not missing_required,
        "missing_required_field_examples": missing_required[:10],
        "no_duplicate_query_paper_pairs": not duplicate_pairs,
        "duplicate_query_paper_pair_examples": duplicate_pairs[:10],
        "no_existing_v2_1_labels_duplicated": not existing_label_duplicates,
        "existing_label_duplicate_examples": existing_label_duplicates[:10],
        "all_query_ids_valid_v2_topics": not invalid_query_ids,
        "invalid_query_ids": sorted(invalid_query_ids),
        "target_count_around_96": target_count_ok,
        "target_count_rule": "80 <= count <= 112",
        "selected_240_file_unchanged": selected_before == selected_after == EXPECTED_SELECTED_240_SHA256,
        "labels_file_unchanged": labels_before == labels_after == EXPECTED_LABELS_SHA256,
        "no_label_fields_created": not rows_with_label_fields,
        "rows_with_label_fields_examples": rows_with_label_fields[:10],
    }


def build_packet(
    *,
    labels_path: Path,
    packet_path: Path,
    v24_path: Path,
    model_path: Path,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = load_jsonl(labels_path)
    packet_rows = load_jsonl(packet_path)
    v24_report = load_json(v24_path)
    labels_lookup = labels_by_query(labels)
    packet_lookup = packet_by_query(packet_rows)
    db_assets = load_db_assets()
    vector_assets = load_vector_assets(db_assets["settings"], db_assets["paper_by_id"])
    documents = build_documents(db_assets["papers"])
    runs = build_retrieval_runs(
        packet_lookup=packet_lookup,
        paper_by_id=db_assets["paper_by_id"],
        documents=documents,
        vector_assets=vector_assets,
        top_k=top_k,
    )
    candidate_rows_by_q = build_candidate_rows(
        runs=runs,
        paper_by_id=db_assets["paper_by_id"],
        identifiers_by_id=db_assets["identifiers_by_id"],
        sources_by_id=db_assets["sources_by_id"],
        source_urls_by_id=db_assets["source_urls_by_id"],
    )
    ranked = add_learned_scores(candidate_rows_by_q, model_path)
    exported: list[dict[str, Any]] = []
    for query_id in sorted(packet_lookup):
        exported.extend(
            select_for_query(
                query_id=query_id,
                rows=candidate_rows_by_q[query_id],
                ranked=ranked[query_id],
                labels_by_q=labels_lookup,
                v24_context=topic_context(v24_report, query_id),
            )
        )
    diagnostics = {
        "db_diagnostics": db_assets["db_diagnostics"],
        "vector_diagnostics": vector_assets["diagnostics"],
        "valid_query_ids": sorted(packet_lookup),
        "candidate_pool_sizes": {
            query_id: len(rows)
            for query_id, rows in sorted(candidate_rows_by_q.items())
        },
    }
    return exported, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export V2.5 hard-negative labeling packet candidates.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--v24-report", default=str(DEFAULT_V24.relative_to(REPO_ROOT)))
    parser.add_argument("--model", default=str(DEFAULT_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--jsonl-out", default=str(DEFAULT_JSONL_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON.relative_to(REPO_ROOT)))
    parser.add_argument("--summary-md", default=str(DEFAULT_SUMMARY_MD.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = resolve_repo_path(args.labels)
    packet_path = resolve_repo_path(args.packet)
    v24_path = resolve_repo_path(args.v24_report)
    model_path = resolve_repo_path(args.model)
    jsonl_out = resolve_repo_path(args.jsonl_out)
    summary_json = resolve_repo_path(args.summary_json)
    summary_md = resolve_repo_path(args.summary_md)
    labels_hash_before = file_sha256(labels_path)
    selected_hash_before = file_sha256(SELECTED_240)
    rows, diagnostics = build_packet(
        labels_path=labels_path,
        packet_path=packet_path,
        v24_path=v24_path,
        model_path=model_path,
        top_k=args.top_k,
    )
    write_jsonl(jsonl_out, rows)
    labels_hash_after = file_sha256(labels_path)
    selected_hash_after = file_sha256(SELECTED_240)
    labels_lookup = labels_by_query(load_jsonl(labels_path))
    validation = validate_rows(
        rows=rows,
        labels_by_q=labels_lookup,
        valid_query_ids=set(diagnostics["valid_query_ids"]),
        selected_before=selected_hash_before,
        selected_after=selected_hash_after,
        labels_before=labels_hash_before,
        labels_after=labels_hash_after,
    )
    command = (
        f".\\backend\\.venv\\Scripts\\python.exe scripts\\export_v2_5_hard_negative_labeling_packet.py "
        f"--labels {args.labels} --packet {args.packet} --v24-report {args.v24_report} "
        f"--model {args.model} --jsonl-out {args.jsonl_out} --summary-json {args.summary_json} "
        f"--summary-md {args.summary_md} --top-k {args.top_k} --eval-k {args.eval_k}"
    )
    per_topic = Counter(str(row["query_id"]) for row in rows)
    by_bucket = Counter(str(row["selection_bucket"]) for row in rows)
    learned_examples = [
        {
            "query_id": row["query_id"],
            "paper_id": row["paper_id"],
            "title": row["title"],
            "learned_hybrid_rank": row["retrieval_ranks_by_method"].get("learned_hybrid"),
            "why_selected": row["why_selected"],
        }
        for row in rows
        if row["selection_bucket"] == "learned_promoted_unjudged"
    ][:12]
    ready = all(
        bool(value)
        for key, value in validation.items()
        if not key.endswith("_examples") and key not in {"invalid_query_ids", "target_count_rule"}
    )
    summary = {
        "schema_version": "v2.5_hard_negative_packet_summary",
        "created_at": datetime.now(UTC).isoformat(),
        "command": command,
        "files_created_or_modified": [
            "scripts/export_v2_5_hard_negative_labeling_packet.py",
            args.jsonl_out,
            args.summary_json,
            args.summary_md,
        ],
        "candidate_count": len(rows),
        "candidates_per_topic": dict(sorted(per_topic.items())),
        "candidates_by_selection_bucket": dict(sorted(by_bucket.items())),
        "top_learned_promoted_examples": learned_examples,
        "reproducibility_checks": {
            "labels_sha256_before": labels_hash_before,
            "labels_sha256_after": labels_hash_after,
            "labels_hash_unchanged": labels_hash_before == labels_hash_after == EXPECTED_LABELS_SHA256,
            "selected_240_sha256_before": selected_hash_before,
            "selected_240_sha256_after": selected_hash_after,
            "selected_240_hash_unchanged": selected_hash_before == selected_hash_after == EXPECTED_SELECTED_240_SHA256,
        },
        "validation": validation,
        "diagnostics": diagnostics,
        "ready_for_labeling": ready,
        "label_creation_status": "no labels created; label fields intentionally omitted",
    }
    write_json(summary_json, summary)
    write_text(summary_md, build_summary_markdown(summary))
    print("V2.5 hard-negative labeling packet export complete")
    print(f"Candidates exported: {len(rows)}")
    print(f"Packet JSONL: {jsonl_out}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    print(f"Ready for labeling: {ready}")


if __name__ == "__main__":
    main()
