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

from evaluate_v2_4_true_production_candidate_pool import (  # noqa: E402
    EXPECTED_SELECTED_240_SHA256,
    SELECTED_240,
    file_sha256,
    load_json,
    load_jsonl,
    resolve_repo_path,
    write_json,
    write_text,
)
from evaluate_v2_7_score_blends import EXPECTED_V25_LABELS_SHA256  # noqa: E402
from train_v2_6_production_aware_learned_hybrid import EXPECTED_V21_LABELS_SHA256  # noqa: E402


DEFAULT_V34_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_production_pool_report.json"
DEFAULT_V34_EXAMPLES = REPO_ROOT / "data" / "eval" / "results" / "v3_4_v3_3_error_examples.jsonl"
DEFAULT_V32_CANDIDATES = REPO_ROOT / "data" / "eval" / "labeling" / "v3_2_800_active_learning_candidates.jsonl"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
DEFAULT_V32_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
DEFAULT_V33_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v3_3_ltr_dataset_800.jsonl"
DEFAULT_CANDIDATES_OUT = REPO_ROOT / "data" / "eval" / "labeling" / "v3_5_unjudged_top10_audit_candidates.jsonl"
DEFAULT_PACKET_MD_OUT = REPO_ROOT / "data" / "eval" / "labeling" / "v3_5_unjudged_top10_audit_packet.md"
DEFAULT_REPORT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_5_unjudged_top10_audit_packet_report.json"
DEFAULT_REPORT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_5_unjudged_top10_audit_packet_report.md"
DEFAULT_LABELS_OUT = REPO_ROOT / "data" / "eval" / "manual_labels_v3_5_unjudged_audit.jsonl"

EXPECTED_V32_LABELS_SHA256 = "84AFFD9480FB28758A258B85D89AE7A19C8C9620708AAD9050C72D6D34DD0A24"
WEAK_TOPICS = {
    "v2_large_language_model_agents",
    "v2_ai_for_scientific_discovery",
    "v2_causal_representation_learning",
    "v2_transformer_architecture",
    "v2_retrieval_augmented_generation",
    "v2_llm_evaluation",
}
REASON_PRIORITY = {
    "v3_3_high_scoring_unjudged_top10": 0,
    "v3_3_promoted_not_v2_7": 1,
    "weak_topic": 2,
    "v2_7_promoted_not_v3_3": 3,
}
SCORE_METHODS = [
    "v3_3_random_forest_no_v27_compact",
    "v2_7",
    "v2_6",
    "hybrid",
    "old_v2_2b",
    "embedding",
    "faiss_embedding",
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["query_id"]), int(row["paper_id"]))


def load_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["query_id"]), int(row["paper_id"])) for row in rows}


def title_is_valid(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) >= 5


def abstract_is_valid(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip().split()) >= 30


def top10_rank_map(v34_report: dict[str, Any], query_id: str, paper_id: int) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    per_topic = v34_report.get("evaluation", {}).get("per_topic", {})
    for method in SCORE_METHODS:
        rows = per_topic.get(method) or []
        topic_row = next((row for row in rows if row.get("query_id") == query_id), None)
        rank = None
        if topic_row:
            retrieved = [int(value) for value in topic_row.get("retrieved_ids_at_k", [])]
            if paper_id in retrieved:
                rank = retrieved.index(paper_id) + 1
        ranks[method] = rank
    return ranks


def selection_reasons(example: dict[str, Any]) -> list[str]:
    reasons = [str(reason) for reason in example.get("reasons", [])]
    if example.get("query_id") in WEAK_TOPICS:
        reasons.append("weak_topic")
    ordered = []
    for reason in sorted(set(reasons), key=lambda value: REASON_PRIORITY.get(value, 99)):
        if reason in REASON_PRIORITY:
            ordered.append(reason)
    return ordered


def priority_tuple(row: dict[str, Any]) -> tuple[int, int, str, int]:
    reasons = row.get("diagnostic_reasons") or []
    priority = min(REASON_PRIORITY.get(reason, 99) for reason in reasons)
    rank = row.get("ranks", {}).get("v3_3_random_forest_no_v27_compact")
    rank_value = int(rank) if isinstance(rank, int) else 99
    return (priority, rank_value, str(row["query_id"]), int(row["paper_id"]))


def build_candidate(
    *,
    example: dict[str, Any],
    v34_report: dict[str, Any],
    v32_candidate_lookup: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    query_id = str(example["query_id"])
    paper_id = int(example["paper_id"])
    v32_candidate = v32_candidate_lookup.get((query_id, paper_id), {})
    ranks = top10_rank_map(v34_report, query_id, paper_id)
    observations = example.get("observations") or []
    for observation in observations:
        method = str(observation.get("method"))
        if method in ranks and ranks[method] is None:
            ranks[method] = int(observation["rank"])
    scores = {method: example.get("scores", {}).get(method) for method in SCORE_METHODS}
    reasons = selection_reasons(example)
    return {
        "schema_version": "v3.5_unjudged_top10_audit_candidate",
        "query_id": query_id,
        "query": v32_candidate.get("query") or query_id.replace("v2_", "").replace("_", " "),
        "topic": v32_candidate.get("topic") or query_id,
        "paper_id": paper_id,
        "title": example.get("title"),
        "year": v32_candidate.get("year"),
        "venue": v32_candidate.get("venue"),
        "abstract": example.get("abstract"),
        "source_url": v32_candidate.get("source_url"),
        "pdf_url": v32_candidate.get("pdf_url"),
        "diagnostic_reasons": reasons,
        "primary_diagnostic_reason": reasons[0] if reasons else "unknown",
        "observations": observations,
        "scores": scores,
        "ranks": ranks,
        "v3_3_rank": ranks.get("v3_3_random_forest_no_v27_compact"),
        "v3_3_score": scores.get("v3_3_random_forest_no_v27_compact"),
        "v2_7_rank": ranks.get("v2_7"),
        "v2_7_score": scores.get("v2_7"),
        "v2_6_rank": ranks.get("v2_6"),
        "v2_6_score": scores.get("v2_6"),
        "hybrid_rank": ranks.get("hybrid"),
        "hybrid_score": scores.get("hybrid"),
        "source_methods": v32_candidate.get("source_methods") or [],
        "retrieval_ranks": v32_candidate.get("retrieval_ranks") or {},
        "candidate_pool_metadata_available": bool(v32_candidate),
        "labeling_instruction": "Manual label only. Do not infer labels from diagnostic reasons.",
    }


def protected_hashes(v21_labels_path: Path, v25_labels_path: Path, v32_labels_path: Path) -> dict[str, Any]:
    v21_hash = file_sha256(v21_labels_path)
    v25_hash = file_sha256(v25_labels_path)
    v32_hash = file_sha256(v32_labels_path)
    selected_hash = file_sha256(SELECTED_240)
    return {
        "v2_1_labels_sha256": v21_hash,
        "v2_1_labels_hash_unchanged": v21_hash == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": v25_hash,
        "v2_5_labels_hash_unchanged": v25_hash == EXPECTED_V25_LABELS_SHA256,
        "v3_2_labels_sha256": v32_hash,
        "v3_2_labels_hash_unchanged": v32_hash == EXPECTED_V32_LABELS_SHA256,
        "selected_240_sha256": selected_hash,
        "selected_240_hash_unchanged": selected_hash == EXPECTED_SELECTED_240_SHA256,
    }


def build_packet(
    *,
    v34_report_path: Path,
    v34_examples_path: Path,
    v32_candidates_path: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v33_dataset_path: Path,
    labels_out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v34_report = load_json(v34_report_path)
    examples = load_jsonl(v34_examples_path)
    v32_candidate_lookup = {row_key(row): row for row in load_jsonl(v32_candidates_path)}
    existing_keys = (
        load_keys(load_jsonl(v21_labels_path))
        | load_keys(load_jsonl(v25_labels_path))
        | load_keys(load_jsonl(v32_labels_path))
    )
    v33_dataset_keys = load_keys(load_jsonl(v33_dataset_path))
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    for example in examples:
        if example.get("judged"):
            excluded["already_labeled_in_v3_4_example"] += 1
            continue
        key = row_key(example)
        if key in existing_keys or key in v33_dataset_keys:
            excluded["overlap_existing_labels"] += 1
            continue
        if not title_is_valid(example.get("title")):
            excluded["missing_or_short_title"] += 1
            continue
        if not abstract_is_valid(example.get("abstract")):
            excluded["missing_or_short_abstract"] += 1
            continue
        candidate = build_candidate(example=example, v34_report=v34_report, v32_candidate_lookup=v32_candidate_lookup)
        if not candidate["diagnostic_reasons"]:
            excluded["no_selected_diagnostic_reason"] += 1
            continue
        selected[key] = candidate

    rows = sorted(selected.values(), key=priority_tuple)
    if labels_out.exists() and labels_out.stat().st_size > 0:
        raise ValueError(f"Refusing to overwrite non-empty V3.5 label file: {labels_out}")
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.touch(exist_ok=True)
    labelable = all(title_is_valid(row.get("title")) and abstract_is_valid(row.get("abstract")) for row in rows)
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row["diagnostic_reasons"])
    report = {
        "schema_version": "v3.5_unjudged_top10_audit_packet_report",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_changed_by_this_script": False,
        "candidate_count": len(rows),
        "target_minimum_high_scoring_unjudged_top10": 27,
        "target_preferred_range": [80, 120],
        "target_maximum": 160,
        "available_unjudged_diagnostic_examples": len(rows),
        "count_by_topic": dict(sorted(Counter(row["query_id"] for row in rows).items())),
        "count_by_primary_reason": dict(sorted(Counter(row["primary_diagnostic_reason"] for row in rows).items())),
        "count_by_diagnostic_reason": dict(sorted(reason_counts.items())),
        "v3_3_high_scoring_unjudged_top10_count": reason_counts.get("v3_3_high_scoring_unjudged_top10", 0),
        "v3_3_promoted_not_v2_7_count": reason_counts.get("v3_3_promoted_not_v2_7", 0),
        "v2_7_promoted_not_v3_3_count": reason_counts.get("v2_7_promoted_not_v3_3", 0),
        "weak_topic_count": reason_counts.get("weak_topic", 0),
        "weak_topics_covered": sorted({row["query_id"] for row in rows if row["query_id"] in WEAK_TOPICS}),
        "duplicate_query_paper_rows": [
            {"query_id": query_id, "paper_id": paper_id, "count": count}
            for (query_id, paper_id), count in Counter(row_key(row) for row in rows).items()
            if count > 1
        ],
        "labelable_candidates": labelable,
        "excluded_candidate_counts": dict(sorted(excluded.items())),
        "all_27_v3_3_high_scoring_unjudged_top10_included": reason_counts.get("v3_3_high_scoring_unjudged_top10", 0) == 27,
        "note": "V3.4 diagnostics contained only 31 unique unjudged labelable examples; packet is intentionally below the preferred 80-120 size.",
        "protected_hashes": protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path),
    }
    return rows, report


def build_packet_markdown(rows: list[dict[str, Any]], report: dict[str, Any]) -> str:
    lines = [
        "# V3.5 Unjudged Top-10 Audit Packet",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "Manual labeling packet for unjudged production-pool papers promoted by V3.3/V2.7 diagnostics. Do not infer labels from diagnostic reasons.",
        "",
        f"- Candidates: `{report['candidate_count']}`",
        f"- Labelable candidates: `{report['labelable_candidates']}`",
        f"- V3.3 high-scoring unjudged top-10 included: `{report['v3_3_high_scoring_unjudged_top10_count']}` / `27`",
        f"- Weak topics covered: `{report['weak_topics_covered']}`",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {index}. {row['query_id']} / {row['paper_id']}",
                "",
                f"**Query:** {row['query']}",
                "",
                f"**Title:** {row['title']}",
                "",
                f"**Year/Venue:** {row.get('year') or ''} / {row.get('venue') or ''}",
                "",
                f"**Reasons:** {', '.join(row['diagnostic_reasons'])}",
                "",
                f"**Ranks:** V3.3 `{row.get('v3_3_rank')}`, V2.7 `{row.get('v2_7_rank')}`, V2.6 `{row.get('v2_6_rank')}`, hybrid `{row.get('hybrid_rank')}`",
                "",
                f"**Scores:** V3.3 `{row.get('v3_3_score')}`, V2.7 `{row.get('v2_7_score')}`, V2.6 `{row.get('v2_6_score')}`, hybrid `{row.get('hybrid_score')}`",
                "",
                "**Abstract:**",
                "",
                str(row.get("abstract") or ""),
                "",
                "**Manual label fields:**",
                "",
                "- topic_match_score:",
                "- reading_value_score:",
                "- beginner_fit_score:",
                "- intermediate_fit_score:",
                "- advanced_fit_score:",
                "- expert_fit_score:",
                "- intent_scores:",
                "- primary_role:",
                "- secondary_roles:",
                "- duplicate_status:",
                "- duplicate_of_paper_id:",
                "- evidence_level:",
                "- full_text_available:",
                "- label_confidence:",
                "- notes:",
                "",
            ]
        )
    return "\n".join(lines)


def build_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3.5 Unjudged Top-10 Audit Packet Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Total candidates exported: `{report['candidate_count']}`",
        f"- Labelable candidates: `{report['labelable_candidates']}`",
        f"- V3.3 high-scoring unjudged top-10 included: `{report['v3_3_high_scoring_unjudged_top10_count']}` / `27`",
        f"- V3.3 promoted-not-V2.7 count: `{report['v3_3_promoted_not_v2_7_count']}`",
        f"- V2.7 promoted-not-V3.3 count: `{report['v2_7_promoted_not_v3_3_count']}`",
        f"- Weak-topic count: `{report['weak_topic_count']}`",
        f"- Duplicate query-paper rows: `{len(report['duplicate_query_paper_rows'])}`",
        "",
        "## Counts by Topic",
        "",
        "| topic | candidates |",
        "|---|---:|",
    ]
    for topic, count in report["count_by_topic"].items():
        lines.append(f"| `{topic}` | {count} |")
    lines.extend(
        [
            "",
            "## Counts by Diagnostic Reason",
            "",
            "| reason | candidates |",
            "|---|---:|",
        ]
    )
    for reason, count in report["count_by_diagnostic_reason"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            f"`{report['excluded_candidate_counts']}`",
            "",
            "## Protected Hashes",
            "",
            f"- V2.1 labels unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
            f"- V3.2 labels unchanged: `{report['protected_hashes']['v3_2_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V3.5 unjudged top-10 audit packet.")
    parser.add_argument("--v34-report", default=str(DEFAULT_V34_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--v34-examples", default=str(DEFAULT_V34_EXAMPLES.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-candidates", default=str(DEFAULT_V32_CANDIDATES.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v33-dataset", default=str(DEFAULT_V33_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--candidates-out", default=str(DEFAULT_CANDIDATES_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--packet-md-out", default=str(DEFAULT_PACKET_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--report-json-out", default=str(DEFAULT_REPORT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--report-md-out", default=str(DEFAULT_REPORT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--labels-out", default=str(DEFAULT_LABELS_OUT.relative_to(REPO_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report = build_packet(
        v34_report_path=resolve_repo_path(args.v34_report),
        v34_examples_path=resolve_repo_path(args.v34_examples),
        v32_candidates_path=resolve_repo_path(args.v32_candidates),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        v33_dataset_path=resolve_repo_path(args.v33_dataset),
        labels_out=resolve_repo_path(args.labels_out),
    )
    write_jsonl(resolve_repo_path(args.candidates_out), rows)
    write_text(resolve_repo_path(args.packet_md_out), build_packet_markdown(rows, report))
    write_json(resolve_repo_path(args.report_json_out), report)
    write_text(resolve_repo_path(args.report_md_out), build_report_markdown(report))
    print("V3.5 unjudged top-10 audit packet built")
    print(f"Candidates: {report['candidate_count']}")
    print(f"V3.3 high-scoring unjudged top-10 included: {report['v3_3_high_scoring_unjudged_top10_count']} / 27")
    print(f"Labelable: {report['labelable_candidates']}")
    print(f"Candidate file: {resolve_repo_path(args.candidates_out)}")
    print(f"Report: {resolve_repo_path(args.report_md_out)}")


if __name__ == "__main__":
    main()
