import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from analyze_v3_0_ltr_errors import (  # noqa: E402
    SELECTED_V27_METHOD,
    active_learning_packet_row,
    add_candidate_ranks,
    add_model_scores,
    build_unjudged_candidate_rows,
    load_model_bundle,
    normalize,
    source_methods,
    topic_priority_weights,
    write_jsonl,
)
from audit_v3_1b_feature_packet import (  # noqa: E402
    multicollinearity_audit,
)
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
from evaluate_v2_9_score_columns import duplicate_rows  # noqa: E402
from train_v2_6_production_aware_learned_hybrid import EXPECTED_V21_LABELS_SHA256  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v2_9_ltr_dataset.jsonl"
DEFAULT_V21_PACKET = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_V30_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v3_0_ltr_training_report.json"
DEFAULT_V31_PACKET = REPO_ROOT / "data" / "eval" / "labeling" / "v3_1_active_learning_candidates.jsonl"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "processed" / "models" / "v3_0_ltr"
DEFAULT_OLD_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_V26_MODEL = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"
DEFAULT_PACKET_OUT = REPO_ROOT / "data" / "eval" / "labeling" / "v3_2_800_active_learning_candidates.jsonl"
DEFAULT_TOPIC_PACKET_DIR = REPO_ROOT / "data" / "eval" / "labeling" / "v3_2_800_packets"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_2_800_active_learning_report.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_2_800_active_learning_report.md"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"

TARGET_TOTAL = 464
TARGET_PER_TOPIC = 29
BAD_CANDIDATE = ("v2_ai_for_scientific_discovery", 5115)
MIN_TITLE_CHARS = 8
MIN_ABSTRACT_CHARS = 120
MODEL_COLUMNS = {
    "hybrid": "hybrid_score",
    "v2_6_score": "v2_6_score",
    "v2_7_score": "v2_7_score",
    "ridge_no_v27": "v3_0_ridge_no_v27_score",
    "pairwise_logistic_no_v27": "v3_0_pairwise_logistic_no_v27_score",
}
REQUIRED_SCORE_FIELDS = [
    "bm25",
    "tfidf",
    "embedding",
    "faiss_embedding",
    "hybrid",
    "old_v2_2b",
    "v2_6",
    "v2_7",
    "ridge_no_v27",
    "pairwise_logistic_no_v27",
]
REQUIRED_RANK_FIELDS = ["hybrid", "v2_6_score", "v2_7_score", "ridge_no_v27", "pairwise_logistic_no_v27"]
RETRIEVERS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]


def as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def text_len(value: Any) -> int:
    return len(str(value or "").strip())


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "topic"


def protected_hashes(v21_labels: Path, v25_labels: Path) -> dict[str, Any]:
    v21_hash = file_sha256(v21_labels)
    v25_hash = file_sha256(v25_labels)
    selected_hash = file_sha256(SELECTED_240)
    return {
        "v2_1_labels_sha256": v21_hash,
        "v2_1_labels_hash_unchanged": v21_hash == EXPECTED_V21_LABELS_SHA256,
        "v2_5_labels_sha256": v25_hash,
        "v2_5_labels_hash_unchanged": v25_hash == EXPECTED_V25_LABELS_SHA256,
        "selected_240_sha256": selected_hash,
        "selected_240_hash_unchanged": selected_hash == EXPECTED_SELECTED_240_SHA256,
    }


def valid_text(row: dict[str, Any]) -> bool:
    return text_len(row.get("title")) >= MIN_TITLE_CHARS and text_len(row.get("abstract")) >= MIN_ABSTRACT_CHARS


def load_valid_v31_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    for row in load_jsonl(path):
        key = (str(row["query_id"]), int(row["paper_id"]))
        if key == BAD_CANDIDATE:
            continue
        if valid_text(row):
            keys.add(key)
    return keys


def enrich_candidates(
    rows: list[dict[str, Any]],
    *,
    v30_report: dict[str, Any],
    v31_valid_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    ranks_by_key = add_candidate_ranks(rows, MODEL_COLUMNS)
    normalized = {
        "ridge_no_v27": normalize([as_float(row.get("v3_0_ridge_no_v27_score")) for row in rows]),
        "pairwise_logistic_no_v27": normalize([as_float(row.get("v3_0_pairwise_logistic_no_v27_score")) for row in rows]),
        "v2_7_score": normalize([as_float(row.get("v2_7_score")) for row in rows]),
        "v2_6_score": normalize([as_float(row.get("v2_6_score")) for row in rows]),
        "hybrid": normalize([as_float(row.get("hybrid_score")) for row in rows]),
    }
    priority = topic_priority_weights(v30_report)
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        key = (str(row["query_id"]), int(row["paper_id"]))
        if key == BAD_CANDIDATE:
            continue
        if not valid_text(row):
            continue
        row_ranks = ranks_by_key[key]
        score_values = [values[index] for values in normalized.values()]
        score_disagreement = max(score_values) - min(score_values)
        rank_disagreement = max(row_ranks.values()) - min(row_ranks.values())
        top10_boundary = any(8 <= rank <= 12 for rank in row_ranks.values())
        top20_boundary = any(17 <= rank <= 23 for rank in row_ranks.values())
        v31_carryover = key in v31_valid_keys
        promoted_methods = [method for method, rank in row_ranks.items() if rank <= 20]
        rejected_methods = [method for method, rank in row_ranks.items() if rank > 50]
        suspected_hard_negative = min(row_ranks["ridge_no_v27"], row_ranks["pairwise_logistic_no_v27"]) <= 10 and (
            row_ranks["v2_7_score"] > 30 or row_ranks["hybrid"] > 50
        )
        suspected_hidden_positive = (
            min(row_ranks["v2_7_score"], row_ranks["v2_6_score"], row_ranks["hybrid"]) <= 15
            and min(row_ranks["ridge_no_v27"], row_ranks["pairwise_logistic_no_v27"]) > 25
        )
        reasons = []
        if v31_carryover:
            reasons.append("valid_v3_1_active_learning_candidate")
        if min(row_ranks["ridge_no_v27"], row_ranks["pairwise_logistic_no_v27"]) <= 20:
            reasons.append("high_scoring_unjudged_by_v3_model")
        if score_disagreement >= 0.30 or rank_disagreement >= 25:
            reasons.append("high_disagreement_between_v3_v2_7_v2_6_hybrid")
        if top10_boundary:
            reasons.append("top10_boundary_candidate")
        if top20_boundary:
            reasons.append("top20_boundary_candidate")
        if suspected_hard_negative:
            reasons.append("suspected_hard_negative")
        if suspected_hidden_positive:
            reasons.append("suspected_hidden_positive")
        if promoted_methods and rejected_methods:
            reasons.append("promoted_by_one_method_rejected_by_others")
        if priority.get(str(row["query_id"]), 1.0) > 1.0:
            reasons.append("v3_0_unstable_or_loss_topic")
        if not reasons:
            reasons.append("balanced_production_pool_coverage")
        selection_score = (
            0.45 * (1.0 if v31_carryover else 0.0)
            + 0.16 * normalized["ridge_no_v27"][index]
            + 0.16 * normalized["pairwise_logistic_no_v27"][index]
            + 0.14 * score_disagreement
            + 0.12 * min(rank_disagreement / 75.0, 1.0)
            + 0.05 * (1.0 if top10_boundary else 0.0)
            + 0.03 * (1.0 if top20_boundary else 0.0)
            + 0.04 * min(priority.get(str(row["query_id"]), 1.0) - 1.0, 1.0)
        )
        enriched.append(
            {
                **row,
                "schema_version": "v3.2_800_active_learning_candidate",
                "method_ranks": row_ranks,
                "score_disagreement": score_disagreement,
                "rank_disagreement": rank_disagreement,
                "selection_score": selection_score,
                "selection_reasons": reasons,
                "v3_1_carryover": v31_carryover,
                "already_labeled": False,
                "diagnostic_only": False,
            }
        )
    return enriched


def packet_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "v3.2_800_active_learning_candidate",
        "query_id": row["query_id"],
        "query": row.get("query"),
        "topic": row.get("topic") or row.get("query"),
        "paper_id": int(row["paper_id"]),
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "year": row.get("year"),
        "authors": row.get("authors") or [],
        "venue": row.get("venue"),
        "citation_count": row.get("citation_count"),
        "identifiers": row.get("identifiers") or {},
        "source_url": row.get("source_url"),
        "pdf_url": row.get("pdf_url"),
        "sources_provenance": row.get("sources_provenance") or [],
        "source_methods": source_methods(row),
        "already_labeled": False,
        "diagnostic_only": False,
        "v3_1_carryover": bool(row.get("v3_1_carryover")),
        "selection_score": row["selection_score"],
        "why_selected": row["selection_reasons"],
        "scores": {
            "bm25": row.get("bm25_score"),
            "tfidf": row.get("tfidf_score"),
            "embedding": row.get("embedding_score"),
            "faiss_embedding": row.get("faiss_embedding_score"),
            "hybrid": row.get("hybrid_score"),
            "old_v2_2b": row.get("old_v2_2b_score"),
            "v2_6": row.get("v2_6_score"),
            "v2_7": row.get("v2_7_score"),
            "ridge_no_v27": row.get("v3_0_ridge_no_v27_score"),
            "pairwise_logistic_no_v27": row.get("v3_0_pairwise_logistic_no_v27_score"),
        },
        "ranks": row["method_ranks"],
        "retrieval_ranks": {
            method: row.get(f"{method}_rank")
            for method in RETRIEVERS
        },
        "score_disagreement": row["score_disagreement"],
        "rank_disagreement": row["rank_disagreement"],
        "labeling_instruction": "Manual label only. Do not infer labels from selection reasons.",
    }


def select_balanced_packet(enriched: list[dict[str, Any]], *, topics: list[str], per_topic: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_topic[str(row["query_id"])].append(row)
    selected: list[dict[str, Any]] = []
    candidate_counts = {}
    shortfalls = {}
    for topic in sorted(topics):
        ranked = sorted(
            by_topic.get(topic, []),
            key=lambda row: (
                row["selection_score"],
                row.get("v3_1_carryover", False),
                -row["method_ranks"]["ridge_no_v27"],
                -int(row["paper_id"]),
            ),
            reverse=True,
        )
        candidate_counts[topic] = len(ranked)
        selected.extend(ranked[:per_topic])
        if len(ranked) < per_topic:
            shortfalls[topic] = {"available": len(ranked), "target": per_topic}
    packet = [packet_row(row) for row in selected]
    return packet, {
        "available_labelable_candidates_by_topic": candidate_counts,
        "shortfalls": shortfalls,
        "per_topic_counts": dict(Counter(row["query_id"] for row in packet)),
        "v3_1_carryover_count": sum(1 for row in packet if row.get("v3_1_carryover")),
    }


def validate_packet(packet: list[dict[str, Any]], *, labeled_keys: set[tuple[str, int]], topics: list[str]) -> dict[str, Any]:
    duplicates = duplicate_rows(packet)
    labeled_rows = [
        {"query_id": row["query_id"], "paper_id": row["paper_id"]}
        for row in packet
        if (str(row["query_id"]), int(row["paper_id"])) in labeled_keys
    ]
    missing_required = []
    missing_scores = []
    missing_ranks = []
    short_titles = []
    short_abstracts = []
    bad_candidate_present = []
    for row in packet:
        key = {"query_id": row["query_id"], "paper_id": row["paper_id"], "title": row.get("title")}
        for field in ["query_id", "query", "topic", "paper_id", "title", "abstract", "scores", "ranks", "retrieval_ranks", "why_selected"]:
            if row.get(field) in (None, "", [], {}):
                missing_required.append({**key, "field": field})
        for field in REQUIRED_SCORE_FIELDS:
            if (row.get("scores") or {}).get(field) is None:
                missing_scores.append({**key, "field": field})
        for field in REQUIRED_RANK_FIELDS:
            if (row.get("ranks") or {}).get(field) is None:
                missing_ranks.append({**key, "field": field})
        if text_len(row.get("title")) < MIN_TITLE_CHARS:
            short_titles.append({**key, "title_length": text_len(row.get("title"))})
        if text_len(row.get("abstract")) < MIN_ABSTRACT_CHARS:
            short_abstracts.append({**key, "abstract_length": text_len(row.get("abstract"))})
        if (str(row["query_id"]), int(row["paper_id"])) == BAD_CANDIDATE:
            bad_candidate_present.append(key)
    counts = dict(Counter(row["query_id"] for row in packet))
    return {
        "target_total": TARGET_TOTAL,
        "target_per_topic": TARGET_PER_TOPIC,
        "actual_total": len(packet),
        "per_topic_counts": counts,
        "topics_represented": sorted(counts),
        "duplicate_query_paper_rows": duplicates,
        "already_labeled_rows": labeled_rows,
        "missing_required_fields": missing_required,
        "missing_score_fields": missing_scores,
        "missing_rank_fields": missing_ranks,
        "short_titles": short_titles,
        "short_abstracts": short_abstracts,
        "known_bad_candidate_present": bad_candidate_present,
        "all_16_topics_represented": set(topics) <= set(counts),
        "all_topics_have_29_candidates": all(counts.get(topic) == TARGET_PER_TOPIC for topic in topics),
        "all_candidates_labelable": (
            len(packet) == TARGET_TOTAL
            and set(topics) <= set(counts)
            and all(counts.get(topic) == TARGET_PER_TOPIC for topic in topics)
            and not duplicates
            and not labeled_rows
            and not missing_required
            and not missing_scores
            and not missing_ranks
            and not short_titles
            and not short_abstracts
            and not bad_candidate_present
        ),
    }


def feature_set_recommendations(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    collinearity = multicollinearity_audit(dataset_rows)
    rank_features = [f"{retriever}_reciprocal_rank" for retriever in RETRIEVERS]
    normalized = [f"{retriever}_score" for retriever in RETRIEVERS]
    compact_base = normalized + rank_features + ["old_v2_2b_score", "v2_6_score", "citation_count_log1p", "full_text_available_bool"]
    return {
        "collinear_raw_rank_normalized_groups": collinearity["groups"],
        "high_correlation_pairs": collinearity["high_correlation_pairs"],
        "drop_raw_score_features": [f"{retriever}_raw_score" for retriever in RETRIEVERS],
        "recommended_variants": {
            "rank_features_only": rank_features + ["old_v2_2b_score", "v2_6_score"],
            "normalized_scores_only": normalized + ["old_v2_2b_score", "v2_6_score"],
            "compact_combined": compact_base,
            "no_v27_compact": compact_base,
            "with_v27_compact": compact_base + ["v2_7_score"],
        },
        "recommendation": (
            "Use compact ablations in V3.3. Drop raw score features first, then compare normalized-only, "
            "rank-only, no-V2.7 compact, and with-V2.7 compact variants under topic-held-out evaluation."
        ),
    }


def markdown_escape(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def format_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else ""


def candidate_block(row: dict[str, Any], index: int) -> list[str]:
    scores = row["scores"]
    ranks = row["ranks"]
    retrieval_ranks = row["retrieval_ranks"]
    return [
        f"## Candidate {index}: {markdown_escape(row.get('title'))}",
        "",
        f"- query_id: `{row['query_id']}`",
        f"- paper_id: `{row['paper_id']}`",
        f"- year: `{row.get('year')}`",
        f"- venue: `{markdown_escape(row.get('venue'))}`",
        f"- V3.1 carryover: `{row.get('v3_1_carryover')}`",
        f"- source methods: `{format_list(row.get('source_methods') or [])}`",
        f"- why_selected: `{format_list(row.get('why_selected') or [])}`",
        f"- scores: ridge `{scores.get('ridge_no_v27')}`, pairwise `{scores.get('pairwise_logistic_no_v27')}`, V2.7 `{scores.get('v2_7')}`, V2.6 `{scores.get('v2_6')}`, hybrid `{scores.get('hybrid')}`",
        f"- ranks: ridge `{ranks.get('ridge_no_v27')}`, pairwise `{ranks.get('pairwise_logistic_no_v27')}`, V2.7 `{ranks.get('v2_7_score')}`, V2.6 `{ranks.get('v2_6_score')}`, hybrid `{ranks.get('hybrid')}`",
        f"- retrieval ranks: BM25 `{retrieval_ranks.get('bm25')}`, TF-IDF `{retrieval_ranks.get('tfidf')}`, embedding `{retrieval_ranks.get('embedding')}`, FAISS `{retrieval_ranks.get('faiss_embedding')}`, hybrid `{retrieval_ranks.get('hybrid')}`",
        f"- source_url: {markdown_escape(row.get('source_url'))}",
        f"- pdf_url: {markdown_escape(row.get('pdf_url'))}",
        "",
        "**Abstract**",
        "",
        markdown_escape(row.get("abstract")),
        "",
        "**Manual Labels**",
        "",
        "- topic_match_score:",
        "- reading_value_score:",
        "- beginner_fit_score:",
        "- intermediate_fit_score:",
        "- advanced_fit_score:",
        "- expert_fit_score:",
        "- primary_role:",
        "- secondary_roles:",
        "- duplicate_status:",
        "- label_confidence:",
        "- notes:",
        "",
    ]


def export_topic_packets(packet: list[dict[str, Any]], packet_dir: Path) -> dict[str, Any]:
    packet_dir.mkdir(parents=True, exist_ok=True)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packet:
        by_topic[str(row["query_id"])].append(row)
    outputs = []
    for topic, rows in sorted(by_topic.items()):
        rows = sorted(rows, key=lambda row: as_float(row["selection_score"]), reverse=True)
        path = packet_dir / f"topic_{slugify(topic)}.md"
        lines = [
            f"# V3.2 800-Row Active-Learning Packet: {topic}",
            "",
            f"Query: {rows[0].get('query') or rows[0].get('topic')}",
            "",
            "Manual labeling packet. Do not infer labels from selection reasons.",
            "",
        ]
        for index, row in enumerate(rows, start=1):
            lines.extend(candidate_block(row, index))
        write_text(path, "\n".join(lines))
        outputs.append({"topic": topic, "path": str(path), "candidate_count": len(rows)})
    return {
        "packet_dir": str(packet_dir),
        "file_count": len(outputs),
        "outputs": outputs,
        "all_topics_have_29_candidates": all(row["candidate_count"] == TARGET_PER_TOPIC for row in outputs),
    }


def build_markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# V3.2 800-Row Active-Learning Packet Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "This report exports one unified active-learning packet intended to expand ResearchPath from 336 judged rows toward 800 judged rows. No labels were created or modified.",
        "",
        "## Packet",
        "",
        f"- Current judged rows: `{report['target']['current_judged_rows']}`",
        f"- Target judged rows: `{report['target']['target_judged_rows']}`",
        f"- New candidates requested: `{report['target']['new_candidates_needed']}`",
        f"- New candidates exported: `{validation['actual_total']}`",
        f"- Topics represented: `{len(validation['topics_represented'])}`",
        f"- Candidates per topic target met: `{validation['all_topics_have_29_candidates']}`",
        f"- All candidates labelable: `{validation['all_candidates_labelable']}`",
        f"- V3.1 carryover candidates: `{report['selection_summary']['v3_1_carryover_count']}`",
        "",
        "| topic | candidates | available labelable pool |",
        "|---|---:|---:|",
    ]
    for topic, count in sorted(validation["per_topic_counts"].items()):
        available = report["selection_summary"]["available_labelable_candidates_by_topic"].get(topic, 0)
        lines.append(f"| `{topic}` | {count} | {available} |")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Duplicate query-paper rows: `{len(validation['duplicate_query_paper_rows'])}`",
            f"- Already-labeled rows: `{len(validation['already_labeled_rows'])}`",
            f"- Missing required fields: `{len(validation['missing_required_fields'])}`",
            f"- Missing score fields: `{len(validation['missing_score_fields'])}`",
            f"- Missing rank fields: `{len(validation['missing_rank_fields'])}`",
            f"- Short titles: `{len(validation['short_titles'])}`",
            f"- Short abstracts: `{len(validation['short_abstracts'])}`",
            f"- Known bad candidate present: `{len(validation['known_bad_candidate_present'])}`",
            "",
            "## Feature-Set Recommendations for V3.3",
            "",
            "- Drop raw score features first: `bm25_raw_score`, `tfidf_raw_score`, `embedding_raw_score`, `faiss_embedding_raw_score`, `hybrid_raw_score`.",
            "- Compare compact variants under topic-held-out evaluation:",
            "  - `rank_features_only`",
            "  - `normalized_scores_only`",
            "  - `compact_combined`",
            "  - `no_v27_compact`",
            "  - `with_v27_compact`",
            f"- High-correlation pairs detected: `{len(report['feature_set_recommendations']['high_correlation_pairs'])}`",
            "",
            "## Protected Hashes",
            "",
            f"- V2.1 labels unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(
    *,
    dataset_path: Path,
    v21_packet_path: Path,
    v30_report_path: Path,
    v31_packet_path: Path,
    model_dir: Path,
    old_model_path: Path,
    v26_model_path: Path,
    topic_packet_dir: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_rows = load_jsonl(dataset_path)
    topics = sorted({str(row["query_id"]) for row in dataset_rows})
    labeled_keys = {(str(row["query_id"]), int(row["paper_id"])) for row in dataset_rows}
    v30_report = load_json(v30_report_path)
    v31_valid_keys = load_valid_v31_keys(v31_packet_path)
    ridge = load_model_bundle(model_dir, "ridge_no_v27")
    pairwise = load_model_bundle(model_dir, "pairwise_logistic_no_v27")
    candidate_rows, candidate_pool_diagnostics = build_unjudged_candidate_rows(
        packet_rows=load_jsonl(v21_packet_path),
        labeled_keys=labeled_keys,
        old_model_path=old_model_path,
        v26_model_path=v26_model_path,
        top_k=top_k,
    )
    scored = add_model_scores(candidate_rows, [ridge, pairwise])
    enriched = enrich_candidates(scored, v30_report=v30_report, v31_valid_keys=v31_valid_keys)
    packet, selection_summary = select_balanced_packet(enriched, topics=topics, per_topic=TARGET_PER_TOPIC)
    validation = validate_packet(packet, labeled_keys=labeled_keys, topics=topics)
    topic_export = export_topic_packets(packet, topic_packet_dir)
    hashes = protected_hashes(v21_labels_path, v25_labels_path)
    feature_recs = feature_set_recommendations(dataset_rows)
    if not all(
        [
            hashes["v2_1_labels_hash_unchanged"],
            hashes["v2_5_labels_hash_unchanged"],
            hashes["selected_240_hash_unchanged"],
        ]
    ):
        raise ValueError("Protected hash check failed.")
    if not validation["all_candidates_labelable"]:
        raise ValueError(f"V3.2 packet failed validation: {validation}")
    report = {
        "schema_version": "v3.2_800_active_learning_report",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "auto_labeled_by_this_script": False,
        "models_trained_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_changed_by_this_script": False,
        "target": {
            "current_judged_rows": len(dataset_rows),
            "target_judged_rows": 800,
            "new_candidates_needed": TARGET_TOTAL,
            "topics": len(topics),
            "new_candidates_per_topic": TARGET_PER_TOPIC,
        },
        "inputs": {
            "dataset": str(dataset_path),
            "v2_1_packet": str(v21_packet_path),
            "v3_0_report": str(v30_report_path),
            "v3_1_packet": str(v31_packet_path),
            "model_dir": str(model_dir),
            "selected_v2_7_method": SELECTED_V27_METHOD,
            "top_k_per_retriever": top_k,
        },
        "excluded_candidates": [
            {
                "query_id": BAD_CANDIDATE[0],
                "paper_id": BAD_CANDIDATE[1],
                "reason": "abstract length is too short for reliable labeling",
            }
        ],
        "candidate_pool_diagnostics": candidate_pool_diagnostics,
        "selection_summary": selection_summary,
        "validation": validation,
        "topic_packet_export": topic_export,
        "feature_set_recommendations": feature_recs,
        "protected_hashes": hashes,
        "ready_for_labeling": validation["all_candidates_labelable"],
    }
    return report, packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified V3.2 800-row active-learning packet.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-packet", default=str(DEFAULT_V21_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--v30-report", default=str(DEFAULT_V30_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--v31-packet", default=str(DEFAULT_V31_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--old-model", default=str(DEFAULT_OLD_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26_MODEL.relative_to(REPO_ROOT)))
    parser.add_argument("--packet-out", default=str(DEFAULT_PACKET_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--topic-packet-dir", default=str(DEFAULT_TOPIC_PACKET_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, packet = build_report(
        dataset_path=resolve_repo_path(args.dataset),
        v21_packet_path=resolve_repo_path(args.v21_packet),
        v30_report_path=resolve_repo_path(args.v30_report),
        v31_packet_path=resolve_repo_path(args.v31_packet),
        model_dir=resolve_repo_path(args.model_dir),
        old_model_path=resolve_repo_path(args.old_model),
        v26_model_path=resolve_repo_path(args.v26_model),
        topic_packet_dir=resolve_repo_path(args.topic_packet_dir),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        top_k=args.top_k,
    )
    report["command"] = " ".join(sys.argv)
    packet_out = resolve_repo_path(args.packet_out)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    write_jsonl(packet_out, packet)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    print("V3.2 800-row active-learning packet build complete")
    print(f"Packet: {packet_out}")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")
    print(f"Topic packet dir: {resolve_repo_path(args.topic_packet_dir)}")
    print(f"Candidates exported: {report['validation']['actual_total']}")
    print(f"All candidates labelable: {report['validation']['all_candidates_labelable']}")
    print(f"Protected hashes unchanged: {report['protected_hashes']}")


if __name__ == "__main__":
    main()
