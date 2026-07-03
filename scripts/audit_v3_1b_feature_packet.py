import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import statistics
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
from evaluate_v2_9_score_columns import duplicate_rows  # noqa: E402
from train_v2_6_production_aware_learned_hybrid import EXPECTED_V21_LABELS_SHA256  # noqa: E402


DEFAULT_PACKET = REPO_ROOT / "data" / "eval" / "labeling" / "v3_1_active_learning_candidates.jsonl"
DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "training" / "v2_9_ltr_dataset.jsonl"
DEFAULT_V30_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v3_0_ltr_training_report.json"
DEFAULT_RIDGE_METADATA = REPO_ROOT / "data" / "processed" / "models" / "v3_0_ltr" / "ridge_no_v27.metadata.json"
DEFAULT_JSON_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_1b_feature_packet_audit.json"
DEFAULT_MD_OUT = REPO_ROOT / "data" / "eval" / "results" / "v3_1b_feature_packet_audit.md"
DEFAULT_TOPIC_PACKET_DIR = REPO_ROOT / "data" / "eval" / "labeling" / "v3_1_packets"
DEFAULT_V21_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
DEFAULT_V25_LABELS = REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"

RETRIEVERS = ["bm25", "tfidf", "embedding", "faiss_embedding", "hybrid"]
REQUIRED_PACKET_FIELDS = [
    "query_id",
    "query",
    "topic",
    "paper_id",
    "title",
    "abstract",
    "scores",
    "ranks",
    "retrieval_ranks",
    "why_selected",
]
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
]
REQUIRED_RANK_FIELDS = ["hybrid", "v2_6_score", "v2_7_score", "ridge_no_v27"]
SUSPICIOUS_FEATURES = ["bm25_rank", "hybrid_raw_score", "embedding_raw_score", "tfidf_score"]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return 0.0
    return numerator / (x_den * y_den)


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


def packet_metadata_audit(packet_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_ranks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    short_titles: list[dict[str, Any]] = []
    short_abstracts: list[dict[str, Any]] = []
    empty_why_selected: list[dict[str, Any]] = []

    for row in packet_rows:
        key = {"query_id": row.get("query_id"), "paper_id": row.get("paper_id"), "title": row.get("title")}
        for field in REQUIRED_PACKET_FIELDS:
            value = row.get(field)
            if value is None or value == "" or value == [] or value == {}:
                missing_by_field[field].append(key)
        scores = row.get("scores") or {}
        ranks = row.get("ranks") or {}
        for field in REQUIRED_SCORE_FIELDS:
            if scores.get(field) is None:
                missing_scores[field].append(key)
        for field in REQUIRED_RANK_FIELDS:
            if ranks.get(field) is None:
                missing_ranks[field].append(key)
        if text_len(row.get("title")) < 8:
            short_titles.append(key | {"title_length": text_len(row.get("title"))})
        if text_len(row.get("abstract")) < 120:
            short_abstracts.append(key | {"abstract_length": text_len(row.get("abstract"))})
        if not row.get("why_selected"):
            empty_why_selected.append(key)

    duplicates = duplicate_rows(packet_rows)
    topic_counts = dict(Counter(str(row["query_id"]) for row in packet_rows))
    blocking_issue_count = (
        len(duplicates)
        + sum(len(rows) for rows in missing_by_field.values())
        + sum(len(rows) for rows in missing_scores.values())
        + sum(len(rows) for rows in missing_ranks.values())
        + len(short_titles)
        + len(short_abstracts)
        + len(empty_why_selected)
    )
    complete = (
        len(packet_rows) == 192
        and not duplicates
        and not missing_by_field
        and not missing_scores
        and not missing_ranks
        and not short_titles
        and not short_abstracts
        and not empty_why_selected
        and len(topic_counts) == 16
        and all(count == 12 for count in topic_counts.values())
    )
    return {
        "row_count": len(packet_rows),
        "topic_counts": topic_counts,
        "duplicate_query_paper_rows": duplicates,
        "missing_required_fields": {field: rows for field, rows in sorted(missing_by_field.items())},
        "missing_score_fields": {field: rows for field, rows in sorted(missing_scores.items())},
        "missing_rank_fields": {field: rows for field, rows in sorted(missing_ranks.items())},
        "short_title_threshold_chars": 8,
        "short_abstract_threshold_chars": 120,
        "short_titles": short_titles,
        "short_abstracts": short_abstracts,
        "empty_why_selected": empty_why_selected,
        "blocking_labelability_issue_count": blocking_issue_count,
        "labelable_candidate_count": max(0, len(packet_rows) - len(short_titles) - len(short_abstracts)),
        "all_192_candidates_labelable": complete,
    }


def feature_values(rows: list[dict[str, Any]], feature: str) -> list[float]:
    return [as_float(row.get(feature)) for row in rows]


def feature_direction_audit(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevance = [as_float(row.get("relevance_label")) for row in dataset_rows]
    retriever_rows: dict[str, Any] = {}
    normalized_checks: dict[str, Any] = {}
    rank_checks: dict[str, Any] = {}
    for retriever in RETRIEVERS:
        score = feature_values(dataset_rows, f"{retriever}_score")
        raw = feature_values(dataset_rows, f"{retriever}_raw_score")
        rank = feature_values(dataset_rows, f"{retriever}_rank")
        reciprocal = feature_values(dataset_rows, f"{retriever}_reciprocal_rank")
        retriever_rows[retriever] = {
            "score_relevance_corr": pearson(score, relevance),
            "raw_score_relevance_corr": pearson(raw, relevance),
            "rank_relevance_corr": pearson(rank, relevance),
            "reciprocal_rank_relevance_corr": pearson(reciprocal, relevance),
            "score_raw_corr": pearson(score, raw),
            "score_rank_corr": pearson(score, rank),
            "score_reciprocal_rank_corr": pearson(score, reciprocal),
            "rank_reciprocal_rank_corr": pearson(rank, reciprocal),
        }
        normalized_checks[retriever] = {
            "larger_normalized_score_appears_better": pearson(score, relevance) >= -0.05,
            "normalized_score_increases_with_raw_score": pearson(score, raw) >= 0.5,
            "normalized_score_decreases_with_rank": pearson(score, rank) <= -0.1,
            "min_score": min(score) if score else None,
            "max_score": max(score) if score else None,
        }
        rank_checks[retriever] = {
            "smaller_rank_appears_better": pearson(rank, score) <= -0.1,
            "reciprocal_rank_increases_as_rank_improves": pearson(rank, reciprocal) <= -0.1,
            "min_rank": min(rank) if rank else None,
            "max_rank": max(rank) if rank else None,
        }
    return {
        "interpretation": {
            "ranks": "smaller numeric rank is intended to mean better; missing items use 999999.",
            "scores": "larger score is intended to mean better.",
            "normalized_scores": "normalized score columns are min-max style and preserve larger-is-better direction.",
        },
        "by_retriever": retriever_rows,
        "rank_direction_checks": rank_checks,
        "normalized_score_direction_checks": normalized_checks,
        "feature_directions_consistent": all(
            row["normalized_score_decreases_with_rank"] and row["normalized_score_increases_with_raw_score"]
            for row in normalized_checks.values()
        ),
    }


def coefficient_lookup(ridge_metadata: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["feature"]): as_float(row["weight"])
        for row in ridge_metadata["estimator_info"].get("coefficients", [])
    }


def feature_correlation_matrix(dataset_rows: list[dict[str, Any]], features: list[str]) -> dict[str, dict[str, float]]:
    values = {feature: feature_values(dataset_rows, feature) for feature in features}
    return {
        left: {right: pearson(values[left], values[right]) for right in features}
        for left in features
    }


def multicollinearity_audit(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    high_pairs: list[dict[str, Any]] = []
    for retriever in RETRIEVERS:
        features = [
            f"{retriever}_score",
            f"{retriever}_raw_score",
            f"{retriever}_rank",
            f"{retriever}_reciprocal_rank",
        ]
        matrix = feature_correlation_matrix(dataset_rows, features)
        pairs = []
        for index, left in enumerate(features):
            for right in features[index + 1 :]:
                corr = matrix[left][right]
                pair = {"left": left, "right": right, "pearson": corr, "abs_pearson": abs(corr)}
                pairs.append(pair)
                if abs(corr) >= 0.85:
                    high_pairs.append({"retriever": retriever, **pair})
        groups[retriever] = {
            "features": features,
            "correlation_pairs": sorted(pairs, key=lambda row: row["abs_pearson"], reverse=True),
            "has_high_multicollinearity": any(abs(row["pearson"]) >= 0.85 for row in pairs),
        }
    return {
        "groups": groups,
        "high_correlation_threshold": 0.85,
        "high_correlation_pairs": sorted(high_pairs, key=lambda row: row["abs_pearson"], reverse=True),
        "multicollinearity_detected": bool(high_pairs),
    }


def suspicious_coefficient_audit(
    dataset_rows: list[dict[str, Any]],
    ridge_metadata: dict[str, Any],
    multicollinearity: dict[str, Any],
) -> dict[str, Any]:
    relevance = [as_float(row.get("relevance_label")) for row in dataset_rows]
    weights = coefficient_lookup(ridge_metadata)
    audits: dict[str, Any] = {}
    for feature in SUSPICIOUS_FEATURES:
        values = feature_values(dataset_rows, feature)
        prefix = feature.split("_")[0]
        related_pairs = [
            row
            for row in multicollinearity["high_correlation_pairs"]
            if str(row["left"]).startswith(prefix) or str(row["right"]).startswith(prefix)
        ]
        corr = pearson(values, relevance)
        weight = weights.get(feature, 0.0)
        if feature.endswith("_rank"):
            expected = "negative_or_near_zero"
            suspicious = weight > 0.001
        elif feature.endswith("_raw_score") or feature.endswith("_score"):
            expected = "positive_or_near_zero"
            suspicious = weight < -0.001
        else:
            expected = "near_zero_unknown"
            suspicious = False
        audits[feature] = {
            "ridge_weight": weight,
            "univariate_relevance_corr": corr,
            "expected_direction": expected,
            "coefficient_direction_suspicious": suspicious,
            "likely_explanation": (
                "Coefficient sign is not directly interpretable because raw, normalized, rank, and reciprocal-rank "
                "features for the same retriever are strongly collinear."
                if related_pairs
                else "No strong same-retriever collinearity detected; inspect feature engineering before reuse."
            ),
            "related_high_correlation_pairs": related_pairs[:8],
        }
    return audits


def recommended_feature_actions(
    suspicious: dict[str, Any],
    multicollinearity: dict[str, Any],
) -> list[dict[str, Any]]:
    actions = [
        {
            "recommendation": "For V3.2, drop or ablate raw score features when normalized score and rank features are present.",
            "reason": "Raw score scales vary by retriever and show unstable signs under multicollinearity.",
            "features": ["bm25_raw_score", "tfidf_raw_score", "embedding_raw_score", "faiss_embedding_raw_score", "hybrid_raw_score"],
        },
        {
            "recommendation": "Keep one directionally clear rank signal per retriever, preferably reciprocal_rank or a capped rank transform, not both rank and reciprocal_rank.",
            "reason": "Rank and reciprocal-rank encode the same signal with opposite direction and high correlation.",
            "features": [f"{retriever}_rank/{retriever}_reciprocal_rank" for retriever in RETRIEVERS],
        },
        {
            "recommendation": "Treat `ridge_no_v27` coefficients as diagnostic, not causal feature importance.",
            "reason": "V3.0 was trained on a small judged table with heavily correlated retrieval features.",
            "features": list(suspicious),
        },
    ]
    if multicollinearity["multicollinearity_detected"]:
        actions.append(
            {
                "recommendation": "Use grouped feature ablations or regularized feature sets in V3.2.",
                "reason": "High same-retriever correlations make individual coefficient signs unstable.",
                "features": sorted({row["retriever"] for row in multicollinearity["high_correlation_pairs"]}),
            }
        )
    return actions


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


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "topic"


def format_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else ""


def markdown_escape(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def candidate_label_block(row: dict[str, Any], index: int) -> list[str]:
    scores = row.get("scores") or {}
    ranks = row.get("ranks") or {}
    retrieval_ranks = row.get("retrieval_ranks") or {}
    lines = [
        f"## Candidate {index}: {markdown_escape(row.get('title'))}",
        "",
        f"- query_id: `{row.get('query_id')}`",
        f"- paper_id: `{row.get('paper_id')}`",
        f"- year: `{row.get('year')}`",
        f"- venue: `{markdown_escape(row.get('venue'))}`",
        f"- source methods: `{format_list(row.get('source_methods') or [])}`",
        f"- why_selected: `{format_list(row.get('why_selected') or [])}`",
        f"- scores: ridge `{scores.get('ridge_no_v27')}`, V2.7 `{scores.get('v2_7')}`, V2.6 `{scores.get('v2_6')}`, hybrid `{scores.get('hybrid')}`",
        f"- ranks: ridge `{ranks.get('ridge_no_v27')}`, V2.7 `{ranks.get('v2_7_score')}`, V2.6 `{ranks.get('v2_6_score')}`, hybrid `{ranks.get('hybrid')}`",
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
    return lines


def export_topic_packets(packet_rows: list[dict[str, Any]], packet_dir: Path) -> dict[str, Any]:
    packet_dir.mkdir(parents=True, exist_ok=True)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in packet_rows:
        by_topic[str(row["query_id"])].append(row)
    outputs = []
    for topic, rows in sorted(by_topic.items()):
        rows = sorted(rows, key=lambda row: as_float(row.get("selection_score")), reverse=True)
        path = packet_dir / f"topic_{slugify(topic)}.md"
        lines = [
            f"# V3.1 Active-Learning Packet: {topic}",
            "",
            f"Query: {rows[0].get('query') or rows[0].get('topic')}",
            "",
            "Manual labeling packet. Do not infer labels from selection reasons.",
            "",
        ]
        for index, row in enumerate(rows, start=1):
            lines.extend(candidate_label_block(row, index))
        write_text(path, "\n".join(lines))
        outputs.append({"topic": topic, "path": str(path), "candidate_count": len(rows)})
    return {
        "packet_dir": str(packet_dir),
        "file_count": len(outputs),
        "outputs": outputs,
        "all_topics_have_12_candidates": all(row["candidate_count"] == 12 for row in outputs),
    }


def build_markdown(report: dict[str, Any]) -> str:
    packet = report["packet_audit"]
    directions = report["feature_direction_audit"]
    multicollinearity = report["multicollinearity_audit"]
    suspicious = report["suspicious_coefficients"]
    lines = [
        "# V3.1b Feature and Packet Sanity Audit",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "This audit checks whether the V3.1 active-learning packet is ready for manual labeling and whether the V3.0 feature directions are trustworthy enough to interpret before V3.2 tuning.",
        "",
        "## Packet Readiness",
        "",
        f"- Candidates: `{packet['row_count']}`",
        f"- Topics: `{len(packet['topic_counts'])}`",
        f"- Duplicate query-paper rows: `{len(packet['duplicate_query_paper_rows'])}`",
        f"- Missing required fields: `{sum(len(rows) for rows in packet['missing_required_fields'].values())}`",
        f"- Missing score fields: `{sum(len(rows) for rows in packet['missing_score_fields'].values())}`",
        f"- Missing rank fields: `{sum(len(rows) for rows in packet['missing_rank_fields'].values())}`",
        f"- Short titles: `{len(packet['short_titles'])}`",
        f"- Short abstracts: `{len(packet['short_abstracts'])}`",
        f"- Labelable candidate count: `{packet['labelable_candidate_count']}`",
        f"- All 192 candidates labelable: `{packet['all_192_candidates_labelable']}`",
        "",
    ]
    if packet["short_abstracts"]:
        lines.append("Short abstract candidates:")
        for row in packet["short_abstracts"][:10]:
            lines.append(
                f"- `{row['query_id']}` / `{row['paper_id']}`: abstract length `{row['abstract_length']}`, title `{row['title']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Feature Direction Checks",
            "",
            f"- Feature directions consistent: `{directions['feature_directions_consistent']}`",
            "- Smaller rank means better: confirmed by score/rank and rank/reciprocal-rank direction checks.",
            "- Larger normalized score means better: confirmed by positive raw-score correlation and negative rank correlation for each retriever.",
            "",
            "| retriever | score/raw corr | score/rank corr | rank/relevance corr | score/relevance corr |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for retriever, row in directions["by_retriever"].items():
        lines.append(
            f"| `{retriever}` | {row['score_raw_corr']:.3f} | {row['score_rank_corr']:.3f} | "
            f"{row['rank_relevance_corr']:.3f} | {row['score_relevance_corr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Suspicious Coefficients",
            "",
            "| feature | ridge weight | univariate relevance corr | suspicious | interpretation |",
            "|---|---:|---:|---|---|",
        ]
    )
    for feature, row in suspicious.items():
        lines.append(
            f"| `{feature}` | {row['ridge_weight']:.4f} | {row['univariate_relevance_corr']:.3f} | "
            f"`{row['coefficient_direction_suspicious']}` | {row['likely_explanation']} |"
        )
    lines.extend(
        [
            "",
            "## Multicollinearity",
            "",
            f"- High-correlation threshold: `{multicollinearity['high_correlation_threshold']}`",
            f"- High-correlation pairs: `{len(multicollinearity['high_correlation_pairs'])}`",
            f"- Multicollinearity detected: `{multicollinearity['multicollinearity_detected']}`",
            "",
            "Top correlated pairs:",
        ]
    )
    for row in multicollinearity["high_correlation_pairs"][:12]:
        lines.append(f"- `{row['left']}` vs `{row['right']}`: `{row['pearson']:.3f}`")
    lines.extend(
        [
            "",
            "## V3.2 Feature Recommendations",
            "",
        ]
    )
    for action in report["recommended_v3_2_feature_actions"]:
        lines.append(f"- {action['recommendation']} Reason: {action['reason']}")
    lines.extend(
        [
            "",
            "## Human-Friendly Topic Packets",
            "",
            f"- Files exported: `{report['topic_packet_export']['file_count']}`",
            f"- All topic files have 12 candidates: `{report['topic_packet_export']['all_topics_have_12_candidates']}`",
            "- These files are exported for review; do not label the short-abstract candidate without enrichment or explicit acceptance.",
            "",
            "## Protected Hashes",
            "",
            f"- V2.1 labels unchanged: `{report['protected_hashes']['v2_1_labels_hash_unchanged']}`",
            f"- V2.5 labels unchanged: `{report['protected_hashes']['v2_5_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
            "",
            "## Decision",
            "",
            f"- Packet ready for labeling: `{report['packet_ready_for_labeling']}`",
            f"- Feature directions consistent: `{report['feature_directions_consistent']}`",
            f"- All 192 candidates labelable: `{packet['all_192_candidates_labelable']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(
    *,
    packet_path: Path,
    dataset_path: Path,
    v30_report_path: Path,
    ridge_metadata_path: Path,
    topic_packet_dir: Path,
    v21_labels_path: Path,
    v25_labels_path: Path,
) -> dict[str, Any]:
    packet_rows = load_jsonl(packet_path)
    dataset_rows = load_jsonl(dataset_path)
    v30_report = load_json(v30_report_path)
    ridge_metadata = load_json(ridge_metadata_path)

    packet_audit = packet_metadata_audit(packet_rows)
    feature_directions = feature_direction_audit(dataset_rows)
    multicollinearity = multicollinearity_audit(dataset_rows)
    suspicious = suspicious_coefficient_audit(dataset_rows, ridge_metadata, multicollinearity)
    actions = recommended_feature_actions(suspicious, multicollinearity)
    topic_export = export_topic_packets(packet_rows, topic_packet_dir)
    hashes = protected_hashes(v21_labels_path, v25_labels_path)
    packet_ready = (
        packet_audit["all_192_candidates_labelable"]
        and not packet_audit["duplicate_query_paper_rows"]
        and all(
            [
                hashes["v2_1_labels_hash_unchanged"],
                hashes["v2_5_labels_hash_unchanged"],
                hashes["selected_240_hash_unchanged"],
            ]
        )
    )
    return {
        "schema_version": "v3.1b_feature_packet_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "labels_created_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_candidates_modified_by_this_script": False,
        "models_trained_by_this_script": False,
        "neural_models_trained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_changed_by_this_script": False,
        "auto_labeled_by_this_script": False,
        "inputs": {
            "packet": str(packet_path),
            "dataset": str(dataset_path),
            "v3_0_report": str(v30_report_path),
            "ridge_metadata": str(ridge_metadata_path),
        },
        "v3_0_context": {
            "best_model_by_loto_reading_ndcg": v30_report.get("best_model_by_loto_reading_ndcg"),
            "success_summary": v30_report.get("success_summary"),
            "failure_summary": v30_report.get("failure_summary"),
        },
        "packet_audit": packet_audit,
        "feature_direction_audit": feature_directions,
        "suspicious_coefficients": suspicious,
        "multicollinearity_audit": multicollinearity,
        "recommended_v3_2_feature_actions": actions,
        "topic_packet_export": topic_export,
        "protected_hashes": hashes,
        "packet_ready_for_labeling": packet_ready,
        "feature_directions_consistent": feature_directions["feature_directions_consistent"],
        "all_192_candidates_labelable": packet_audit["all_192_candidates_labelable"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V3.1 packet and V3.0 feature directions before labeling.")
    parser.add_argument("--packet", default=str(DEFAULT_PACKET.relative_to(REPO_ROOT)))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET.relative_to(REPO_ROOT)))
    parser.add_argument("--v30-report", default=str(DEFAULT_V30_REPORT.relative_to(REPO_ROOT)))
    parser.add_argument("--ridge-metadata", default=str(DEFAULT_RIDGE_METADATA.relative_to(REPO_ROOT)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--topic-packet-dir", default=str(DEFAULT_TOPIC_PACKET_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        packet_path=resolve_repo_path(args.packet),
        dataset_path=resolve_repo_path(args.dataset),
        v30_report_path=resolve_repo_path(args.v30_report),
        ridge_metadata_path=resolve_repo_path(args.ridge_metadata),
        topic_packet_dir=resolve_repo_path(args.topic_packet_dir),
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
    )
    report["command"] = " ".join(sys.argv)
    json_out = resolve_repo_path(args.json_out)
    md_out = resolve_repo_path(args.md_out)
    write_json(json_out, report)
    write_text(md_out, build_markdown(report))
    print("V3.1b feature and packet audit complete")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")
    print(f"Topic packet dir: {resolve_repo_path(args.topic_packet_dir)}")
    print(f"Packet ready for labeling: {report['packet_ready_for_labeling']}")
    print(f"All 192 candidates labelable: {report['all_192_candidates_labelable']}")
    print(f"Feature directions consistent: {report['feature_directions_consistent']}")


if __name__ == "__main__":
    main()
