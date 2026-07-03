import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
import re
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from app.db.session import SessionLocal  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.services.recommendation_service import build_v3_3_ltr_retriever  # noqa: E402
from evaluate_v2_4_true_production_candidate_pool import write_json, write_text  # noqa: E402
from evaluate_v3_6_revealed_v3_5_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    DEFAULT_V21_LABELS,
    DEFAULT_V25_LABELS,
    DEFAULT_V32_LABELS,
    DEFAULT_V35_LABELS,
    protected_hashes,
)


TOPICS = [
    "v2_ai_for_scientific_discovery",
    "v2_bayesian_optimization",
    "v2_causal_representation_learning",
    "v2_contrastive_learning",
    "v2_diffusion_image_generation",
    "v2_efficient_transformers",
    "v2_graph_neural_networks",
    "v2_graph_recommendation",
    "v2_large_language_model_agents",
    "v2_llm_evaluation",
    "v2_multimodal_learning",
    "v2_recommendation_systems",
    "v2_retrieval_augmented_generation",
    "v2_robot_learning",
    "v2_self_supervised_vision",
    "v2_transformer_architecture",
]
WEAK_TOPICS = {
    "v2_ai_for_scientific_discovery",
    "v2_causal_representation_learning",
    "v2_large_language_model_agents",
    "v2_llm_evaluation",
    "v2_retrieval_augmented_generation",
    "v2_transformer_architecture",
}
QUERY_TEXT = {
    "v2_ai_for_scientific_discovery": "ai for scientific discovery",
    "v2_bayesian_optimization": "bayesian optimization",
    "v2_causal_representation_learning": "causal representation learning",
    "v2_contrastive_learning": "contrastive learning",
    "v2_diffusion_image_generation": "diffusion image generation",
    "v2_efficient_transformers": "efficient transformers",
    "v2_graph_neural_networks": "graph neural networks",
    "v2_graph_recommendation": "graph recommendation",
    "v2_large_language_model_agents": "large language model agents",
    "v2_llm_evaluation": "llm evaluation",
    "v2_multimodal_learning": "multimodal learning",
    "v2_recommendation_systems": "recommendation systems",
    "v2_retrieval_augmented_generation": "retrieval augmented generation",
    "v2_robot_learning": "robot learning",
    "v2_self_supervised_vision": "self supervised vision",
    "v2_transformer_architecture": "transformer architecture",
}
SCORE_METHODS = [
    "bm25",
    "tfidf",
    "embedding",
    "faiss_embedding",
    "hybrid",
    "old_v2_2b",
    "v2_6",
    "v2_7",
    "v3_3_ltr",
    "v3_8_cross_encoder",
]
DIAGNOSTIC_QUOTAS = {
    "hidden_positive_candidate": 0.30,
    "v3_3_v2_7_disagreement": 0.20,
    "v3_3_v2_6_disagreement": 0.10,
    "hard_negative_candidate": 0.20,
    "role_diversity_candidate": 0.10,
}

DEFAULT_LABEL_OUT = REPO_ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl"
DEFAULT_CANDIDATES_OUT = REPO_ROOT / "data" / "eval" / "labeling" / "v3_9_semantic_expansion_candidates.jsonl"
DEFAULT_PACKET_OUT = REPO_ROOT / "data" / "eval" / "labeling" / "v3_9_semantic_expansion_packet.md"
DEFAULT_BATCH_DIR = REPO_ROOT / "data" / "eval" / "labeling" / "v3_9_semantic_expansion_batches"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v3_9_semantic_expansion_packet_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v3_9_semantic_expansion_packet_report.md"
DEFAULT_V38_CACHE = REPO_ROOT / "data" / "eval" / "cache" / "v3_8_text_reranker_scores.jsonl"
TARGET_TOTAL_LABELS = 2400
TARGET_PER_TOPIC = 150
MIN_NEW_LABELS = 1569


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def label_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["query_id"]), int(row["paper_id"])


def title_is_valid(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) >= 5


def abstract_is_valid(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip().split()) >= 30


def normalized_title(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return " ".join(text.split())


def score_rank(scores: dict[int, float]) -> dict[int, int]:
    return {
        paper_id: rank
        for rank, (paper_id, _score) in enumerate(
            sorted(scores.items(), key=lambda item: (item[1], -item[0]), reverse=True),
            start=1,
        )
    }


def load_existing_labels(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    by_source: dict[str, int] = {}
    for path in paths:
        source = path.stem.replace("manual_labels_", "")
        source_rows = load_jsonl(path)
        by_source[source] = len(source_rows)
        rows.extend(source_rows)
    by_topic = dict(sorted(Counter(str(row["query_id"]) for row in rows).items()))
    return rows, by_source, by_topic


def compute_topic_quotas(current_by_topic: dict[str, int]) -> dict[str, int]:
    return {topic: max(0, TARGET_PER_TOPIC - int(current_by_topic.get(topic, 0))) for topic in TOPICS}


def load_v38_cache(path: Path) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for row in load_jsonl(path, missing_ok=True):
        if row.get("model_name") != "cross-encoder/ms-marco-MiniLM-L-6-v2":
            continue
        scores[(str(row["query_id"]), int(row["paper_id"]))] = float(row["score"])
    return scores


def false_friend_terms(query_id: str, title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    terms_by_topic = {
        "v2_diffusion_image_generation": [
            "ecg",
            "arrhythmia",
            "traffic",
            "surface",
            "mesh",
            "pde",
            "diffusion convolutional",
            "random walk",
        ],
        "v2_transformer_architecture": ["time series", "load forecasting", "traffic forecasting", "power load"],
        "v2_retrieval_augmented_generation": ["biomedicine", "healthcare survey", "embedding model"],
        "v2_large_language_model_agents": ["protein language", "bias", "social impact"],
        "v2_causal_representation_learning": ["gwas", "genome", "mediation"],
        "v2_ai_for_scientific_discovery": ["academic life", "social impact", "education"],
    }
    return [term for term in terms_by_topic.get(query_id, []) if term in text]


def role_diversity_reasons(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    reasons = []
    if any(term in text for term in ["survey", "review", "overview", "taxonomy"]):
        reasons.append("role_diversity_survey_background")
    if any(term in text for term in ["benchmark", "evaluation", "dataset", "metric"]):
        reasons.append("role_diversity_evaluation_benchmark")
    if any(term in text for term in ["application", "case study", "deployed", "real-world"]):
        reasons.append("role_diversity_application")
    if any(term in text for term in ["foundation", "foundational", "seminal", "first "]):
        reasons.append("role_diversity_foundational")
    return reasons


def source_methods_for(debug: dict[str, Any], paper_id: int) -> list[str]:
    methods = []
    for method, rows in (debug.get("generation_runs") or {}).items():
        if any(int(row["paper_id"]) == paper_id for row in rows):
            methods.append(str(method))
    return sorted(methods)


def build_topic_candidates(
    *,
    query_id: str,
    query: str,
    retriever: Any,
    top_k: int,
    v38_cache: dict[tuple[str, int], float],
    existing_keys: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    debug = retriever.score_query(query=query, top_k=top_k)
    row_by_id = {int(row["paper_id"]): row for row in debug.get("rows", [])}
    score_values = debug.get("scores", {})
    scores_by_method: dict[str, dict[int, float]] = {method: {} for method in SCORE_METHODS}
    for paper_id, row in row_by_id.items():
        retrieval_scores = row.get("retrieval_scores_by_method") or {}
        values = score_values.get(paper_id, {})
        scores_by_method["bm25"][paper_id] = float(retrieval_scores.get("bm25", 0.0))
        scores_by_method["tfidf"][paper_id] = float(retrieval_scores.get("tfidf", 0.0))
        scores_by_method["embedding"][paper_id] = float(retrieval_scores.get("embedding", 0.0))
        scores_by_method["faiss_embedding"][paper_id] = float(retrieval_scores.get("faiss_embedding", 0.0))
        scores_by_method["hybrid"][paper_id] = float(values.get("hybrid_score", retrieval_scores.get("hybrid", 0.0)))
        scores_by_method["old_v2_2b"][paper_id] = float(values.get("old_v2_2b_score", 0.0))
        scores_by_method["v2_6"][paper_id] = float(values.get("v2_6_score", 0.0))
        scores_by_method["v2_7"][paper_id] = float(values.get("v2_7_score", 0.0))
        scores_by_method["v3_3_ltr"][paper_id] = float(values.get("v3_3_ltr_score", 0.0))
        ce_score = v38_cache.get((query_id, paper_id))
        if ce_score is not None:
            scores_by_method["v3_8_cross_encoder"][paper_id] = float(ce_score)

    ranks_by_method = {
        method: score_rank(scores)
        for method, scores in scores_by_method.items()
        if scores
    }
    candidates: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    seen_titles: set[str] = set()
    for paper_id, row in sorted(row_by_id.items()):
        key = (query_id, paper_id)
        if key in existing_keys:
            excluded["already_labeled"] += 1
            continue
        if not title_is_valid(row.get("title")):
            excluded["missing_or_short_title"] += 1
            continue
        if not abstract_is_valid(row.get("abstract")):
            excluded["missing_or_short_abstract"] += 1
            continue
        title_key = normalized_title(str(row.get("title") or ""))
        if title_key in seen_titles:
            excluded["near_duplicate_title_within_topic"] += 1
            continue
        seen_titles.add(title_key)

        title = str(row.get("title") or "")
        abstract = str(row.get("abstract") or "")
        source_methods = source_methods_for(debug, paper_id)
        ranks = {method: ranks_by_method.get(method, {}).get(paper_id) for method in SCORE_METHODS}
        scores = {
            method: scores_by_method.get(method, {}).get(paper_id)
            for method in SCORE_METHODS
            if paper_id in scores_by_method.get(method, {})
        }
        v33_rank = ranks.get("v3_3_ltr") or 999999
        v27_rank = ranks.get("v2_7") or 999999
        v26_rank = ranks.get("v2_6") or 999999
        ce_rank = ranks.get("v3_8_cross_encoder") or 999999
        reasons = set()
        if query_id in WEAK_TOPICS:
            reasons.add("weak_topic")
        if v33_rank <= 75:
            reasons.add("hidden_positive_candidate")
            reasons.add("v3_3_high_scoring_unjudged")
        if (v33_rank <= 75 and v27_rank > 175) or (v27_rank <= 75 and v33_rank > 175):
            reasons.add("v3_3_v2_7_disagreement")
        if (v33_rank <= 75 and v26_rank > 175) or (v26_rank <= 75 and v33_rank > 175):
            reasons.add("v3_3_v2_6_disagreement")
        if "v3_8_cross_encoder" in scores and ((v33_rank <= 75 and ce_rank > 175) or (ce_rank <= 75 and v33_rank > 175)):
            reasons.add("v3_3_cross_encoder_disagreement")
        false_friends = false_friend_terms(query_id, title, abstract)
        if false_friends or (ranks.get("bm25", 999999) <= 60 and min(ranks.get("embedding", 999999), v33_rank) > 150):
            reasons.add("hard_negative_candidate")
        diversity_reasons = role_diversity_reasons(title, abstract)
        if diversity_reasons:
            reasons.add("role_diversity_candidate")
            reasons.update(diversity_reasons)
        if len(source_methods) >= 3:
            reasons.add("multi_retriever_agreement_unjudged")
        if not reasons:
            reasons.add("semantic_pool_candidate")

        priority = packet_priority(
            query_id=query_id,
            reasons=reasons,
            ranks=ranks,
            source_methods=source_methods,
            abstract=abstract,
        )
        candidates.append(
            {
                "schema_version": "v3.9_semantic_expansion_candidate",
                "query_id": query_id,
                "query": query,
                "topic": query_id,
                "paper_id": paper_id,
                "title": title,
                "year": row.get("year"),
                "venue": row.get("venue"),
                "authors": row.get("authors") or [],
                "abstract": abstract,
                "source_url": row.get("source_url"),
                "pdf_url": row.get("pdf_url"),
                "diagnostic_reasons": sorted(reasons),
                "primary_diagnostic_reason": primary_reason(reasons),
                "candidate_source_methods": source_methods,
                "weak_topic": query_id in WEAK_TOPICS,
                "false_friend_terms": false_friends,
                "scores": scores,
                "ranks": ranks,
                "retrieval_scores_by_method": row.get("retrieval_scores_by_method") or {},
                "retrieval_ranks_by_method": row.get("retrieval_ranks_by_method") or {},
                "identifiers": row.get("identifiers") or {},
                "evidence_availability": row.get("evidence_availability") or {},
                "priority_score": priority,
                "labeling_instruction": "Manual label only. Do not infer labels from diagnostic reasons or priority score.",
            }
        )
    return candidates, excluded


def packet_priority(
    *,
    query_id: str,
    reasons: set[str],
    ranks: dict[str, int | None],
    source_methods: list[str],
    abstract: str,
) -> float:
    v33_rank = ranks.get("v3_3_ltr") or 999999
    v27_rank = ranks.get("v2_7") or 999999
    v26_rank = ranks.get("v2_6") or 999999
    score = 0.0
    score += 35.0 / math.sqrt(max(v33_rank, 1))
    score += 8.0 if query_id in WEAK_TOPICS else 0.0
    score += 10.0 if "hidden_positive_candidate" in reasons else 0.0
    score += 9.0 if "hard_negative_candidate" in reasons else 0.0
    score += 8.0 if "v3_3_v2_7_disagreement" in reasons else 0.0
    score += 5.0 if "v3_3_v2_6_disagreement" in reasons else 0.0
    score += 4.0 if "v3_3_cross_encoder_disagreement" in reasons else 0.0
    score += 2.0 if "role_diversity_candidate" in reasons else 0.0
    score += min(3.0, len(source_methods) * 0.75)
    score += min(2.0, len(abstract.split()) / 250.0)
    if v27_rank <= 75 and v33_rank > 175:
        score += 4.0
    if v26_rank <= 75 and v33_rank > 175:
        score += 3.0
    return round(score, 6)


def primary_reason(reasons: set[str]) -> str:
    order = [
        "hard_negative_candidate",
        "v3_3_v2_7_disagreement",
        "hidden_positive_candidate",
        "v3_3_v2_6_disagreement",
        "v3_3_cross_encoder_disagreement",
        "role_diversity_candidate",
        "multi_retriever_agreement_unjudged",
        "weak_topic",
        "semantic_pool_candidate",
    ]
    for reason in order:
        if reason in reasons:
            return reason
    return sorted(reasons)[0]


def select_topic_packet(candidates: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    candidates_sorted = sorted(candidates, key=lambda row: (float(row["priority_score"]), -int(row["paper_id"])), reverse=True)
    for reason, fraction in DIAGNOSTIC_QUOTAS.items():
        target = min(quota - len(selected), math.ceil(quota * fraction))
        if target <= 0:
            continue
        for row in candidates_sorted:
            if len([item for item in selected.values() if reason in item["diagnostic_reasons"]]) >= target:
                break
            if int(row["paper_id"]) in selected or reason not in row["diagnostic_reasons"]:
                continue
            selected[int(row["paper_id"])] = row
    for row in candidates_sorted:
        if len(selected) >= quota:
            break
        selected.setdefault(int(row["paper_id"]), row)
    return sorted(selected.values(), key=lambda row: (str(row["query_id"]), -float(row["priority_score"]), int(row["paper_id"])))


def label_template(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "v3.9_semantic_expansion_manual_label",
        "query_id": row["query_id"],
        "query": row["query"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "topic_match_score": None,
        "reading_value_score": None,
        "beginner_fit_score": None,
        "intermediate_fit_score": None,
        "advanced_fit_score": None,
        "expert_fit_score": None,
        "intent_scores": {
            "background": None,
            "foundational": None,
            "core_methods": None,
            "recent_frontier": None,
            "evaluation_benchmark": None,
            "application": None,
        },
        "primary_role": None,
        "secondary_roles": [],
        "duplicate_status": None,
        "duplicate_of_paper_id": None,
        "evidence_level": None,
        "full_text_available": None,
        "label_confidence": None,
        "notes": "",
    }


def format_score(value: Any) -> str:
    return "None" if value is None else f"{float(value):.6f}"


def build_candidate_markdown(rows: list[dict[str, Any]], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Manual labeling packet. Do not infer labels from diagnostic reasons, scores, ranks, or priority.",
        "",
        f"- Candidates: `{len(rows)}`",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        ranks = row.get("ranks") or {}
        scores = row.get("scores") or {}
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
                f"**Authors:** {', '.join(row.get('authors') or [])}",
                "",
                f"**Reasons:** {', '.join(row.get('diagnostic_reasons') or [])}",
                "",
                f"**Source methods:** {', '.join(row.get('candidate_source_methods') or [])}",
                "",
                "**Ranks:** "
                f"V3.3 `{ranks.get('v3_3_ltr')}`, V2.7 `{ranks.get('v2_7')}`, "
                f"V2.6 `{ranks.get('v2_6')}`, hybrid `{ranks.get('hybrid')}`, "
                f"BM25 `{ranks.get('bm25')}`, TF-IDF `{ranks.get('tfidf')}`, "
                f"embedding `{ranks.get('embedding')}`, CE `{ranks.get('v3_8_cross_encoder')}`",
                "",
                "**Scores:** "
                f"V3.3 `{format_score(scores.get('v3_3_ltr'))}`, V2.7 `{format_score(scores.get('v2_7'))}`, "
                f"V2.6 `{format_score(scores.get('v2_6'))}`, hybrid `{format_score(scores.get('hybrid'))}`, "
                f"CE `{format_score(scores.get('v3_8_cross_encoder'))}`",
                "",
                "**Abstract:**",
                "",
                str(row.get("abstract") or ""),
                "",
                "**Blank manual label JSONL template:**",
                "",
                "```json",
                json.dumps(label_template(row), ensure_ascii=False, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_batches(rows: list[dict[str, Any]], batch_dir: Path, batch_size: int) -> dict[str, Any]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    batches = [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]
    index_rows = []
    for batch_index, batch_rows in enumerate(batches, start=1):
        path = batch_dir / f"batch_{batch_index:02d}.md"
        write_text(path, build_candidate_markdown(batch_rows, title=f"V3.9 Semantic Expansion Batch {batch_index:02d}"))
        index_rows.append(
            {
                "batch": batch_index,
                "path": str(path.relative_to(REPO_ROOT)),
                "candidate_count": len(batch_rows),
                "first_candidate_index": (batch_index - 1) * batch_size + 1,
                "last_candidate_index": (batch_index - 1) * batch_size + len(batch_rows),
                "topics": dict(sorted(Counter(row["query_id"] for row in batch_rows).items())),
            }
        )
    write_json(batch_dir / "index.json", {"schema_version": "v3.9_semantic_expansion_batch_index", "batches": index_rows})
    lines = [
        "# V3.9 Semantic Expansion Batch Index",
        "",
        "| batch | candidates | range | topics | file |",
        "|---:|---:|---|---|---|",
    ]
    for row in index_rows:
        lines.append(
            f"| {row['batch']} | {row['candidate_count']} | {row['first_candidate_index']}-{row['last_candidate_index']} | "
            f"`{row['topics']}` | `{row['path']}` |"
        )
    write_text(batch_dir / "index.md", "\n".join(lines) + "\n")
    return {"batch_size": batch_size, "batch_count": len(batches), "batches": index_rows}


def build_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3.9 Semantic Expansion Packet Report",
        "",
        f"Created: `{report['created_at']}`",
        "",
        f"- Current judged rows: `{report['current_judged_total']}`",
        f"- V3.9 candidates: `{report['candidate_count']}`",
        f"- Projected judged rows after complete labeling: `{report['projected_total_after_labeling']}`",
        f"- Minimum target reached: `{report['target_reached']}`",
        f"- Labelable candidates: `{report['labelable_candidates']}`",
        f"- Batch files: `{report['batching']['batch_count']}`",
        f"- Corpus expanded by this script: `{report['corpus_expanded_by_this_script']}`",
        f"- Models retrained by this script: `{report['models_retrained_by_this_script']}`",
        "",
        "## Labels And Quotas",
        "",
        "| topic | existing | quota | selected | projected |",
        "|---|---:|---:|---:|---:|",
    ]
    for topic in TOPICS:
        lines.append(
            f"| `{topic}` | {report['current_labels_by_topic'].get(topic, 0)} | "
            f"{report['topic_quotas'].get(topic, 0)} | {report['candidates_by_topic'].get(topic, 0)} | "
            f"{report['projected_labels_by_topic'].get(topic, 0)} |"
        )
    lines.extend(["", "## Diagnostic Reasons", "", "| reason | candidates |", "|---|---:|"])
    for reason, count in report["candidates_by_diagnostic_reason"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Source Methods", "", "| method | candidates |", "|---|---:|"])
    for method, count in report["candidates_by_source_method"].items():
        lines.append(f"| `{method}` | {count} |")
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
            f"- V3.5 labels unchanged: `{report['protected_hashes']['v3_5_labels_hash_unchanged']}`",
            f"- Selected 240 unchanged: `{report['protected_hashes']['selected_240_hash_unchanged']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_packet(
    *,
    v21_labels_path: Path,
    v25_labels_path: Path,
    v32_labels_path: Path,
    v35_labels_path: Path,
    selected_240_path: Path,
    labels_out: Path,
    top_k: int,
    batch_size: int,
    v38_cache_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_paths = [v21_labels_path, v25_labels_path, v32_labels_path, v35_labels_path]
    existing_labels, labels_by_source, current_by_topic = load_existing_labels(label_paths)
    existing_keys = {label_key(row) for row in existing_labels}
    quotas = compute_topic_quotas(current_by_topic)
    v38_cache = load_v38_cache(v38_cache_path)
    excluded = Counter()
    available_by_topic: dict[str, int] = {}
    selected_rows: list[dict[str, Any]] = []

    with SessionLocal() as db:
        papers = list(db.query(Paper).order_by(Paper.id).all())
        retriever = build_v3_3_ltr_retriever(papers)
        for topic in TOPICS:
            candidates, topic_excluded = build_topic_candidates(
                query_id=topic,
                query=QUERY_TEXT[topic],
                retriever=retriever,
                top_k=top_k,
                v38_cache=v38_cache,
                existing_keys=existing_keys,
            )
            available_by_topic[topic] = len(candidates)
            excluded.update({f"{topic}:{key}": count for key, count in topic_excluded.items()})
            selected_rows.extend(select_topic_packet(candidates, quotas[topic]))

    selected_rows = sorted(selected_rows, key=lambda row: (TOPICS.index(str(row["query_id"])), -float(row["priority_score"]), int(row["paper_id"])))
    for index, row in enumerate(selected_rows, start=1):
        row["packet_index"] = index
        row["batch_index"] = math.ceil(index / batch_size)

    if labels_out.exists() and labels_out.stat().st_size > 0:
        raise ValueError(f"Refusing to overwrite non-empty V3.9 label file: {labels_out}")
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.touch(exist_ok=True)

    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in selected_rows:
        reason_counts.update(row.get("diagnostic_reasons") or [])
        source_counts.update(row.get("candidate_source_methods") or [])
    candidates_by_topic = dict(sorted(Counter(row["query_id"] for row in selected_rows).items()))
    projected_by_topic = {
        topic: int(current_by_topic.get(topic, 0)) + int(candidates_by_topic.get(topic, 0))
        for topic in TOPICS
    }
    batch_info = write_batches(selected_rows, DEFAULT_BATCH_DIR, batch_size)
    report = {
        "schema_version": "v3.9_semantic_expansion_packet_report",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "current_judged_total": len(existing_labels),
        "labels_by_source": labels_by_source,
        "current_labels_by_topic": current_by_topic,
        "target_total_labels": TARGET_TOTAL_LABELS,
        "target_per_topic": TARGET_PER_TOPIC,
        "minimum_new_candidates": MIN_NEW_LABELS,
        "topic_quotas": quotas,
        "candidate_count": len(selected_rows),
        "projected_total_after_labeling": len(existing_labels) + len(selected_rows),
        "projected_labels_by_topic": projected_by_topic,
        "target_reached": len(existing_labels) + len(selected_rows) >= TARGET_TOTAL_LABELS,
        "candidates_by_topic": candidates_by_topic,
        "available_candidates_by_topic_after_exclusions": available_by_topic,
        "candidates_by_diagnostic_reason": dict(sorted(reason_counts.items())),
        "candidates_by_primary_reason": dict(sorted(Counter(row["primary_diagnostic_reason"] for row in selected_rows).items())),
        "candidates_by_source_method": dict(sorted(source_counts.items())),
        "weak_topic_coverage": {
            "weak_topics": sorted(WEAK_TOPICS),
            "candidate_count": sum(1 for row in selected_rows if row["weak_topic"]),
            "count_by_topic": dict(sorted(Counter(row["query_id"] for row in selected_rows if row["weak_topic"]).items())),
        },
        "disagreement_coverage": {
            "v3_3_v2_7": reason_counts.get("v3_3_v2_7_disagreement", 0),
            "v3_3_v2_6": reason_counts.get("v3_3_v2_6_disagreement", 0),
            "v3_3_cross_encoder": reason_counts.get("v3_3_cross_encoder_disagreement", 0),
        },
        "hard_negative_candidate_count": reason_counts.get("hard_negative_candidate", 0),
        "hidden_positive_candidate_count": reason_counts.get("hidden_positive_candidate", 0),
        "excluded_candidate_counts": dict(sorted(excluded.items())),
        "duplicate_query_paper_rows": [
            {"query_id": query_id, "paper_id": paper_id, "count": count}
            for (query_id, paper_id), count in Counter(label_key(row) for row in selected_rows).items()
            if count > 1
        ],
        "labelable_candidates": all(title_is_valid(row.get("title")) and abstract_is_valid(row.get("abstract")) for row in selected_rows),
        "batching": batch_info,
        "protected_hashes": protected_hashes(v21_labels_path, v25_labels_path, v32_labels_path, v35_labels_path, selected_240_path),
    }
    return selected_rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V3.9 semantic label expansion packet.")
    parser.add_argument("--v21-labels", default=str(DEFAULT_V21_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v25-labels", default=str(DEFAULT_V25_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v32-labels", default=str(DEFAULT_V32_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--v35-labels", default=str(DEFAULT_V35_LABELS.relative_to(REPO_ROOT)))
    parser.add_argument("--selected-240", default=str(DEFAULT_SELECTED_240.relative_to(REPO_ROOT)))
    parser.add_argument("--labels-out", default=str(DEFAULT_LABEL_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--candidates-out", default=str(DEFAULT_CANDIDATES_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--packet-out", default=str(DEFAULT_PACKET_OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--report-json-out", default=str(DEFAULT_REPORT_JSON.relative_to(REPO_ROOT)))
    parser.add_argument("--report-md-out", default=str(DEFAULT_REPORT_MD.relative_to(REPO_ROOT)))
    parser.add_argument("--v38-cache", default=str(DEFAULT_V38_CACHE.relative_to(REPO_ROOT)))
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report = build_packet(
        v21_labels_path=resolve_repo_path(args.v21_labels),
        v25_labels_path=resolve_repo_path(args.v25_labels),
        v32_labels_path=resolve_repo_path(args.v32_labels),
        v35_labels_path=resolve_repo_path(args.v35_labels),
        selected_240_path=resolve_repo_path(args.selected_240),
        labels_out=resolve_repo_path(args.labels_out),
        top_k=args.top_k,
        batch_size=args.batch_size,
        v38_cache_path=resolve_repo_path(args.v38_cache),
    )
    write_jsonl(resolve_repo_path(args.candidates_out), rows)
    write_text(resolve_repo_path(args.packet_out), build_candidate_markdown(rows, title="V3.9 Semantic Expansion Packet"))
    write_json(resolve_repo_path(args.report_json_out), report)
    write_text(resolve_repo_path(args.report_md_out), build_report_markdown(report))
    print("V3.9 semantic expansion packet built")
    print(f"Current judged rows: {report['current_judged_total']}")
    print(f"Candidates: {report['candidate_count']}")
    print(f"Projected judged rows: {report['projected_total_after_labeling']}")
    print(f"Batch files: {report['batching']['batch_count']}")
    print(f"Report: {resolve_repo_path(args.report_json_out)}")


if __name__ == "__main__":
    main()
