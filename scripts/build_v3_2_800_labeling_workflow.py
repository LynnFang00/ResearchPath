import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_v2_4_true_production_candidate_pool import load_jsonl, resolve_repo_path, write_text  # noqa: E402


DEFAULT_CANDIDATES = REPO_ROOT / "data" / "eval" / "labeling" / "v3_2_800_active_learning_candidates.jsonl"
DEFAULT_BATCH_DIR = REPO_ROOT / "data" / "eval" / "labeling" / "v3_2_800_batches"
DEFAULT_GUIDE = REPO_ROOT / "data" / "eval" / "labeling" / "v3_2_800_labeling_guide.md"

BATCH_TOPICS = [
    [
        "v2_ai_for_scientific_discovery",
        "v2_bayesian_optimization",
        "v2_causal_representation_learning",
        "v2_contrastive_learning",
    ],
    [
        "v2_diffusion_image_generation",
        "v2_efficient_transformers",
        "v2_graph_neural_networks",
        "v2_graph_recommendation",
    ],
    [
        "v2_large_language_model_agents",
        "v2_llm_evaluation",
        "v2_multimodal_learning",
        "v2_recommendation_systems",
    ],
    [
        "v2_retrieval_augmented_generation",
        "v2_robot_learning",
        "v2_self_supervised_vision",
        "v2_transformer_architecture",
    ],
]


def markdown_escape(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def format_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else ""


def candidate_block(row: dict[str, Any], index: int) -> list[str]:
    scores = row.get("scores") or {}
    ranks = row.get("ranks") or {}
    retrieval_ranks = row.get("retrieval_ranks") or {}
    return [
        f"### Candidate {index}: {markdown_escape(row.get('title'))}",
        "",
        f"- query_id: `{row.get('query_id')}`",
        f"- paper_id: `{row.get('paper_id')}`",
        f"- year: `{row.get('year')}`",
        f"- venue: `{markdown_escape(row.get('venue'))}`",
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
        "**Manual Label JSONL Fields**",
        "",
        "```json",
        '{"schema_version":"v3.2_800_manual_label","query_id":"' + str(row.get("query_id")) + '","query":"' + str(row.get("query") or row.get("topic")) + '","paper_id":' + str(row.get("paper_id")) + ',"title":"' + str(row.get("title", "")).replace('"', '\\"') + '","topic_match_score":null,"reading_value_score":null,"beginner_fit_score":null,"intermediate_fit_score":null,"advanced_fit_score":null,"expert_fit_score":null,"intent_scores":{"background":null,"foundational":null,"core_methods":null,"recent_frontier":null,"evaluation_benchmark":null,"application":null},"primary_role":"","secondary_roles":[],"duplicate_status":"none","duplicate_of_paper_id":null,"evidence_level":"title_abstract","full_text_available":false,"label_confidence":"","notes":""}',
        "```",
        "",
    ]


def export_batches(candidates: list[dict[str, Any]], batch_dir: Path) -> dict[str, Any]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_topic[str(row["query_id"])].append(row)
    outputs = []
    for batch_index, topics in enumerate(BATCH_TOPICS, start=1):
        lines = [
            f"# V3.2 800-Row Labeling Batch {batch_index:02d}",
            "",
            "Copy completed JSON objects into `data/eval/manual_labels_v3_2_800.jsonl`, one object per line. Do not infer labels from selection reasons.",
            "",
        ]
        candidate_count = 0
        for topic in topics:
            rows = sorted(by_topic.get(topic, []), key=lambda row: (-float(row.get("selection_score") or 0.0), int(row["paper_id"])))
            if not rows:
                continue
            lines.extend([f"## Topic: {topic}", "", f"Query: {rows[0].get('query') or rows[0].get('topic')}", ""])
            for row in rows:
                candidate_count += 1
                lines.extend(candidate_block(row, candidate_count))
        path = batch_dir / f"batch_{batch_index:02d}.md"
        write_text(path, "\n".join(lines))
        outputs.append({"batch": f"batch_{batch_index:02d}", "path": str(path), "topics": topics, "candidate_count": candidate_count})
    return {"batch_dir": str(batch_dir), "file_count": len(outputs), "outputs": outputs}


def build_guide() -> str:
    return """# V3.2 800-Row Labeling Guide

Label into `data/eval/manual_labels_v3_2_800.jsonl`, one JSON object per line. Do not edit V2.1, V2.5, or selected-240 files.

## Required Fields

Use the existing manual-label schema:

- `schema_version`: use `v3.2_800_manual_label`
- `query_id`, `query`, `paper_id`, `title`
- `topic_match_score`, `reading_value_score`
- `beginner_fit_score`, `intermediate_fit_score`, `advanced_fit_score`, `expert_fit_score`
- `intent_scores.background`, `foundational`, `core_methods`, `recent_frontier`, `evaluation_benchmark`, `application`
- `primary_role`, `secondary_roles`
- `duplicate_status`, `duplicate_of_paper_id`
- `evidence_level`, `full_text_available`, `label_confidence`, `notes`

## Scores

All score fields are continuous numbers from `0.0` to `1.0`.

- `topic_match_score`: how directly the paper addresses the topic query.
- `reading_value_score`: how useful the paper is for a reading path on this topic.
- Difficulty fit scores: how appropriate the paper is for each reader level.
- Intent scores: how well the paper serves each reading-path role.

Use `0.0` for no fit, around `0.25` for weak fit, around `0.5` for partial fit, around `0.75` for strong fit, and `1.0` for canonical or excellent fit.

## Roles

Allowed `primary_role` and `secondary_roles` values:

- `background`
- `foundational`
- `core_methods`
- `recent_frontier`
- `evaluation_benchmark`
- `application`
- `negative`
- `duplicate`
- `uncertain`

Use `negative` when the paper should not be promoted for the topic. Use `duplicate` only when the candidate is substantially redundant with another paper.

## Duplicate Status

Allowed values:

- `none`
- `near_duplicate`
- `exact_duplicate`
- `uncertain`

Set `duplicate_of_paper_id` when the target duplicate paper is known; otherwise leave it `null`.

## Evidence and Confidence

Allowed `evidence_level` values:

- `title_only`
- `title_abstract`
- `title_abstract_intro_conclusion`
- `fulltext_available`

Allowed `label_confidence` values:

- `low`
- `medium`
- `high`

Use `notes` for a short rationale, especially for high-confidence positives, hard negatives, duplicates, and uncertain cases.

## Recommended Labeling Order

1. `batch_01.md`: scientific discovery, Bayesian optimization, causal representation, contrastive learning.
2. `batch_02.md`: diffusion, efficient transformers, graph neural networks, graph recommendation.
3. `batch_03.md`: LLM agents/evaluation, multimodal, recommendation systems.
4. `batch_04.md`: RAG, robot learning, self-supervised vision, transformer architecture.

Run `python scripts/validate_v3_2_800_labels.py` after each batch.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V3.2 800-row manual-labeling workflow assets.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES.relative_to(REPO_ROOT)))
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--guide-out", default=str(DEFAULT_GUIDE.relative_to(REPO_ROOT)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_jsonl(resolve_repo_path(args.candidates))
    batch_report = export_batches(candidates, resolve_repo_path(args.batch_dir))
    write_text(resolve_repo_path(args.guide_out), build_guide())
    print("V3.2 labeling workflow assets built")
    print(f"Batches: {batch_report['file_count']}")
    print(f"Guide: {resolve_repo_path(args.guide_out)}")


if __name__ == "__main__":
    main()
