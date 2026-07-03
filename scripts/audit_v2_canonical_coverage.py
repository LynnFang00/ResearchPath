import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = REPO_ROOT / "data" / "eval" / "v2_labeling_candidate_pool.jsonl"
DEFAULT_SELECTED = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data" / "eval" / "v2_canonical_coverage_audit.md"

REPLACEMENTS = [
    {
        "query_id": "v2_retrieval_augmented_generation",
        "old_paper_id": 9920,
        "new_paper_id": 46348,
        "reason": "Replaces an old retrieval-failure indexing paper with a clearer retrieval-enhanced neural generation paper from the candidate pool.",
        "source": "candidate_pool",
    },
    {
        "query_id": "v2_diffusion_image_generation",
        "old_paper_id": 11847,
        "new_paper_id": 313,
        "reason": "Replaces diffusion-weighted image quality benchmarking with a stronger text-to-image diffusion control paper.",
        "source": "candidate_pool",
    },
    {
        "query_id": "v2_diffusion_image_generation",
        "old_paper_id": 30257,
        "new_paper_id": 3876,
        "reason": "Replaces a generic vector-learning row with a text-to-image generation taxonomy/background paper.",
        "source": "candidate_pool",
    },
    {
        "query_id": "v2_efficient_transformers",
        "old_paper_id": 49227,
        "new_paper_id": 49828,
        "reason": "Replaces a NAS/agent transfer result with a more directly efficient-transformer candidate about shared attention weights.",
        "source": "candidate_pool",
    },
    {
        "query_id": "v2_robot_learning",
        "old_paper_id": 9017,
        "new_paper_id": 33621,
        "reason": "Replaces an unrelated older robot-control row with a core imitation-learning background paper.",
        "source": "candidate_pool",
    },
    {
        "query_id": "v2_ai_for_scientific_discovery",
        "old_paper_id": 1850,
        "new_paper_id": 3953,
        "reason": "Replaces a generic AI-management paper with a self-driving laboratory paper directly tied to autonomous scientific discovery.",
        "source": "candidate_pool",
    },
    {
        "query_id": "v2_ai_for_scientific_discovery",
        "old_paper_id": 35425,
        "new_paper_id": 36,
        "reason": "Replaces a general AI ethics row with a stronger autonomous-chemical-research paper using large language models.",
        "source": "candidate_pool",
    },
]

QUERY_TERMS = {
    "v2_transformer_architecture": {"attention", "transformer", "architecture", "vision", "sequence"},
    "v2_retrieval_augmented_generation": {"retrieval", "augmented", "generation", "rag", "knowledge", "dialogue"},
    "v2_graph_neural_networks": {"graph", "neural", "network", "gnn", "convolution"},
    "v2_contrastive_learning": {"contrastive", "self-supervised", "representation", "embedding"},
    "v2_bayesian_optimization": {"bayesian", "optimization", "surrogate", "acquisition"},
    "v2_large_language_model_agents": {"language", "model", "agent", "tool", "planning", "reasoning"},
    "v2_recommendation_systems": {"recommendation", "recommender", "collaborative", "ranking"},
    "v2_diffusion_image_generation": {"diffusion", "image", "generation", "generative", "text-to-image", "denoising"},
    "v2_ai_for_scientific_discovery": {"scientific", "discovery", "laboratory", "chemistry", "materials", "autonomous"},
    "v2_multimodal_learning": {"multimodal", "vision", "language", "representation", "cross-modal"},
    "v2_graph_recommendation": {"graph", "recommendation", "recommender", "user", "item"},
    "v2_efficient_transformers": {"efficient", "transformer", "attention", "lightweight", "fast"},
    "v2_llm_evaluation": {"language", "model", "evaluation", "benchmark", "llm"},
    "v2_self_supervised_vision": {"self-supervised", "vision", "contrastive", "representation", "image"},
    "v2_causal_representation_learning": {"causal", "representation", "learning", "invariant"},
    "v2_robot_learning": {"robot", "demonstration", "imitation", "learning", "reinforcement"},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def text_for(row: dict[str, Any]) -> str:
    return f"{row.get('title') or ''} {row.get('abstract') or ''}".lower()


def term_overlap(row: dict[str, Any]) -> int:
    terms = QUERY_TERMS.get(row["query_id"], set())
    text = text_for(row)
    return sum(1 for term in terms if term in text)


def likely_coverage(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").lower()
    year = row.get("year")
    citations = row.get("citation_count") or 0
    reasons = set(row.get("selection_reasons") or [])
    tags = set(row.get("candidate_source") or [])
    overlap = term_overlap(row)

    if "random_weak_negative" in reasons or "random_weak_negative" in tags:
        return "likely random/irrelevant negative"
    if "canonical_foundational_seed" in reasons:
        return "likely core/foundational positive"
    if any(token in title for token in ("survey", "taxonomy", "overview", "benchmark", "benchmarks", "evaluation")) and overlap >= 1:
        return "likely relevant survey/background"
    if citations >= 1000 and overlap >= 1:
        return "likely core/foundational positive"
    if isinstance(year, int) and year >= 2022 and overlap >= 2:
        return "likely recent frontier/application"
    if overlap >= 3 and citations >= 100:
        return "likely core/foundational positive"
    if overlap >= 2:
        return "likely recent frontier/application" if isinstance(year, int) and year >= 2018 else "likely relevant survey/background"
    if "hard_negative_candidate" in reasons or "hard_negative_candidate" in tags:
        return "likely hard negative"
    return "likely hard negative"


def apply_replacements(
    *,
    selected_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_by_query_paper = {(row["query_id"], int(row["paper_id"])): row for row in pool_rows}
    selected_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        selected_by_topic[row["query_id"]].append(row)

    changes: list[dict[str, Any]] = []
    for replacement in REPLACEMENTS:
        query_id = replacement["query_id"]
        topic_rows = selected_by_topic[query_id]
        old_index = next(
            (index for index, row in enumerate(topic_rows) if int(row["paper_id"]) == replacement["old_paper_id"]),
            None,
        )
        if old_index is None:
            continue
        replacement_row = pool_by_query_paper.get((query_id, replacement["new_paper_id"]))
        if replacement_row is None:
            continue
        old_row = topic_rows[old_index]
        new_row = dict(replacement_row)
        new_row["selection_reasons"] = ["canonical_coverage_replacement"]
        new_row["selection_policy"] = "balanced_v2_prelabeling_15_per_topic_canonical_audited"
        topic_rows[old_index] = new_row
        changes.append(
            {
                **replacement,
                "old_title": old_row["title"],
                "new_title": new_row["title"],
            }
        )

    output_rows: list[dict[str, Any]] = []
    for query_id in sorted(selected_by_topic):
        for index, row in enumerate(selected_by_topic[query_id], start=1):
            row = dict(row)
            row["labeling_packet_rank"] = index
            row["likely_coverage"] = likely_coverage(row)
            output_rows.append(row)
    return output_rows, changes


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_topic[row["query_id"]].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for query_id, topic_rows in rows_by_topic.items():
        coverage = Counter(row.get("likely_coverage") or likely_coverage(row) for row in topic_rows)
        provenance = Counter(source for row in topic_rows for source in row.get("source_provenance", []))
        summaries[query_id] = {
            "count": len(topic_rows),
            "coverage": dict(coverage),
            "provenance": dict(provenance),
            "core_positive_like": coverage.get("likely core/foundational positive", 0)
            + coverage.get("likely relevant survey/background", 0)
            + coverage.get("likely recent frontier/application", 0),
            "negative_like": coverage.get("likely hard negative", 0)
            + coverage.get("likely random/irrelevant negative", 0),
        }
    return summaries


def write_report(
    *,
    report_path: Path,
    rows: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> None:
    selected_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pool_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        selected_by_topic[row["query_id"]].append(row)
    for row in pool_rows:
        pool_by_topic[row["query_id"]].append(row)
    summaries = summarize(rows)

    lines = [
        "# V2 Canonical Coverage Audit",
        "",
        f"- Created at: `{datetime.now(UTC).isoformat()}`",
        "- This audit does not create relevance labels.",
        "- The selected file remains exactly `16` topics x `15` papers = `240` rows.",
        "- Likely coverage marks are heuristic pre-labeling triage, not gold labels.",
        "",
        "## Replacement Summary",
        "",
    ]
    if not changes:
        lines.append("No replacements were made.")
    else:
        lines.extend(["| Topic | Old Paper | New Paper | Source | Reason |", "|---|---|---|---|---|"])
        for change in changes:
            lines.append(
                f"| {change['query_id']} | `{change['old_paper_id']}` {change['old_title']} | "
                f"`{change['new_paper_id']}` {change['new_title']} | {change['source']} | {change['reason']} |"
            )

    lines.extend(
        [
            "",
            "## Topic Coverage Summary",
            "",
            "| Topic | Selected | Likely Positive/Background/Frontier | Likely Negative | Coverage Counts | Provenance |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for query_id in sorted(summaries):
        summary = summaries[query_id]
        lines.append(
            f"| {query_id} | {summary['count']} | {summary['core_positive_like']} | {summary['negative_like']} | "
            f"`{summary['coverage']}` | `{summary['provenance']}` |"
        )

    lines.extend(["", "## Per-Topic Selected Papers", ""])
    for query_id in sorted(selected_by_topic):
        lines.extend(
            [
                f"### {query_id}",
                "",
                f"- Candidate pool size: `{len(pool_by_topic[query_id])}`",
                "",
                "| Rank | Paper ID | Year | Likely Coverage | Selection Reasons | Title |",
                "|---:|---:|---:|---|---|---|",
            ]
        )
        for row in sorted(selected_by_topic[query_id], key=lambda item: item["labeling_packet_rank"]):
            lines.append(
                f"| {row['labeling_packet_rank']} | {row['paper_id']} | {row.get('year') or ''} | "
                f"{row.get('likely_coverage')} | `{row.get('selection_reasons')}` | {row['title']} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["query_id"] for row in rows)
    if len(rows) != 240:
        raise RuntimeError(f"Expected 240 selected rows, got {len(rows)}")
    if len(counts) != 16:
        raise RuntimeError(f"Expected 16 topics, got {len(counts)}")
    bad = {query_id: count for query_id, count in counts.items() if count != 15}
    if bad:
        raise RuntimeError(f"Expected 15 rows per topic, got {bad}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V2 selected labeling packets for canonical/core coverage.")
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--selected", default=str(DEFAULT_SELECTED))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_path = Path(args.pool)
    selected_path = Path(args.selected)
    report_path = Path(args.report)
    if not pool_path.is_absolute():
        pool_path = REPO_ROOT / pool_path
    if not selected_path.is_absolute():
        selected_path = REPO_ROOT / selected_path
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path

    pool_rows = load_jsonl(pool_path)
    selected_rows = load_jsonl(selected_path)
    audited_rows, changes = apply_replacements(selected_rows=selected_rows, pool_rows=pool_rows)
    validate(audited_rows)
    write_jsonl(audited_rows, selected_path)
    write_report(report_path=report_path, rows=audited_rows, changes=changes, pool_rows=pool_rows)
    print(json.dumps({"selected_rows": len(audited_rows), "replacement_count": len(changes), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
