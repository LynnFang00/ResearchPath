import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "eval" / "v2_labeling_candidate_pool.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data" / "eval" / "v2_labeling_selection_report.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def decade_bucket(year: Any) -> str:
    if not isinstance(year, int):
        return "unknown"
    if year < 2010:
        return "pre-2010"
    if year < 2015:
        return "2010-2014"
    if year < 2020:
        return "2015-2019"
    if year < 2023:
        return "2020-2022"
    return "2023+"


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("best_rank") if row.get("best_rank") is not None else 9999,
        -(row.get("appears_in_n_methods") or 0),
        -(row.get("citation_count") or 0),
        row.get("paper_id") or 0,
    )


def method_rank(row: dict[str, Any], method: str) -> int:
    ranks = row.get("retrieval_ranks_by_method") or {}
    value = ranks.get(method)
    return int(value) if value is not None else 9999


def has_tag(row: dict[str, Any], tag: str) -> bool:
    return tag in set(row.get("candidate_source") or [])


def has_source(row: dict[str, Any], source: str) -> bool:
    return source in set(row.get("source_provenance") or [])


def summarize_topic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_method_counter: Counter[str] = Counter()
    candidate_source_counter: Counter[str] = Counter()
    provenance_counter: Counter[str] = Counter()
    year_counter: Counter[str] = Counter()
    for row in rows:
        source_method_counter.update(row.get("source_methods") or [])
        candidate_source_counter.update(row.get("candidate_source") or [])
        provenance_counter.update(row.get("source_provenance") or [])
        year_counter[decade_bucket(row.get("year"))] += 1

    duplicate_rows = [row for row in rows if (row.get("duplicate_title_cluster_count") or 0) > 1]
    duplicate_clusters = {
        row.get("duplicate_title_key")
        for row in duplicate_rows
        if row.get("duplicate_title_key")
    }
    return {
        "candidate_count": len(rows),
        "source_methods": dict(sorted(source_method_counter.items())),
        "candidate_source": dict(sorted(candidate_source_counter.items())),
        "source_provenance": dict(sorted(provenance_counter.items())),
        "year_distribution": dict(sorted(year_counter.items())),
        "duplicate_title_row_count": len(duplicate_rows),
        "duplicate_title_cluster_count": len(duplicate_clusters),
        "has_abstract_count": sum(1 for row in rows if str(row.get("abstract") or "").strip()),
        "citation_count_available": sum(1 for row in rows if row.get("citation_count") is not None),
        "citation_count_positive": sum(1 for row in rows if (row.get("citation_count") or 0) > 0),
    }


def add_first(
    *,
    selected: list[dict[str, Any]],
    selected_ids: set[int],
    rows: list[dict[str, Any]],
    limit: int,
    reason: str,
    predicate: Callable[[dict[str, Any]], bool],
    key: Callable[[dict[str, Any]], Any] = sort_key,
) -> None:
    for row in sorted((item for item in rows if predicate(item)), key=key):
        if len(selected) >= 15:
            return
        paper_id = int(row["paper_id"])
        if paper_id in selected_ids:
            continue
        selected_row = dict(row)
        selected_row.setdefault("selection_reasons", [])
        selected_row["selection_reasons"] = list(selected_row["selection_reasons"]) + [reason]
        selected.append(selected_row)
        selected_ids.add(paper_id)
        if sum(reason in item.get("selection_reasons", []) for item in selected) >= limit:
            return


def ensure_condition(
    *,
    selected: list[dict[str, Any]],
    selected_ids: set[int],
    rows: list[dict[str, Any]],
    reason: str,
    selected_predicate: Callable[[dict[str, Any]], bool],
    candidate_predicate: Callable[[dict[str, Any]], bool],
    key: Callable[[dict[str, Any]], Any] = sort_key,
) -> None:
    if any(selected_predicate(row) for row in selected):
        return
    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=1,
        reason=reason,
        predicate=candidate_predicate,
        key=key,
    )


def select_topic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=3,
        reason="canonical_foundational_seed",
        predicate=lambda row: has_tag(row, "canonical_seed"),
        key=lambda row: (row.get("year") or 9999, sort_key(row)),
    )
    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=2,
        reason="top_bm25",
        predicate=lambda row: method_rank(row, "bm25") <= 10,
        key=lambda row: (method_rank(row, "bm25"), sort_key(row)),
    )
    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=2,
        reason="top_embedding_or_faiss",
        predicate=lambda row: min(method_rank(row, "embedding"), method_rank(row, "faiss_embedding")) <= 10,
        key=lambda row: (min(method_rank(row, "embedding"), method_rank(row, "faiss_embedding")), sort_key(row)),
    )
    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=2,
        reason="top_hybrid",
        predicate=lambda row: method_rank(row, "hybrid") <= 10,
        key=lambda row: (method_rank(row, "hybrid"), sort_key(row)),
    )
    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=2,
        reason="deeper_rank_candidate",
        predicate=lambda row: any(str(tag).endswith("_deeper") for tag in row.get("candidate_source") or []),
        key=lambda row: (row.get("best_rank") or 9999, row.get("paper_id") or 0),
    )
    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=2,
        reason="hard_negative_candidate",
        predicate=lambda row: has_tag(row, "hard_negative_candidate"),
        key=lambda row: (row.get("best_rank") or 9999, row.get("appears_in_n_methods") or 0, row.get("paper_id") or 0),
    )
    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=2,
        reason="random_weak_negative",
        predicate=lambda row: has_tag(row, "random_weak_negative"),
        key=lambda row: (row.get("paper_id") or 0),
    )

    ensure_condition(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        reason="source_balance_arxiv",
        selected_predicate=lambda row: has_source(row, "arxiv"),
        candidate_predicate=lambda row: has_source(row, "arxiv"),
    )
    ensure_condition(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        reason="source_balance_openalex",
        selected_predicate=lambda row: has_source(row, "openalex"),
        candidate_predicate=lambda row: has_source(row, "openalex"),
    )
    ensure_condition(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        reason="year_balance_older",
        selected_predicate=lambda row: isinstance(row.get("year"), int) and row["year"] <= 2017,
        candidate_predicate=lambda row: isinstance(row.get("year"), int) and row["year"] <= 2017,
        key=lambda row: (row.get("year") or 9999, sort_key(row)),
    )
    ensure_condition(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        reason="year_balance_recent",
        selected_predicate=lambda row: isinstance(row.get("year"), int) and row["year"] >= 2022,
        candidate_predicate=lambda row: isinstance(row.get("year"), int) and row["year"] >= 2022,
        key=lambda row: (-(row.get("year") or 0), sort_key(row)),
    )

    add_first(
        selected=selected,
        selected_ids=selected_ids,
        rows=rows,
        limit=15,
        reason="quality_diversity_fill",
        predicate=lambda row: True,
    )

    return selected[:15]


def reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get("selection_reasons") or [])
    return dict(sorted(counter.items()))


def write_report(
    *,
    topic_summaries: dict[str, dict[str, Any]],
    selected_by_topic: dict[str, list[dict[str, Any]]],
    output_path: Path,
    input_path: Path,
    selected_path: Path,
) -> None:
    lines = [
        "# V2 Labeling Selection Report",
        "",
        f"- Created at: `{datetime.now(UTC).isoformat()}`",
        f"- Candidate pool: `{input_path}`",
        f"- Selected output: `{selected_path}`",
        f"- Topics: `{len(topic_summaries)}`",
        f"- Selected candidates: `{sum(len(rows) for rows in selected_by_topic.values())}`",
        "",
        "## Selection Policy",
        "",
        "Each topic is capped at 15 papers. The selector first includes canonical/foundational seed papers where available, then adds top BM25, top embedding/FAISS, and top hybrid candidates. It then adds deeper-rank candidates, hard-negative candidates, random weak negatives, and source/year balance rows where available. Remaining slots are filled by high-quality diverse retrieval candidates.",
        "",
        "No relevance labels are created in this file; these are packets for human labeling.",
        "",
        "## Topic Pool Summary",
        "",
        "| Query ID | Pool | Selected | Abstracts | Citation Count Available | Citation Count > 0 | Duplicate Title Rows | arXiv | OpenAlex | Seed | Year Buckets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for query_id, summary in topic_summaries.items():
        provenance = summary["source_provenance"]
        lines.append(
            f"| {query_id} | {summary['candidate_count']} | {len(selected_by_topic[query_id])} | "
            f"{summary['has_abstract_count']} | {summary['citation_count_available']} | "
            f"{summary['citation_count_positive']} | "
            f"{summary['duplicate_title_row_count']} | {provenance.get('arxiv', 0)} | "
            f"{provenance.get('openalex', 0)} | {provenance.get('seed', 0)} | "
            f"`{summary['year_distribution']}` |"
        )
    lines.extend(["", "## Topic Selection Breakdown", ""])
    for query_id, rows in selected_by_topic.items():
        summary = topic_summaries[query_id]
        lines.extend(
            [
                f"### {query_id}",
                "",
                f"- Pool size: `{summary['candidate_count']}`",
                f"- Source methods: `{summary['source_methods']}`",
                f"- Candidate source tags: `{summary['candidate_source']}`",
                f"- Selected reason counts: `{reason_counts(rows)}`",
                "",
                "| Paper ID | Year | Sources | Selection Reasons | Title |",
                "|---:|---:|---|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['paper_id']} | {row.get('year') or ''} | "
                f"`{row.get('source_provenance')}` | `{row.get('selection_reasons')}` | {row['title']} |"
            )
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select balanced V2 labeling packets from candidate pool.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path

    rows = load_jsonl(input_path)
    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_topic[str(row["query_id"])].append(row)

    topic_summaries = {query_id: summarize_topic(topic_rows) for query_id, topic_rows in rows_by_topic.items()}
    selected_by_topic = {query_id: select_topic(topic_rows) for query_id, topic_rows in rows_by_topic.items()}
    selected_rows: list[dict[str, Any]] = []
    for query_id, topic_rows in selected_by_topic.items():
        if len(topic_rows) != 15:
            raise RuntimeError(f"Expected 15 selected rows for {query_id}, got {len(topic_rows)}")
        for index, row in enumerate(topic_rows, start=1):
            row = dict(row)
            row["labeling_packet_rank"] = index
            row["selection_policy"] = "balanced_v2_prelabeling_15_per_topic"
            selected_rows.append(row)

    write_jsonl(selected_rows, output_path)
    write_report(
        topic_summaries=topic_summaries,
        selected_by_topic=selected_by_topic,
        output_path=report_path,
        input_path=input_path,
        selected_path=output_path,
    )
    print(json.dumps(
        {
            "input_rows": len(rows),
            "topics": len(rows_by_topic),
            "selected_rows": len(selected_rows),
            "output": str(output_path),
            "report": str(report_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
