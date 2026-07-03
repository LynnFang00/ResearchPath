import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets_by_topic"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "eval" / "v2_1_chatgpt_upload"
DEFAULT_QUERY_SET = REPO_ROOT / "data" / "eval" / "query_set_v2_seed.json"
DEFAULT_MAX_FILE_CHARS = 180_000

ROLE_VALUES = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
    "negative",
    "duplicate",
    "uncertain",
]
EVIDENCE_LEVELS = ["title_only", "title_abstract", "title_abstract_intro_conclusion", "fulltext_available"]
LABEL_CONFIDENCE_VALUES = ["low", "medium", "high"]
DUPLICATE_STATUS_VALUES = ["none", "near_duplicate", "exact_duplicate", "uncertain"]
INTENT_SCORE_FIELDS = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
SCORE_FIELDS = [
    "topic_match_score",
    "reading_value_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ordered_query_ids(query_set_path: Path) -> list[str]:
    queries = load_json(query_set_path)
    return [str(item["query_id"]) for item in queries]


def render_topic_file(packet: dict[str, Any], *, part_index: int | None = None, part_count: int | None = None) -> str:
    candidates = packet["candidates"]
    lines: list[str] = []
    lines.extend(
        [
            f"# ResearchPath V2.1 Labeling Packet: {packet['query_id']}",
            "",
            "Upload this file to ChatGPT and ask:",
            "",
            "> Label this V2.1 topic packet. Return JSONL labels only.",
            "",
            "Do not return Markdown, explanations, comments, or code fences. Return exactly one JSON object per line.",
            "",
        ]
    )
    if part_index is not None and part_count is not None:
        lines.extend(
            [
                f"Packet part: {part_index} of {part_count}",
                "This topic was split for upload size. Label only the candidates in this file.",
                "",
            ]
        )

    lines.extend(render_schema_section())
    lines.extend(render_topic_section(packet))
    lines.extend(render_anchor_section(packet.get("topic_specific_anchor_calibration") or []))
    lines.extend(render_candidate_section(candidates))
    return "\n".join(lines).rstrip() + "\n"


def render_schema_section() -> list[str]:
    schema_example = {
        "schema_version": "v2.1",
        "query_id": "v2_example_topic",
        "query": "example query",
        "paper_id": 123,
        "title": "Example Paper",
        "topic_match_score": 0.0,
        "reading_value_score": 0.0,
        "beginner_fit_score": 0.0,
        "intermediate_fit_score": 0.0,
        "advanced_fit_score": 0.0,
        "expert_fit_score": 0.0,
        "intent_scores": {field: 0.0 for field in INTENT_SCORE_FIELDS},
        "primary_role": "uncertain",
        "secondary_roles": [],
        "duplicate_status": "none",
        "duplicate_of_paper_id": None,
        "evidence_level": "title_abstract",
        "full_text_available": False,
        "label_confidence": "medium",
        "notes": "Short rationale.",
    }
    return [
        "## V2.1 Label Schema",
        "",
        "Return one JSONL row per candidate with this shape:",
        "",
        "```json",
        json.dumps(schema_example, indent=2, ensure_ascii=False),
        "```",
        "",
        "Required score fields:",
        bullet_list(SCORE_FIELDS),
        "",
        "Required `intent_scores` fields:",
        bullet_list(INTENT_SCORE_FIELDS),
        "",
        "Allowed `primary_role` and `secondary_roles` values:",
        bullet_list(ROLE_VALUES),
        "",
        "Allowed `duplicate_status` values:",
        bullet_list(DUPLICATE_STATUS_VALUES),
        "",
        "Allowed `evidence_level` values:",
        bullet_list(EVIDENCE_LEVELS),
        "",
        "Allowed `label_confidence` values:",
        bullet_list(LABEL_CONFIDENCE_VALUES),
        "",
        "## Score Anchors",
        "",
        "- 0.00 = none / not applicable",
        "- 0.25 = weak",
        "- 0.50 = moderate",
        "- 0.75 = strong",
        "- 1.00 = excellent/core",
        "",
        "All scores must be numbers in `[0, 1]`. Use the anchors by default; intermediate decimals are allowed when useful.",
        "",
    ]


def render_topic_section(packet: dict[str, Any]) -> list[str]:
    return [
        "## Topic",
        "",
        f"- query_id: `{packet['query_id']}`",
        f"- query: {packet['query']}",
        f"- target use: {packet.get('target_use', 'personalized ML/AI reading path')}",
        f"- target audience: {packet.get('target_audience', '')}",
        f"- selected candidates in this file: `{len(packet['candidates'])}`",
        "",
    ]


def render_anchor_section(anchors: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Topic-Specific Anchor Calibration",
        "",
        "These anchors are calibration hints only. They are not gold labels and should not be copied as labels.",
        "",
    ]
    if not anchors:
        lines.extend(["No topic-specific anchors were available.", ""])
        return lines
    for anchor in anchors:
        lines.extend(
            [
                f"### Anchor: {anchor.get('anchor_category')}",
                "",
                f"- paper_id: `{anchor.get('paper_id')}`",
                f"- title: {anchor.get('title')}",
                f"- year: {anchor.get('year')}",
                f"- packet_rank: `{anchor.get('packet_rank')}`",
                f"- likely_coverage heuristic only: {anchor.get('likely_coverage_heuristic_only')}",
                f"- selection_reasons: `{json.dumps(anchor.get('selection_reasons') or [], ensure_ascii=False)}`",
                f"- note: {anchor.get('anchor_note')}",
                "",
            ]
        )
    return lines


def render_candidate_section(candidates: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Selected Candidate Papers",
        "",
        f"Exactly `{len(candidates)}` selected candidates are included below.",
        "",
    ]
    for index, candidate in enumerate(candidates, start=1):
        lines.extend(render_candidate(candidate, index=index))
    return lines


def render_candidate(candidate: dict[str, Any], *, index: int) -> list[str]:
    lines = [
        f"### Candidate {index}: paper_id {candidate.get('paper_id')}",
        "",
        f"- paper_id: `{candidate.get('paper_id')}`",
        f"- title: {candidate.get('title')}",
        f"- year: {value_or_blank(candidate.get('year'))}",
        f"- venue: {value_or_blank(candidate.get('venue'))}",
        f"- source/provenance: `{json.dumps(candidate.get('sources_provenance') or [], ensure_ascii=False)}`",
    ]
    identifiers = candidate.get("identifiers") if isinstance(candidate.get("identifiers"), dict) else {}
    lines.extend(
        [
            f"- arxiv_id: {value_or_blank(identifiers.get('arxiv_id'))}",
            f"- doi: {value_or_blank(identifiers.get('doi'))}",
            f"- openalex_id: {value_or_blank(identifiers.get('openalex_id'))}",
            f"- source_url: {value_or_blank(candidate.get('source_url'))}",
            f"- pdf_url: {value_or_blank(candidate.get('pdf_url'))}",
            f"- citation_count: {value_or_blank(candidate.get('citation_count'))}",
            f"- selection_reasons: `{json.dumps(candidate.get('selection_reasons') or [], ensure_ascii=False)}`",
        ]
    )
    likely = candidate.get("likely_coverage") if isinstance(candidate.get("likely_coverage"), dict) else {}
    lines.extend(
        [
            f"- likely_coverage heuristic only: {value_or_blank(likely.get('value'))}",
            f"- retrieval ranks by method: `{json.dumps(candidate.get('retrieval_ranks_by_method') or {}, ensure_ascii=False)}`",
            f"- retrieval scores by method: `{json.dumps(candidate.get('retrieval_scores_by_method') or {}, ensure_ascii=False)}`",
            f"- duplicate-title cluster info: `{json.dumps(candidate.get('duplicate_title_cluster') or {}, ensure_ascii=False)}`",
            f"- evidence availability: `{json.dumps(candidate.get('evidence_availability') or {}, ensure_ascii=False)}`",
        ]
    )
    if candidate.get("anchor_note"):
        lines.append(f"- canonical/anchor note: {candidate['anchor_note']}")
    lines.extend(["", "#### Abstract", "", value_or_blank(candidate.get("abstract")), ""])

    fulltext_sections = candidate.get("fulltext_sections")
    if isinstance(fulltext_sections, dict) and fulltext_sections:
        lines.extend(["#### Included Full-Text Evidence", ""])
        section_headings = fulltext_sections.get("section_headings")
        if section_headings:
            lines.extend(["Section headings:", "", "```text", "\n".join(str(item) for item in section_headings), "```", ""])
        introduction = fulltext_sections.get("introduction")
        if introduction:
            lines.extend(["Introduction:", "", str(introduction), ""])
        conclusion = fulltext_sections.get("conclusion")
        if conclusion:
            lines.extend(["Conclusion:", "", str(conclusion), ""])
    return lines


def split_packet_if_needed(packet: dict[str, Any], max_file_chars: int) -> list[str]:
    full_content = render_topic_file(packet)
    if len(full_content) <= max_file_chars:
        return [full_content]

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for candidate in packet["candidates"]:
        trial_packet = {**packet, "candidates": current + [candidate]}
        if current and len(render_topic_file(trial_packet)) > max_file_chars:
            batches.append(current)
            current = [candidate]
        else:
            current.append(candidate)
    if current:
        batches.append(current)

    rendered: list[str] = []
    for index, batch in enumerate(batches, start=1):
        part_packet = {**packet, "candidates": batch}
        rendered.append(render_topic_file(part_packet, part_index=index, part_count=len(batches)))
    return rendered


def topic_filename(topic_index: int, query_id: str, *, part_index: int | None = None) -> str:
    base = f"topic_{topic_index:02d}_{safe_slug(query_id)}"
    if part_index is not None:
        base = f"{base}_part{part_index}"
    return f"{base}.md"


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def bullet_list(values: list[str]) -> str:
    return "\n".join(f"- `{value}`" for value in values)


def value_or_blank(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.md"):
        child.unlink()


def build_readme(output_dir: Path, generated_files: list[dict[str, Any]]) -> str:
    file_lines = "\n".join(f"- `{item['name']}`: {item['candidate_count']} candidates" for item in generated_files)
    return f"""# ResearchPath V2.1 ChatGPT Upload Packets

Generated at: `{datetime.now(UTC).isoformat()}`

These files are upload-ready topic packets for assistant-assisted, human-reviewed V2.1 labeling.

## Workflow

1. Upload one topic file to ChatGPT.
2. Ask: "Label this V2.1 topic packet. Return JSONL labels only."
3. Save ChatGPT's returned JSONL to a temporary file, for example `data/eval/tmp_v2_1_topic_labels.jsonl`.
4. Append using the V2.1 append script:

```powershell
.\\backend\\.venv\\Scripts\\python.exe scripts\\append_v2_1_labels.py --input data\\eval\\tmp_v2_1_topic_labels.jsonl
```

5. Run the V2.1 validation script:

```powershell
.\\backend\\.venv\\Scripts\\python.exe scripts\\validate_v2_1_labels.py --labels data\\eval\\manual_labels_v2_1.jsonl --packet data\\eval\\v2_1_labeling_packets.jsonl
```

6. Repeat for all 16 topics.

Do not edit `data/eval/v2_labeling_selected_240.jsonl`. Do not treat unreviewed auto-generated labels as gold labels.

## Upload Files

{file_lines}
"""


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChatGPT-upload-ready Markdown files for V2.1 labeling.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--query-set", default=str(DEFAULT_QUERY_SET))
    parser.add_argument("--max-file-chars", type=int, default=DEFAULT_MAX_FILE_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_repo_path(args.input_dir)
    output_dir = resolve_repo_path(args.output_dir)
    query_set_path = resolve_repo_path(args.query_set)

    clean_output_dir(output_dir)
    query_ids = ordered_query_ids(query_set_path)
    generated_files: list[dict[str, Any]] = []
    for topic_index, query_id in enumerate(query_ids, start=1):
        packet_path = input_dir / f"{query_id}.json"
        if not packet_path.exists():
            raise FileNotFoundError(f"Missing topic packet: {packet_path}")
        packet = load_json(packet_path)
        parts = split_packet_if_needed(packet, args.max_file_chars)
        for part_number, content in enumerate(parts, start=1):
            part_index = part_number if len(parts) > 1 else None
            filename = topic_filename(topic_index, query_id, part_index=part_index)
            write_text(output_dir / filename, content)
            generated_files.append(
                {
                    "name": filename,
                    "query_id": query_id,
                    "candidate_count": len(packet["candidates"]) if len(parts) == 1 else content.count("### Candidate "),
                    "characters": len(content),
                }
            )

    readme = build_readme(output_dir, generated_files)
    write_text(output_dir / "README.md", readme)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": generated_files,
                "readme": str(output_dir / "README.md"),
                "labels_created": False,
                "models_trained": False,
                "evaluation_run": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
