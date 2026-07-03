import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "eval" / "v2_labeling_selected_240.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl"
DEFAULT_META_OUTPUT = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.meta.json"
DEFAULT_TOPIC_PACKET_DIR = REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets_by_topic"
DEFAULT_FULLTEXT_MANIFEST_GLOB = "data/fulltext/fulltext_manifest*.jsonl"

SCHEMA_VERSION = "v2.1"
MIN_ABSTRACT_WORDS = 80
DEFAULT_MAX_TOPIC_PACKET_CHARS = 750_000

REQUIRED_SCORE_FIELDS = [
    "topic_match_score",
    "reading_value_score",
    "beginner_fit_score",
    "intermediate_fit_score",
    "advanced_fit_score",
    "expert_fit_score",
]
INTENT_SCORE_FIELDS = [
    "background",
    "foundational",
    "core_methods",
    "recent_frontier",
    "evaluation_benchmark",
    "application",
]
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
DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]
DUPLICATE_STATUS_VALUES = ["none", "near_duplicate", "exact_duplicate", "uncertain"]
EVIDENCE_LEVELS = ["title_only", "title_abstract", "title_abstract_intro_conclusion", "fulltext_available"]
LABEL_CONFIDENCE_VALUES = ["low", "medium", "high"]

ANCHOR_CATEGORY_BY_LIKELY_COVERAGE = {
    "likely core/foundational positive": "core_or_foundational",
    "likely relevant survey/background": "background_or_survey",
    "likely recent frontier/application": "frontier_or_application",
    "likely hard negative": "hard_negative",
    "likely random/irrelevant negative": "random_or_irrelevant_negative",
}

ANCHOR_ORDER = [
    "core_or_foundational",
    "background_or_survey",
    "frontier_or_application",
    "hard_negative",
    "random_or_irrelevant_negative",
]

NO_TRUNCATION_POLICY = {
    "paragraphs_and_sections_are_never_truncated": True,
    "snippet_fields_are_omitted": True,
    "split_packets_instead_of_shortening_text": True,
    "full_text_default": "Do not include full text unless the abstract is missing, very short, or ambiguous.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected JSON object.")
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(payload: dict[str, Any], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text)


def load_fulltext_manifests(pattern: str) -> dict[int, dict[str, Any]]:
    manifest_lookup: dict[int, dict[str, Any]] = {}
    for path in sorted(REPO_ROOT.glob(pattern)):
        for row in load_jsonl(path):
            paper_id = row.get("paper_id")
            if isinstance(paper_id, int):
                existing = manifest_lookup.get(paper_id)
                if existing is None or (row.get("full_text_available") and not existing.get("full_text_available")):
                    manifest_lookup[paper_id] = row
    return manifest_lookup


def labeling_schema_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_path": "data/eval/v2_1_labeling_schema.md",
        "guide_path": "data/eval/v2_1_labeling_guide.md",
        "anchor_calibration_path": "data/eval/v2_1_anchor_calibration.md",
        "required_score_fields": REQUIRED_SCORE_FIELDS,
        "intent_score_fields": INTENT_SCORE_FIELDS,
        "role_values": ROLE_VALUES,
        "difficulty_levels": DIFFICULTY_LEVELS,
        "duplicate_status_values": DUPLICATE_STATUS_VALUES,
        "evidence_levels": EVIDENCE_LEVELS,
        "label_confidence_values": LABEL_CONFIDENCE_VALUES,
        "score_range": [0.0, 1.0],
        "no_truncation_policy": NO_TRUNCATION_POLICY,
    }


def score_anchor_summary() -> dict[str, str]:
    return {
        "0.00": "none / not applicable",
        "0.25": "weak",
        "0.50": "moderate",
        "0.75": "strong",
        "1.00": "excellent/core",
    }


def anchor_category(row: dict[str, Any]) -> str | None:
    coverage = str(row.get("likely_coverage") or "")
    return ANCHOR_CATEGORY_BY_LIKELY_COVERAGE.get(coverage)


def build_topic_anchors(topic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors_by_category: dict[str, dict[str, Any]] = {}
    for row in sorted(topic_rows, key=lambda item: int(item.get("labeling_packet_rank") or 9999)):
        category = anchor_category(row)
        if category is None or category in anchors_by_category:
            continue
        anchors_by_category[category] = {
            "anchor_category": category,
            "paper_id": row.get("paper_id"),
            "title": row.get("title"),
            "year": row.get("year"),
            "packet_rank": row.get("labeling_packet_rank"),
            "likely_coverage_heuristic_only": row.get("likely_coverage"),
            "selection_reasons": row.get("selection_reasons") or [],
            "anchor_note": "Calibration hint only; not a gold label.",
        }
    return [anchors_by_category[category] for category in ANCHOR_ORDER if category in anchors_by_category]


def build_packets(
    rows: list[dict[str, Any]],
    *,
    fulltext_lookup: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_topic[str(row["query_id"])].append(row)

    fulltext_lookup = fulltext_lookup or {}
    anchors_by_topic = {query_id: build_topic_anchors(topic_rows) for query_id, topic_rows in rows_by_topic.items()}

    packets: list[dict[str, Any]] = []
    for row in rows:
        packet = build_candidate_packet(row, fulltext_lookup=fulltext_lookup)
        packet["v2_1_packet_version"] = SCHEMA_VERSION
        packet["anchor_category_hint"] = anchor_category(row)
        packet["anchor_note"] = anchor_note(row)
        packet["labeling_instructions"] = {
            "do_not_edit_source_packet": "data/eval/v2_labeling_selected_240.jsonl",
            "write_labels_to": "data/eval/manual_labels_v2_1.jsonl",
            "do_not_use_anchor_hints_as_labels": True,
            "do_not_truncate_text": True,
        }
        packets.append(packet)

    return packets, anchors_by_topic


def build_candidate_packet(row: dict[str, Any], *, fulltext_lookup: dict[int, dict[str, Any]]) -> dict[str, Any]:
    paper_id = int(row["paper_id"])
    abstract = normalize_optional_text(row.get("abstract"))
    manifest = fulltext_lookup.get(paper_id)
    fulltext_sections = fulltext_sections_for(row, manifest)
    evidence = evidence_payload(abstract, manifest, fulltext_sections)
    identifiers = normalized_identifiers(row)
    urls = normalized_urls(row, identifiers, manifest)
    categories = list_value(row.get("categories"))

    packet: dict[str, Any] = {
        "query_id": row.get("query_id"),
        "query": row.get("query"),
        "target_audience": row.get("target_audience"),
        "paper_id": paper_id,
        "title": row.get("title"),
        "year": row.get("year"),
        "venue": row.get("venue"),
        "abstract": abstract,
        "sources_provenance": row.get("source_provenance") or [],
        "identifiers": identifiers,
        "source_url": urls["source_url"],
        "pdf_url": urls["pdf_url"],
        "selection_reasons": row.get("selection_reasons") or [],
        "likely_coverage": {
            "value": row.get("likely_coverage"),
            "heuristic_only": True,
            "warning": "Pre-labeling triage hint only; do not use as a gold label.",
        },
        "retrieval_ranks_by_method": row.get("retrieval_ranks_by_method") or {},
        "retrieval_scores_by_method": row.get("retrieval_scores_by_method") or {},
        "citation_count": row.get("citation_count"),
        "labeling_packet_rank": row.get("labeling_packet_rank"),
        "selection_policy": row.get("selection_policy"),
        "duplicate_title_cluster": {
            "duplicate_title_key": row.get("duplicate_title_key"),
            "duplicate_title_cluster_count": row.get("duplicate_title_cluster_count"),
        },
        "evidence_availability": evidence,
        "fulltext_sections": fulltext_sections,
    }

    add_optional(packet, "authors", row.get("authors"))
    add_optional(packet, "arxiv_categories", categories if has_arxiv_source(row) else None)
    add_optional(packet, "openalex_concepts_topics", categories if has_openalex_source(row) else None)
    add_optional(packet, "keywords", row.get("keywords"))
    add_optional(packet, "source_specific_metadata", source_specific_metadata(row, manifest))
    return packet


def normalized_identifiers(row: dict[str, Any]) -> dict[str, str | None]:
    identifiers = row.get("identifiers") if isinstance(row.get("identifiers"), dict) else {}
    return {
        "arxiv_id": first_identifier(identifiers, "arxiv") or extract_arxiv_id(row),
        "doi": normalize_optional_text(row.get("doi")),
        "openalex_id": first_identifier(identifiers, "openalex") or extract_openalex_id(row),
    }


def first_identifier(identifiers: dict[str, Any], key: str) -> str | None:
    value = identifiers.get(key)
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return None


def extract_arxiv_id(row: dict[str, Any]) -> str | None:
    for value in (row.get("external_id"), row.get("source_url"), row.get("url")):
        text = str(value or "")
        if text.startswith("arxiv:"):
            return text.removeprefix("arxiv:")
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#\s]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).removesuffix(".pdf")
    return None


def extract_openalex_id(row: dict[str, Any]) -> str | None:
    for value in (row.get("external_id"), row.get("source_url"), row.get("url")):
        text = str(value or "")
        if text.startswith("openalex:"):
            return text.removeprefix("openalex:")
        match = re.search(r"openalex\.org/(W\d+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def normalized_urls(
    row: dict[str, Any],
    identifiers: dict[str, str | None],
    manifest: dict[str, Any] | None,
) -> dict[str, str | None]:
    source_url = normalize_optional_text(row.get("source_url") or row.get("url"))
    pdf_url = normalize_optional_text(row.get("pdf_url"))

    if manifest:
        source_url = source_url or normalize_optional_text(manifest.get("source_url"))
        pdf_url = pdf_url or normalize_optional_text(manifest.get("source_url") if manifest.get("source_type") else None)

    arxiv_id = identifiers.get("arxiv_id")
    doi = identifiers.get("doi")
    openalex_id = identifiers.get("openalex_id")
    if not source_url and arxiv_id:
        source_url = f"https://arxiv.org/abs/{arxiv_id}"
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    if not source_url and openalex_id:
        source_url = f"https://openalex.org/{openalex_id}"
    if not source_url and doi:
        source_url = f"https://doi.org/{doi}"
    return {"source_url": source_url, "pdf_url": pdf_url}


def evidence_payload(
    abstract: str | None,
    manifest: dict[str, Any] | None,
    fulltext_sections: dict[str, Any],
) -> dict[str, Any]:
    has_abstract = bool(abstract)
    has_included_sections = bool(fulltext_sections)
    fulltext_available = bool(manifest and manifest.get("full_text_available"))
    if has_included_sections:
        level = "title_abstract_intro_conclusion" if has_abstract else "fulltext_available"
    elif has_abstract:
        level = "title_abstract"
    elif fulltext_available:
        level = "fulltext_available"
    else:
        level = "title_only"
    return {
        "level": level,
        "fulltext_available": fulltext_available,
        "included_fulltext_sections": sorted(fulltext_sections),
        "fulltext_rule": "Full text is included only when the abstract is missing, very short, or ambiguous.",
        "no_truncation": True,
    }


def fulltext_sections_for(row: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    abstract = normalize_optional_text(row.get("abstract"))
    if not should_include_fulltext_sections(abstract):
        return {}
    if not manifest or not manifest.get("full_text_available") or not manifest.get("text_path"):
        return {}
    text_path = Path(str(manifest["text_path"]))
    if not text_path.exists():
        return {}
    text = text_path.read_text(encoding="utf-8", errors="replace")
    sections: dict[str, Any] = {}
    headings = extract_section_headings(text)
    if headings:
        sections["section_headings"] = headings
    introduction = extract_named_section(text, ("introduction", "1 introduction", "1. introduction"))
    if introduction:
        sections["introduction"] = introduction
    conclusion = extract_named_section(
        text,
        ("conclusion", "conclusions", "discussion and conclusion", "conclusion and future work"),
    )
    if conclusion:
        sections["conclusion"] = conclusion
    return sections


def should_include_fulltext_sections(abstract: str | None) -> bool:
    if not abstract:
        return True
    words = re.findall(r"\w+", abstract)
    if len(words) < MIN_ABSTRACT_WORDS:
        return True
    lower = abstract.lower()
    ambiguous_phrases = [
        "abstract unavailable",
        "no abstract",
        "not available",
        "this paper is about",
        "we present a method",
    ]
    return any(phrase in lower for phrase in ambiguous_phrases)


def extract_section_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        clean = line.strip()
        if is_heading(clean):
            headings.append(clean)
    return dedupe_preserve_order(headings)


def extract_named_section(text: str, names: tuple[str, ...]) -> str | None:
    lines = text.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        clean = normalize_heading(line)
        if clean in names:
            start_index = index + 1
            break
    if start_index is None:
        return None

    section_lines: list[str] = []
    for line in lines[start_index:]:
        clean = line.strip()
        if section_lines and is_heading(clean):
            break
        section_lines.append(line.rstrip())
    section = "\n".join(section_lines).strip()
    return section or None


def is_heading(value: str) -> bool:
    if not value or len(value) > 120:
        return False
    normalized = normalize_heading(value)
    known = {
        "abstract",
        "introduction",
        "1 introduction",
        "1. introduction",
        "related work",
        "method",
        "methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
    }
    if normalized in known:
        return True
    return bool(re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,:;()/-]{2,}$", value))


def normalize_heading(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.rstrip(":")


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def source_specific_metadata(row: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field_name in ("external_id", "source_methods", "candidate_source", "appears_in_n_methods", "best_rank", "mean_rank"):
        if field_name in row:
            metadata[field_name] = row.get(field_name)
    if manifest:
        metadata["fulltext_manifest"] = {
            "source_type": manifest.get("source_type"),
            "source_url": manifest.get("source_url"),
            "status": manifest.get("status"),
            "text_char_count": manifest.get("text_char_count"),
        }
    return metadata


def anchor_note(row: dict[str, Any]) -> str | None:
    category = anchor_category(row)
    if not category:
        return None
    return f"Calibration anchor candidate: {category}. Heuristic only; not a gold label."


def add_optional(packet: dict[str, Any], field_name: str, value: Any) -> None:
    if value is None:
        return
    if value == [] or value == {}:
        return
    packet[field_name] = value


def list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def has_arxiv_source(row: dict[str, Any]) -> bool:
    return "arxiv" in set(row.get("source_provenance") or [])


def has_openalex_source(row: dict[str, Any]) -> bool:
    return "openalex" in set(row.get("source_provenance") or [])


def normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def build_topic_packet(query_id: str, rows: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> dict[str, Any]:
    query = rows[0].get("query") if rows else ""
    target_audience = rows[0].get("target_audience") if rows else ""
    return {
        "packet_version": SCHEMA_VERSION,
        "query_id": query_id,
        "query": query,
        "target_use": "personalized ML/AI reading path",
        "target_audience": target_audience,
        "label_schema_summary": labeling_schema_payload(),
        "score_anchor_summary": score_anchor_summary(),
        "topic_specific_anchor_calibration": anchors,
        "candidate_count": len(rows),
        "candidates": rows,
        "batching_policy": {
            "preferred_topics_per_packet": 1,
            "max_topics_per_packet": 2,
            "do_not_shorten_text_to_fit_batch": True,
            "split_packets_instead_of_truncating": True,
        },
    }


def write_topic_packets(
    *,
    packets: list[dict[str, Any]],
    anchors_by_topic: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    max_packet_chars: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        rows_by_topic[str(packet["query_id"])].append(packet)

    written_files: list[dict[str, Any]] = []
    for query_id, topic_rows in sorted(rows_by_topic.items()):
        topic_rows = sorted(topic_rows, key=lambda row: int(row.get("labeling_packet_rank") or 9999))
        topic_packet = build_topic_packet(query_id, topic_rows, anchors_by_topic.get(query_id, []))
        serialized = json.dumps(topic_packet, indent=2, ensure_ascii=True) + "\n"
        if len(serialized) <= max_packet_chars:
            path = output_dir / f"{query_id}.json"
            size = write_json(topic_packet, path)
            written_files.append({"path": str(path), "query_id": query_id, "rows": len(topic_rows), "characters": size})
            continue

        for part_index, batch_rows in enumerate(split_rows_without_truncating(topic_packet, topic_rows, max_packet_chars), start=1):
            packet_part = build_topic_packet(query_id, batch_rows, anchors_by_topic.get(query_id, []))
            packet_part["packet_part"] = part_index
            packet_part["split_reason"] = "Packet exceeded preferred size; text was not shortened."
            path = output_dir / f"{query_id}_part_{part_index:02d}.json"
            size = write_json(packet_part, path)
            written_files.append({"path": str(path), "query_id": query_id, "rows": len(batch_rows), "characters": size})
    return {
        "output_dir": str(output_dir),
        "file_count": len(written_files),
        "files": written_files,
    }


def split_rows_without_truncating(topic_packet: dict[str, Any], rows: list[dict[str, Any]], max_packet_chars: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        trial = current + [row]
        trial_packet = {**topic_packet, "candidates": trial, "candidate_count": len(trial)}
        trial_size = len(json.dumps(trial_packet, indent=2, ensure_ascii=True)) + 1
        if current and trial_size > max_packet_chars:
            batches.append(current)
            current = [row]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def validate_same_candidates(source_rows: list[dict[str, Any]], packet_rows: list[dict[str, Any]]) -> None:
    source_keys = [(row.get("query_id"), row.get("paper_id")) for row in source_rows]
    packet_keys = [(row.get("query_id"), row.get("paper_id")) for row in packet_rows]
    if source_keys != packet_keys:
        raise RuntimeError("V2.1 packet changed the selected candidate order or membership.")


def validate_no_truncated_generated_fields(packet_rows: list[dict[str, Any]]) -> None:
    forbidden_fields = {"abstract_snippet", "snippet"}
    for row in packet_rows:
        present = forbidden_fields & set(row)
        if present:
            raise RuntimeError(f"Generated packet contains forbidden truncated fields for paper_id={row.get('paper_id')}: {sorted(present)}")
        abstract = row.get("abstract")
        if isinstance(abstract, str) and abstract.endswith("..."):
            raise RuntimeError(f"Generated packet abstract appears truncated for paper_id={row.get('paper_id')}")


def build_report(
    *,
    input_path: Path,
    output_path: Path,
    source_rows: list[dict[str, Any]],
    packet_rows: list[dict[str, Any]],
    anchors_by_topic: dict[str, list[dict[str, Any]]],
    topic_packet_report: dict[str, Any],
) -> dict[str, Any]:
    rows_by_topic = defaultdict(int)
    evidence_levels = defaultdict(int)
    fulltext_section_rows = 0
    for row in packet_rows:
        rows_by_topic[str(row["query_id"])] += 1
        evidence = row.get("evidence_availability") or {}
        evidence_levels[str(evidence.get("level"))] += 1
        if row.get("fulltext_sections"):
            fulltext_section_rows += 1
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "source_selected_file": str(input_path),
        "output_packet_file": str(output_path),
        "source_selected_file_frozen": True,
        "labels_created": False,
        "models_trained": False,
        "row_count": len(packet_rows),
        "topic_count": len(rows_by_topic),
        "rows_by_topic": dict(sorted(rows_by_topic.items())),
        "anchors_by_topic": {query_id: len(anchors) for query_id, anchors in sorted(anchors_by_topic.items())},
        "evidence_levels": dict(sorted(evidence_levels.items())),
        "rows_with_included_fulltext_sections": fulltext_section_rows,
        "topic_packets": topic_packet_report,
        "label_schema": labeling_schema_payload(),
        "no_truncation_policy": NO_TRUNCATION_POLICY,
        "notes": [
            "The V2 selected 240 file is read-only input for this builder.",
            "Anchors are pre-labeling calibration hints, not labels.",
            "Candidate query_id/paper_id order and membership are preserved exactly.",
            "Generated packets omit snippet fields and do not truncate abstracts or included full-text sections.",
        ],
    }


def clean_topic_packet_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.json"):
        child.unlink()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V2.1 labeling packets from the frozen V2 selected packet.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--meta-output", default=str(DEFAULT_META_OUTPUT))
    parser.add_argument("--topic-packet-dir", default=str(DEFAULT_TOPIC_PACKET_DIR))
    parser.add_argument("--fulltext-manifest-glob", default=DEFAULT_FULLTEXT_MANIFEST_GLOB)
    parser.add_argument("--max-topic-packet-chars", type=int, default=DEFAULT_MAX_TOPIC_PACKET_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    output_path = resolve_repo_path(args.output)
    meta_output_path = resolve_repo_path(args.meta_output)
    topic_packet_dir = resolve_repo_path(args.topic_packet_dir)

    source_rows = load_jsonl(input_path)
    fulltext_lookup = load_fulltext_manifests(args.fulltext_manifest_glob)
    packet_rows, anchors_by_topic = build_packets(source_rows, fulltext_lookup=fulltext_lookup)
    validate_same_candidates(source_rows, packet_rows)
    validate_no_truncated_generated_fields(packet_rows)

    write_jsonl(packet_rows, output_path)
    clean_topic_packet_dir(topic_packet_dir)
    topic_packet_report = write_topic_packets(
        packets=packet_rows,
        anchors_by_topic=anchors_by_topic,
        output_dir=topic_packet_dir,
        max_packet_chars=args.max_topic_packet_chars,
    )
    report = build_report(
        input_path=input_path,
        output_path=output_path,
        source_rows=source_rows,
        packet_rows=packet_rows,
        anchors_by_topic=anchors_by_topic,
        topic_packet_report=topic_packet_report,
    )
    write_json(report, meta_output_path)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
