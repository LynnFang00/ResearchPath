import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = REPO_ROOT / "data" / "eval" / "manual_label_pool_v1.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "fulltext" / "fulltext_manifest_v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "eval" / "fulltext_label_packets_v1.jsonl"
DEFAULT_EXCERPT_CHARS = 1800
DEFAULT_QUERY_EXCERPT_CHARS = 1800
DEFAULT_MIN_FULLTEXT_TOP_K = 7
DEFAULT_EXPAND_TO = 20
DEFAULT_MIN_SELECTED = 8
MIN_USEFUL_TEXT_CHARS = 50
MIN_TITLE_TOKEN_OVERLAP = 0.18
MIN_TITLE_TOKEN_HITS = 2
MIN_TITLE_TERMS_FOR_SANITY_CHECK = 3
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
TITLE_STOPWORDS = QUERY_STOPWORDS | {
    "using",
    "via",
    "based",
    "toward",
    "towards",
    "learning",
    "model",
    "models",
    "generation",
    "neural",
    "large",
    "language",
}


@dataclass(frozen=True)
class Excerpt:
    text: str
    found: bool


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File was not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_by_paper_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for row in rows:
        paper_id = int(row["paper_id"])
        existing = lookup.get(paper_id)
        if existing is None or manifest_row_priority(row) >= manifest_row_priority(existing):
            lookup[paper_id] = row
    return lookup


def manifest_row_priority(row: dict[str, Any]) -> int:
    if bool(row.get("full_text_available", False)):
        return 3
    if row.get("status") == "error":
        return 2
    return 1


def build_packets(
    *,
    pool_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    per_query: int = 10,
    min_fulltext_top_k: int = DEFAULT_MIN_FULLTEXT_TOP_K,
    expand_to: int = DEFAULT_EXPAND_TO,
    min_selected: int = DEFAULT_MIN_SELECTED,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    query_excerpt_chars: int = DEFAULT_QUERY_EXCERPT_CHARS,
    query_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    manifest_lookup = manifest_by_paper_id(manifest_rows)
    rows_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        if query_ids is not None and str(row["query_id"]) not in query_ids:
            continue
        rows_by_query[str(row["query_id"])].append(row)

    packets: list[dict[str, Any]] = []
    for query_rows in rows_by_query.values():
        for row, reason in select_rows_for_labeling(
            query_rows,
            manifest_lookup,
            per_query=per_query,
            min_fulltext_top_k=min_fulltext_top_k,
            expand_to=expand_to,
            min_selected=min_selected,
        ):
            manifest = manifest_lookup.get(int(row["paper_id"]), {})
            packets.append(
                packet_for_candidate(
                    row,
                    manifest,
                    selection_reason=reason,
                    excerpt_chars=excerpt_chars,
                    query_excerpt_chars=query_excerpt_chars,
                )
            )
    return packets


def select_rows_for_labeling(
    query_rows: list[dict[str, Any]],
    manifest_lookup: dict[int, dict[str, Any]],
    *,
    per_query: int,
    min_fulltext_top_k: int,
    expand_to: int,
    min_selected: int,
) -> list[tuple[dict[str, Any], str]]:
    if per_query <= 0:
        return []

    top_rows = query_rows[:per_query]
    top_fulltext_count = sum(1 for row in top_rows if has_full_text(row, manifest_lookup))
    if top_fulltext_count >= min_fulltext_top_k:
        return [(row, "top_k_has_enough_full_text") for row in top_rows]

    window = query_rows[: max(expand_to, per_query)]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(window):
        if not has_full_text(row, manifest_lookup) and abstract_too_vague(str(row.get("abstract") or "")):
            continue
        scored.append((labeling_priority_score(row, manifest_lookup, index), index, row))

    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:per_query]
    if len(selected) < min_selected:
        selected = scored[:per_query]
    selected_by_original_order = sorted(selected, key=lambda item: item[1])
    return [(row, "expanded_for_low_full_text_coverage") for _, _, row in selected_by_original_order]


def has_full_text(row: dict[str, Any], manifest_lookup: dict[int, dict[str, Any]]) -> bool:
    return bool(manifest_lookup.get(int(row["paper_id"]), {}).get("full_text_available", False))


def abstract_too_vague(abstract: str) -> bool:
    normalized = normalize_text(abstract).lower()
    if len(normalized) < 50:
        return True
    vague_markers = {
        "no abstract",
        "abstract not available",
        "not available",
        "none",
        "n/a",
    }
    return normalized in vague_markers


def labeling_priority_score(row: dict[str, Any], manifest_lookup: dict[int, dict[str, Any]], original_index: int) -> float:
    source_methods = row.get("source_methods") or []
    ranks = row.get("retrieval_ranks_by_method") or {}
    rank_values = [int(value) for value in ranks.values() if isinstance(value, int | float)]
    best_rank = min(rank_values) if rank_values else original_index + 1
    full_text_bonus = 100.0 if has_full_text(row, manifest_lookup) else 25.0
    method_bonus = min(len(source_methods), 5) * 12.0
    rank_bonus = max(0.0, 25.0 - float(best_rank))
    top_order_bonus = max(0.0, 20.0 - float(original_index))
    return full_text_bonus + method_bonus + rank_bonus + top_order_bonus


def packet_for_candidate(
    row: dict[str, Any],
    manifest: dict[str, Any],
    *,
    selection_reason: str = "top_k",
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    query_excerpt_chars: int = DEFAULT_QUERY_EXCERPT_CHARS,
) -> dict[str, Any]:
    full_text_available = bool(manifest.get("full_text_available", False))
    text_path = manifest.get("text_path")
    full_text = read_text_if_available(text_path) if full_text_available else ""
    warnings = packet_warnings(row, manifest, full_text)
    usable_full_text = full_text_available and not any(warning.startswith("full_text_unusable") for warning in warnings)
    intro = intro_excerpt(full_text, max_chars=excerpt_chars) if usable_full_text else Excerpt("", False)
    method = method_or_contribution_excerpt(full_text, max_chars=excerpt_chars) if usable_full_text else Excerpt("", False)
    conclusion = conclusion_excerpt(full_text, max_chars=excerpt_chars) if usable_full_text else Excerpt("", False)
    query_relevant = (
        query_relevant_excerpt(full_text, str(row.get("query") or ""), max_chars=query_excerpt_chars)
        if usable_full_text
        else Excerpt("", False)
    )
    excerpt_char_count_total = sum(
        len(item.text)
        for item in (
            intro,
            method,
            conclusion,
            query_relevant,
        )
    )
    if usable_full_text and excerpt_char_count_total <= 0:
        warnings.append("full_text_unusable:no_excerpts_extracted")
        usable_full_text = False
        intro = Excerpt("", False)
        method = Excerpt("", False)
        conclusion = Excerpt("", False)
        query_relevant = Excerpt("", False)
        excerpt_char_count_total = 0
    return {
        "query_id": row["query_id"],
        "query": row["query"],
        "target_audience": row["target_audience"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "year": row.get("year"),
        "venue": row.get("venue"),
        "citation_count": row.get("citation_count"),
        "source_methods": row.get("source_methods", []),
        "retrieval_ranks_by_method": row.get("retrieval_ranks_by_method", {}),
        "full_text_available": full_text_available,
        "text_path": text_path if usable_full_text else None,
        "evidence_level": "full_text_skim" if usable_full_text else "abstract_only",
        "packet_warnings": warnings,
        "selection_policy": "full_text_balanced_v1",
        "selection_reason": selection_reason,
        "intro_excerpt": intro.text,
        "method_or_contribution_excerpt": method.text,
        "conclusion_excerpt": conclusion.text,
        "query_relevant_excerpt": query_relevant.text,
        "intro_found": intro.found,
        "method_found": method.found,
        "conclusion_found": conclusion.found,
        "query_excerpt_found": query_relevant.found,
        "excerpt_char_count_total": excerpt_char_count_total,
        "full_text_excerpt": compact_excerpt(full_text, max_chars=excerpt_chars) if usable_full_text else "",
    }


def packet_warnings(row: dict[str, Any], manifest: dict[str, Any], full_text: str) -> list[str]:
    warnings: list[str] = []
    if not bool(manifest.get("full_text_available", False)):
        status = manifest.get("status")
        error = str(manifest.get("error") or "")
        if status:
            warnings.append(f"full_text_unavailable:{status}")
        if error:
            warnings.append(f"full_text_error:{error}")
        return warnings

    normalized_text = normalize_text(full_text)
    if len(normalized_text) < MIN_USEFUL_TEXT_CHARS:
        warnings.append(f"full_text_unusable:too_short:{len(normalized_text)}")
        return warnings

    title_terms = content_terms(str(row.get("title") or ""), TITLE_STOPWORDS)
    if len(title_terms) >= MIN_TITLE_TERMS_FOR_SANITY_CHECK:
        text_lower = normalized_text[:12000].lower()
        hits = {term for term in title_terms if term in text_lower}
        overlap = len(hits) / len(title_terms)
        if len(hits) < MIN_TITLE_TOKEN_HITS and overlap < MIN_TITLE_TOKEN_OVERLAP:
            warnings.append(
                "full_text_unusable:title_content_mismatch:"
                f"hits={len(hits)}/{len(title_terms)}"
            )

    if not query_relevant_excerpt(normalized_text, str(row.get("query") or ""), max_chars=600).found:
        warnings.append("query_terms_not_found_in_full_text")
    return warnings


def read_text_if_available(path_value: Any) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def intro_excerpt(text: str, *, max_chars: int) -> Excerpt:
    excerpt = section_excerpt(text, ["introduction", "background"], max_chars=max_chars)
    if excerpt.found:
        return excerpt
    return Excerpt(compact_excerpt(text, max_chars=max_chars), False)


def method_or_contribution_excerpt(text: str, *, max_chars: int) -> Excerpt:
    excerpt = section_excerpt(
        text,
        ["method", "methods", "approach", "architecture", "model", "contribution", "proposed"],
        max_chars=max_chars,
    )
    if excerpt.found:
        return excerpt
    return keyword_excerpt(
        text,
        ["propose", "proposed", "contribution", "method", "architecture", "model", "approach"],
        max_chars=max_chars,
    )


def conclusion_excerpt(text: str, *, max_chars: int) -> Excerpt:
    excerpt = section_excerpt(text, ["conclusion", "conclusions", "discussion"], max_chars=max_chars)
    if excerpt.found:
        return excerpt
    normalized = normalize_text(text)
    return Excerpt(compact_excerpt(normalized[-max_chars:], max_chars=max_chars), False)


def query_relevant_excerpt(text: str, query: str, *, max_chars: int) -> Excerpt:
    terms = query_terms(query)
    if not terms:
        return Excerpt("", False)
    normalized = normalize_text(text)
    lower = normalized.lower()
    best_index: int | None = None
    best_score = 0
    for term in terms:
        for match in re.finditer(re.escape(term), lower):
            start = max(0, match.start() - max_chars // 2)
            end = min(len(lower), start + max_chars)
            window = lower[start:end]
            score = sum(window.count(candidate) for candidate in terms)
            if score > best_score:
                best_score = score
                best_index = match.start()
    if best_index is None:
        return Excerpt("", False)
    start = max(0, best_index - max_chars // 2)
    return Excerpt(compact_excerpt(normalized[start : start + max_chars * 2], max_chars=max_chars), True)


def query_terms(query: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 2 and term not in QUERY_STOPWORDS
    ]


def content_terms(text: str, stopwords: set[str]) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 3 and term not in stopwords
    ]


def section_excerpt(text: str, headings: list[str], *, max_chars: int = DEFAULT_EXCERPT_CHARS) -> Excerpt:
    if not text.strip():
        return Excerpt("", False)
    normalized = normalize_text(text)
    lower = normalized.lower()
    for heading in headings:
        match = re.search(rf"(?:^|\n|\s){re.escape(heading)}(?:\s|\n|:)", lower)
        if match:
            return Excerpt(compact_excerpt(normalized[match.start() : match.start() + max_chars * 2], max_chars=max_chars), True)
    return Excerpt("", False)


def keyword_excerpt(text: str, keywords: list[str], *, max_chars: int = DEFAULT_EXCERPT_CHARS) -> Excerpt:
    if not text.strip():
        return Excerpt("", False)
    normalized = normalize_text(text)
    lower = normalized.lower()
    for keyword in keywords:
        match = re.search(rf"\b{re.escape(keyword)}\w*\b", lower)
        if match:
            start = max(0, match.start() - max_chars // 3)
            return Excerpt(compact_excerpt(normalized[start : start + max_chars * 2], max_chars=max_chars), True)
    return Excerpt("", False)


def compact_excerpt(text: str, *, max_chars: int = DEFAULT_EXCERPT_CHARS) -> str:
    compact = normalize_text(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def export_packets(
    *,
    pool_path: Path = DEFAULT_POOL,
    manifest_paths: list[Path] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    per_query: int = 10,
    min_fulltext_top_k: int = DEFAULT_MIN_FULLTEXT_TOP_K,
    expand_to: int = DEFAULT_EXPAND_TO,
    min_selected: int = DEFAULT_MIN_SELECTED,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    query_excerpt_chars: int = DEFAULT_QUERY_EXCERPT_CHARS,
    query_ids: set[str] | None = None,
) -> dict[str, Any]:
    manifest_paths = manifest_paths or [DEFAULT_MANIFEST]
    manifest_rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest_rows.extend(load_jsonl(manifest_path))
    packets = build_packets(
        pool_rows=load_jsonl(pool_path),
        manifest_rows=manifest_rows,
        per_query=per_query,
        min_fulltext_top_k=min_fulltext_top_k,
        expand_to=expand_to,
        min_selected=min_selected,
        excerpt_chars=excerpt_chars,
        query_excerpt_chars=query_excerpt_chars,
        query_ids=query_ids,
    )
    write_jsonl(packets, output_path)
    counts_by_query: dict[str, int] = defaultdict(int)
    fulltext_by_query: dict[str, int] = defaultdict(int)
    warnings_by_query: dict[str, int] = defaultdict(int)
    for packet in packets:
        query_id = str(packet["query_id"])
        counts_by_query[query_id] += 1
        if packet["evidence_level"] == "full_text_skim":
            fulltext_by_query[query_id] += 1
        if packet.get("packet_warnings"):
            warnings_by_query[query_id] += 1
    return {
        "packet_count": len(packets),
        "full_text_count": sum(1 for packet in packets if packet["evidence_level"] == "full_text_skim"),
        "counts_by_query": dict(counts_by_query),
        "fulltext_by_query": dict(fulltext_by_query),
        "warnings_by_query": dict(warnings_by_query),
        "output_path": str(output_path),
    }


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-text-assisted label packets.")
    parser.add_argument("--pool", default=str(DEFAULT_POOL))
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--per-query", type=int, default=10)
    parser.add_argument("--min-fulltext-top-k", type=int, default=DEFAULT_MIN_FULLTEXT_TOP_K)
    parser.add_argument("--expand-to", type=int, default=DEFAULT_EXPAND_TO)
    parser.add_argument("--min-selected", type=int, default=DEFAULT_MIN_SELECTED)
    parser.add_argument("--excerpt-chars", type=int, default=DEFAULT_EXCERPT_CHARS)
    parser.add_argument("--query-excerpt-chars", type=int, default=DEFAULT_QUERY_EXCERPT_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_packets(
        pool_path=resolve_repo_path(args.pool),
        manifest_paths=[resolve_repo_path(path) for path in args.manifest] if args.manifest else [DEFAULT_MANIFEST],
        output_path=resolve_repo_path(args.out),
        per_query=args.per_query,
        min_fulltext_top_k=args.min_fulltext_top_k,
        expand_to=args.expand_to,
        min_selected=args.min_selected,
        excerpt_chars=args.excerpt_chars,
        query_excerpt_chars=args.query_excerpt_chars,
        query_ids=set(args.query_id) if args.query_id else None,
    )
    print(f"Packets: {report['packet_count']}")
    print(f"Packets with full text: {report['full_text_count']}")
    print(f"Packets by query: {json.dumps(report['counts_by_query'], sort_keys=True)}")
    print(f"Full-text packets by query: {json.dumps(report['fulltext_by_query'], sort_keys=True)}")
    print(f"Packet warnings by query: {json.dumps(report['warnings_by_query'], sort_keys=True)}")
    print(f"Output: {report['output_path']}")


if __name__ == "__main__":
    main()
