import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from build_v4_3_text_reranker_dataset import protected_status  # noqa: E402
from evaluate_v3_4_v3_3_production_pool import resolve_repo_path, write_json, write_text  # noqa: E402
from evaluate_v6_1_3_exact_learned_baselines import load_jsonl, write_jsonl  # noqa: E402
from reconstruct_v6_1_6_canonical_corpus_metadata import normalize_title, raw_row_number_alignment, safe_int  # noqa: E402
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


BACKEND_DEFAULT_DATABASE_URL = "postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath"
EXPECTED_CORPUS_SIZE = 50424

DEFAULT_V6_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_paper_metadata.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "processed" / "corpus" / "v6_1_6b_canonical_corpus_manifest.json"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6b_canonical_postgres_snapshot_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6b_canonical_postgres_snapshot_report.md"
DEFAULT_COVERAGE = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6b_label_coverage_against_canonical.jsonl"
DEFAULT_AMBIGUOUS = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6b_ambiguous_title_year_groups.jsonl"

LABEL_SOURCES = {
    "v2_1": REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl",
    "v2_5": REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl",
    "v3_2": REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl",
    "v3_5": REPO_ROOT / "data" / "eval" / "manual_labels_v3_5_unjudged_audit.jsonl",
    "v3_9": REPO_ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl",
    "v4_8_override": REPO_ROOT / "data" / "eval" / "manual_labels_v4_8_targeted_contrastive.jsonl",
    "v6_0_2_neural_examples": DEFAULT_V6_EXAMPLES,
}


def load_postgres_rows(database_url: str) -> tuple[list[dict[str, Any]], list[str], str | None]:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            fields = [
                row[0]
                for row in conn.execute(
                    text("select column_name from information_schema.columns where table_name='papers' order by ordinal_position")
                ).fetchall()
            ]
            wanted = [
                col
                for col in (
                    "id",
                    "title",
                    "abstract",
                    "year",
                    "venue",
                    "source",
                    "external_id",
                    "doi",
                    "source_url",
                    "url",
                    "citation_count",
                    "abstract_word_count",
                )
                if col in fields
            ]
            if not fields or not wanted:
                return [], fields, "papers table or exportable columns were not found"
            rows = [dict(row._mapping) for row in conn.execute(text(f"select {', '.join(wanted)} from papers order by id")).fetchall()]
            return rows, fields, None
    except Exception as exc:
        return [], [], str(exc)


def title_year_key(row: dict[str, Any]) -> tuple[str, int | None]:
    return normalize_title(row.get("title")), safe_int(row.get("year"))


def ambiguous_title_year_groups(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, int | None], int]]:
    counts = Counter(title_year_key(row) for row in rows if normalize_title(row.get("title")))
    examples: dict[tuple[str, int | None], list[int]] = {}
    for row in rows:
        key = title_year_key(row)
        if counts[key] > 1:
            examples.setdefault(key, [])
            if len(examples[key]) < 10:
                examples[key].append(int(row["id"]))
    groups = [
        {
            "schema_version": "v6.1.6b_ambiguous_title_year_group",
            "normalized_title": title,
            "year": year,
            "group_size": count,
            "sample_paper_ids": examples.get((title, year), []),
            "identity_policy": "warning_only_explicit_paper_id_is_primary",
        }
        for (title, year), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or 0))
        if count > 1
    ]
    return groups, dict(counts)


def validate_examples(rows_by_id: dict[int, dict[str, Any]], examples: list[dict[str, Any]], group_sizes: dict[tuple[str, int | None], int]) -> dict[str, Any]:
    unique = {int(row["paper_id"]): row for row in examples}
    resolved = 0
    missing: list[int] = []
    mismatches: list[dict[str, Any]] = []
    ambiguous_v6_ids: list[int] = []
    for paper_id, example in sorted(unique.items()):
        canonical = rows_by_id.get(paper_id)
        if canonical is None:
            missing.append(paper_id)
            continue
        resolved += 1
        title_match = str(canonical.get("title") or "") == str(example.get("title") or "")
        normalized_year_match = (
            normalize_title(canonical.get("title")) == normalize_title(example.get("title"))
            and safe_int(canonical.get("year")) == safe_int(example.get("year"))
        )
        if not (title_match or normalized_year_match):
            mismatches.append(
                {
                    "paper_id": paper_id,
                    "canonical_title": canonical.get("title"),
                    "canonical_year": canonical.get("year"),
                    "example_title": example.get("title"),
                    "example_year": example.get("year"),
                }
            )
        if group_sizes.get(title_year_key(canonical), 0) > 1:
            ambiguous_v6_ids.append(paper_id)
    return {
        "total_v6_unique_paper_ids": len(unique),
        "resolved_by_explicit_paper_id": resolved,
        "missing_paper_ids": missing,
        "missing_count": len(missing),
        "true_title_year_mismatch_count": len(mismatches),
        "sample_true_title_year_mismatches": mismatches[:50],
        "ambiguous_normalized_title_year_groups_are_blockers": False,
        "ambiguous_v6_row_count": len(ambiguous_v6_ids),
        "sample_ambiguous_v6_paper_ids": ambiguous_v6_ids[:50],
    }


def canonical_row(row: dict[str, Any], group_sizes: dict[tuple[str, int | None], int]) -> dict[str, Any]:
    paper_id = int(row["id"])
    external_ids = {
        key: row.get(key)
        for key in ("external_id", "doi")
        if row.get(key) not in (None, "")
    }
    key = title_year_key(row)
    group_size = int(group_sizes.get(key, 0))
    return {
        "schema_version": "v6.1.6b_canonical_paper_metadata",
        "paper_id": paper_id,
        "title": str(row.get("title") or ""),
        "abstract": str(row.get("abstract") or ""),
        "year": safe_int(row.get("year")),
        "venue": row.get("venue"),
        "source": row.get("source"),
        "source_url": row.get("source_url") or row.get("url"),
        "external_ids": external_ids,
        "citation_count": safe_int(row.get("citation_count")) or 0,
        "abstract_word_count": safe_int(row.get("abstract_word_count")),
        "source_table": "papers",
        "identity_key": "paper_id",
        "normalized_title": key[0],
        "normalized_title_year_group_size": group_size,
        "identity_validation_status": "explicit_paper_id_primary_ambiguous_title_year_warning" if group_size > 1 else "passed",
    }


def write_snapshot(rows: list[dict[str, Any]], group_sizes: dict[tuple[str, int | None], int], path: Path) -> tuple[list[dict[str, Any]], str]:
    output_rows = [canonical_row(row, group_sizes) for row in rows]
    output_rows.sort(key=lambda row: row["paper_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output_rows, hashlib.sha256(path.read_bytes()).hexdigest().upper()


def label_title_year_mismatch(canonical: dict[str, Any], label: dict[str, Any]) -> bool:
    label_title = normalize_title(label.get("title"))
    label_year = safe_int(label.get("year"))
    title_available = bool(label_title)
    year_available = label_year is not None
    title_mismatch = title_available and normalize_title(canonical.get("title")) != label_title
    year_mismatch = year_available and safe_int(canonical.get("year")) != label_year
    return title_mismatch or year_mismatch


def label_coverage(rows_by_id: dict[int, dict[str, Any]], group_sizes: dict[tuple[str, int | None], int]) -> list[dict[str, Any]]:
    output = []
    for source, path in LABEL_SOURCES.items():
        labels = load_jsonl(path) if path.exists() else []
        unique_ids = {safe_int(row.get("paper_id")) for row in labels}
        unique_ids.discard(None)
        resolved = 0
        missing: list[int] = []
        mismatches: list[dict[str, Any]] = []
        ambiguous_rows = 0
        for label in labels:
            paper_id = safe_int(label.get("paper_id"))
            canonical = rows_by_id.get(paper_id) if paper_id is not None else None
            if canonical is None:
                if paper_id is not None and len(missing) < 50:
                    missing.append(paper_id)
                continue
            resolved += 1
            if label_title_year_mismatch(canonical, label):
                if len(mismatches) < 50:
                    mismatches.append(
                        {
                            "paper_id": paper_id,
                            "canonical_title": canonical.get("title"),
                            "canonical_year": canonical.get("year"),
                            "label_title": label.get("title"),
                            "label_year": label.get("year"),
                        }
                    )
            if group_sizes.get(title_year_key(canonical), 0) > 1:
                ambiguous_rows += 1
        output.append(
            {
                "schema_version": "v6.1.6b_label_coverage_against_canonical",
                "source": source,
                "path": str(path),
                "rows_checked": len(labels),
                "unique_paper_ids": len(unique_ids),
                "resolved_by_explicit_paper_id": resolved,
                "missing_paper_id_count": max(0, len(labels) - resolved),
                "sample_missing_paper_ids": missing,
                "true_title_year_mismatch_count": len(mismatches),
                "sample_true_title_year_mismatches": mismatches,
                "ambiguous_normalized_title_year_warning_row_count": ambiguous_rows,
                "identity_policy": "explicit_paper_id_primary_no_fuzzy_remapping",
            }
        )
    return output


def build_manifest(
    *,
    output_rows: list[dict[str, Any]],
    corpus_hash: str,
    database_url: str,
    source_schema: list[str],
    duplicate_paper_id_count: int,
    ambiguous_groups: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "v6.1.6b_canonical_corpus_manifest",
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": len(output_rows),
        "corpus_hash": corpus_hash,
        "source_database_url": database_url,
        "source_table": "papers",
        "source_schema": source_schema,
        "id_policy": {
            "primary_identity_key": "explicit ResearchPath paper_id from papers.id",
            "normalized_title_year_role": "diagnostic_only_warning_not_blocker",
            "raw_jsonl_row_numbers_used_as_paper_id": False,
            "fuzzy_title_matching_used": False,
            "paper_id_mapping_guessed": False,
        },
        "duplicate_paper_id_count": duplicate_paper_id_count,
        "missing_title_count": sum(1 for row in output_rows if not row["title"].strip()),
        "missing_abstract_count": sum(1 for row in output_rows if not row["abstract"].strip()),
        "missing_year_count": sum(1 for row in output_rows if row["year"] is None),
        "ambiguous_normalized_title_year_group_count": len(ambiguous_groups),
        "validation_summary": validation,
        "known_limitations": [
            "Normalized title+year duplicates remain in the corpus and are recorded as warnings.",
            "This snapshot does not regenerate BM25, dense, hybrid, V2.2b, V2.6, citation, or full-text features.",
        ],
    }


def acceptance(
    *,
    rows: list[dict[str, Any]],
    connection_error: str | None,
    duplicate_paper_id_count: int,
    validation: dict[str, Any],
    protected: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons = []
    if connection_error:
        reasons.append(f"PostgreSQL connection failed: {connection_error}")
    if len(rows) != EXPECTED_CORPUS_SIZE:
        reasons.append(f"PostgreSQL row count is {len(rows)}, expected {EXPECTED_CORPUS_SIZE}")
    if duplicate_paper_id_count != 0:
        reasons.append(f"duplicate paper_id count is {duplicate_paper_id_count}")
    if validation["resolved_by_explicit_paper_id"] != validation["total_v6_unique_paper_ids"]:
        reasons.append("V6.0.2 paper_ids did not resolve 100% by explicit paper_id")
    if validation["true_title_year_mismatch_count"] != 0:
        reasons.append("V6.0.2 true title/year mismatches are present")
    if not all(value for key, value in protected.items() if key.endswith("_hash_unchanged")):
        reasons.append("protected hashes changed")
    return not reasons, reasons


def render_markdown(report: dict[str, Any], coverage: list[dict[str, Any]]) -> str:
    validation = report["v6_example_identity_validation"]
    lines = [
        "# V6.1.6b Canonical PostgreSQL Snapshot",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- PostgreSQL available: `{report['postgresql_available']}`",
        f"- Snapshot created: `{report['canonical_snapshot_created']}`",
        f"- Row count: `{report['row_count']}`",
        f"- Corpus hash: `{report['corpus_hash']}`",
        f"- Duplicate paper_id count: `{report['duplicate_paper_id_count']}`",
        f"- Ambiguous title/year groups: `{report['ambiguous_normalized_title_year_group_count']}`",
        f"- V6 resolved by paper_id: `{validation['resolved_by_explicit_paper_id']}` / `{validation['total_v6_unique_paper_ids']}`",
        f"- V6 true title/year mismatches: `{validation['true_title_year_mismatch_count']}`",
        f"- Feature regeneration can proceed next: `{report['feature_regeneration_can_proceed_next']}`",
        "",
        "## Label Coverage",
        "",
    ]
    for row in coverage:
        lines.append(
            f"- `{row['source']}`: resolved `{row['resolved_by_explicit_paper_id']}` / `{row['rows_checked']}`, "
            f"missing `{row['missing_paper_id_count']}`, mismatches `{row['true_title_year_mismatch_count']}`, "
            f"ambiguous warnings `{row['ambiguous_normalized_title_year_warning_row_count']}`"
        )
    lines.extend(["", "## Acceptance", ""])
    lines.append(f"- Accepted: `{report['acceptance_passed']}`")
    if report["acceptance_failure_reasons"]:
        for reason in report["acceptance_failure_reasons"]:
            lines.append(f"- Blocker: {reason}")
    lines.extend(["", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows, source_schema, connection_error = load_postgres_rows(args.database_url)
    ids = [safe_int(row.get("id")) for row in rows]
    duplicate_paper_id_count = len(ids) - len(set(ids))
    ambiguous_groups, group_sizes = ambiguous_title_year_groups(rows)
    rows_by_id = {int(row["id"]): row for row in rows if row.get("id") is not None}
    v6_examples = load_jsonl(resolve_repo_path(args.v6_examples))
    validation = validate_examples(rows_by_id, v6_examples, group_sizes)
    coverage = label_coverage(rows_by_id, group_sizes)
    protected = protected_status(resolve_repo_path(args.v39_labels))
    accepted, failure_reasons = acceptance(
        rows=rows,
        connection_error=connection_error,
        duplicate_paper_id_count=duplicate_paper_id_count,
        validation=validation,
        protected=protected,
    )
    output_rows: list[dict[str, Any]] = []
    corpus_hash: str | None = None
    manifest: dict[str, Any] | None = None
    if accepted:
        output_rows, corpus_hash = write_snapshot(rows, group_sizes, resolve_repo_path(args.snapshot_out))
        manifest = build_manifest(
            output_rows=output_rows,
            corpus_hash=corpus_hash,
            database_url=args.database_url,
            source_schema=source_schema,
            duplicate_paper_id_count=duplicate_paper_id_count,
            ambiguous_groups=ambiguous_groups,
            validation=validation,
        )
        write_json(resolve_repo_path(args.manifest_out), manifest)
    report = {
        "schema_version": "v6.1.6b_canonical_postgres_snapshot_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_trained_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "feature_artifacts_regenerated": False,
        "external_api_calls_made": False,
        "fuzzy_remapping_performed": False,
        "paper_id_mapping_guessed": False,
        "raw_row_number_alignment": raw_row_number_alignment(REPO_ROOT / "data" / "raw" / "arxiv_ml_ai_50k_incremental.jsonl", v6_examples),
        "postgresql_available": connection_error is None,
        "postgresql_connection_error": connection_error,
        "source_database_url": args.database_url,
        "source_table": "papers",
        "source_schema": source_schema,
        "row_count": len(rows),
        "expected_row_count": EXPECTED_CORPUS_SIZE,
        "canonical_snapshot_created": accepted,
        "canonical_snapshot_path": str(resolve_repo_path(args.snapshot_out)) if accepted else None,
        "canonical_manifest_path": str(resolve_repo_path(args.manifest_out)) if accepted else None,
        "corpus_hash": corpus_hash,
        "duplicate_paper_id_count": duplicate_paper_id_count,
        "missing_title_count": sum(1 for row in rows if not str(row.get("title") or "").strip()),
        "missing_abstract_count": sum(1 for row in rows if not str(row.get("abstract") or "").strip()),
        "missing_year_count": sum(1 for row in rows if safe_int(row.get("year")) is None),
        "ambiguous_normalized_title_year_group_count": len(ambiguous_groups),
        "ambiguous_title_year_groups_path": str(resolve_repo_path(args.ambiguous_out)),
        "ambiguous_normalized_title_year_groups_are_warnings": True,
        "id_policy": {
            "primary_identity_key": "explicit ResearchPath paper_id from PostgreSQL papers.id",
            "normalized_title_year_role": "diagnostic_only_warning_not_blocker",
            "raw_jsonl_row_numbers_used_as_paper_id": False,
            "fuzzy_title_matching_used": False,
            "paper_id_mapping_guessed": False,
        },
        "v6_example_identity_validation": validation,
        "label_coverage_path": str(resolve_repo_path(args.coverage_out)),
        "acceptance_passed": accepted,
        "acceptance_failure_reasons": failure_reasons,
        "feature_regeneration_can_proceed_next": accepted,
        "protected_hashes": protected,
    }
    return report, coverage, ambiguous_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=BACKEND_DEFAULT_DATABASE_URL)
    parser.add_argument("--v6-examples", default=str(DEFAULT_V6_EXAMPLES))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--snapshot-out", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--coverage-out", default=str(DEFAULT_COVERAGE))
    parser.add_argument("--ambiguous-out", default=str(DEFAULT_AMBIGUOUS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, coverage, ambiguous_groups = build_report(args)
    write_jsonl(resolve_repo_path(args.coverage_out), coverage)
    write_jsonl(resolve_repo_path(args.ambiguous_out), ambiguous_groups)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report, coverage))
    print(f"Wrote V6.1.6b canonical PostgreSQL snapshot report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
