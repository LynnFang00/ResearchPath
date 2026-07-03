import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
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
from validate_v3_9_semantic_expansion_labels import DEFAULT_LABELS as DEFAULT_V39_LABELS  # noqa: E402


DEFAULT_V6_EXAMPLES = REPO_ROOT / "data" / "eval" / "neural_reranker" / "v6_0_2_neural_reranker_examples.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6_canonical_corpus_metadata_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6_canonical_corpus_metadata_report.md"
DEFAULT_CANDIDATES_OUT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6_canonical_source_candidates.jsonl"
DEFAULT_COVERAGE_OUT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_6_label_coverage_against_canonical.jsonl"
DEFAULT_CANONICAL_OUT = REPO_ROOT / "data" / "processed" / "corpus" / "v6_1_6_canonical_paper_metadata.jsonl"
DEFAULT_MANIFEST_OUT = REPO_ROOT / "data" / "processed" / "corpus" / "v6_1_6_canonical_corpus_manifest.json"
DEFAULT_RUNTIME_DB = REPO_ROOT / "backend" / "data" / "researchpath.db"
DEFAULT_EMBEDDINGS = REPO_ROOT / "data" / "processed" / "embeddings" / "all_minilm_l6_v2_50k.npz"
DEFAULT_FAISS_IDS = REPO_ROOT / "data" / "processed" / "faiss" / "all_minilm_l6_v2_50k.ids.npz"

BACKEND_DEFAULT_DATABASE_URL = "postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath"
EXPECTED_CORPUS_SIZE = 50424

LABEL_SOURCES = {
    "v2_1": REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl",
    "v2_5": REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl",
    "v3_2": REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl",
    "v3_5": REPO_ROOT / "data" / "eval" / "manual_labels_v3_5_unjudged_audit.jsonl",
    "v3_9": REPO_ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl",
    "v4_8_override": REPO_ROOT / "data" / "eval" / "manual_labels_v4_8_targeted_contrastive.jsonl",
    "v6_0_2_neural_examples": DEFAULT_V6_EXAMPLES,
}

CODE_TRACE_FILES = [
    "backend/app/core/config.py",
    "backend/app/db/session.py",
    "backend/app/main.py",
    "backend/app/models/paper.py",
    "backend/app/services/ingestion.py",
    "scripts/ingest_papers.py",
    "scripts/fetch_arxiv_papers.py",
    "scripts/enrich_arxiv_with_openalex.py",
    "scripts/validate_corpus_provenance.py",
    "scripts/export_labeling_candidates.py",
    "scripts/export_v2_labeling_candidates.py",
    "scripts/build_v3_9_semantic_label_expansion_packet.py",
    "scripts/build_v6_0_neural_reranker_dataset.py",
    "scripts/build_v6_0_2_neural_reranker_dataset.py",
]


def normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def jsonl_count_and_fields(path: Path) -> tuple[int, set[str], list[dict[str, Any]]]:
    count = 0
    fields: set[str] = set()
    samples: list[dict[str, Any]] = []
    if not path.exists():
        return 0, fields, samples
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            fields.update(row.keys())
            if len(samples) < 10:
                samples.append(row)
    return count, fields, samples


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            output: dict[str, Any] = {"exists": True, "keys": list(data.files)}
            if "document_ids" in data:
                ids = data["document_ids"].astype(int)
                output.update(
                    {
                        "row_count": int(len(ids)),
                        "document_id_count": int(len(ids)),
                        "document_id_min": int(ids.min()) if len(ids) else None,
                        "document_id_max": int(ids.max()) if len(ids) else None,
                        "document_id_unique_count": int(len(set(ids.tolist()))),
                    }
                )
            if "embeddings" in data:
                output["embedding_shape"] = [int(x) for x in data["embeddings"].shape]
                output["row_count"] = int(data["embeddings"].shape[0])
            return output
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def candidate_base(
    *,
    name: str,
    source_kind: str,
    locator: str,
    exists: bool,
    row_count: int | None,
    fields: set[str] | list[str],
) -> dict[str, Any]:
    field_set = set(fields)
    return {
        "schema_version": "v6.1.6_canonical_source_candidate",
        "name": name,
        "source_kind": source_kind,
        "path_or_connection": locator,
        "exists_or_available": exists,
        "row_count": row_count,
        "fields_available": sorted(field_set),
        "has_explicit_paper_id": bool({"paper_id", "id"} & field_set),
        "has_title": "title" in field_set,
        "has_abstract": "abstract" in field_set,
        "has_year": "year" in field_set,
        "has_venue_or_source": bool({"venue", "source"} & field_set),
        "has_external_ids": bool({"external_id", "openalex_id", "doi", "identifiers"} & field_set),
        "v6_identity_validation": {},
        "can_safely_become_canonical": False,
        "canonical_rejection_reasons": [],
    }


def sqlite_candidate(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        return candidate_base(name=name, source_kind="sqlite", locator=str(path), exists=False, row_count=None, fields=set()) | {
            "canonical_rejection_reasons": ["missing"]
        }
    try:
        with sqlite3.connect(path) as conn:
            tables = [row[0] for row in conn.execute("select name from sqlite_master where type='table'").fetchall()]
            if "papers" not in tables:
                return candidate_base(name=name, source_kind="sqlite", locator=str(path), exists=True, row_count=0, fields=set()) | {
                    "tables": tables,
                    "canonical_rejection_reasons": ["no papers table"],
                }
            fields = [row[1] for row in conn.execute("pragma table_info(papers)").fetchall()]
            count = int(conn.execute("select count(*) from papers").fetchone()[0])
            candidate = candidate_base(name=name, source_kind="sqlite", locator=str(path), exists=True, row_count=count, fields=fields)
            candidate["tables"] = tables
            return candidate
    except Exception as exc:
        return candidate_base(name=name, source_kind="sqlite", locator=str(path), exists=True, row_count=None, fields=set()) | {
            "canonical_rejection_reasons": [str(exc)]
        }


def jsonl_candidate(path: Path, name: str) -> dict[str, Any]:
    count, fields, _ = jsonl_count_and_fields(path)
    return candidate_base(name=name, source_kind="jsonl", locator=str(path), exists=path.exists(), row_count=count, fields=fields)


def npz_candidate(path: Path, name: str) -> dict[str, Any]:
    info = inspect_npz(path)
    fields = set(info.get("keys") or [])
    candidate = candidate_base(
        name=name,
        source_kind="npz",
        locator=str(path),
        exists=bool(info.get("exists")),
        row_count=info.get("row_count"),
        fields=fields,
    )
    candidate.update({k: v for k, v in info.items() if k not in {"exists", "keys", "row_count"}})
    if "document_ids" in fields:
        candidate["has_explicit_paper_id"] = True
    return candidate


def postgres_candidate(database_url: str, v6_examples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    candidate = candidate_base(
        name="backend_default_postgresql",
        source_kind="postgresql",
        locator=database_url,
        exists=False,
        row_count=None,
        fields=set(),
    )
    rows: list[dict[str, Any]] | None = None
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            fields = [row[0] for row in conn.execute(text("select column_name from information_schema.columns where table_name='papers' order by ordinal_position")).fetchall()]
            if not fields:
                candidate["canonical_rejection_reasons"] = ["connected but papers table was not found"]
                candidate["exists_or_available"] = True
                return candidate, None
            count = int(conn.execute(text("select count(*) from papers")).scalar_one())
            candidate = candidate_base(
                name="backend_default_postgresql",
                source_kind="postgresql",
                locator=database_url,
                exists=True,
                row_count=count,
                fields=fields,
            )
            wanted = [
                col for col in (
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
            query = text(f"select {', '.join(wanted)} from papers order by id")
            rows = [dict(row._mapping) for row in conn.execute(query).fetchall()]
    except Exception as exc:
        candidate["connection_error"] = str(exc)
        candidate["canonical_rejection_reasons"] = [f"PostgreSQL read-only connection failed: {exc}"]
        return candidate, None
    candidate["v6_identity_validation"] = validate_candidate_rows(rows or [], v6_examples)
    finalize_candidate(candidate)
    return candidate, rows


def candidate_rows_from_sqlite(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        fields = [row[1] for row in conn.execute("pragma table_info(papers)").fetchall()]
        wanted = [
            col for col in (
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
        if not wanted:
            return []
        return [dict(row) for row in conn.execute(f"select {', '.join(wanted)} from papers order by id").fetchall()]


def validate_candidate_rows(candidate_rows: list[dict[str, Any]], v6_examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {safe_int(row.get("id") if row.get("id") is not None else row.get("paper_id")): row for row in candidate_rows}
    by_id.pop(None, None)
    unique_v6 = {}
    for row in v6_examples:
        unique_v6.setdefault(int(row["paper_id"]), row)
    resolved = 0
    exact_title_matches = 0
    normalized_title_year_matches = 0
    accepted_identity_matches = 0
    mismatches: list[dict[str, Any]] = []
    missing: list[int] = []
    for paper_id, example in sorted(unique_v6.items()):
        candidate = by_id.get(paper_id)
        if candidate is None:
            missing.append(paper_id)
            continue
        resolved += 1
        title_match = str(candidate.get("title") or "") == str(example.get("title") or "")
        normalized_year_match = (
            normalize_title(candidate.get("title")) == normalize_title(example.get("title"))
            and safe_int(candidate.get("year")) == safe_int(example.get("year"))
        )
        if title_match:
            exact_title_matches += 1
        if normalized_year_match:
            normalized_title_year_matches += 1
        if title_match or normalized_year_match:
            accepted_identity_matches += 1
        else:
            if len(mismatches) < 50:
                mismatches.append(
                    {
                        "paper_id": paper_id,
                        "candidate_title": candidate.get("title"),
                        "candidate_year": candidate.get("year"),
                        "v6_title": example.get("title"),
                        "v6_year": example.get("year"),
                    }
                )
    title_year_counts = Counter(
        (normalize_title(row.get("title")), safe_int(row.get("year")))
        for row in candidate_rows
        if normalize_title(row.get("title"))
    )
    ambiguous = [
        {"title": title, "year": year, "count": count}
        for (title, year), count in title_year_counts.items()
        if count > 1
    ]
    return {
        "v6_unique_paper_ids": len(unique_v6),
        "resolved_by_paper_id": resolved,
        "missing_by_paper_id": len(missing),
        "exact_title_matches": exact_title_matches,
        "normalized_title_year_matches": normalized_title_year_matches,
        "accepted_identity_matches": accepted_identity_matches,
        "mismatch_count": resolved - accepted_identity_matches,
        "sample_mismatches": mismatches,
        "sample_missing_paper_ids": missing[:50],
        "ambiguous_normalized_title_year_count": len(ambiguous),
        "sample_ambiguous_normalized_title_year": ambiguous[:20],
    }


def finalize_candidate(candidate: dict[str, Any]) -> None:
    reasons: list[str] = list(candidate.get("canonical_rejection_reasons") or [])
    validation = candidate.get("v6_identity_validation") or {}
    if not candidate.get("exists_or_available"):
        reasons.append("source unavailable")
    if not candidate.get("has_explicit_paper_id"):
        reasons.append("missing explicit ResearchPath paper_id/id")
    if not candidate.get("has_title"):
        reasons.append("missing title")
    if not candidate.get("has_abstract"):
        reasons.append("missing abstract")
    if not candidate.get("has_year"):
        reasons.append("missing year")
    if (candidate.get("row_count") or 0) != EXPECTED_CORPUS_SIZE:
        reasons.append(f"row count is {candidate.get('row_count')}, expected {EXPECTED_CORPUS_SIZE}")
    if validation:
        if validation.get("resolved_by_paper_id") != validation.get("v6_unique_paper_ids"):
            reasons.append("not all V6 paper_ids resolve")
        if validation.get("mismatch_count", 1) != 0:
            reasons.append("V6 title/year mismatches are present")
        if validation.get("ambiguous_normalized_title_year_count", 1) != 0:
            reasons.append("ambiguous normalized title+year rows are present")
    else:
        reasons.append("identity validation was not available")
    candidate["canonical_rejection_reasons"] = sorted(set(reasons))
    candidate["can_safely_become_canonical"] = not candidate["canonical_rejection_reasons"]


def raw_row_number_alignment(raw_path: Path, v6_examples: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [1, 2, 5053, 30046]
    v6_titles = {int(row["paper_id"]): row["title"] for row in v6_examples if int(row["paper_id"]) in samples}
    raw_titles: dict[int, str] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if index in v6_titles and line.strip():
                    raw_titles[index] = json.loads(line).get("title") or ""
                if index > max(samples):
                    break
    mismatches = [
        {"paper_id": paper_id, "v6_title": v6_titles.get(paper_id), "raw_row_title": raw_titles.get(paper_id)}
        for paper_id in samples
        if paper_id in v6_titles and v6_titles.get(paper_id) != raw_titles.get(paper_id)
    ]
    return {
        "raw_jsonl_row_numbers_used_as_paper_id": False,
        "raw_jsonl_row_number_safe_as_paper_id": not mismatches,
        "mismatches": mismatches,
        "conclusion": "unsafe" if mismatches else "not_disproven_by_samples",
    }


def discover_jsonl_candidates() -> list[Path]:
    roots = [REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "eval", REPO_ROOT / "data" / "processed"]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            name = path.name.lower()
            if name.startswith("v6_1_6_"):
                continue
            if any(token in name for token in ("pairwise", "prediction", "failure_examples")):
                continue
            if path.stat().st_size >= 50_000 or any(token in name for token in ("corpus", "candidate", "examples", "arxiv", "openalex")):
                candidates.append(path)
    return sorted(set(candidates))


def code_trace() -> list[dict[str, Any]]:
    output = []
    keywords = ("paper_id", "papers", "database_url", "embedding", "faiss", "jsonl", "ingest", "bm25", "tfidf")
    for relative in CODE_TRACE_FILES:
        path = REPO_ROOT / relative
        if not path.exists():
            output.append({"path": relative, "exists": False, "matched_keywords": []})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        output.append(
            {
                "path": relative,
                "exists": True,
                "matched_keywords": [keyword for keyword in keywords if keyword in text],
                "role": role_for_trace_file(relative),
            }
        )
    return output


def role_for_trace_file(relative: str) -> str:
    if "config" in relative:
        return "runtime DB/model artifact path configuration"
    if "session" in relative or "main" in relative:
        return "runtime database engine/session initialization"
    if "ingestion" in relative or "ingest" in relative:
        return "JSONL/CSV corpus ingestion and deduplication"
    if "fetch" in relative or "enrich" in relative:
        return "raw API export generation"
    if "provenance" in relative:
        return "database provenance validation"
    if "candidate" in relative or "label" in relative or "reranker" in relative:
        return "evaluation candidate/dataset construction"
    return "related corpus code"


def build_candidates(args: argparse.Namespace, v6_examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    candidates: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] | None = None
    postgres, postgres_rows = postgres_candidate(args.database_url, v6_examples)
    candidates.append(postgres)
    if postgres["can_safely_become_canonical"]:
        canonical_rows = postgres_rows

    for path in sorted((REPO_ROOT / "backend" / "data").glob("*.db")):
        candidate = sqlite_candidate(path, f"sqlite_{path.stem}")
        try:
            rows = candidate_rows_from_sqlite(path)
            candidate["v6_identity_validation"] = validate_candidate_rows(rows, v6_examples) if rows else {}
        except Exception as exc:
            candidate["v6_identity_validation_error"] = str(exc)
        finalize_candidate(candidate)
        candidates.append(candidate)
        if candidate["can_safely_become_canonical"] and canonical_rows is None:
            canonical_rows = candidate_rows_from_sqlite(path)

    for path in discover_jsonl_candidates():
        candidate = jsonl_candidate(path, f"jsonl_{path.stem}")
        rows: list[dict[str, Any]] = []
        if candidate["has_explicit_paper_id"] and candidate["has_title"]:
            try:
                rows = load_jsonl(path)
                converted = [{**row, "id": row.get("paper_id", row.get("id"))} for row in rows]
                candidate["v6_identity_validation"] = validate_candidate_rows(converted, v6_examples)
            except Exception as exc:
                candidate["v6_identity_validation_error"] = str(exc)
        finalize_candidate(candidate)
        candidates.append(candidate)
        if candidate["can_safely_become_canonical"] and canonical_rows is None:
            canonical_rows = [{**row, "id": row.get("paper_id", row.get("id"))} for row in rows]

    for path, name in ((resolve_repo_path(args.embeddings), "embeddings_50k"), (resolve_repo_path(args.faiss_ids), "faiss_id_map_50k")):
        candidate = npz_candidate(path, name)
        finalize_candidate(candidate)
        candidates.append(candidate)

    report_path = REPO_ROOT / "data" / "processed" / "reports" / "corpus_provenance_validation_v2_50k.json"
    if report_path.exists():
        report = load_json(report_path)
        candidate = candidate_base(
            name="corpus_provenance_validation_v2_50k_report",
            source_kind="provenance_report",
            locator=str(report_path),
            exists=True,
            row_count=safe_int(report.get("paper_count")),
            fields=set(report.keys()),
        )
        candidate["canonical_rejection_reasons"] = ["provenance report is not row-level paper metadata"]
        finalize_candidate(candidate)
        candidates.append(candidate)
    return candidates, canonical_rows


def canonical_metadata_row(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    identifiers = {
        key: row.get(key)
        for key in ("external_id", "doi", "openalex_id")
        if row.get(key) not in (None, "")
    }
    return {
        "schema_version": "v6.1.6_canonical_paper_metadata",
        "paper_id": int(row.get("id") if row.get("id") is not None else row.get("paper_id")),
        "title": str(row.get("title") or ""),
        "abstract": str(row.get("abstract") or ""),
        "year": safe_int(row.get("year")),
        "venue": row.get("venue"),
        "source": row.get("source"),
        "source_url": row.get("source_url") or row.get("url"),
        "external_ids": identifiers,
        "citation_count": safe_int(row.get("citation_count")) or 0,
        "abstract_word_count": safe_int(row.get("abstract_word_count")),
        "source_path_or_table": source["path_or_connection"],
        "identity_validation_status": "passed",
    }


def write_canonical(rows: list[dict[str, Any]], source: dict[str, Any], metadata_out: Path, manifest_out: Path, validation: dict[str, Any]) -> dict[str, Any]:
    output_rows = [canonical_metadata_row(row, source) for row in rows]
    output_rows.sort(key=lambda row: row["paper_id"])
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with metadata_out.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    corpus_hash = hashlib.sha256(metadata_out.read_bytes()).hexdigest().upper()
    manifest = {
        "schema_version": "v6.1.6_canonical_corpus_manifest",
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": len(output_rows),
        "corpus_hash": corpus_hash,
        "source_path_or_table": source["path_or_connection"],
        "field_schema": sorted(output_rows[0].keys()) if output_rows else [],
        "id_policy": "Explicit ResearchPath paper_id/id from accepted canonical source; no fuzzy remapping.",
        "validation_summary": validation,
        "known_limitations": [],
    }
    write_json(manifest_out, manifest)
    return manifest


def coverage_against_canonical(label_sources: dict[str, Path], canonical_rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    canonical_by_id = {}
    if canonical_rows is not None:
        canonical_by_id = {
            int(row.get("id") if row.get("id") is not None else row.get("paper_id")): row
            for row in canonical_rows
            if row.get("id") is not None or row.get("paper_id") is not None
        }
    rows = []
    for source, path in label_sources.items():
        labels = load_jsonl(path) if path.exists() else []
        resolved = 0
        mismatches = 0
        exact_title_year = 0
        missing = 0
        for label in labels:
            paper_id = safe_int(label.get("paper_id"))
            if paper_id is None or paper_id not in canonical_by_id:
                missing += 1
                continue
            resolved += 1
            canonical = canonical_by_id[paper_id]
            title_match = normalize_title(canonical.get("title")) == normalize_title(label.get("title"))
            year_match = safe_int(label.get("year")) is None or safe_int(canonical.get("year")) == safe_int(label.get("year"))
            if title_match and year_match:
                exact_title_year += 1
            else:
                mismatches += 1
        rows.append(
            {
                "schema_version": "v6.1.6_label_coverage_against_canonical",
                "source": source,
                "path": str(path),
                "canonical_available": canonical_rows is not None,
                "total_rows": len(labels),
                "rows_resolved": resolved,
                "rows_missing": missing,
                "exact_title_year_match_count": exact_title_year,
                "exact_title_year_match_rate": exact_title_year / resolved if resolved else 0.0,
                "ambiguous_match_count": 0 if canonical_rows is not None else None,
                "mismatch_count": mismatches,
                "status": "evaluated" if canonical_rows is not None else "not_evaluated_no_canonical_source",
            }
        )
    return rows


def decision(canonical_source: dict[str, Any] | None, postgres: dict[str, Any]) -> str:
    if canonical_source is not None:
        if canonical_source["source_kind"] == "postgresql":
            return "postgres_full_corpus_available_use_that"
        return "canonical_metadata_created_ready_for_feature_regeneration"
    return "no_safe_metadata_source_found_rebuild_from_ingestion_required"


def render_markdown(report: dict[str, Any], candidates: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> str:
    lines = [
        "# V6.1.6 Canonical Corpus Metadata Reconstruction",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Safe canonical source found: `{report['safe_canonical_metadata_source_found']}`",
        f"- PostgreSQL full corpus available: `{report['postgresql']['full_corpus_available']}`",
        f"- Canonical metadata snapshot created: `{report['canonical_metadata_snapshot_created']}`",
        f"- Decision: `{report['decision_recommendation']}`",
        f"- Feature regeneration can proceed next: `{report['feature_regeneration_can_proceed_next']}`",
        f"- Raw row numbers used as paper_id: `{report['raw_row_number_alignment']['raw_jsonl_row_numbers_used_as_paper_id']}`",
        "",
        "## Source Candidates",
        "",
        "| name | kind | rows | id | title | abstract | year | safe canonical | reasons |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for candidate in candidates:
        reasons = "; ".join(candidate.get("canonical_rejection_reasons") or [])
        lines.append(
            f"| `{candidate['name']}` | `{candidate['source_kind']}` | `{candidate.get('row_count')}` | "
            f"`{candidate['has_explicit_paper_id']}` | `{candidate['has_title']}` | `{candidate['has_abstract']}` | "
            f"`{candidate['has_year']}` | `{candidate['can_safely_become_canonical']}` | {reasons[:180]} |"
        )
    lines.extend(["", "## Label Coverage", ""])
    for row in coverage:
        lines.append(
            f"- `{row['source']}`: `{row['status']}`, resolved `{row['rows_resolved']}` / `{row['total_rows']}`, "
            f"mismatches `{row['mismatch_count']}`"
        )
    lines.extend(["", "## Recommended Next Step", "", report["recommended_next_step"], "", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    v6_examples = load_jsonl(resolve_repo_path(args.v6_examples))
    candidates, canonical_rows = build_candidates(args, v6_examples)
    canonical_source = next((candidate for candidate in candidates if candidate["can_safely_become_canonical"]), None)
    manifest = None
    if canonical_source is not None and canonical_rows is not None:
        manifest = write_canonical(
            canonical_rows,
            canonical_source,
            resolve_repo_path(args.canonical_out),
            resolve_repo_path(args.manifest_out),
            canonical_source.get("v6_identity_validation") or {},
        )
    coverage = coverage_against_canonical(LABEL_SOURCES, canonical_rows if canonical_source is not None else None)
    postgres = next(candidate for candidate in candidates if candidate["name"] == "backend_default_postgresql")
    report = {
        "schema_version": "v6.1.6_canonical_corpus_metadata_report",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_integration_status": "offline_only_not_integrated",
        "default_ranker_status": "unchanged_bm25",
        "runtime_defaults_changed_by_this_script": False,
        "frontend_defaults_changed_by_this_script": False,
        "corpus_expanded_by_this_script": False,
        "labels_modified_by_this_script": False,
        "protected_labels_modified_by_this_script": False,
        "selected_240_modified_by_this_script": False,
        "models_retrained_by_this_script": False,
        "existing_model_artifacts_overwritten": False,
        "production_candidate_created": False,
        "external_api_calls_made": False,
        "fuzzy_remapping_performed": False,
        "raw_row_number_alignment": raw_row_number_alignment(REPO_ROOT / "data" / "raw" / "arxiv_ml_ai_50k_incremental.jsonl", v6_examples),
        "code_provenance_trace": code_trace(),
        "source_candidate_count": len(candidates),
        "source_candidates_path": str(resolve_repo_path(args.candidates_out)),
        "label_coverage_path": str(resolve_repo_path(args.coverage_out)),
        "safe_canonical_metadata_source_found": canonical_source is not None,
        "safe_canonical_source": canonical_source,
        "postgresql": {
            "connection_string": args.database_url,
            "available": postgres.get("exists_or_available"),
            "connection_error": postgres.get("connection_error"),
            "paper_count": postgres.get("row_count"),
            "schema": postgres.get("fields_available"),
            "full_corpus_available": postgres.get("exists_or_available") is True and postgres.get("row_count") == EXPECTED_CORPUS_SIZE,
            "v6_identity_validation": postgres.get("v6_identity_validation") or {},
        },
        "canonical_metadata_snapshot_created": manifest is not None,
        "canonical_metadata_path": str(resolve_repo_path(args.canonical_out)) if manifest is not None else None,
        "canonical_manifest_path": str(resolve_repo_path(args.manifest_out)) if manifest is not None else None,
        "canonical_row_count": manifest.get("row_count") if manifest else None,
        "canonical_corpus_hash": manifest.get("corpus_hash") if manifest else None,
        "v6_example_coverage": (canonical_source or {}).get("v6_identity_validation") or None,
        "feature_regeneration_can_proceed_next": manifest is not None,
        "decision_recommendation": decision(canonical_source, postgres),
        "recommended_next_step": (
            "Proceed to offline feature regeneration keyed by the canonical metadata snapshot."
            if manifest is not None
            else "Rebuild or make available a read-only full-corpus metadata source from the ingestion database with explicit paper_id, title, abstract, year, venue/source, and external IDs."
        ),
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }
    return report, candidates, coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", BACKEND_DEFAULT_DATABASE_URL))
    parser.add_argument("--v6-examples", default=str(DEFAULT_V6_EXAMPLES))
    parser.add_argument("--runtime-db", default=str(DEFAULT_RUNTIME_DB))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--faiss-ids", default=str(DEFAULT_FAISS_IDS))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--candidates-out", default=str(DEFAULT_CANDIDATES_OUT))
    parser.add_argument("--coverage-out", default=str(DEFAULT_COVERAGE_OUT))
    parser.add_argument("--canonical-out", default=str(DEFAULT_CANONICAL_OUT))
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST_OUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, candidates, coverage = build_report(args)
    write_jsonl(resolve_repo_path(args.candidates_out), candidates)
    write_jsonl(resolve_repo_path(args.coverage_out), coverage)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report, candidates, coverage))
    print(f"Wrote V6.1.6 canonical corpus metadata report to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
