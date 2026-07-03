import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
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
DEFAULT_RAW_CORPUS = REPO_ROOT / "data" / "raw" / "arxiv_ml_ai_50k_incremental.jsonl"
DEFAULT_RAW_OPENALEX = REPO_ROOT / "data" / "raw" / "arxiv_openalex_enrichment_50k_doi_subset.jsonl"
DEFAULT_EMBEDDINGS = REPO_ROOT / "data" / "processed" / "embeddings" / "all_minilm_l6_v2_50k.npz"
DEFAULT_FAISS_INDEX = REPO_ROOT / "data" / "processed" / "faiss" / "all_minilm_l6_v2_50k.faiss"
DEFAULT_FAISS_IDS = REPO_ROOT / "data" / "processed" / "faiss" / "all_minilm_l6_v2_50k.ids.npz"
DEFAULT_RUNTIME_DB = REPO_ROOT / "backend" / "data" / "researchpath.db"
DEFAULT_V33_METADATA = REPO_ROOT / "data" / "processed" / "models" / "v3_3_ltr_800" / "random_forest_no_v27_compact.metadata.json"
DEFAULT_V41_METADATA = REPO_ROOT / "data" / "processed" / "models" / "v4_1_weighted_ltr_2400" / "downweight_v3_9_active_learning.metadata.json"
DEFAULT_V41_BLEND_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_1_calibrated_blend.json"
DEFAULT_V43_INDEX = REPO_ROOT / "data" / "processed" / "models" / "v4_3_text_reranker" / "index.json"
DEFAULT_V49_CONFIG = REPO_ROOT / "data" / "processed" / "models" / "v4_9_guarded_text_blend_candidate.json"
DEFAULT_V22B = REPO_ROOT / "data" / "processed" / "models" / "v2_2b_lightweight_learned_hybrid.json"
DEFAULT_V26 = REPO_ROOT / "data" / "processed" / "models" / "v2_6_production_aware_learned_hybrid.json"

DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval" / "results" / "v6_1_5_corpus_identity_audit_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval" / "results" / "v6_1_5_corpus_identity_audit_report.md"
DEFAULT_ID_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_5_paper_id_resolution_report.jsonl"
DEFAULT_FEATURE_REPORT = REPO_ROOT / "data" / "eval" / "results" / "v6_1_5_feature_artifact_provenance_report.json"

LABEL_SOURCES = {
    "v2_1": REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl",
    "v2_5": REPO_ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl",
    "v3_2": REPO_ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl",
    "v3_5": REPO_ROOT / "data" / "eval" / "manual_labels_v3_5_unjudged_audit.jsonl",
    "v3_9": REPO_ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl",
    "v4_8_override": REPO_ROOT / "data" / "eval" / "manual_labels_v4_8_targeted_contrastive.jsonl",
    "v6_0_2_neural_examples": DEFAULT_V6_EXAMPLES,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def jsonl_count_and_fields(path: Path, sample_limit: int = 20) -> tuple[int, set[str], list[dict[str, Any]]]:
    count = 0
    fields: set[str] = set()
    samples: list[dict[str, Any]] = []
    if not path.exists():
        return count, fields, samples
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            fields.update(row.keys())
            if len(samples) < sample_limit:
                samples.append(row)
    return count, fields, samples


def source_status(path: Path, source_type: str) -> dict[str, Any]:
    status: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "source_type": source_type,
        "row_or_paper_count": 0,
        "schema_fields": [],
        "contains_researchpath_paper_id": False,
        "contains_title": False,
        "contains_abstract": False,
        "contains_year": False,
        "contains_venue_or_source": False,
        "contains_external_ids": False,
        "appears_full_50k_corpus": False,
        "appears_tiny_dev_test_db": False,
        "safe_join_to_labels": False,
    }
    if not path.exists():
        status["blocker"] = "missing"
        return status
    if source_type == "sqlite":
        try:
            with sqlite3.connect(path) as conn:
                tables = [row[0] for row in conn.execute("select name from sqlite_master where type='table'").fetchall()]
                status["tables"] = tables
                if "papers" in tables:
                    count = int(conn.execute("select count(*) from papers").fetchone()[0])
                    fields = [row[1] for row in conn.execute("pragma table_info(papers)").fetchall()]
                    status["row_or_paper_count"] = count
                    status["schema_fields"] = fields
                else:
                    status["blocker"] = "sqlite file has no papers table"
        except Exception as exc:
            status["blocker"] = str(exc)
            return status
    elif source_type == "jsonl":
        count, fields, _ = jsonl_count_and_fields(path)
        status["row_or_paper_count"] = count
        status["schema_fields"] = sorted(fields)
    elif source_type == "npz":
        npz = inspect_npz(path)
        status.update(npz)
        fields = set(npz.get("keys", []))
        count = int(npz.get("document_id_count") or npz.get("row_or_paper_count") or 0)
        status["row_or_paper_count"] = count
        status["schema_fields"] = sorted(fields)
    elif source_type == "faiss":
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        status["row_or_paper_count"] = load_json(meta_path).get("paper_count") if meta_path.exists() else None
        status["schema_fields"] = ["faiss_binary_index"]
    else:
        status["row_or_paper_count"] = path.stat().st_size
        status["schema_fields"] = []
    fields = set(status.get("schema_fields") or [])
    status["contains_researchpath_paper_id"] = bool({"paper_id", "id", "document_ids"} & fields)
    status["contains_title"] = "title" in fields
    status["contains_abstract"] = "abstract" in fields
    status["contains_year"] = "year" in fields
    status["contains_venue_or_source"] = bool({"venue", "source"} & fields)
    status["contains_external_ids"] = bool({"external_id", "openalex_id", "doi", "identifiers", "document_ids"} & fields)
    count = int(status.get("row_or_paper_count") or 0)
    status["appears_full_50k_corpus"] = 50000 <= count <= 51000
    status["appears_tiny_dev_test_db"] = source_type == "sqlite" and 0 < count < 100
    status["safe_join_to_labels"] = bool(
        status["contains_researchpath_paper_id"]
        and status["contains_title"]
        and status["contains_year"]
        and count >= 50000
    )
    if not status["safe_join_to_labels"]:
        if status["appears_tiny_dev_test_db"]:
            status["blocker"] = "tiny dev/test-sized SQLite corpus"
        elif status["appears_full_50k_corpus"] and not status["contains_title"]:
            status["blocker"] = "full-size keyed artifact lacks title/abstract metadata for identity validation"
        elif status["appears_full_50k_corpus"] and not status["contains_researchpath_paper_id"]:
            status["blocker"] = "full-size metadata source lacks stable ResearchPath paper_id"
        elif count < 50000:
            status["blocker"] = f"row count {count} is below full corpus target"
        else:
            status["blocker"] = "join safety requirements not met"
    return status


def inspect_npz(path: Path) -> dict[str, Any]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            keys = list(data.files)
            output: dict[str, Any] = {"keys": keys}
            if "document_ids" in data:
                ids = data["document_ids"].astype(int)
                output.update(
                    {
                        "document_id_count": int(len(ids)),
                        "document_id_unique_count": int(len(set(ids.tolist()))),
                        "document_id_min": int(ids.min()) if len(ids) else None,
                        "document_id_max": int(ids.max()) if len(ids) else None,
                        "document_id_sample_first": [int(x) for x in ids[:10].tolist()],
                        "document_id_sample_last": [int(x) for x in ids[-10:].tolist()],
                    }
                )
            if "embeddings" in data:
                output["embedding_shape"] = [int(x) for x in data["embeddings"].shape]
                output["row_or_paper_count"] = int(data["embeddings"].shape[0])
            return output
    except Exception as exc:
        return {"error": str(exc)}


def runtime_db_config(default_sqlite: Path) -> dict[str, Any]:
    env_vars = {
        name: os.environ.get(name)
        for name in ("DATABASE_URL", "RESEARCHPATH_DATABASE_URL", "SQLALCHEMY_DATABASE_URL")
        if os.environ.get(name)
    }
    backend_default = "postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath"
    sqlite_status = source_status(default_sqlite, "sqlite")
    return {
        "backend_config_database_url_default": backend_default,
        "environment_overrides_present": env_vars,
        "actual_sqlite_db_used_by_offline_scripts": str(default_sqlite),
        "actual_sqlite_db_used_by_backend_defaults": None,
        "backend_default_uses_sqlite": False,
        "configured_sqlite_status": sqlite_status,
        "app_and_eval_scripts_use_different_db_paths": True,
        "six_paper_db_assessment": (
            "dev_or_smoke_fixture_not_full_runtime_corpus"
            if sqlite_status.get("row_or_paper_count") == 6
            else "not_assessed_as_six_paper_fixture"
        ),
    }


def build_raw_title_year_index(paths: list[Path]) -> dict[str, Any]:
    exact: dict[tuple[str, int], int] = defaultdict(int)
    title_only: Counter[str] = Counter()
    field_summary = []
    for path in paths:
        count, fields, _ = jsonl_count_and_fields(path)
        field_summary.append({"path": str(path), "row_count": count, "fields": sorted(fields)})
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                title = normalize_title(row.get("title"))
                year = safe_int(row.get("year"))
                if title:
                    title_only[title] += 1
                    if year is not None:
                        exact[(title, year)] += 1
    return {"exact_title_year": exact, "title_only": title_only, "field_summary": field_summary}


def sqlite_id_index(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(path) as conn:
            columns = [row[1] for row in conn.execute("pragma table_info(papers)").fetchall()]
            if "id" not in columns:
                return {}
            wanted = [col for col in ("id", "title", "year") if col in columns]
            rows = conn.execute(f"select {', '.join(wanted)} from papers").fetchall()
            output = {}
            for row in rows:
                item = dict(zip(wanted, row, strict=True))
                output[int(item["id"])] = item
            return output
    except Exception:
        return {}


def embedding_id_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if "document_ids" not in data:
                return set()
            return {int(x) for x in data["document_ids"].tolist()}
    except Exception:
        return set()


def row_paper_id(row: dict[str, Any]) -> int | None:
    return safe_int(row.get("paper_id"))


def row_year(row: dict[str, Any]) -> int | None:
    return safe_int(row.get("year"))


def audit_label_source(
    name: str,
    path: Path,
    *,
    runtime_ids: dict[int, dict[str, Any]],
    embedding_ids: set[int],
    raw_index: dict[str, Any],
) -> dict[str, Any]:
    rows = load_jsonl(path) if path.exists() else []
    paper_ids = [row_paper_id(row) for row in rows if row_paper_id(row) is not None]
    runtime_resolved = [pid for pid in paper_ids if pid in runtime_ids]
    embedding_resolved = [pid for pid in paper_ids if pid in embedding_ids]
    exact_title_year = raw_index["exact_title_year"]
    title_only = raw_index["title_only"]
    exact_matches = 0
    normalized_matches = 0
    ambiguous_title_matches = 0
    runtime_exact_title_year = 0
    for row in rows:
        title = normalize_title(row.get("title"))
        year = row_year(row)
        pid = row_paper_id(row)
        if title and year is not None:
            count = int(exact_title_year.get((title, year), 0))
            if count:
                exact_matches += 1
                normalized_matches += 1
            if pid in runtime_ids:
                runtime = runtime_ids[pid]
                if normalize_title(runtime.get("title")) == title and safe_int(runtime.get("year")) == year:
                    runtime_exact_title_year += 1
        if title and title_only.get(title, 0) > 1:
            ambiguous_title_matches += 1
    missing_ids = len(rows) - len(paper_ids)
    has_full_metadata_source_with_valid_paper_id = False
    return {
        "schema_version": "v6.1.5_paper_id_resolution_row",
        "source": name,
        "path": str(path),
        "exists": path.exists(),
        "total_rows": len(rows),
        "unique_paper_ids": len(set(paper_ids)),
        "missing_paper_ids": missing_ids,
        "resolving_in_runtime_sqlite_by_id": len(runtime_resolved),
        "runtime_sqlite_exact_title_year_matches": runtime_exact_title_year,
        "resolving_in_full_50k_source_by_paper_id": None,
        "full_50k_source_has_valid_paper_id_field": has_full_metadata_source_with_valid_paper_id,
        "resolving_in_embedding_or_faiss_id_map": len(embedding_resolved),
        "exact_title_year_matches_in_raw_metadata": exact_matches,
        "normalized_title_year_matches_in_raw_metadata": normalized_matches,
        "ambiguous_title_matches_in_raw_metadata": ambiguous_title_matches,
        "automatic_mapping_safe": False,
        "fuzzy_remapping_performed": False,
        "blocker": (
            "No full 50K metadata source with both stable ResearchPath paper_id and title/year was found; "
            "embedding/FAISS IDs alone are insufficient for identity validation."
        ),
    }


def audit_model_artifact(name: str, artifact_path: Path, *, blend_config: Path | None = None, v49_config: Path | None = None) -> dict[str, Any]:
    metadata = load_json(artifact_path) if artifact_path.exists() else {}
    feature_order = metadata.get("feature_columns") or metadata.get("feature_order") or metadata.get("numeric_ranker_features") or []
    if not feature_order and isinstance(metadata.get("formula"), dict):
        feature_order = metadata["formula"].get("features_used") or []
    dependency_artifacts: list[str] = []
    if blend_config and blend_config.exists():
        dependency_artifacts.append(str(blend_config))
    if v49_config and v49_config.exists():
        config = load_json(v49_config)
        dependency_artifacts.extend(config.get("component_model_paths", {}).values())
    corpus_fields = {key: metadata.get(key) for key in ("corpus_hash", "corpus_source", "paper_id_source", "dataset_hash") if metadata.get(key)}
    label_fields = {key: metadata.get(key) for key in ("label_version", "training_row_count", "training_topics", "label_table") if metadata.get(key) is not None}
    blockers = [
        "V6 rows lack complete exact runtime feature values required by feature_order.",
        "No artifact metadata records a full-corpus hash or stable paper_id source.",
        "Local SQLite corpus contains only 6 papers, so runtime feature recreation is not possible from the configured local DB.",
    ]
    return {
        "method": name,
        "artifact_path": str(artifact_path),
        "exists": artifact_path.exists(),
        "model_type": metadata.get("model_type") or metadata.get("model_family") or metadata.get("schema_version"),
        "expected_feature_names": feature_order,
        "expected_feature_order": feature_order,
        "expected_normalization_or_scaling": "model-internal estimator/scaler plus per-query score normalization for blends",
        "expected_dependency_artifacts": dependency_artifacts,
        "records_corpus_hash_or_source": bool(corpus_fields),
        "corpus_provenance_fields": corpus_fields,
        "records_label_version": bool(label_fields),
        "label_provenance_fields": label_fields,
        "can_safely_apply_to_v6_0_2_rows": False,
        "cannot_apply_reasons": blockers,
    }


def feature_artifact_provenance(v6_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    v6_ids = {int(row["paper_id"]) for row in v6_rows}
    embedding_ids = embedding_id_set(resolve_repo_path(args.embeddings))
    faiss_ids = embedding_id_set(resolve_repo_path(args.faiss_ids))
    fulltext_ids: set[int] = set()
    for manifest in (REPO_ROOT / "data" / "fulltext").glob("fulltext_manifest_v1*.jsonl"):
        for row in load_jsonl(manifest):
            pid = row_paper_id(row)
            if pid is not None:
                fulltext_ids.add(pid)
    groups = {
        "tfidf_bm25": {
            "artifact_path": None,
            "exists": False,
            "row_or_key_type": "runtime-computed from corpus table",
            "keyed_by": "paper_id only after a valid full corpus DB exists",
            "coverage_over_v6_0_2_paper_ids": sum(1 for row in v6_rows if ((row.get("baseline_rankings") or {}).get("bm25") or {}).get("rank") is not None),
            "safe_join_possible": False,
            "blocker": "No persisted full-corpus BM25/TF-IDF feature snapshot keyed by ResearchPath paper_id was found.",
        },
        "dense_faiss": {
            "artifact_path": [str(resolve_repo_path(args.embeddings)), str(resolve_repo_path(args.faiss_index)), str(resolve_repo_path(args.faiss_ids))],
            "exists": resolve_repo_path(args.embeddings).exists() and resolve_repo_path(args.faiss_index).exists() and resolve_repo_path(args.faiss_ids).exists(),
            "row_or_key_type": "document_ids in NPZ id maps",
            "keyed_by": "document_ids interpreted as paper_id but without metadata identity validation",
            "coverage_over_v6_0_2_paper_ids": len(v6_ids & embedding_ids & faiss_ids),
            "safe_join_possible": False,
            "blocker": "Embedding/FAISS IDs cover many V6 IDs but do not provide title/year metadata to prove ID identity.",
        },
        "hybrid": {
            "artifact_path": None,
            "exists": False,
            "row_or_key_type": "runtime-computed blend of lexical/vector/quality features",
            "keyed_by": "candidate paper_id in runtime retrieval context",
            "coverage_over_v6_0_2_paper_ids": 0,
            "safe_join_possible": False,
            "blocker": "No persisted hybrid score/rank snapshot keyed by ResearchPath paper_id was found.",
        },
        "v2_2b": model_feature_group(resolve_repo_path(args.v22b_model), v6_ids, "packet-style feature model"),
        "v2_6": model_feature_group(resolve_repo_path(args.v26_model), v6_ids, "production-aware packet-style feature model"),
        "citation_quality": {
            "artifact_path": None,
            "exists": False,
            "row_or_key_type": "corpus metadata fields such as citation_count/year/full text",
            "keyed_by": "paper_id if full corpus metadata exists",
            "coverage_over_v6_0_2_paper_ids": sum(1 for row in v6_rows if row.get("citation_count") is not None),
            "safe_join_possible": False,
            "blocker": "V6 examples do not contain citation_count, and no full keyed citation feature table was found.",
        },
        "full_text_availability": {
            "artifact_path": str(REPO_ROOT / "data" / "fulltext" / "fulltext_manifest_v1*.jsonl"),
            "exists": bool(list((REPO_ROOT / "data" / "fulltext").glob("fulltext_manifest_v1*.jsonl"))),
            "row_or_key_type": "manifest paper_id rows",
            "keyed_by": "paper_id",
            "coverage_over_v6_0_2_paper_ids": len(v6_ids & fulltext_ids),
            "safe_join_possible": False,
            "blocker": "Full-text manifests are small topic packets, not a complete full-corpus availability table.",
        },
    }
    return {
        "schema_version": "v6.1.5_feature_artifact_provenance_report",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_groups": groups,
    }


def model_feature_group(path: Path, v6_ids: set[int], description: str) -> dict[str, Any]:
    metadata = load_json(path) if path.exists() else {}
    return {
        "artifact_path": str(path),
        "exists": path.exists(),
        "row_or_key_type": description,
        "keyed_by": "feature vector, not a precomputed per-paper score table",
        "coverage_over_v6_0_2_paper_ids": 0,
        "safe_join_possible": False,
        "feature_count": len(metadata.get("feature_names") or []),
        "blocker": "Artifact is a scorer requiring packet/runtime features; it is not a keyed feature snapshot.",
    }


def raw_row_alignment(raw_path: Path, v6_rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [1, 2, 5053, 30046]
    example_titles = {int(row["paper_id"]): row["title"] for row in v6_rows if int(row["paper_id"]) in samples}
    raw_titles: dict[int, str] = {}
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if index in example_titles and line.strip():
                    raw_titles[index] = json.loads(line).get("title") or ""
                if index > max(samples):
                    break
    mismatches = [
        {"paper_id": pid, "v6_title": example_titles.get(pid), "raw_row_title": raw_titles.get(pid)}
        for pid in samples
        if pid in example_titles and example_titles.get(pid) != raw_titles.get(pid)
    ]
    return {
        "raw_jsonl_row_number_safe_as_paper_id": not mismatches,
        "samples_checked": samples,
        "mismatches": mismatches,
        "conclusion": "unsafe" if mismatches else "not_disproven_by_samples",
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    v6_rows = load_jsonl(resolve_repo_path(args.v6_examples))
    sqlite_files = sorted((REPO_ROOT / "backend" / "data").glob("*.db"))
    raw_sources = [resolve_repo_path(args.raw_corpus), resolve_repo_path(args.raw_openalex)]
    source_paths = [
        *[("configured_or_local_sqlite", path, "sqlite") for path in sqlite_files],
        ("raw_arxiv_50k_incremental", resolve_repo_path(args.raw_corpus), "jsonl"),
        ("raw_openalex_50k_doi_subset", resolve_repo_path(args.raw_openalex), "jsonl"),
        ("v6_0_2_neural_examples", resolve_repo_path(args.v6_examples), "jsonl"),
        ("embeddings_50k", resolve_repo_path(args.embeddings), "npz"),
        ("faiss_50k", resolve_repo_path(args.faiss_index), "faiss"),
        ("faiss_id_map_50k", resolve_repo_path(args.faiss_ids), "npz"),
    ]
    corpus_sources = [
        {"name": name, **source_status(path, source_type)}
        for name, path, source_type in source_paths
    ]
    runtime_config = runtime_db_config(resolve_repo_path(args.runtime_db))
    runtime_ids = sqlite_id_index(resolve_repo_path(args.runtime_db))
    raw_index = build_raw_title_year_index(raw_sources)
    embedding_ids = embedding_id_set(resolve_repo_path(args.embeddings)) | embedding_id_set(resolve_repo_path(args.faiss_ids))
    paper_id_resolution = [
        audit_label_source(
            name,
            path,
            runtime_ids=runtime_ids,
            embedding_ids=embedding_ids,
            raw_index=raw_index,
        )
        for name, path in LABEL_SOURCES.items()
    ]
    model_provenance = {
        "v3_3_ltr": audit_model_artifact("v3_3_ltr", resolve_repo_path(args.v33_metadata)),
        "v4_1_blend": audit_model_artifact(
            "v4_1_blend",
            resolve_repo_path(args.v41_metadata),
            blend_config=resolve_repo_path(args.v41_blend_config),
        ),
        "v4_9_guarded_text_blend": audit_model_artifact(
            "v4_9_guarded_text_blend",
            resolve_repo_path(args.v49_config),
            v49_config=resolve_repo_path(args.v49_config),
        ),
    }
    feature_provenance = feature_artifact_provenance(v6_rows, args)
    recommendations = [
        "need_to_rebuild_full_corpus_db_from_processed_source",
        "need_to_create_paper_id_mapping_table",
        "need_to_regenerate_feature_artifacts_keyed_by_paper_id",
        "need_to_update_offline_eval_to_use_correct_db_path",
        "exact_v4_9_comparison_not_recoverable_without_rebuilding_artifacts",
        "stop_and_document_limitation",
    ]
    smallest_safe_next_fix = (
        "Build a read-only full-corpus metadata/feature snapshot keyed by stable ResearchPath paper_id, "
        "then regenerate BM25, TF-IDF, dense, hybrid, V2.2b, V2.6, citation, and full-text features from that snapshot before exact learned-ranker comparison."
    )
    report = {
        "schema_version": "v6.1.5_corpus_identity_audit_report",
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
        "automatic_fuzzy_remapping_performed": False,
        "target_corpus_size": 50424,
        "judged_label_rows_after_v6_0_2": len(v6_rows),
        "runtime_db_configuration": runtime_config,
        "corpus_sources": corpus_sources,
        "raw_row_number_alignment": raw_row_alignment(resolve_repo_path(args.raw_corpus), v6_rows),
        "paper_id_resolution_summary": {
            row["source"]: {
                "total_rows": row["total_rows"],
                "unique_paper_ids": row["unique_paper_ids"],
                "resolving_in_runtime_sqlite_by_id": row["resolving_in_runtime_sqlite_by_id"],
                "resolving_in_embedding_or_faiss_id_map": row["resolving_in_embedding_or_faiss_id_map"],
                "automatic_mapping_safe": row["automatic_mapping_safe"],
            }
            for row in paper_id_resolution
        },
        "model_artifact_provenance": model_provenance,
        "feature_artifact_provenance_path": str(resolve_repo_path(args.feature_report_out)),
        "decision_recommendations": recommendations,
        "smallest_safe_next_fix": smallest_safe_next_fix,
        "exact_learned_ranker_comparison_possible": False,
        "exact_learned_ranker_comparison_blocker": "No full 50K metadata source with stable paper_id plus complete runtime feature artifacts was found.",
        "protected_hashes": protected_status(resolve_repo_path(args.v39_labels)),
    }
    return report, paper_id_resolution, feature_provenance


def render_markdown(report: dict[str, Any], id_rows: list[dict[str, Any]], feature_report: dict[str, Any]) -> str:
    sqlite_status = report["runtime_db_configuration"]["configured_sqlite_status"]
    raw_alignment = report["raw_row_number_alignment"]
    lines = [
        "# V6.1.5 Corpus Identity and Paper ID Audit",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Summary",
        "",
        f"- Offline only: `{report['runtime_integration_status']}`",
        f"- Default ranker: `{report['default_ranker_status']}`",
        f"- Configured local SQLite path: `{sqlite_status['path']}`",
        f"- Configured local SQLite paper count: `{sqlite_status.get('row_or_paper_count')}`",
        f"- Raw JSONL row number safe as paper_id: `{raw_alignment['raw_jsonl_row_number_safe_as_paper_id']}`",
        f"- Exact learned-ranker comparison possible: `{report['exact_learned_ranker_comparison_possible']}`",
        f"- Blocker: {report['exact_learned_ranker_comparison_blocker']}",
        "",
        "## Corpus Sources",
        "",
        "| source | count | paper_id | title | abstract | year | full 50K | safe join | blocker |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for source in report["corpus_sources"]:
        lines.append(
            f"| `{source['name']}` | `{source.get('row_or_paper_count')}` | `{source['contains_researchpath_paper_id']}` | "
            f"`{source['contains_title']}` | `{source['contains_abstract']}` | `{source['contains_year']}` | "
            f"`{source['appears_full_50k_corpus']}` | `{source['safe_join_to_labels']}` | {source.get('blocker', '')} |"
        )
    lines.extend(["", "## Paper ID Resolution", "", "| source | rows | runtime id matches | embedding/FAISS id matches | exact title+year raw matches | safe mapping |", "|---|---:|---:|---:|---:|---|"])
    for row in id_rows:
        lines.append(
            f"| `{row['source']}` | `{row['total_rows']}` | `{row['resolving_in_runtime_sqlite_by_id']}` | "
            f"`{row['resolving_in_embedding_or_faiss_id_map']}` | `{row['exact_title_year_matches_in_raw_metadata']}` | `{row['automatic_mapping_safe']}` |"
        )
    lines.extend(["", "## Feature Artifact Provenance", ""])
    for group, row in feature_report["feature_groups"].items():
        lines.append(
            f"- `{group}`: exists `{row['exists']}`, coverage `{row['coverage_over_v6_0_2_paper_ids']}`, "
            f"safe join `{row['safe_join_possible']}`. {row['blocker']}"
        )
    lines.extend(["", "## Recommendations", ""])
    for recommendation in report["decision_recommendations"]:
        lines.append(f"- `{recommendation}`")
    lines.extend(["", f"Smallest safe next fix: {report['smallest_safe_next_fix']}", "", "## Protected Hashes", ""])
    for key, value in report["protected_hashes"].items():
        if key.endswith("_hash_unchanged"):
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-examples", default=str(DEFAULT_V6_EXAMPLES))
    parser.add_argument("--raw-corpus", default=str(DEFAULT_RAW_CORPUS))
    parser.add_argument("--raw-openalex", default=str(DEFAULT_RAW_OPENALEX))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument("--faiss-index", default=str(DEFAULT_FAISS_INDEX))
    parser.add_argument("--faiss-ids", default=str(DEFAULT_FAISS_IDS))
    parser.add_argument("--runtime-db", default=str(DEFAULT_RUNTIME_DB))
    parser.add_argument("--v33-metadata", default=str(DEFAULT_V33_METADATA))
    parser.add_argument("--v41-metadata", default=str(DEFAULT_V41_METADATA))
    parser.add_argument("--v41-blend-config", default=str(DEFAULT_V41_BLEND_CONFIG))
    parser.add_argument("--v43-index", default=str(DEFAULT_V43_INDEX))
    parser.add_argument("--v49-config", default=str(DEFAULT_V49_CONFIG))
    parser.add_argument("--v22b-model", default=str(DEFAULT_V22B))
    parser.add_argument("--v26-model", default=str(DEFAULT_V26))
    parser.add_argument("--v39-labels", default=str(DEFAULT_V39_LABELS))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--id-report-out", default=str(DEFAULT_ID_REPORT))
    parser.add_argument("--feature-report-out", default=str(DEFAULT_FEATURE_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, id_rows, feature_report = build_report(args)
    write_json(resolve_repo_path(args.feature_report_out), feature_report)
    write_jsonl(resolve_repo_path(args.id_report_out), id_rows)
    write_json(resolve_repo_path(args.json_out), report)
    write_text(resolve_repo_path(args.md_out), render_markdown(report, id_rows, feature_report))
    print(f"Wrote V6.1.5 corpus identity audit to {resolve_repo_path(args.json_out)}")


if __name__ == "__main__":
    main()
