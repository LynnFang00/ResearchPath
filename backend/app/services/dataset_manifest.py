from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.citation_edge import CitationEdge
from app.models.paper import Paper


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "data" / "processed" / "manifests"
DEFAULT_MODEL_INDEX_VERSION = "heuristic-ranking-v1"


def safe_manifest_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe or "dataset"


def latest_manifest_path(manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> Path | None:
    if not manifest_dir.exists():
        return None
    candidates = [path for path in manifest_dir.glob("*.json") if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if _looks_like_dataset_manifest(path):
            return path
    return candidates[0]


def _looks_like_dataset_manifest(path: Path) -> bool:
    try:
        manifest = read_manifest(path)
    except (OSError, json.JSONDecodeError):
        return False
    has_paper_count = "paper_count" in manifest or "number_of_papers" in manifest
    has_edge_count = "citation_edge_count" in manifest or "number_of_citation_edges" in manifest
    return bool(manifest.get("dataset_name") and manifest.get("source") and has_paper_count and has_edge_count)


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_timestamp(manifest: dict[str, Any]) -> str | None:
    return (
        manifest.get("last_updated_timestamp")
        or manifest.get("last_updated")
        or manifest.get("updated_at")
        or manifest.get("date_created")
        or manifest.get("created_at")
    )


def _manifest_paper_count(manifest: dict[str, Any]) -> int:
    value = manifest.get("paper_count", manifest.get("number_of_papers", 0))
    return int(value or 0)


def _manifest_edge_count(manifest: dict[str, Any]) -> int:
    value = manifest.get("citation_edge_count", manifest.get("number_of_citation_edges", 0))
    return int(value or 0)


def _count_rows(db: Session, model: type[Paper] | type[CitationEdge]) -> int:
    return int(db.scalar(select(func.count()).select_from(model)) or 0)


def _latest_paper_update(db: Session) -> str | None:
    latest = db.scalar(select(func.max(Paper.updated_at)))
    if latest is None:
        return None
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=UTC)
    return latest.isoformat()


def build_dataset_manifest(
    *,
    dataset_name: str,
    source: str,
    paper_count: int,
    citation_edge_count: int,
    last_updated_timestamp: str | None = None,
    model_index_version: str = DEFAULT_MODEL_INDEX_VERSION,
    embedding_model_name: str | None = None,
    faiss_index_path: str | None = None,
    fields_available: list[str] | None = None,
    filters_used: dict[str, Any] | None = None,
    notes: str | None = None,
    inserted: int = 0,
    skipped: int = 0,
    citation_edges_inserted: int = 0,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    timestamp = last_updated_timestamp or datetime.now(UTC).isoformat()
    return {
        "dataset_name": dataset_name,
        "source": source,
        "paper_count": int(paper_count),
        "citation_edge_count": int(citation_edge_count),
        "last_updated_timestamp": timestamp,
        "model_index_version": model_index_version,
        "embedding_model_name": embedding_model_name or settings.embedding_model_name,
        "faiss_index_path": faiss_index_path or settings.faiss_index_path,
        "date_created": timestamp,
        "number_of_papers": int(paper_count),
        "number_of_citation_edges": int(citation_edge_count),
        "fields_available": fields_available or [],
        "filters_used": filters_used or {},
        "notes": notes or "",
        "inserted": int(inserted),
        "skipped": int(skipped),
        "citation_edges_inserted": int(citation_edges_inserted),
        "errors": errors or [],
        "warnings": warnings or [],
    }


def write_dataset_manifest(
    manifest: dict[str, Any],
    *,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _manifest_timestamp(manifest) or datetime.now(UTC).isoformat()
    dataset_name = str(manifest.get("dataset_name") or "dataset")
    suffix = timestamp.replace(":", "").replace("+", "_").replace(".", "_")
    output_path = manifest_dir / f"{safe_manifest_name(dataset_name)}_{suffix}.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def dataset_status_from_manifest(
    manifest: dict[str, Any] | None,
    *,
    manifest_path: Path | None = None,
    db: Session | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    paper_count = _manifest_paper_count(manifest or {})
    citation_edge_count = _manifest_edge_count(manifest or {})
    last_updated = _manifest_timestamp(manifest or {})

    if db is not None:
        paper_count = _count_rows(db, Paper)
        citation_edge_count = _count_rows(db, CitationEdge)
        last_updated = _latest_paper_update(db) or last_updated

    return {
        "dataset_name": (manifest or {}).get("dataset_name", "unknown"),
        "source": (manifest or {}).get("source", "unknown"),
        "paper_count": paper_count,
        "citation_edge_count": citation_edge_count,
        "last_updated_timestamp": last_updated,
        "model_index_version": (manifest or {}).get("model_index_version", DEFAULT_MODEL_INDEX_VERSION),
        "embedding_model_name": (manifest or {}).get("embedding_model_name", settings.embedding_model_name),
        "faiss_index_path": (manifest or {}).get("faiss_index_path", settings.faiss_index_path),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
    }


def get_dataset_status(
    *,
    db: Session,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
) -> dict[str, Any]:
    path = latest_manifest_path(manifest_dir)
    manifest = read_manifest(path) if path is not None else None
    return dataset_status_from_manifest(manifest, manifest_path=path, db=db)
