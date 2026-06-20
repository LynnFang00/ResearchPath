from pathlib import Path

from app.services.dataset_manifest import (
    build_dataset_manifest,
    dataset_status_from_manifest,
    latest_manifest_path,
    write_dataset_manifest,
)


def test_manifest_creation_records_freshness_fields(tmp_path: Path) -> None:
    manifest = build_dataset_manifest(
        dataset_name="unit_dataset",
        source="unit",
        paper_count=12,
        citation_edge_count=34,
        last_updated_timestamp="2026-06-20T10:00:00+00:00",
        model_index_version="test-v1",
        embedding_model_name="test-embedding",
        faiss_index_path="data/processed/faiss/test.faiss",
    )
    path = write_dataset_manifest(manifest, manifest_dir=tmp_path)

    assert path.exists()
    assert latest_manifest_path(tmp_path) == path
    assert manifest["paper_count"] == 12
    assert manifest["citation_edge_count"] == 34
    assert manifest["last_updated_timestamp"] == "2026-06-20T10:00:00+00:00"
    assert manifest["model_index_version"] == "test-v1"
    assert manifest["embedding_model_name"] == "test-embedding"
    assert manifest["faiss_index_path"] == "data/processed/faiss/test.faiss"


def test_dataset_status_reads_legacy_manifest_names() -> None:
    status = dataset_status_from_manifest(
        {
            "dataset_name": "legacy",
            "source": "openalex",
            "number_of_papers": 7,
            "number_of_citation_edges": 9,
            "date_created": "2026-06-19T00:00:00+00:00",
        }
    )

    assert status["paper_count"] == 7
    assert status["citation_edge_count"] == 9
    assert status["last_updated_timestamp"] == "2026-06-19T00:00:00+00:00"
