import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_openalex_dataset import (  # noqa: E402
    compact_ingestion_result,
    dataset_paths,
    write_pipeline_summary,
)


def test_dataset_paths_are_named_consistently():
    paths = dataset_paths("openalex_test")

    assert paths["seed"].name == "openalex_test.jsonl"
    assert paths["references"].name == "openalex_test_references.jsonl"
    assert paths["weak_labels"].name == "openalex_test_weak_labels.jsonl"
    assert paths["run_summary"].name == "openalex_test_pipeline_summary.json"


def test_write_pipeline_summary(tmp_path):
    output_path = tmp_path / "summary.json"
    write_pipeline_summary(output_path, {"dataset_name": "openalex_test", "paper_count": 3})

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {"dataset_name": "openalex_test", "paper_count": 3}


def test_compact_ingestion_result_keeps_warning_preview():
    result = compact_ingestion_result(
        {
            "inserted": 2,
            "skipped": 1,
            "citation_edges_inserted": 5,
            "errors": [],
            "warnings": ["first", "second", "third"],
            "manifest_path": "manifest.json",
        },
        max_warnings=2,
    )

    assert result["warnings_count"] == 3
    assert result["warnings_preview"] == ["first", "second"]
    assert result["citation_edges_inserted"] == 5
