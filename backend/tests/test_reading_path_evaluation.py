import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_reading_paths import compute_metrics, load_labels  # noqa: E402


def test_manual_label_parsing(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "paper_id": 10,
                "relevance_score": 3,
                "section_correct": True,
                "duplicate": False,
                "too_advanced": False,
                "too_narrow": True,
                "notes": "Good but narrow.",
            }
        ),
        encoding="utf-8",
    )

    labels = load_labels(path)

    assert labels[0]["relevance_score"] == 3
    assert labels[0]["section_correct"] is True
    assert labels[0]["too_narrow"] is True


def test_reading_path_evaluation_metrics_output() -> None:
    rows = [
        {
            "reading_path": {
                "sections": {
                    "background": [{"final_path_score": 0.8}],
                    "foundational": [{"final_path_score": 0.6}],
                }
            }
        }
    ]
    labels = [
        {
            "query_id": "q1",
            "paper_id": 1,
            "relevance_score": 3,
            "section_correct": True,
            "duplicate": False,
            "too_advanced": False,
            "too_narrow": True,
        },
        {
            "query_id": "q1",
            "paper_id": 2,
            "relevance_score": 1,
            "section_correct": False,
            "duplicate": True,
            "too_advanced": True,
            "too_narrow": False,
        },
    ]

    metrics = compute_metrics(rows=rows, labels=labels)

    assert metrics["average_relevance"] == 2.0
    assert metrics["section_accuracy"] == 0.5
    assert metrics["duplicate_rate"] == 0.5
    assert metrics["too_narrow_rate"] == 0.5
    assert metrics["too_advanced_rate"] == 0.5
    assert metrics["average_score_by_section"]["background"] == 0.8
