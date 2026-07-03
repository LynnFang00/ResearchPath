import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_v2_1_labeling_packets import build_packets, validate_same_candidates  # noqa: E402
from scripts.evaluate_v2_1_benchmark import build_report  # noqa: E402
from scripts.validate_v2_1_labels import validate_labels  # noqa: E402


def _packet_row(query_id: str, paper_id: int, *, rank: int, likely_coverage: str) -> dict:
    return {
        "query_id": query_id,
        "query": "transformer architecture",
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "This is a complete synthetic abstract for packet generation tests.",
        "labeling_packet_rank": rank,
        "likely_coverage": likely_coverage,
        "selection_reasons": ["test"],
        "retrieval_ranks_by_method": {"bm25": rank, "hybrid": 3 - rank},
        "abstract_snippet": "This field must not appear in generated V2.1 packets...",
    }


def _valid_label(query_id: str, paper_id: int, *, role: str = "foundational", reading_value: float = 1.0) -> dict:
    return {
        "schema_version": "v2.1",
        "query_id": query_id,
        "query": "transformer architecture",
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "topic_match_score": 1.0,
        "reading_value_score": reading_value,
        "beginner_fit_score": 0.8,
        "intermediate_fit_score": 1.0,
        "advanced_fit_score": 0.6,
        "expert_fit_score": 0.3,
        "intent_scores": {
            "background": 0.2,
            "foundational": 1.0 if role == "foundational" else 0.0,
            "core_methods": 1.0 if role == "core_methods" else 0.0,
            "recent_frontier": 0.0,
            "evaluation_benchmark": 0.0,
            "application": 0.0,
        },
        "primary_role": role,
        "secondary_roles": [],
        "duplicate_status": "none",
        "duplicate_of_paper_id": None,
        "evidence_level": "title_abstract",
        "full_text_available": False,
        "label_confidence": "medium",
        "notes": "Synthetic label with enough rationale for validation.",
    }


def test_v2_1_packet_builder_preserves_selected_candidates_and_adds_anchors() -> None:
    rows = [
        _packet_row("q1", 1, rank=1, likely_coverage="likely core/foundational positive"),
        _packet_row("q1", 2, rank=2, likely_coverage="likely random/irrelevant negative"),
    ]

    packets, anchors_by_topic = build_packets(rows)

    validate_same_candidates(rows, packets)
    assert [(row["query_id"], row["paper_id"]) for row in packets] == [("q1", 1), ("q1", 2)]
    assert packets[0]["v2_1_packet_version"] == "v2.1"
    assert "abstract_snippet" not in packets[0]
    assert packets[0]["abstract"] == rows[0]["abstract"]
    assert packets[0]["likely_coverage"]["heuristic_only"] is True
    assert len(anchors_by_topic["q1"]) == 2


def test_v2_1_validator_reports_score_errors_and_soft_warnings() -> None:
    label = _valid_label("q1", 1, role="negative", reading_value=0.9)
    label["topic_match_score"] = 1.2

    report = validate_labels([label], packet_keys={("q1", 1)})

    assert report["is_valid"] is False
    assert any(item["field"] == "topic_match_score" for item in report["invalid_values"])
    assert any(item["code"] == "negative_role_high_reading_value" for item in report["warnings"])
    assert report["per_topic"]["q1"]["label_count"] == 1


def test_v2_1_evaluator_reports_requested_scaffolding_metrics() -> None:
    packet_rows = [
        _packet_row("q1", 1, rank=1, likely_coverage="likely core/foundational positive"),
        _packet_row("q1", 2, rank=2, likely_coverage="likely recent frontier/application"),
    ]
    labels = [
        _valid_label("q1", 1, role="foundational", reading_value=1.0),
        _valid_label("q1", 2, role="core_methods", reading_value=0.5),
    ]

    report = build_report(
        labels=labels,
        packet_rows=packet_rows,
        requested_methods=["bm25", "hybrid"],
        k=2,
        labels_path=REPO_ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl",
        packet_path=REPO_ROOT / "data" / "eval" / "v2_1_labeling_packets.jsonl",
    )

    bm25 = report["methods"]["bm25"]["averages"]
    assert bm25["topic_match_ndcg"] == 1.0
    assert bm25["reading_value_ndcg"] == 1.0
    assert "intent_foundational_ndcg" in bm25
    assert "beginner_ndcg" in bm25
    assert "path_level_coverage_at_k" in bm25
    assert "duplicate_penalty_at_k" in bm25
