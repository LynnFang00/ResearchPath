import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "training" / "v4_3_text_reranker" / "v4_3_text_reranker_dataset.jsonl"
REPORT = ROOT / "data" / "eval" / "results" / "v4_3_text_reranker_dataset_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v4_3_dataset_shape_and_dedup() -> None:
    rows = load_jsonl(DATASET)
    assert len(rows) == 2400
    assert len({(row["query_id"], row["paper_id"]) for row in rows}) == 2400
    assert len({row["query_id"] for row in rows}) == 16
    assert {row["split"] for row in rows} == {"train", "dev", "test"}


def test_v4_3_dataset_text_and_feature_contract() -> None:
    rows = load_jsonl(DATASET)
    assert all(row["title"] for row in rows)
    assert all(row["text_input"] for row in rows)
    assert any(row["large_score_disagreement"] or row["v3_3_promoted_v4_demoted"] or row["v4_fixed_v3_hard_negative"] for row in rows)
    for row in rows:
        features = row["numeric_ranker_features"]
        assert "v2_7_score" not in features
        assert not any(name.endswith("_raw_score") for name in features)


def test_v4_3_dataset_report_protected_hashes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["dataset"]["row_count"] == 2400
    assert report["dataset"]["duplicate_query_paper_rows"] == []
    unchanged = {key: value for key, value in report["protected_hashes"].items() if key.endswith("_hash_unchanged")}
    assert unchanged
    assert all(unchanged.values())
