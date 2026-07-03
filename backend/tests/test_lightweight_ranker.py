import json
from pathlib import Path
import sys

from app.models.paper import Paper
from app.services.learned_ranker import LinearRanker, load_linear_ranker, save_linear_ranker
from app.services.retrievers.hybrid import HybridRetriever


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_lightweight_ranker import build_training_examples, train_linear_ranker  # noqa: E402


def test_linear_ranker_round_trips_and_scores_components(tmp_path: Path) -> None:
    path = tmp_path / "ranker.json"
    ranker = LinearRanker(
        feature_names=["bm25_score", "citation_score"],
        weights=[0.4, 0.1],
        intercept=0.2,
        version="test",
    )

    save_linear_ranker(ranker, path)
    loaded = load_linear_ranker(path)

    assert loaded is not None
    assert loaded.version == "test"
    assert loaded.score({"bm25_score": 1.0, "citation_score": 2.0}) == 0.8


def test_train_lightweight_ranker_from_review_labels() -> None:
    review_rows = [
        {
            "query_id": "q1",
            "reading_path": {
                "sections": {
                    "background": [
                        {"paper_id": 1, "bm25_score": 1.0, "citation_score": 0.2},
                        {"paper_id": 2, "bm25_score": 0.1, "citation_score": 0.0},
                    ]
                }
            },
        }
    ]
    labels = {
        ("q1", 1): {"relevance_score": 3},
        ("q1", 2): {"relevance_score": 0},
    }

    examples = build_training_examples(
        review_rows=review_rows,
        labels=labels,
        feature_names=["bm25_score", "citation_score"],
    )
    ranker, metrics = train_linear_ranker(examples, feature_names=["bm25_score", "citation_score"])

    assert metrics["example_count"] == 2
    assert ranker.score({"bm25_score": 1.0, "citation_score": 0.2}) > ranker.score(
        {"bm25_score": 0.1, "citation_score": 0.0}
    )


def test_hybrid_retriever_applies_learned_ranker_adjustment(tmp_path: Path) -> None:
    model_path = tmp_path / "ranker.json"
    model_path.write_text(
        json.dumps(
            {
                "version": "test",
                "feature_names": ["bm25_score"],
                "weights": [1.0],
                "intercept": 0.0,
            }
        ),
        encoding="utf-8",
    )
    papers = [
        Paper(
            id=1,
            title="Transformer Survey",
            abstract="survey of transformer models",
            authors="A",
            year=2020,
            citation_count=100,
        ),
        Paper(
            id=2,
            title="Transformer Application",
            abstract="transformer for a narrow application",
            authors="B",
            year=2024,
            citation_count=5,
        ),
    ]

    retriever = HybridRetriever(papers, learned_ranker_path=model_path, current_year=2026)
    results = retriever.search("transformer", k=2)
    components = retriever.components_for(results[0].document_id)

    assert components["learned_ranker_version"] == "test"
    assert "learned_ranker_adjustment" in components
