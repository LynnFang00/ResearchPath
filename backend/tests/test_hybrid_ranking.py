from app.models.paper import Paper
from app.services.difficulty import score_difficulty
from app.services.quality_signals import compute_quality_signals, normalize_scores, section_score
from app.services.retrievers.hybrid import HybridRetriever


def test_hybrid_score_normalization_bounds_components() -> None:
    normalized = normalize_scores({1: 10.0, 2: 5.0, 3: 0.0})

    assert normalized[1] == 1.0
    assert normalized[2] == 0.5
    assert normalized[3] == 0.0


def test_hybrid_retriever_combines_lexical_and_quality_components() -> None:
    papers = [
        Paper(
            id=1,
            title="A Survey of Transformer Architectures",
            abstract="A survey and tutorial overview of transformer architecture methods.",
            authors="A",
            year=2021,
            categories="cs.CL",
            citation_count=500,
            references_count=50,
        ),
        Paper(
            id=2,
            title="Medical Transformer for a Narrow Dataset",
            abstract="A clinical case study for a narrow medical dataset.",
            authors="B",
            year=2025,
            categories="cs.LG",
            citation_count=3,
            references_count=10,
        ),
    ]

    retriever = HybridRetriever(papers, current_year=2026)
    results = retriever.search("transformer", k=2)

    assert results[0].document_id == 1
    assert 0.0 <= retriever.components_for(1)["bm25_score"] <= 1.0
    assert "too_narrow_penalty" in retriever.components_for(2)


def test_section_score_prefers_matching_section_signals() -> None:
    survey = Paper(
        id=1,
        title="A Survey of Large Language Models",
        abstract="A tutorial overview and survey for beginners.",
        authors="A",
        year=2023,
        categories="cs.CL",
        citation_count=200,
    )
    method = Paper(
        id=2,
        title="An Efficient Transformer Architecture",
        abstract="This paper introduces an algorithm, benchmark, and architecture.",
        authors="B",
        year=2024,
        categories="cs.CL",
        citation_count=80,
    )
    survey_signals = compute_quality_signals(
        query="large language model",
        paper=survey,
        difficulty=score_difficulty(survey, current_year=2026),
        background_level="basic_ml",
        bm25_score=1.0,
    )
    method_signals = compute_quality_signals(
        query="large language model",
        paper=method,
        difficulty=score_difficulty(method, current_year=2026),
        background_level="basic_ml",
        bm25_score=1.0,
    )

    assert section_score("background", survey_signals, query="large language model") > section_score(
        "background", method_signals, query="large language model"
    )
    assert section_score("core_methods", method_signals, query="large language model") > section_score(
        "core_methods", survey_signals, query="large language model"
    )
