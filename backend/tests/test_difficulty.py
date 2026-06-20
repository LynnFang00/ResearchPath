from app.models.paper import Paper
from app.services.difficulty import score_difficulty, technical_term_density


def test_difficulty_scores_survey_as_beginner_friendly():
    paper = Paper(
        id=1,
        title="A Survey and Tutorial on AI Agents",
        abstract="This introduction gives an overview for beginner researchers.",
        authors="",
        year=2021,
        categories="cs.AI",
        citation_count=1200,
        references_count=20,
    )

    result = score_difficulty(paper, current_year=2026)

    assert result.label == "beginner"
    assert "survey" in result.explanation


def test_difficulty_scores_recent_dense_method_as_advanced():
    paper = Paper(
        id=2,
        title="Contrastive Transformer Optimization for Graph Neural Agents",
        abstract=(
            "We study transformer attention, contrastive representation learning, "
            "policy gradient optimization, latent embeddings, and graph neural methods. "
        )
        * 6,
        authors="",
        year=2026,
        categories="cs.LG",
        citation_count=5,
        references_count=95,
    )

    result = score_difficulty(paper, current_year=2026)

    assert result.label == "advanced"
    assert result.score >= 0.67


def test_technical_term_density_handles_empty_text():
    assert technical_term_density("") == 0.0
