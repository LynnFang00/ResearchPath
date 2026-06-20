from app.services.retrievers.base import PaperDocument
from app.services.retrievers.citation_recency import CitationRecencyRetriever
from app.services.retrievers.tfidf import TfidfRetriever


def test_tfidf_retriever_ranks_relevant_document_first() -> None:
    retriever = TfidfRetriever(
        [
            PaperDocument(1, "graph neural networks for molecules"),
            PaperDocument(2, "AI agents use tools and reasoning"),
            PaperDocument(3, "database indexing internals"),
        ]
    )

    results = retriever.search("agents tools", k=2)

    assert results[0].document_id == 2
    assert results[0].score > 0


def test_citation_recency_retriever_adds_metadata_signal() -> None:
    documents = [
        PaperDocument(1, "AI agents", year=2020, citation_count=1),
        PaperDocument(2, "AI agents", year=2025, citation_count=1000),
    ]
    retriever = CitationRecencyRetriever(documents, alpha=0.2, beta=0.5)

    results = retriever.search("AI agents", k=2)

    assert results[0].document_id == 2
    assert results[0].score > results[1].score
