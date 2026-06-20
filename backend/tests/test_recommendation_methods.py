import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, build_retriever
from app.services.retrievers.base import PaperDocument
from app.services.retrievers.citation_recency import CitationRecencyRetriever
from app.services.retrievers.tfidf import TfidfRetriever


def test_recommend_methods_endpoint_lists_supported_methods() -> None:
    client = TestClient(app)
    response = client.get("/recommend/methods")

    assert response.status_code == 200
    assert response.json()["methods"] == sorted(SUPPORTED_RETRIEVAL_METHODS)


def test_recommend_query_rejects_unknown_method() -> None:
    client = TestClient(app)
    response = client.get("/recommend/query?query=agents&method=unknown")

    assert response.status_code == 400
    assert "Unsupported recommendation method" in response.json()["detail"]


def test_build_retriever_supports_tfidf_and_citation_recency() -> None:
    documents = [
        PaperDocument(1, "AI agents", year=2024, citation_count=10),
        PaperDocument(2, "database systems", year=2020, citation_count=1),
    ]

    assert isinstance(build_retriever("tfidf", documents), TfidfRetriever)
    assert isinstance(build_retriever("citation_recency", documents), CitationRecencyRetriever)


def test_build_retriever_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported recommendation method"):
        build_retriever("unknown", [])
