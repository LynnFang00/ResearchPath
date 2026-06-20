import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app
from app.models.paper import Paper


def test_path_query_endpoint_returns_grouped_sections():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    with TestingSessionLocal() as db:
        db.add_all(
            [
                Paper(
                    title="A Survey of AI Agents",
                    abstract="A survey and tutorial introduction to AI agents.",
                    authors="Ada Lovelace",
                    year=2021,
                    categories="cs.AI",
                    citation_count=500,
                    references_count=40,
                ),
                Paper(
                    title="Tool-Using Agents for Scientific Discovery",
                    abstract="AI agents use tools for scientific discovery and machine learning workflows.",
                    authors="Alan Turing",
                    year=2026,
                    categories="cs.LG",
                    citation_count=12,
                    references_count=50,
                ),
            ]
        )
        db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.get("/path/query?query=AI%20agents&method=bm25&k=1&candidate_k=10")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "AI agents"
    assert set(payload["sections"]) == {"background", "foundational", "core_methods", "recent_frontier"}
    assert set(payload["section_status"]) == {"background", "foundational", "core_methods", "recent_frontier"}
    assert any(section for section in payload["sections"].values())
    first_paper = next(papers[0] for papers in payload["sections"].values() if papers)
    assert "difficulty_label" in first_paper
    assert "path_reason" in first_paper
