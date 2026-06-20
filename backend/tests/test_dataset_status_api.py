import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app
from app.models.citation_edge import CitationEdge
from app.models.paper import Paper


def test_dataset_status_endpoint_returns_counts() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    with TestingSessionLocal() as db:
        first = Paper(title="Paper A", abstract="Abstract", authors="", year=2024, categories="")
        second = Paper(title="Paper B", abstract="Abstract", authors="", year=2025, categories="")
        db.add_all([first, second])
        db.flush()
        db.add(CitationEdge(source_paper_id=first.id, target_paper_id=second.id, relationship_type="reference"))
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
        response = client.get("/dataset/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_count"] == 2
    assert payload["citation_edge_count"] == 1
    assert payload["last_updated_timestamp"] is not None
    assert "embedding_model_name" in payload
    assert "faiss_index_path" in payload
