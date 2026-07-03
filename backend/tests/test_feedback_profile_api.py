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


def _client_with_db() -> tuple[TestClient, sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    with TestingSessionLocal() as db:
        db.add(Paper(id=1, title="A Survey of Agents", abstract="Survey paper.", authors="A", categories="cs.AI"))
        db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app), TestingSessionLocal


def test_feedback_creation_and_summary() -> None:
    client, _ = _client_with_db()
    try:
        response = client.post(
            "/feedback",
            json={
                "paper_id": 1,
                "query": "agents",
                "section": "background",
                "action": "save",
                "method": "hybrid",
                "background_level": "basic_ml",
            },
        )
        summary = client.get("/feedback/summary")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["action"] == "save"
    assert summary.status_code == 200
    assert summary.json()["total_events"] == 1
    assert summary.json()["action_counts"]["save"] == 1
    assert summary.json()["saved_paper_ids"] == [1]


def test_profile_endpoint_get_and_patch() -> None:
    client, _ = _client_with_db()
    try:
        initial = client.get("/profile")
        patched = client.patch(
            "/profile",
            json={"background_level": "intermediate", "preferred_topics": ["agents"], "saved_paper_ids": [1]},
        )
    finally:
        app.dependency_overrides.clear()

    assert initial.status_code == 200
    assert initial.json()["background_level"] == "basic_ml"
    assert patched.status_code == 200
    assert patched.json()["background_level"] == "intermediate"
    assert patched.json()["preferred_topics"] == ["agents"]
    assert patched.json()["saved_paper_ids"] == [1]
