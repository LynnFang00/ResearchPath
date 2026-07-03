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


def _client() -> TestClient:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    with TestingSessionLocal() as db:
        db.add(Paper(id=1, title="Transformer Paper", abstract="A transformer method.", authors="A", year=2024, categories="cs.LG"))
        db.commit()

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_library_save_and_filter_by_user_tags() -> None:
    client = _client()
    try:
        saved = client.post("/library/items", json={"paper_id": 1, "tags": ["Transformer", "resnet"]})
        all_items = client.get("/library")
        transformer_items = client.get("/library?tag=transformer")
        missing_items = client.get("/library?tag=agents")
    finally:
        app.dependency_overrides.clear()

    assert saved.status_code == 200
    assert saved.json()["tags"] == ["resnet", "transformer"]
    assert all_items.json()["tags"] == ["resnet", "transformer"]
    assert len(transformer_items.json()["items"]) == 1
    assert missing_items.json()["items"] == []


def test_library_item_can_be_unsaved() -> None:
    client = _client()
    try:
        client.post("/library/items", json={"paper_id": 1, "tags": ["Transformer"]})
        removed = client.delete("/library/items/1")
        library = client.get("/library")
        profile = client.get("/profile")
    finally:
        app.dependency_overrides.clear()

    assert removed.status_code == 200
    assert library.json()["items"] == []
    assert library.json()["tags"] == []
    assert profile.json()["saved_paper_ids"] == []


def test_save_feedback_creates_library_item_with_tags() -> None:
    client = _client()
    try:
        feedback = client.post(
            "/feedback",
            json={
                "paper_id": 1,
                "query": "transformer",
                "section": "core_methods",
                "action": "save",
                "method": "hybrid",
                "background_level": "basic_ml",
                "tags": ["transformer", "vision"],
            },
        )
        library = client.get("/library?tag=vision")
    finally:
        app.dependency_overrides.clear()

    assert feedback.status_code == 200
    assert len(library.json()["items"]) == 1
    assert library.json()["items"][0]["tags"] == ["transformer", "vision"]
