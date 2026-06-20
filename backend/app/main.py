from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.models  # noqa: F401
from app.api.routes import router
from app.core.config import get_settings
from app.db.schema import ensure_runtime_schema
from app.db.session import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ResearchPath API",
        description="Retrieval and recommendation API for beginner-friendly research reading paths.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    return app


app = create_app()
