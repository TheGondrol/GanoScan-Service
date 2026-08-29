"""FastAPI application factory for the GanoScan backend.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 5005
    python run.py                                   (dev, with --reload)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.security import ApiKeyMiddleware
from app.services.inference import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = get_model()  # load weights (or fall back to random mode) on startup
    print(f"[ganoscan] model: {model.info()}")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="GanoScan Service",
        version="1.0.0",
        description="Stateless Ganoderma detection inference for the GanoScan Android app. "
        "Nothing is persisted server-side — the app owns its own scan history.",
        lifespan=lifespan,
        servers=[{"url": f"http://127.0.0.1:{settings.port}", "description": "Local development"}],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ApiKeyMiddleware)
    app.include_router(api_router)
    return app


app = create_app()
