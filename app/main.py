"""FastAPI application factory for the GanoScan backend.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 5005
    python run.py                                   (dev, with --reload)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import settings
from app.core.security import ApiKeyMiddleware
from app.db import store
from app.services.inference import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    store.init_db()
    model = get_model()  # load weights (or fall back to random mode) on startup
    print(f"[ganoscan] model: {model.info()}")
    yield


def create_app() -> FastAPI:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="GanoScan Service",
        version="1.0.0",
        description="Ganoderma detection inference + scan history for the GanoScan Android app.",
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
    app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")
    return app


app = create_app()
