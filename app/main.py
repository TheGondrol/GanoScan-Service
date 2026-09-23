"""FastAPI application factory for the GanoScan backend.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 5005
    python run.py                                   (dev, with --reload)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
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
        # No hardcoded `servers` here on purpose: with it unset, the OpenAPI
        # doc has no "servers" key at all, and Swagger UI's "Try it out" then
        # targets whatever origin is currently serving /docs (relative URL) —
        # correct for both local dev and the deployed Render URL. A hardcoded
        # http://127.0.0.1:<port> here used to make "Try it out" on the
        # deployed /docs page send requests to the *tester's own machine*
        # (and get blocked as mixed content, since the page is HTTPS) instead
        # of the real server.
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
