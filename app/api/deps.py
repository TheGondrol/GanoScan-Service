"""Shared FastAPI dependencies."""

from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.services.inference import GanodermaModel, get_model


def model_dep() -> GanodermaModel:
    """Injects the singleton model into route handlers."""
    return get_model()


def require_admin(x_admin_token: str | None = Header(default=None)):
    """Guards destructive endpoints (scan deletion). No-op unless
    GANOSCAN_ADMIN_TOKEN is set, so local dev is unaffected; set it before
    exposing the service on a public server."""
    if settings.admin_token and x_admin_token != settings.admin_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing X-Admin-Token")
