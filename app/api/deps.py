"""Shared FastAPI dependencies."""

from app.services.inference import GanodermaModel, get_model


def model_dep() -> GanodermaModel:
    """Injects the singleton model into route handlers."""
    return get_model()
