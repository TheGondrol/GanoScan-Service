"""Aggregates all route modules into a single router."""

from fastapi import APIRouter

from app.api.routes import health, predict, scans

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(scans.router)
