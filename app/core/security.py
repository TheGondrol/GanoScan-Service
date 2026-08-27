"""API-key auth middleware.

Every request needs a matching ``X-API-Key`` header once GANOSCAN_API_KEY is
set; unset (default), the service stays open for local development. Health
checks and the interactive docs are always reachable without a key.
"""

from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import settings

_OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

# Documents the X-API-Key requirement in the OpenAPI schema (so /docs and any
# generated contract show it). auto_error=False because enforcement actually
# happens in ApiKeyMiddleware below, not through this dependency.
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if (
            not settings.api_key
            or request.method == "OPTIONS"
            or request.url.path in _OPEN_PATHS
        ):
            return await call_next(request)

        if request.headers.get("x-api-key") != settings.api_key:
            return JSONResponse({"detail": "invalid or missing X-API-Key"}, status_code=401)

        return await call_next(request)
