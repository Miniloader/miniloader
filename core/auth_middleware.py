"""
auth_middleware.py — Bearer token authentication for internal HTTP servers
==========================================================================
Reusable Starlette/FastAPI middleware that guards ``/api/*`` and ``/v1/*``
routes with a vault-derived bearer token.

The middleware can be toggled at runtime (e.g. auto-enabled when ngrok
connects).  Health endpoints are always exempt.
"""

from __future__ import annotations

import hmac
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

log = logging.getLogger(__name__)

_HEALTH_PATHS = frozenset({"/health", "/v1/health"})
_GUARDED_PREFIXES = ("/api/", "/v1/", "/db/", "/rag/", "/ws")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests lacking a valid ``Authorization: Bearer <token>`` header.

    Parameters
    ----------
    app
        The ASGI application to wrap.
    token_provider
        Callable that returns the current valid token string.
        Called per-request so the token can be rotated without restarting.
    enabled_check
        Callable returning True when auth should be enforced.
        Defaults to always-on.
    """

    def __init__(
        self,
        app: object,
        token_provider: Callable[[], str],
        enabled_check: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._token_provider = token_provider
        self._enabled_check = enabled_check or (lambda: True)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._enabled_check():
            return await call_next(request)

        path = request.url.path
        if path in _HEALTH_PATHS:
            return await call_next(request)

        if not any(path.startswith(prefix) for prefix in _GUARDED_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": {"message": "Missing bearer token", "type": "auth_error"}},
                status_code=401,
            )

        token = auth_header[7:]
        expected = self._token_provider()
        if not hmac.compare_digest(token, expected):
            return JSONResponse(
                {"error": {"message": "Invalid bearer token", "type": "auth_error"}},
                status_code=401,
            )

        return await call_next(request)
