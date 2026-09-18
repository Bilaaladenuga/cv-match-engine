"""
Request Middleware — Phase 22.

Provides:
- Request ID injection
- Performance timing
- Structured request logging
"""

from __future__ import annotations

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import (
    generate_request_id,
    log_request,
    request_id_var,
)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds request ID and performance tracking."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or extract request ID
        req_id = request.headers.get("X-Request-ID") or generate_request_id()
        request_id_var.set(req_id)

        # Start timing
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add headers to response
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time"] = f"{duration_ms:.0f}ms"

        # Log the request
        log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=req_id,
        )

        return response
