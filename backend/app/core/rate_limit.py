"""
IP-based rate limiting — Phase 20 (security hardening).

The API is public and unauthenticated, and the expensive endpoints
(POST /api/matches, POST /api/ranking) run the embedding model plus the
trained classifier on every call — real CPU work on a 512 MB instance.
Without a limiter, a single abusive client (or one misbehaving script)
can degrade the service for everyone.

Design
------
- Fixed-window counting keyed by client IP + path bucket, in memory.
  Sufficient for the single-worker free deployment; if the app ever runs
  multi-worker or multi-instance, this must move to Redis (each process
  would otherwise keep its own counters).
- Cheap endpoints get a generous default; the ML endpoints get a tight one.
- GET requests (health checks, docs) and OPTIONS preflights are exempt:
  Render's health pings must never be throttled, and CORS answers
  preflights before this layer sees them.
- 429 responses carry Retry-After and are logged with the offending IP.
- The client IP comes from X-Forwarded-For / X-Real-IP (Render's proxy
  sets these). We trust the proxy because this service is only ever
  reached through it; a directly-exposed deployment should pin this down.

Exceeded requests never reach the pipeline, so an abusive loop costs the
server almost nothing — that is the point.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import Settings

logger = logging.getLogger("ratelimit")

# Path prefixes of the expensive ML endpoints, mapped to the settings key
# holding their rule. Matched with startswith so query strings don't matter.
_EXPENSIVE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("/api/matches", "RATE_LIMIT_MATCHES"),
    ("/api/ranking", "RATE_LIMIT_RANKING"),
)

_RATE_RE = re.compile(r"^(\d+)\s*/\s*(sec|second|s|min|minute|m|hour|h)$")
_PERIOD_SECONDS = {
    "sec": 1, "second": 1, "s": 1,
    "min": 60, "minute": 60, "m": 60,
    "hour": 3600, "h": 3600,
}


def parse_rate(spec: str) -> tuple[int, float]:
    """Parse a rate limit string like "5/min" into (limit, window_seconds).

    Raises ValueError with a human-readable message on malformed input so a
    bad environment variable fails loudly at startup, not silently at 3am.
    """
    match = _RATE_RE.fullmatch((spec or "").strip().lower())
    if not match:
        raise ValueError(
            f"Invalid rate limit {spec!r}: expected '<int>/<period>' "
            f"with period in sec|min|hour (e.g. '5/min')"
        )
    limit = int(match.group(1))
    if limit < 1:
        raise ValueError(f"Invalid rate limit {spec!r}: limit must be >= 1")
    return limit, float(_PERIOD_SECONDS[match.group(2)])


class RateLimiter:
    """In-memory fixed-window counters.

    Windows are aligned to wall-clock buckets (`int(now / window)`), so the
    window a client lands in is stable across processes seeing the same
    clock — irrelevant today (single worker) but keeps semantics simple.
    Counters are pruned opportunistically when the table grows past a
    threshold, so memory stays bounded under scanner traffic.
    """

    def __init__(self) -> None:
        # (bucket_key) -> (count, window_end_epoch)
        self._windows: dict[str, tuple[int, float]] = {}

    def check(self, key: str, limit: int, window: float) -> tuple[bool, int, float]:
        """Record a hit for `key`; decide whether it is within the limit.

        Returns (allowed, remaining, retry_after_seconds). A blocked request
        still counts toward its (already exhausted) window — it must not
        extend or reset the punishment, that would let a constant requester
        lock themselves (and only themselves) out forever.
        """
        now = time.time()
        window_end = (int(now / window) + 1) * window
        count, current_end = self._windows.get(key, (0, window_end))
        if now >= current_end:
            # Previous window expired: start fresh.
            count, current_end = 0, window_end

        count += 1
        self._windows[key] = (count, current_end)
        self._maybe_prune(now)

        allowed = count <= limit
        remaining = max(0, limit - count)
        retry_after = max(0.0, current_end - now) if not allowed else 0.0
        return allowed, remaining, retry_after

    def _maybe_prune(self, now: float, threshold: int = 10_000) -> None:
        """Drop expired windows once the table gets large."""
        if len(self._windows) < threshold:
            return
        expired = [k for k, (_, end) in self._windows.items() if now >= end]
        for k in expired:
            del self._windows[k]


def client_ip(request: Request) -> str:
    """Best-effort client IP behind Render's proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # First entry is the original client; the rest are proxies.
        return forwarded.split(",")[0].strip()
    return request.headers.get("X-Real-IP") or (
        request.client.host if request.client else "unknown"
    )


def rule_for_path(path: str, settings: Settings) -> str | None:
    """Return the rate-limit spec applying to this path, or None to exempt it.

    GETs (health checks, docs) are never limited; the expensive ML POSTs get
    their dedicated tight rules; everything else falls under the default.
    """
    if path == "/health" or path == "/":
        return None
    for prefix, setting_name in _EXPENSIVE_PREFIXES:
        if path.startswith(prefix):
            return getattr(settings, setting_name)
    return settings.RATE_LIMIT_DEFAULT


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed the per-IP fixed-window allowance.

    Registered inside CORS and RequestTracking (see app/main.py) so 429
    responses still carry Access-Control-* headers and appear in the
    request logs with a request ID.
    """

    # Shared counter store for middleware instances that don't inject their
    # own. Module-level so the test suite can clear state between tests
    # (counters are in-process memory by design).
    _shared_limiter = RateLimiter()

    def __init__(self, app, settings: Settings, limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self._settings = settings
        self._limiter = limiter if limiter is not None else self._shared_limiter
        # Parse eagerly: a malformed env var should fail at boot, not on the
        # first abusive request.
        self._cache: dict[str, tuple[int, float]] = {}

    @classmethod
    def reset(cls) -> None:
        """Clear all window counters (used between tests; harmless in prod)."""
        cls._shared_limiter._windows.clear()

    def _rule(self, spec: str) -> tuple[int, float]:
        if spec not in self._cache:
            self._cache[spec] = parse_rate(spec)
        return self._cache[spec]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        spec = rule_for_path(request.url.path, self._settings)
        if spec is None:
            return await call_next(request)

        limit, window = self._rule(spec)
        ip = client_ip(request)
        # Bucket per IP + rule so the default allowance and the ML allowance
        # are independent budgets for the same client.
        key = f"{ip}:{spec}"

        allowed, remaining, retry_after = self._limiter.check(key, limit, window)
        if not allowed:
            logger.warning(
                "Rate limit exceeded: ip=%s path=%s limit=%s retry_after=%.0fs",
                ip,
                request.url.path,
                spec,
                retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please wait before trying again."
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
