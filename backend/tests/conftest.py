"""Shared test configuration.

The real app (app.main) attaches its middleware at import time, so the API
test suite would otherwise share one in-memory rate-limit window across all
tests: after five cumulative POST /api/matches calls, every later test would
start receiving 429s.

The autouse fixture below therefore (a) clears the limiter's counters before
each test and (b) disables limiting on the *real* app for the duration of
each test. The limiter's own behavior has a dedicated test module
(test_rate_limit.py) that constructs isolated middleware instances with
explicit settings, so it is unaffected by the flag.
"""

from __future__ import annotations

import pytest

from app.core.rate_limit import RateLimitMiddleware
from app.main import settings as app_settings


@pytest.fixture(autouse=True)
def _isolate_rate_limiter():
    RateLimitMiddleware.reset()
    original = app_settings.RATE_LIMIT_ENABLED
    app_settings.RATE_LIMIT_ENABLED = False
    yield
    app_settings.RATE_LIMIT_ENABLED = original
