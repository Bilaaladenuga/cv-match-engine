"""Tests for the IP-based rate limiter (app/core/rate_limit.py).

Covers the pure pieces (rate parsing, client IP extraction, path rule
selection, window expiry) and the middleware end-to-end through a minimal
FastAPI app — 429 behavior, headers, the enabled flag, and per-rule budgets.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    client_ip,
    parse_rate,
    rule_for_path,
)

# ---------------------------------------------------------------------------
# parse_rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("5/min", (5, 60.0)),
        ("60/min", (60, 60.0)),
        ("100/hour", (100, 3600.0)),
        ("2/sec", (2, 1.0)),
        ("10 / min", (10, 60.0)),  # tolerant of whitespace
        ("3/MIN", (3, 60.0)),  # case-insensitive
    ],
)
def test_parse_rate_valid(spec, expected):
    assert parse_rate(spec) == expected


@pytest.mark.parametrize("spec", ["", "5", "5/hourly", "min/5", "0/min", "-1/min", "5/decade"])
def test_parse_rate_invalid(spec):
    with pytest.raises(ValueError):
        parse_rate(spec)


# ---------------------------------------------------------------------------
# client_ip
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Duck-typed stand-in: only the attributes client_ip() touches."""

    def __init__(self, headers=None, host="203.0.113.9"):
        self.headers = headers or {}
        if host is not None:
            self.client = type("C", (), {"host": host})()
        else:
            self.client = None


def test_client_ip_prefers_forwarded_for():
    req = _FakeRequest({"X-Forwarded-For": "198.51.100.7, 10.0.0.1"})
    assert client_ip(req) == "198.51.100.7"


def test_client_ip_falls_back_to_real_ip_then_socket():
    assert client_ip(_FakeRequest({"X-Real-IP": "198.51.100.8"})) == "198.51.100.8"
    assert client_ip(_FakeRequest()) == "203.0.113.9"
    assert client_ip(_FakeRequest(host=None)) == "unknown"


# ---------------------------------------------------------------------------
# rule_for_path
# ---------------------------------------------------------------------------


def _settings(**overrides) -> Settings:
    return Settings(RATE_LIMIT_ENABLED=True, **overrides)


def test_health_and_root_are_exempt():
    s = _settings()
    assert rule_for_path("/health", s) is None
    assert rule_for_path("/", s) is None


def test_expensive_endpoints_get_their_rules():
    s = _settings(RATE_LIMIT_MATCHES="5/min", RATE_LIMIT_RANKING="9/min")
    assert rule_for_path("/api/matches", s) == "5/min"
    assert rule_for_path("/api/matches/export-pdf", s) == "5/min"
    assert rule_for_path("/api/ranking", s) == "9/min"


def test_other_paths_get_default():
    s = _settings(RATE_LIMIT_DEFAULT="60/min")
    assert rule_for_path("/api/resumes/extract", s) == "60/min"
    assert rule_for_path("/docs", s) == "60/min"


# ---------------------------------------------------------------------------
# RateLimiter window logic
# ---------------------------------------------------------------------------


def test_limiter_counts_and_blocks_within_window():
    limiter = RateLimiter()
    results = [limiter.check("ip:5/min", limit=3, window=60.0) for _ in range(5)]
    allowed = [r[0] for r in results]
    assert allowed == [True, True, True, False, False]
    # After exhaustion: remaining 0, retry_after set.
    assert results[3][1] == 0
    assert results[3][2] > 0


def test_limiter_new_window_resets_count():
    limiter = RateLimiter()
    window = 1.0
    for _ in range(5):
        limiter.check("k", limit=2, window=window)
    # Force the window to expire and confirm a clean slate.
    key_count, window_end = limiter._windows["k"]
    limiter._windows["k"] = (key_count, window_end - 10)
    allowed, remaining, retry = limiter.check("k", limit=2, window=window)
    assert allowed and remaining == 1 and retry == 0.0


# ---------------------------------------------------------------------------
# Middleware end-to-end
# ---------------------------------------------------------------------------


def _build_app(settings_kwargs=None):
    settings = Settings(**{"RATE_LIMIT_ENABLED": True, **(settings_kwargs or {})})
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, settings=settings)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/api/matches")
    def matches():
        return {"ok": True}

    @app.post("/api/resumes/extract")
    def extract():
        return {"ok": True}

    return TestClient(app)


def test_middleware_blocks_after_limit_and_reports_headers():
    client = _build_app({"RATE_LIMIT_MATCHES": "2/min"})
    assert client.post("/api/matches").status_code == 200
    assert client.post("/api/matches").status_code == 200
    third = client.post("/api/matches")
    assert third.status_code == 429
    assert "wait" in third.json()["detail"].lower()
    assert int(third.headers["Retry-After"]) >= 1
    assert third.headers["X-RateLimit-Remaining"] == "0"


def test_middleware_success_response_carries_headers():
    client = _build_app({"RATE_LIMIT_MATCHES": "2/min"})
    resp = client.post("/api/matches")
    assert resp.headers["X-RateLimit-Limit"] == "2"
    assert resp.headers["X-RateLimit-Remaining"] == "1"


def test_middleware_health_exempt_even_when_spammed():
    client = _build_app()
    for _ in range(10):
        assert client.get("/health").status_code == 200


def test_middleware_disabled_flag_bypasses_everything():
    client = _build_app({"RATE_LIMIT_ENABLED": False, "RATE_LIMIT_MATCHES": "1/min"})
    for _ in range(5):
        assert client.post("/api/matches").status_code == 200


def test_budgets_are_independent_per_rule():
    # Default rule (resumes) should not consume the matches budget.
    client = _build_app({"RATE_LIMIT_MATCHES": "1/min", "RATE_LIMIT_DEFAULT": "50/min"})
    assert client.post("/api/matches").status_code == 200
    assert client.post("/api/matches").status_code == 429
    assert client.post("/api/resumes/extract").status_code == 200


def test_clients_have_independent_budgets():
    client = _build_app({"RATE_LIMIT_MATCHES": "1/min"})
    assert client.post("/api/matches", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/api/matches", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
    assert client.post("/api/matches", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
