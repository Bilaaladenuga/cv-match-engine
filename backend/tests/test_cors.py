"""
Tests for CORS configuration (app/core/config.py + app/main.py).

Covers the failure modes that broke production:
    - CORS_ORIGINS is a JSON array on the hosting platform but a
      comma-separated list locally (and a bare string in simple setups).
    - Wildcard subdomains never match Starlette's literal `allow_origins`.
    - A wildcard origin must NOT be combined with credentials.
    - Disallowed origins must not receive an Access-Control-Allow-Origin
      header (otherwise the browser would happily make the request).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import build_cors_origin_regex, split_cors_origins
from app.core.middleware import ErrorHandlingMiddleware, RequestTrackingMiddleware
from app.main import configure_cors

# ---------------------------------------------------------------------------
# split_cors_origins
# ---------------------------------------------------------------------------


class TestSplitCorsOrigins:
    def test_json_array(self):
        raw = '["https://cv-match-engine.vercel.app", "https://*.onrender.com"]'
        assert split_cors_origins(raw) == [
            "https://cv-match-engine.vercel.app",
            "https://*.onrender.com",
        ]

    def test_comma_separated(self):
        raw = "https://a.com, https://b.com"
        assert split_cors_origins(raw) == ["https://a.com", "https://b.com"]

    def test_single_origin(self):
        assert split_cors_origins("http://localhost:3000") == [
            "http://localhost:3000"
        ]

    def test_wildcard(self):
        assert split_cors_origins("*") == ["*"]

    def test_empty_and_none(self):
        assert split_cors_origins("") == []
        assert split_cors_origins(None) == []
        assert split_cors_origins("   ") == []

    def test_malformed_json_falls_back_to_split(self):
        # A leading bracket that is not valid JSON should not raise; the
        # value is treated as a plain string.
        assert split_cors_origins('["https://a.com"') == ['["https://a.com"']


# ---------------------------------------------------------------------------
# build_cors_origin_regex
# ---------------------------------------------------------------------------


class TestBuildCorsOriginRegex:
    def test_single_wildcard(self):
        regex = build_cors_origin_regex(["https://*.vercel.app"])
        assert regex is not None
        assert regex == r"^(https://.*\.vercel\.app)$"

    def test_multiple_wildcards(self):
        regex = build_cors_origin_regex(
            ["https://*.vercel.app", "https://*.onrender.com"]
        )
        assert regex is not None
        assert regex.startswith("^(") and regex.endswith(")$")
        assert "vercel" in regex and "onrender" in regex

    def test_no_wildcards_returns_none(self):
        assert build_cors_origin_regex(["https://a.com"]) is None

    def test_bare_star_is_not_a_regex(self):
        # The global wildcard is handled by allow_origins, not the regex.
        assert build_cors_origin_regex(["*"]) is None


# ---------------------------------------------------------------------------
# Middleware wiring
# ---------------------------------------------------------------------------


def _client(origins: str) -> TestClient:
    application = FastAPI()
    configure_cors(application, origins)

    @application.get("/ping")
    def ping():  # pragma: no cover - trivial
        return {"ok": True}

    return TestClient(application)


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/ping",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )


class TestCorsMiddleware:
    def test_allowed_literal_origin(self):
        client = _client('["https://cv-match-engine.vercel.app"]')
        resp = _preflight(client, "https://cv-match-engine.vercel.app")
        assert resp.status_code == 200
        assert (
            resp.headers["access-control-allow-origin"]
            == "https://cv-match-engine.vercel.app"
        )

    def test_wildcard_subdomain_matches(self):
        client = _client('["https://*.vercel.app"]')
        resp = _preflight(client, "https://cv-match-engine-abc123.vercel.app")
        assert resp.status_code == 200
        assert (
            resp.headers["access-control-allow-origin"]
            == "https://cv-match-engine-abc123.vercel.app"
        )

    def test_wildcard_subdomain_rejects_other_host(self):
        client = _client('["https://*.vercel.app"]')
        resp = _preflight(client, "https://evil.example.com")
        assert "access-control-allow-origin" not in resp.headers

    def test_disallowed_origin_gets_no_acao(self):
        client = _client('["https://cv-match-engine.vercel.app"]')
        resp = _preflight(client, "https://evil.example.com")
        assert "access-control-allow-origin" not in resp.headers

    def test_explicit_origins_allow_credentials(self):
        client = _client('["https://cv-match-engine.vercel.app"]')
        resp = _preflight(client, "https://cv-match-engine.vercel.app")
        assert resp.headers.get("access-control-allow-credentials") == "true"

    def test_wildcard_does_not_allow_credentials(self):
        # `*` + credentials is invalid; the middleware must downgrade rather
        # than emit a combination the browser rejects.
        client = _client("*")
        resp = _preflight(client, "https://anything.example.com")
        assert resp.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in resp.headers

    def test_actual_request_carries_cors_header(self):
        client = _client('["https://cv-match-engine.vercel.app"]')
        resp = client.get(
            "/ping", headers={"Origin": "https://cv-match-engine.vercel.app"}
        )
        assert resp.status_code == 200
        assert (
            resp.headers["access-control-allow-origin"]
            == "https://cv-match-engine.vercel.app"
        )

    def test_comma_separated_config_works(self):
        client = _client("https://a.com,https://*.vercel.app")
        assert (
            _preflight(client, "https://a.com").headers["access-control-allow-origin"]
            == "https://a.com"
        )
        assert (
            _preflight(client, "https://x.vercel.app").headers[
                "access-control-allow-origin"
            ]
            == "https://x.vercel.app"
        )


# ---------------------------------------------------------------------------
# CORS on error responses
# ---------------------------------------------------------------------------


def _client_with_errors(origins: str) -> TestClient:
    """Mirror app/main.py's middleware composition.

    ErrorHandling must be registered INSIDE CORS for this to work, which is
    exactly the ordering under test here.
    """
    application = FastAPI()

    @application.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    application.add_middleware(ErrorHandlingMiddleware)
    application.add_middleware(RequestTrackingMiddleware)
    configure_cors(application, origins)
    return TestClient(application, raise_server_exceptions=False)


class TestCorsOnErrors:
    def test_unhandled_error_carries_cors_headers(self):
        # A 500 without Access-Control-Allow-Origin is reported by browsers
        # as a CORS failure, hiding the real status code.
        client = _client_with_errors('["https://cv-match-engine-rho.vercel.app"]')
        resp = client.get(
            "/boom", headers={"Origin": "https://cv-match-engine-rho.vercel.app"}
        )
        assert resp.status_code == 500
        assert (
            resp.headers["access-control-allow-origin"]
            == "https://cv-match-engine-rho.vercel.app"
        )
        assert resp.json() == {"detail": "Internal server error"}

    def test_error_body_does_not_leak_internals(self):
        client = _client_with_errors('["https://cv-match-engine-rho.vercel.app"]')
        resp = client.get(
            "/boom", headers={"Origin": "https://cv-match-engine-rho.vercel.app"}
        )
        assert "kaboom" not in resp.text

    def test_disallowed_origin_error_has_no_acao(self):
        client = _client_with_errors('["https://cv-match-engine-rho.vercel.app"]')
        resp = client.get("/boom", headers={"Origin": "https://evil.example.com"})
        assert resp.status_code == 500
        assert "access-control-allow-origin" not in resp.headers
