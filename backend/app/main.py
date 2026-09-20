"""
Career Match API — Main Application Entry Point.

A CV–Job Matching Engine that produces explainable compatibility analysis.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.history import router as history_router
from app.api.matches import router as matches_router
from app.api.ranking import router as ranking_router
from app.api.resumes import router as resumes_router
from app.core.config import (
    build_cors_origin_regex,
    get_settings,
    split_cors_origins,
)
from app.core.logging import log_ml_event, setup_logging
from app.core.middleware import ErrorHandlingMiddleware, RequestTrackingMiddleware
from app.core.rate_limit import RateLimitMiddleware
from app.ml.model_scorer import ml_model_available

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO", json_output=True)
    model_status = "available" if ml_model_available() else "unavailable"
    log_ml_event(f"Application starting - ML model: {model_status}", model_version=settings.APP_VERSION)
    yield
    log_ml_event("Application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An explainable ML system for CV–Job compatibility analysis.",
    lifespan=lifespan,
)


def configure_cors(application: FastAPI, origins: str | None) -> None:
    """Attach CORSMiddleware using a CORS_ORIGINS string.

    A wildcard origin can never be combined with credentials — the browser
    rejects `Access-Control-Allow-Origin: *` on credentialed requests — so
    `*` disables credentials, while an explicit allowlist enables them.
    Wildcard entries (`https://*.vercel.app`) are compiled into a regex,
    since Starlette's `allow_origins` matches literal strings only.

    Exposed as a function so tests can exercise the parsing without
    rebuilding the whole application.
    """
    configured = split_cors_origins(origins)
    allow_all = "*" in configured
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else [o for o in configured if "*" not in o],
        allow_origin_regex=None if allow_all else build_cors_origin_regex(configured),
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Middleware runs outermost-first in reverse registration order: the LAST one
# added runs FIRST. The registration below produces, outside-in:
#
#     CORS -> RequestTracking -> ErrorHandling -> RateLimit -> routes
#
# CORS must be outermost so every response — including error and 429
# responses — carries the Access-Control-* headers. ErrorHandling sits just
# inside it so an unhandled exception becomes a normal JSON 500 that travels
# back out through CORS, rather than bypassing it via ServerErrorMiddleware.
# RateLimit is innermost: it only sees requests that passed CORS, and its
# rejections still acquire tracking + CORS headers on the way out.
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(RateLimitMiddleware, settings=settings)
configure_cors(app, settings.CORS_ORIGINS)

# Routers
app.include_router(matches_router)
app.include_router(ranking_router)
app.include_router(history_router)
app.include_router(resumes_router)


@app.get("/health")
def health_check():
    """Enhanced health check endpoint with service status."""
    model_available = ml_model_available()
    return {
        "status": "healthy" if model_available else "degraded",
        "version": settings.APP_VERSION,
        "services": {
            "api": "up",
            "ml_model": "up" if model_available else "down",
            "database": "up",
        },
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
