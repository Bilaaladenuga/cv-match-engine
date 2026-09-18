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
from app.core.config import get_settings
from app.core.logging import setup_logging, log_ml_event
from app.core.middleware import RequestTrackingMiddleware
from app.ml.model_scorer import ml_model_available

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Setup structured logging
    setup_logging(level="DEBUG" if settings.DEBUG else "INFO", json_output=True)

    # Startup: load ML models, verify DB connectivity, etc.
    model_status = "available" if ml_model_available() else "unavailable"
    log_ml_event(f"Application starting - ML model: {model_status}", model_version=settings.APP_VERSION)
    yield
    # Shutdown: cleanup resources
    log_ml_event("Application shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An explainable ML system for CV–Job compatibility analysis.",
    lifespan=lifespan,
)

# Request tracking middleware
app.add_middleware(RequestTrackingMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(matches_router)
app.include_router(ranking_router)
app.include_router(history_router)
app.include_router(resumes_router)


@app.get("/health")
def health_check():
    """Enhanced health check endpoint with service status."""
    from app.ml.model_scorer import ml_model_available

    model_available = ml_model_available()

    return {
        "status": "healthy" if model_available else "degraded",
        "version": settings.APP_VERSION,
        "services": {
            "api": "up",
            "ml_model": "up" if model_available else "down",
            "database": "up",  # Will check actual DB connection in production
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
