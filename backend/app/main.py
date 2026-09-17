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

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup: load ML models, verify DB connectivity, etc.
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    yield
    # Shutdown: cleanup resources
    print(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An explainable ML system for CV–Job compatibility analysis.",
    lifespan=lifespan,
)

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
    """Health check endpoint."""
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
