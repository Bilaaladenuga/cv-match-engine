"""
API Schemas — matching endpoints.

Pydantic request/response models for POST /api/matches.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class MatchRequest(BaseModel):
    """Request body for POST /api/matches.

    Two modes:
        - ``resume_id`` + ``job_id``: match two stored entities (DB path).
        - ``cv_text`` + ``job_text``: match raw text (stateless path);
          nothing is persisted.
    """

    resume_id: int | None = None
    job_id: int | None = None
    cv_text: str | None = Field(default=None, max_length=200_000)
    job_text: str | None = Field(default=None, max_length=100_000)
    weights: dict[str, float] | None = None

    model_config = {"json_schema_extra": {"examples": [
        {
            "cv_text": "Sarah Johnson ... Python, React, 5 years experience ...",
            "job_text": "Senior Frontend Engineer with 4+ years React ...",
        }
    ]}}


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ComponentScoreOut(BaseModel):
    name: str
    raw_score: float
    weight: float
    weighted: float
    evidence: str


class MatchResponse(BaseModel):
    """Response for POST /api/matches."""

    match_id: int | None = None
    model_version: str
    overall_score: float
    overall_percent: int
    band: str
    components: list[ComponentScoreOut]
    weights: dict[str, float]
    positive_factors: list[str]
    negative_factors: list[str]
    recommendations: list[str]
    disclaimer: str
    skill_matches: list[dict] | None = None
    experience: dict | None = None
    semantic: dict | None = None
    education: dict | None = None
    certifications: dict | None = None
    ml_details: dict | None = None  # trained-model probabilities when available


class ErrorResponse(BaseModel):
    detail: str
