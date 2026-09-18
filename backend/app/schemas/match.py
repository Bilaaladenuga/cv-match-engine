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
    prioritized_improvements: list[dict] | None = None  # ML-powered prioritized actions
    disclaimer: str
    skill_matches: list[dict] | None = None
    experience: dict | None = None
    semantic: dict | None = None
    education: dict | None = None
    certifications: dict | None = None
    skill_evidence: list[dict] | None = None  # Phase 16 per-skill evidence grades
    ml_details: dict | None = None  # trained-model probabilities when available


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Ranking (Phase 15)
# ---------------------------------------------------------------------------


class RankCandidatesRequest(BaseModel):
    """Request body for POST /api/jobs/{job_id}/rank-candidates."""

    resume_ids: list[int] = Field(..., description="Resumes (candidates) to rank", min_length=1)
    weights: dict[str, float] | None = Field(
        None, description="Optional hybrid weight overrides; must sum to 1.0"
    )


class RankedCandidateOut(BaseModel):
    """One row of the ranking slate."""

    rank: int
    candidate_id: int
    resume_id: int
    candidate_name: str | None = None
    match_id: int
    overall_score: float
    fit_score: float | None = None
    ml_label: str | None = None
    band: str
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    top_positive_factors: list[str] = Field(default_factory=list)
    top_negative_factors: list[str] = Field(default_factory=list)


class RankingRunResponse(BaseModel):
    """Version-stamped, reproducible ranking for one job."""

    ranking_run_id: str
    job_id: int
    job_title: str
    model_version: str
    weights: dict[str, float]
    created_at: str
    ranked: list[RankedCandidateOut]
    failed_resume_ids: list[int] = Field(default_factory=list)
    disclaimer: str


# ---------------------------------------------------------------------------
# History (retrieval of stored analyses)
# ---------------------------------------------------------------------------


class HistoryEntryOut(BaseModel):
    """One stored match, list view (no raw documents)."""

    match_id: int
    candidate_id: int
    job_id: int
    candidate_name: str | None = None
    job_title: str | None = None
    overall_score: float
    band: str | None = None
    model_version: str
    created_at: str | None = None
    ml_fit_score: float | None = None
    ml_label: str | None = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class MatchDetailResponse(BaseModel):
    """Full stored analysis for one match (read back, no recomputation)."""

    match_id: int
    candidate_id: int
    job_id: int
    candidate_name: str | None = None
    job_title: str | None = None
    job_company: str | None = None
    overall_score: float
    band: str | None = None
    model_version: str
    created_at: str | None = None
    component_scores: dict[str, float]
    ml: dict
    explanation: dict
    feature_values: dict | None = None
