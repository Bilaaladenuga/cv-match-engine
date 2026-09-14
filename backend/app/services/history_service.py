"""
History Service — retrieval of stored analyses.

Read-only: everything here reads persisted Match/MatchExplanation rows
(plus candidate/job context for display) and never re-runs the pipeline.
Raw CV text and job descriptions are NOT returned in list views; they are
only available through the document endpoints that own them (Phase 20
will gate those behind auth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session, joinedload

from app.models.candidate import CandidateProfile
from app.models.job import Job
from app.models.match import Match, MatchExplanation
from app.services.matching_service import MatchNotFoundError


@dataclass
class HistoryEntry:
    """One stored match, as shown in analysis history lists."""

    match_id: int
    candidate_id: int
    job_id: int
    candidate_name: str | None
    job_title: str | None
    overall_score: float
    band: str | None
    model_version: str
    created_at: str | None
    ml_fit_score: float | None = None
    ml_label: str | None = None
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "candidate_name": self.candidate_name,
            "job_title": self.job_title,
            "overall_score": self.overall_score,
            "band": self.band,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "ml_fit_score": self.ml_fit_score,
            "ml_label": self.ml_label,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
        }


def _band_from_feature_values(feature_values: dict | None) -> str | None:
    if not feature_values:
        return None
    return feature_values.get("band")


def _ml_from_feature_values(feature_values: dict | None) -> tuple[float | None, str | None]:
    if not feature_values:
        return None, None
    ml = feature_values.get("ml_details") or {}
    return ml.get("fit_score"), ml.get("label")


def get_match_detail(db: Session, match_id: int) -> dict:
    """Full stored analysis for one match (no recomputation).

    Raises MatchNotFoundError when the match does not exist.
    """
    row = (
        db.query(Match)
        .options(joinedload(Match.explanation))
        .filter(Match.id == match_id)
        .first()
    )
    if row is None:
        raise MatchNotFoundError(f"Match {match_id} not found")

    candidate = db.get(CandidateProfile, row.candidate_id)
    job = db.get(Job, row.job_id)

    fv = row.feature_values or {}
    explanation: MatchExplanation | None = row.explanation
    ml = fv.get("ml_details") or {}

    return {
        "match_id": row.id,
        "candidate_id": row.candidate_id,
        "job_id": row.job_id,
        "candidate_name": candidate.name if candidate else None,
        "job_title": job.title if job else None,
        "job_company": job.company if job else None,
        "overall_score": row.overall_score,
        "band": fv.get("band"),
        "model_version": row.model_version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "component_scores": {
            "skills": row.skills_score,
            "semantic": row.semantic_score,
            "experience": row.experience_score,
            "education": row.education_score,
        },
        "ml": {
            "fit_score": ml.get("fit_score"),
            "label": ml.get("label"),
            "probabilities": ml.get("probabilities"),
            "calibration_method": ml.get("calibration_method"),
        },
        "explanation": {
            "matched_skills": explanation.matched_skills if explanation else [],
            "missing_skills": explanation.missing_skills if explanation else [],
            "partial_skills": explanation.partial_skills if explanation else [],
            "recommendations": explanation.recommendations if explanation else [],
            "positive_factors": explanation.positive_factors if explanation else [],
            "negative_factors": explanation.negative_factors if explanation else [],
        },
        "feature_values": fv,
    }


def list_history(db: Session, limit: int = 50, offset: int = 0) -> list[HistoryEntry]:
    """Most recent analyses first. Candidate/job names are display context."""
    rows = (
        db.query(Match)
        .options(joinedload(Match.explanation))
        .order_by(Match.created_at.desc(), Match.id.desc())
        .offset(max(offset, 0))
        .limit(min(max(limit, 1), 200))
        .all()
    )
    entries: list[HistoryEntry] = []
    for row in rows:
        candidate = db.get(CandidateProfile, row.candidate_id)
        job = db.get(Job, row.job_id)
        fit, label = _ml_from_feature_values(row.feature_values)
        entries.append(
            HistoryEntry(
                match_id=row.id,
                candidate_id=row.candidate_id,
                job_id=row.job_id,
                candidate_name=candidate.name if candidate else None,
                job_title=job.title if job else None,
                overall_score=row.overall_score,
                band=_band_from_feature_values(row.feature_values),
                model_version=row.model_version,
                created_at=row.created_at.isoformat() if row.created_at else None,
                ml_fit_score=fit,
                ml_label=label,
                matched_skills=(row.explanation.matched_skills if row.explanation else []) or [],
                missing_skills=(row.explanation.missing_skills if row.explanation else []) or [],
            )
        )
    return entries
