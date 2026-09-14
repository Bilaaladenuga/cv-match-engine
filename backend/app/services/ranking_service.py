"""
Ranking Service (Phase 15) — rank multiple candidates against one job.

Recruiter flow: POST /api/jobs/{job_id}/rank-candidates with
{"resume_ids": [...]} runs the SAME pipeline used by /api/matches for
every candidate (no ranking-specific scoring code — what a recruiter sees
is what a candidate sees, up to the pointwise fit model), then orders the
slate by overall score.

Reproducibility contract
------------------------
- Every candidate's match is persisted as a normal Match row carrying the
  combined model version (match-model-v0.1+v0.4.0-baseline), so a ranking
  can be audited candidate-by-candidate.
- The response stamps the whole run: model_version + weights + ISO
  timestamp + per-candidate match_ids + a ranking_run id. Re-running with
  the same inputs and the same model_version reproduces the same ordering
  (deterministic pipeline; no sampling anywhere).
- Ties are broken deterministically by (overall score desc, candidate_id
  asc) so ordering never depends on iteration or DB return order.

Honest framing (per the Phase 13 ranking evaluation): the slate is a
"surface excellent candidates" ordering. Binary shortlist precision at
relevant-minority base rates is NOT established — the response carries
the same decision-support disclaimer as single matches.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.models.candidate import CandidateProfile
from app.models.job import Job
from app.models.resume import Resume
from app.services.matching_service import (
    MatchInputError,
    MatchNotFoundError,
    _persist_match,
    run_pipeline,
)

logger = logging.getLogger(__name__)


class RankingInputError(ValueError):
    """Client provided an empty or malformed candidate list."""


@dataclass
class RankedCandidate:
    """One row of the ranking slate."""

    rank: int
    candidate_id: int
    resume_id: int
    candidate_name: str | None
    match_id: int
    overall_score: float          # 0-100 hybrid+ML percent
    fit_score: float | None       # trained-model fit score 0-1 when available
    ml_label: str | None
    band: str
    matched_skills: list[str]
    missing_skills: list[str]
    top_positive_factors: list[str]
    top_negative_factors: list[str]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "resume_id": self.resume_id,
            "candidate_name": self.candidate_name,
            "match_id": self.match_id,
            "overall_score": self.overall_score,
            "fit_score": round(self.fit_score, 4) if self.fit_score is not None else None,
            "ml_label": self.ml_label,
            "band": self.band,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "top_positive_factors": self.top_positive_factors,
            "top_negative_factors": self.top_negative_factors,
        }


@dataclass
class RankingRun:
    """A reproducible, version-stamped ranking of candidates for one job."""

    ranking_run_id: str
    job_id: int
    job_title: str
    model_version: str
    weights: dict[str, float]
    created_at: str
    ranked: list[RankedCandidate] = field(default_factory=list)
    failed_resume_ids: list[int] = field(default_factory=list)
    disclaimer: str = (
        "Rankings are model-estimated compatibility, not hiring "
        "recommendations. See docs/model-card.md for limitations."
    )

    def to_dict(self) -> dict:
        return {
            "ranking_run_id": self.ranking_run_id,
            "job_id": self.job_id,
            "job_title": self.job_title,
            "model_version": self.model_version,
            "weights": self.weights,
            "created_at": self.created_at,
            "ranked": [c.to_dict() for c in self.ranked],
            "failed_resume_ids": self.failed_resume_ids,
            "disclaimer": self.disclaimer,
        }


def _fit_score_from_details(ml_details: dict | None) -> float | None:
    if not ml_details:
        return None
    fit = ml_details.get("fit_score")
    return float(fit) if fit is not None else None


def rank_candidates_for_job(
    db: Session,
    job_id: int,
    resume_ids: list[int],
    weights: dict | None = None,
) -> RankingRun:
    """Run the match pipeline for every candidate and return an ordered slate.

    Every successfully matched candidate gets a persisted Match row (with
    explanation). Candidates whose resume text is missing/empty are
    reported in ``failed_resume_ids`` instead of failing the whole run.
    """
    if not resume_ids:
        raise RankingInputError("resume_ids must not be empty")
    if len(resume_ids) != len(set(resume_ids)):
        raise RankingInputError("resume_ids contains duplicates")

    job_row = db.get(Job, job_id)
    if job_row is None:
        raise MatchNotFoundError(f"Job {job_id} not found")

    ranked: list[RankedCandidate] = []
    failed: list[int] = []
    model_version: str | None = None
    run_weights: dict[str, float] = {}

    for resume_id in resume_ids:
        resume = db.get(Resume, resume_id)
        if resume is None:
            logger.warning("ranking: resume %s not found; skipping", resume_id)
            failed.append(resume_id)
            continue
        cv_text = resume.raw_text or ""
        if not cv_text.strip():
            logger.warning("ranking: resume %s has no text; skipping", resume_id)
            failed.append(resume_id)
            continue

        try:
            output = run_pipeline(cv_text, job_row.description, weights=weights)
        except MatchInputError:
            raise
        except Exception:  # noqa: BLE001 - one bad CV must not kill the slate
            logger.exception("ranking: pipeline failed for resume %s; skipping", resume_id)
            failed.append(resume_id)
            continue

        # Reuse or create the CandidateProfile for this resume (same rule
        # as create_match_from_entities).
        candidate = (
            db.query(CandidateProfile).filter(CandidateProfile.resume_id == resume.id).first()
        )
        if candidate is None:
            candidate = CandidateProfile(
                resume_id=resume.id,
                name=output.candidate_name,
                extracted_data={"source": "ranking"},
            )
            db.add(candidate)
            db.flush()

        match_row = _persist_match(db, candidate.id, job_row.id, output)
        db.flush()  # assign match_row.id before commit

        # All candidates share the same model version + weights (same
        # pipeline, same request); capture once from the first result.
        if model_version is None:
            model_version = output.result.model_version
            run_weights = dict(output.result.weights)

        ml = output.ml_details or {}
        ranked.append(
            RankedCandidate(
                rank=0,  # assigned after sorting
                candidate_id=candidate.id,
                resume_id=resume.id,
                candidate_name=output.candidate_name,
                match_id=match_row.id,
                overall_score=float(output.result.overall_percent),
                fit_score=_fit_score_from_details(ml),
                ml_label=ml.get("label"),
                band=output.result.band,
                matched_skills=output.matched_skills,
                missing_skills=output.missing_skills,
                top_positive_factors=result_positives(output),
                top_negative_factors=result_negatives(output),
            )
        )

    # Deterministic order: score desc, then candidate_id asc (stable tiebreak).
    ranked.sort(key=lambda rc: (-rc.overall_score, rc.candidate_id))
    for i, rc in enumerate(ranked, start=1):
        rc.rank = i

    db.commit()

    return RankingRun(
        ranking_run_id=uuid.uuid4().hex,
        job_id=job_row.id,
        job_title=job_row.title,
        model_version=model_version or "unknown",
        weights=run_weights,
        created_at=datetime.now(UTC).isoformat(),
        ranked=ranked,
        failed_resume_ids=failed,
    )


def result_positives(output) -> list[str]:
    return list(output.result.positive_factors or [])[:3]


def result_negatives(output) -> list[str]:
    return list(output.result.negative_factors or [])[:3]
