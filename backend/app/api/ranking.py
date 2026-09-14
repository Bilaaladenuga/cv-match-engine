"""
Ranking API routes (Phase 15).

POST /api/jobs/{job_id}/rank-candidates — rank stored candidates against
one stored job with a reproducible, version-stamped run.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.match import (
    ErrorResponse,
    RankCandidatesRequest,
    RankingRunResponse,
)
from app.scoring.weights import WeightsError
from app.services.matching_service import MatchNotFoundError
from app.services.ranking_service import RankingInputError, rank_candidates_for_job

router = APIRouter(prefix="/api/jobs", tags=["ranking"])


@router.post(
    "/{job_id}/rank-candidates",
    response_model=RankingRunResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Malformed ranking input"},
        404: {"model": ErrorResponse, "description": "Job or resumes not found"},
        422: {"model": ErrorResponse, "description": "Invalid weights"},
    },
    summary="Rank candidates against a job",
    description=(
        "Runs the same explainable matching pipeline used by POST /api/matches "
        "for every candidate, orders the slate by overall score (ties broken "
        "deterministically), persists a Match row per candidate, and stamps "
        "the run with the combined model version + weights + timestamp."
    ),
)
def rank_candidates(
    job_id: int,
    payload: RankCandidatesRequest,
    db: Session = Depends(get_db),
) -> RankingRunResponse:
    try:
        run = rank_candidates_for_job(
            db, job_id=job_id, resume_ids=payload.resume_ids, weights=payload.weights
        )
    except RankingInputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except MatchNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except WeightsError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return RankingRunResponse(**run.to_dict())
