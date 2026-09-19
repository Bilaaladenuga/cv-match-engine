"""
Matching API routes.

POST /api/matches — run the explainable matching pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.match import ErrorResponse, MatchRequest, MatchResponse
from app.scoring.weights import WeightsError
from app.services.matching_service import (
    MatchInputError,
    MatchNotFoundError,
    create_match_from_entities,
    create_match_from_texts,
    run_pipeline,
)

# The product is privacy-first free software: the primary /api/matches flow
# is STATELESS (nothing persisted, no accounts). Persistence exists for the
# recruiter ranking service, which needs stored resumes/jobs.
_ENABLE_STATELESS_PERSISTENCE = False

router = APIRouter(prefix="/api", tags=["matches"])


def _build_response(match_row, output) -> MatchResponse:
    result = output.result
    return MatchResponse(
        match_id=match_row.id if match_row is not None else None,
        model_version=result.model_version,
        overall_score=result.overall_score,
        overall_percent=result.overall_percent,
        band=result.band,
        components=[c.to_dict() for c in result.components],
        weights=result.weights,
        positive_factors=result.positive_factors,
        negative_factors=result.negative_factors,
        recommendations=result.recommendations,
        prioritized_improvements=output.prioritized_improvements,
        ats_details=output.ats_details,
        disclaimer=result.disclaimer,
        skill_matches=output.skill_matches,
        experience=output.experience,
        semantic=output.semantic,
        education=output.education,
        certifications=output.certifications,
        skill_evidence=output.skill_evidence,
        ml_details=output.ml_details,
        ml_failure=output.ml_failure,
    )


@router.post(
    "/matches",
    response_model=MatchResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def create_match(request: MatchRequest, db: Session = Depends(get_db)):
    """
    Run the CV–Job matching pipeline and return the explainable result.

    - **resume_id + job_id**: match two stored entities (result persisted).
    - **cv_text + job_text**: match raw text; entities and result are also
      persisted under a demo user until authentication lands (Phase 20).

    Provide one mode, not both.
    """
    has_entities = request.resume_id is not None or request.job_id is not None
    has_texts = bool((request.cv_text or "").strip()) and bool(
        (request.job_text or "").strip()
    )

    if has_entities and has_texts:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Provide either resume_id/job_id or cv_text/job_text, not both.",
        )
    if request.resume_id is not None and request.job_id is None:
        raise HTTPException(400, detail="job_id is required when resume_id is given")
    if request.job_id is not None and request.resume_id is None:
        raise HTTPException(400, detail="resume_id is required when job_id is given")
    if not has_entities and not has_texts:
        raise HTTPException(
            400, detail="Provide resume_id+job_id or cv_text+job_text"
        )

    try:
        if has_entities:
            # Entity mode REQUIRES the database: the inputs live there.
            match_row, output = create_match_from_entities(
                db, request.resume_id, request.job_id, weights=request.weights
            )
        else:
            if _ENABLE_STATELESS_PERSISTENCE:
                # Optional opt-in persistence (self-hosters who want server-
                # side history). Degrades to stateless if the DB is down.
                try:
                    match_row, output = create_match_from_texts(
                        db, request.cv_text, request.job_text,
                        weights=request.weights,
                    )
                except OperationalError:
                    db.rollback()
                    output = run_pipeline(
                        request.cv_text or "", request.job_text or "",
                        weights=request.weights,
                    )
                    match_row = None
            else:
                # Default: fully stateless — the analysis never touches the
                # database and match_id is always None.
                output = run_pipeline(
                    request.cv_text or "", request.job_text or "",
                    weights=request.weights,
                )
                match_row = None
    except (MatchInputError, MatchNotFoundError) as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND if isinstance(exc, MatchNotFoundError) else 400,
            detail=str(exc),
        ) from exc
    except OperationalError as exc:
        # Entity mode with an unreachable database: the inputs live in the
        # DB, so there is nothing to compute from. 503 tells the client to
        # retry later (and keeps DB outages from surfacing as vague 500s).
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable; entity-based matching requires it.",
        ) from exc
    except WeightsError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return _build_response(match_row, output)


@router.post(
    "/matches/export-pdf",
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def export_match_pdf(request: MatchRequest, db: Session = Depends(get_db)):
    """
    Generate and return a PDF report of the CV-Job match.

    Accepts the same parameters as POST /api/matches.
    Returns a PDF file.
    """
    has_texts = bool((request.cv_text or "").strip()) and bool(
        (request.job_text or "").strip()
    )

    if not has_texts:
        raise HTTPException(
            400, detail="cv_text and job_text are required for PDF export"
        )

    try:
        output = run_pipeline(
            request.cv_text or "", request.job_text or "",
            weights=request.weights,
        )
    except (MatchInputError, MatchNotFoundError) as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND if isinstance(exc, MatchNotFoundError) else 400,
            detail=str(exc),
        ) from exc
    except WeightsError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    result = output.result

    # Generate PDF
    from app.services.pdf_generator import generate_match_report_pdf

    # ml_details is already a dict
    ml_dict = result.ml_details if isinstance(result.ml_details, dict) else None

    pdf_bytes = generate_match_report_pdf(
        overall_percent=result.overall_percent,
        band=result.band,
        ml_label=ml_dict.get("label", "N/A") if ml_dict else "N/A",
        ml_score=ml_dict.get("fit_score", 0) if ml_dict else 0,
        skill_matches=output.skill_matches,
        prioritized_improvements=output.prioritized_improvements,
        recommendations=result.recommendations,
        ml_details=ml_dict,
        ats_details=output.ats_details,
        disclaimer=result.disclaimer,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=match-report-{result.overall_percent}pct.pdf"
        },
    )
