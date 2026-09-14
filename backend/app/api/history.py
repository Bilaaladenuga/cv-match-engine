"""
History API routes — retrieval of stored analyses.

GET /api/history            — most recent analyses (list view)
GET /api/matches/{match_id} — full stored analysis for one match
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.match import ErrorResponse, HistoryEntryOut, MatchDetailResponse
from app.services.history_service import get_match_detail, list_history
from app.services.matching_service import MatchNotFoundError

router = APIRouter(prefix="/api", tags=["history"])


@router.get(
    "/history",
    response_model=list[HistoryEntryOut],
    responses={400: {"model": ErrorResponse}},
    summary="List recent analyses",
    description=(
        "Most recent stored matches, newest first. List views do not "
        "include raw CV text or job descriptions."
    ),
)
def history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[HistoryEntryOut]:
    entries = list_history(db, limit=limit, offset=offset)
    return [HistoryEntryOut(**e.to_dict()) for e in entries]


@router.get(
    "/matches/{match_id}",
    response_model=MatchDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get one stored analysis",
    description=(
        "The full stored result for one match: component scores, ML fit "
        "details, and the explanation — read back from the database, no "
        "recomputation."
    ),
)
def match_detail(match_id: int, db: Session = Depends(get_db)) -> MatchDetailResponse:
    try:
        detail = get_match_detail(db, match_id)
    except MatchNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    return MatchDetailResponse(**detail)
