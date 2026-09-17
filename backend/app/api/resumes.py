"""
Resume document endpoints — upload extraction only.

Privacy-first: there is deliberately NO persistence and NO storage endpoint.
The upload is validated, its text extracted, and the file discarded. The
client keeps the text in browser memory/localStorage (no server-side
accounts exist by design).

    POST /api/resumes/extract   multipart upload → normalized text
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, UploadFile

from app.schemas.resume import ExtractResponse
from app.services.document_service import extract_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/resumes", tags=["resumes"])


@router.post("/extract", response_model=ExtractResponse)
async def extract_resume_text(file: UploadFile = File(...)) -> ExtractResponse:
    """Extract normalized text from an uploaded CV (PDF / DOCX / TXT).

    The document is processed in memory/temp storage and deleted immediately;
    nothing is persisted server-side. Returns the text plus basic metadata.
    """
    result = extract_upload(file)
    logger.info(
        "resume.extract ok filename=%r type=%s size=%d chars=%d",
        result.filename,
        result.file_type,
        result.size_bytes,
        len(result.text),
    )
    return ExtractResponse(
        filename=result.filename,
        file_type=result.file_type,
        size_bytes=result.size_bytes,
        char_count=len(result.text),
        text=result.text,
    )
