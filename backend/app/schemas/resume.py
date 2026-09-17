"""
Resume API schemas — upload extraction responses.

The service is stateless and privacy-first: requests are multipart uploads,
responses carry the extracted text and nothing that identifies a user
(no accounts exist by design).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractResponse(BaseModel):
    """Response for POST /api/resumes/extract."""

    filename: str = Field(..., description="Sanitized original filename.")
    file_type: str = Field(..., description="Detected file type: pdf, docx, or txt.")
    size_bytes: int = Field(..., ge=0, description="Upload size in bytes.")
    char_count: int = Field(..., ge=0, description="Length of extracted text.")
    text: str = Field(..., description="Normalized text extracted from the document.")
