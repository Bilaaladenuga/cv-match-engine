"""
Document extraction service — turns an uploaded file into normalized text.

Privacy-first design: uploads are written to a temporary file, parsed, and
DELETED immediately after extraction. Nothing is persisted — no file, no
text, no metadata. The caller receives the text and nothing else.

Validation layers
-----------------
1. Extension whitelist (pdf / docx / txt) — rejected before any parsing.
2. Size limit (settings.MAX_UPLOAD_SIZE_MB) — enforced while streaming the
   upload, so an oversized file is cut off rather than fully buffered.
3. Magic-byte content sniffing — the declared extension is not trusted:
   a ".pdf" that is not a PDF (no %PDF header) or a ".docx" that is not a
   ZIP container (no PK header) is rejected as a disguised/mislabeled file.
4. Empty-content rejection — a file that yields no extractable text
   (scanned image PDFs, blank documents) is rejected with a clear error
   rather than flowing into the pipeline as garbage input.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.parsers.base import parse_document

# Magic-byte signatures for content sniffing (extension is untrusted).
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"  # DOCX is a ZIP (OOXML) container

_CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunks


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a successful upload extraction."""

    text: str
    filename: str
    file_type: str
    size_bytes: int


def _validate_extension(filename: str | None) -> str:
    """Return the lowercased extension, or raise 415 if not whitelisted."""
    settings = get_settings()
    if not filename or "." not in filename:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File has no recognizable extension.",
        )
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(f".{e}" for e in settings.ALLOWED_EXTENSIONS)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '.{ext}'. Allowed: {allowed}.",
        )
    return ext


def _sniff_content(head: bytes, ext: str) -> None:
    """Reject files whose bytes contradict their declared extension."""
    if ext == "pdf" and not head.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File claims to be a PDF but its content is not PDF data.",
        )
    if ext == "docx" and not head.startswith(_ZIP_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File claims to be a DOCX but its content is not a DOCX container.",
        )
    # txt: any bytes are acceptable — decoding issues handled at extraction.


def extract_upload(file: UploadFile) -> ExtractionResult:
    """Validate, extract, and normalize text from an uploaded document.

    Raises HTTPException (413 / 415 / 422) on validation failure.
    """
    ext = _validate_extension(file.filename)
    max_bytes = get_settings().MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Stream the upload with a hard size cap — never buffer unbounded input.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = file.file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File exceeds the {get_settings().MAX_UPLOAD_SIZE_MB} MB limit."
                ),
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    _sniff_content(content[:8], ext)

    text = (
        _decode_text(content) if ext == "txt" else _extract_via_tempfile(content, ext)
    )

    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No extractable text found. The file may be empty, "
                "image-only (scanned), or corrupted."
            ),
        )

    return ExtractionResult(
        text=text,
        filename=os.path.basename(file.filename or f"upload.{ext}"),
        file_type=ext,
        size_bytes=total,
    )


def _decode_text(content: bytes) -> str:
    """Decode TXT content, tolerating the common legacy encodings."""
    from app.parsers.base import normalize_text

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover — latin-1 decodes any byte sequence
        raw = content.decode("latin-1", errors="replace")
    return normalize_text(raw)


def _extract_via_tempfile(content: bytes, ext: str) -> str:
    """Write bytes to a temp file, run the Phase 3 parser, delete the file.

    The parsers are path-based, so binary formats go through a real file —
    created in a private temp directory and removed in a finally block.
    """
    parser_for_ext = {"pdf": "pdf", "docx": "docx"}
    suffix = f".{parser_for_ext[ext]}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return parse_document(tmp_path)
        except Exception as exc:
            # Malformed/password-protected/corrupt documents surface as
            # library errors (e.g. PyMuPDF FileDataError). Translate them
            # into a clear client error instead of a 500.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Could not read the {ext} file; it may be corrupted "
                    "or password-protected."
                ),
            ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):  # best-effort cleanup
                os.remove(tmp_path)
