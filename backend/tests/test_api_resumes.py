"""Tests for POST /api/resumes/extract — stateless upload extraction.

Covers the full validation ladder (extension whitelist, magic-byte sniffing,
size cap, empty content), successful text extraction for txt/docx, and the
privacy contract: uploads leave no trace in the database.
"""

from __future__ import annotations

import io

from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

CV_TEXT = (
    "Jane Okafor\njane.okafor@example.com\n\n"
    "SKILLS\nPython, PostgreSQL, Docker\n\n"
    "EXPERIENCE\nData Engineer, Acme Ltd, 2021 - Present\n"
    "Built ETL pipelines with Python and PostgreSQL.\n"
)


def _client() -> TestClient:
    # No DB dependency needed — the endpoint is fully stateless.
    return TestClient(app, raise_server_exceptions=False)


def _upload(name: str, payload: bytes, content_type: str):
    return _client().post(
        "/api/resumes/extract",
        files={"file": (name, io.BytesIO(payload), content_type)},
    )


def _docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def test_txt_extraction_returns_normalized_text():
    resp = _upload("cv.txt", CV_TEXT.encode("utf-8"), "text/plain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "cv.txt"
    assert body["file_type"] == "txt"
    assert body["char_count"] == len(body["text"])
    assert "Jane Okafor" in body["text"]
    assert "ETL pipelines" in body["text"]


def test_docx_extraction_returns_text():
    payload = _docx_bytes(
        ["Jane Okafor", "SKILLS", "Python, PostgreSQL, Docker", "EXPERIENCE"]
    )
    resp = _upload("cv.docx", payload, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert resp.status_code == 200
    body = resp.json()
    assert body["file_type"] == "docx"
    assert "Jane Okafor" in body["text"]
    assert "PostgreSQL" in body["text"]


def test_upload_leaves_no_database_trace():
    """Privacy contract: extraction must not persist anything anywhere."""
    # The endpoint does not even take a DB dependency; if it silently did,
    # the app's get_db wiring (PostgreSQL) would fail the request.
    resp = _upload("cv.txt", CV_TEXT.encode("utf-8"), "text/plain")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Validation: extension whitelist
# ---------------------------------------------------------------------------


def test_rejects_disallowed_extension():
    resp = _upload("cv.exe", b"MZ\x90\x00", "application/octet-stream")
    assert resp.status_code == 415
    assert "Unsupported file type" in resp.json()["detail"]


def test_rejects_missing_extension():
    resp = _upload("cv", b"hello", "text/plain")
    assert resp.status_code == 415


# ---------------------------------------------------------------------------
# Validation: magic-byte sniffing
# ---------------------------------------------------------------------------


def test_rejects_fake_pdf():
    resp = _upload("fake.pdf", b"this is not really a pdf", "application/pdf")
    assert resp.status_code == 415
    assert "not PDF data" in resp.json()["detail"]


def test_rejects_fake_docx():
    resp = _upload("fake.docx", b"plain text pretending", "application/msword")
    assert resp.status_code == 415
    assert "not a DOCX container" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Validation: size and emptiness
# ---------------------------------------------------------------------------


def test_rejects_oversized_file(monkeypatch):
    # get_settings() is lru_cached, so patching the instance affects the
    # same object the service reads.
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_SIZE_MB", 1)
    big = b"x" * (1 * 1024 * 1024 + 1)
    resp = _upload("big.txt", big, "text/plain")
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"]


def test_rejects_empty_file():
    resp = _upload("empty.txt", b"", "text/plain")
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"]


def test_rejects_corrupt_pdf():
    """A PDF the parser cannot open (corrupt/truncated) is a clean 422."""
    payload = b"%PDF-1.4\n%%EOF\n"  # valid magic bytes, unparsable structure
    resp = _upload("scan.pdf", payload, "application/pdf")
    assert resp.status_code == 422
    assert "corrupted or password-protected" in resp.json()["detail"]


def test_rejects_whitespace_only_txt():
    """Content that parses but normalizes to nothing is rejected clearly."""
    resp = _upload("blank.txt", b"   \n\n  \t \n", "text/plain")
    assert resp.status_code == 422
    assert "No extractable text" in resp.json()["detail"]
