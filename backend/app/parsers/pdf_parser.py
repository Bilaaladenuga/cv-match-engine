"""
PDF document parser using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF

from app.parsers.base import BaseParser


class PDFParser(BaseParser):
    """Extract text from PDF files."""

    def extract_text(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)
