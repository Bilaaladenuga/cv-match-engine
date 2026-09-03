"""
DOCX document parser using python-docx.
"""

from docx import Document

from app.parsers.base import BaseParser


class DOCXParser(BaseParser):
    """Extract text from DOCX files."""

    def extract_text(self, file_path: str) -> str:
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
