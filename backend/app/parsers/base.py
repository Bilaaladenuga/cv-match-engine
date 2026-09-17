"""
Base document parser interface and common utilities.
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize raw extracted text: strip whitespace, fix encoding, collapse spaces."""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract raw text from a document."""
        ...

    def normalize_text(self, text: str) -> str:
        """Normalize extracted text: strip whitespace, fix encoding, collapse spaces."""
        return normalize_text(text)

    def parse(self, file_path: str) -> str:
        """Parse a document and return normalized text."""
        raw = self.extract_text(file_path)
        return self.normalize_text(raw)


def detect_file_type(file_path: str) -> str:
    """Detect file type from extension.

    Returns: 'pdf', 'docx', 'txt', or raises ValueError.
    """
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext == "pdf":
        return "pdf"
    elif ext == "docx":
        return "docx"
    elif ext == "txt":
        return "txt"
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def get_parser(file_path: str) -> BaseParser:
    """Factory: return the appropriate parser for a file type."""
    from app.parsers.docx_parser import DOCXParser
    from app.parsers.pdf_parser import PDFParser
    from app.parsers.txt_parser import TXTParser

    file_type = detect_file_type(file_path)
    parsers = {
        "pdf": PDFParser,
        "docx": DOCXParser,
        "txt": TXTParser,
    }
    return parsers[file_type]()


def parse_document(file_path: str) -> str:
    """Convenience function: detect type, parse, return normalized text."""
    parser = get_parser(file_path)
    return parser.parse(file_path)
