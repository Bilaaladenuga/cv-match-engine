"""
Plain text document parser.
"""

from app.parsers.base import BaseParser


class TXTParser(BaseParser):
    """Read plain text files with encoding detection."""

    # Encodings to try in order
    ENCODINGS = ["utf-8", "latin-1", "cp1252", "ascii"]

    def extract_text(self, file_path: str) -> str:
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Could not decode {file_path} with any supported encoding")
