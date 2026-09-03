"""
Tests for document parsers and CV section detection.
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.parsers.base import (
    BaseParser,
    detect_file_type,
    get_parser,
    parse_document,
)
from app.parsers.section_detector import (
    ContactInfo,
    DetectedSections,
    _classify_heading,
    _extract_contact_info,
    _extract_name,
    _is_heading_line,
    detect_sections,
)
from app.parsers.txt_parser import TXTParser

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"


# --- Helper ---

def _write_temp_file(content: str, suffix: str = ".txt") -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return path


# --- Parser Factory Tests ---

class TestDetectFileType:
    def test_pdf(self):
        assert detect_file_type("resume.pdf") == "pdf"

    def test_docx(self):
        assert detect_file_type("resume.docx") == "docx"

    def test_txt(self):
        assert detect_file_type("resume.txt") == "txt"

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            detect_file_type("resume.jpg")

    def test_uppercase_ext(self):
        assert detect_file_type("Resume.PDF") == "pdf"
        assert detect_file_type("Resume.DOCX") == "docx"


class TestGetParser:
    def test_txt_parser(self):
        parser = get_parser("cv.txt")
        assert isinstance(parser, TXTParser)

    def test_pdf_parser(self):
        from app.parsers.pdf_parser import PDFParser
        parser = get_parser("cv.pdf")
        assert isinstance(parser, PDFParser)

    def test_docx_parser(self):
        from app.parsers.docx_parser import DOCXParser
        parser = get_parser("cv.docx")
        assert isinstance(parser, DOCXParser)


# --- TXT Parser Tests ---

class TestTXTParser:
    def test_basic_parse(self):
        path = _write_temp_file("Hello World\nSecond line")
        try:
            parser = TXTParser()
            result = parser.parse(path)
            assert result == "Hello World\nSecond line"
        finally:
            os.unlink(path)

    def test_normalization(self):
        path = _write_temp_file("Hello   World\t\tTab\n\n\n\n\n\nDouble newlines")
        try:
            parser = TXTParser()
            result = parser.parse(path)
            assert "\t" not in result
            assert "   " not in result
            assert "\n\n\n" not in result
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = _write_temp_file("")
        try:
            parser = TXTParser()
            result = parser.parse(path)
            assert result == ""
        finally:
            os.unlink(path)

    def test_sample_cv(self):
        """Parse the sample CV file."""
        sample_path = str(SAMPLE_DATA_DIR / "sample_cv.txt")
        if not os.path.exists(sample_path):
            pytest.skip("Sample CV not found")
        parser = TXTParser()
        result = parser.parse(sample_path)
        assert len(result) > 100
        assert "Sarah Johnson" in result
        assert "Python" in result


# --- Section Heading Detection Tests ---

class TestHeadingDetection:
    def test_is_heading_caps(self):
        assert _is_heading_line("SKILLS") is True
        assert _is_heading_line("WORK EXPERIENCE") is True

    def test_is_heading_titlecase(self):
        assert _is_heading_line("Education") is True
        assert _is_heading_line("Summary") is True

    def test_not_heading_long_line(self):
        assert _is_heading_line("This is a very long line that is definitely not a section heading") is False

    def test_not_heading_empty(self):
        assert _is_heading_line("") is False

    def test_classify_skills(self):
        assert _classify_heading("SKILLS") == "skills"
        assert _classify_heading("Technical Skills") == "skills"

    def test_classify_experience(self):
        assert _classify_heading("EXPERIENCE") == "experience"
        assert _classify_heading("Work Experience") == "experience"

    def test_classify_education(self):
        assert _classify_heading("EDUCATION") == "education"

    def test_classify_unknown(self):
        assert _classify_heading("random heading") is None


# --- Contact Extraction Tests ---

class TestContactExtraction:
    def test_extract_email(self):
        text = "Contact me at john@example.com or jane@test.org"
        contact = _extract_contact_info(text)
        assert "john@example.com" in contact.emails
        assert "jane@test.org" in contact.emails

    def test_extract_phone(self):
        text = "Call me at +1-555-123-4567 or (555) 987-6543"
        contact = _extract_contact_info(text)
        assert len(contact.phones) >= 1

    def test_extract_linkedin(self):
        text = "Visit linkedin.com/in/johndoe for my profile"
        contact = _extract_contact_info(text)
        assert contact.linkedin is not None
        assert "johndoe" in contact.linkedin

    def test_extract_github(self):
        text = "Code at github.com/johndoe"
        contact = _extract_contact_info(text)
        assert contact.github is not None


# --- Name Extraction Tests ---

class TestNameExtraction:
    def test_name_at_top(self):
        text = "John Smith\nj@email.com\n+123456\n\nSUMMARY\nEngineer"
        name = _extract_name(text, ContactInfo(emails=["j@email.com"]))
        assert name == "John Smith"

    def test_name_not_email_line(self):
        text = "john@email.com\nJohn Smith\nSUMMARY"
        name = _extract_name(text, ContactInfo(emails=["john@email.com"]))
        assert name == "John Smith"

    def test_no_name_found(self):
        text = "j@email.com\n+1234567890\nSKILLS\nPython"
        name = _extract_name(text, ContactInfo(emails=["j@email.com"]))
        assert name is None


# --- Full Section Detection Tests ---

class TestSectionDetection:
    @pytest.fixture
    def sample_cv(self):
        """Parse the sample CV for section detection tests."""
        sample_path = str(SAMPLE_DATA_DIR / "sample_cv.txt")
        if not os.path.exists(sample_path):
            pytest.skip("Sample CV not found")
        with open(sample_path, "r") as f:
            return f.read()

    def test_detects_name(self, sample_cv):
        result = detect_sections(sample_cv)
        assert result.name == "Sarah Johnson"

    def test_detects_email(self, sample_cv):
        result = detect_sections(sample_cv)
        assert "sarah.johnson@email.com" in result.contact.emails

    def test_detects_phone(self, sample_cv):
        result = detect_sections(sample_cv)
        assert len(result.contact.phones) >= 1

    def test_detects_linkedin(self, sample_cv):
        result = detect_sections(sample_cv)
        assert result.contact.linkedin is not None

    def test_detects_github(self, sample_cv):
        result = detect_sections(sample_cv)
        assert result.contact.github is not None

    def test_detects_summary(self, sample_cv):
        result = detect_sections(sample_cv)
        assert result.summary is not None
        assert "Senior software engineer" in result.summary

    def test_detects_skills_section(self, sample_cv):
        result = detect_sections(sample_cv)
        skills_section = result.get_section("skills")
        assert skills_section is not None
        assert "Python" in skills_section.content
        assert "React" in skills_section.content

    def test_detects_experience_section(self, sample_cv):
        result = detect_sections(sample_cv)
        exp_section = result.get_section("experience")
        assert exp_section is not None
        assert "TechCorp" in exp_section.content

    def test_detects_education_section(self, sample_cv):
        result = detect_sections(sample_cv)
        edu_section = result.get_section("education")
        assert edu_section is not None
        assert "Berkeley" in edu_section.content

    def test_detects_certifications(self, sample_cv):
        result = detect_sections(sample_cv)
        cert_section = result.get_section("certifications")
        assert cert_section is not None
        assert "AWS" in cert_section.content

    def test_section_count(self, sample_cv):
        result = detect_sections(sample_cv)
        assert len(result.sections) >= 4  # at least summary, skills, experience, education

    def test_empty_text(self):
        result = detect_sections("")
        assert result.name is None
        assert len(result.sections) == 0
