"""
CV Section Detector — identifies structural sections in CV text.

Uses a combination of regex patterns and keyword matching to detect:
- Contact info (email, phone)
- Name (heuristic)
- Summary / Objective
- Skills
- Experience / Work History
- Education
- Certifications
- Projects
- Languages
"""

import re
from dataclasses import dataclass, field


# Common section headings (case-insensitive matching)
SECTION_PATTERNS: dict[str, list[str]] = {
    "summary": [
        r"^\s*(summary|professional\s+summary|career\s+summary|objective|profile|about\s+me)\s*$",
    ],
    "skills": [
        r"^\s*(skills?|technical\s+skills?|competencies|technologies|tools|proficiencies)\s*$",
    ],
    "experience": [
        r"^\s*(experience|work\s+experience|employment|work\s+history|professional\s+experience|career\s+history)\s*$",
    ],
    "education": [
        r"^\s*(education|academic|qualifications?|degree|university|studies)\s*$",
    ],
    "certifications": [
        r"^\s*(certifications?|licenses?|credentials|certificates?)\s*$",
    ],
    "projects": [
        r"^\s*(projects?|portfolio|personal\s+projects?|key\s+projects?)\s*$",
    ],
    "languages": [
        r"^\s*(languages?|foreign\s+languages?)\s*$",
    ],
    "contact": [
        r"^\s*(contact|contact\s+info|personal\s+info|reach)\s*$",
    ],
}

# Regex patterns for extracting contact info
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
)
# LinkedIn and GitHub patterns
LINKEDIN_PATTERN = re.compile(r"linkedin\.com/in/[a-zA-Z0-9_-]+", re.IGNORECASE)
GITHUB_PATTERN = re.compile(r"github\.com/[a-zA-Z0-9_-]+", re.IGNORECASE)


@dataclass
class ContactInfo:
    """Extracted contact information."""
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    linkedin: str | None = None
    github: str | None = None


@dataclass
class CVSection:
    """A detected section in a CV."""
    name: str  # section type: summary, skills, experience, etc.
    heading: str  # original heading text
    content: str  # section content
    start_line: int  # line number where section starts
    end_line: int  # line number where section ends


@dataclass
class DetectedSections:
    """All detected sections from a CV."""
    contact: ContactInfo
    sections: list[CVSection]
    full_text: str
    # Convenience accessors
    name: str | None = None
    summary: str | None = None

    def get_section(self, section_type: str) -> CVSection | None:
        """Get a section by type name."""
        for s in self.sections:
            if s.name == section_type:
                return s
        return None

    def get_section_text(self, section_type: str) -> str | None:
        """Get the content text of a section by type."""
        section = self.get_section(section_type)
        return section.content if section else None


def _is_heading_line(line: str) -> bool:
    """Heuristic: a heading is a short, possibly ALL CAPS or Title Case line."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    # ALL CAPS
    if stripped.isupper() and len(stripped.split()) <= 5:
        return True
    # Title Case or single word
    if stripped.istitle() and len(stripped.split()) <= 5:
        return True
    return False


def _classify_heading(heading: str) -> str | None:
    """Try to match a heading line against known section patterns."""
    for section_type, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, heading, re.IGNORECASE):
                return section_type
    return None


def _extract_contact_info(text: str) -> ContactInfo:
    """Extract contact information from CV text."""
    contact = ContactInfo()

    # Emails
    contact.emails = list(set(EMAIL_PATTERN.findall(text)))

    # Phones (filter out things that are clearly not phone numbers)
    raw_phones = PHONE_PATTERN.findall(text)
    contact.phones = [
        p.strip() for p in raw_phones
        if len(re.sub(r"[^\d]", "", p)) >= 7
    ]

    # LinkedIn
    linkedin_match = LINKEDIN_PATTERN.search(text)
    if linkedin_match:
        contact.linkedin = linkedin_match.group()

    # GitHub
    github_match = GITHUB_PATTERN.search(text)
    if github_match:
        contact.github = github_match.group()

    return contact


def _extract_name(text: str, contact: ContactInfo) -> str | None:
    """Heuristic to extract the candidate name from the first lines of a CV."""
    lines = text.strip().split("\n")
    for line in lines[:10]:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that look like section headings
        if _classify_heading(stripped):
            continue
        # Skip lines containing email or phone
        if EMAIL_PATTERN.search(stripped) or PHONE_PATTERN.search(stripped):
            continue
        # Skip lines with URLs
        if "http" in stripped.lower() or "www." in stripped.lower():
            continue
        # Name heuristic: 2-4 words, mostly alphabetic, not too long
        words = stripped.split()
        if 2 <= len(words) <= 4 and all(w.replace(".", "").replace("'", "").isalpha() for w in words):
            return stripped
    return None


def detect_sections(text: str) -> DetectedSections:
    """
    Analyze CV text and detect structural sections.

    Returns DetectedSections with parsed sections, contact info, and name.
    """
    lines = text.split("\n")
    contact = _extract_contact_info(text)
    name = _extract_name(text, contact)

    sections: list[CVSection] = []
    current_section: CVSection | None = None
    current_content_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check if this line is a section heading
        section_type = _classify_heading(stripped) if _is_heading_line(stripped) else None

        if section_type:
            # Save previous section if it exists
            if current_section:
                current_section.content = "\n".join(current_content_lines).strip()
                current_section.end_line = i - 1
                sections.append(current_section)

            # Start new section
            current_section = CVSection(
                name=section_type,
                heading=stripped,
                content="",
                start_line=i,
                end_line=i,
            )
            current_content_lines = []
        elif current_section is not None:
            # Add line to current section content
            current_content_lines.append(line)
        # Lines before any section heading are pre-section content (ignored for now)

    # Don't forget the last section
    if current_section:
        current_section.content = "\n".join(current_content_lines).strip()
        current_section.end_line = len(lines) - 1
        sections.append(current_section)

    # Extract summary if available
    summary = None
    summary_section = next((s for s in sections if s.name == "summary"), None)
    if summary_section:
        summary = summary_section.content

    return DetectedSections(
        contact=contact,
        sections=sections,
        full_text=text,
        name=name,
        summary=summary,
    )
