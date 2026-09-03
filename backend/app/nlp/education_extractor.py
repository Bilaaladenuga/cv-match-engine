"""
Education Extractor — extracts structured education data from CV text.

Extracts:
- Degree level (Bachelor's, Master's, PhD, etc.)
- Field of study
- Institution name
- Graduation year
"""

import re
from dataclasses import dataclass


@dataclass
class EducationEntry:
    """A single education entry."""
    degree: str | None  # e.g., "Bachelor of Science"
    field_of_study: str | None  # e.g., "Computer Science"
    institution: str | None  # e.g., "MIT"
    year: str | None  # e.g., "2018"
    raw_text: str  # Original text


# Degree keywords by level
DEGREE_LEVELS = {
    "phd": ["phd", "ph.d", "doctorate", "doctoral"],
    "master": ["master", "masters", "m.s.", "m.a.", "mba", "m.sc.", "m.eng.", "m.b.a."],
    "bachelor": ["bachelor", "bachelors", "b.s.", "b.a.", "b.sc.", "b.eng.", "bs", "ba"],
    "associate": ["associate", "associates", "a.s.", "a.a."],
    "diploma": ["diploma", "certificate", "cert"],
}

# Common fields of study
FIELDS_OF_STUDY = [
    "computer science", "computer engineering", "software engineering",
    "information technology", "information systems", "data science",
    "electrical engineering", "mechanical engineering", "mathematics",
    "physics", "chemistry", "biology", "economics", "finance",
    "business administration", "marketing", "psychology",
    "statistics", "applied mathematics", "artificial intelligence",
    "cybersecurity", "network engineering", "web development",
]

# Year pattern
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _detect_degree_level(text: str) -> str | None:
    """Detect the degree level from text."""
    text_lower = text.lower()
    for level, keywords in DEGREE_LEVELS.items():
        for kw in keywords:
            if kw in text_lower:
                return level
    return None


def _detect_field(text: str) -> str | None:
    """Detect the field of study from text."""
    text_lower = text.lower()
    for field in FIELDS_OF_STUDY:
        if field in text_lower:
            # Return properly capitalized
            return field.title()
    return None


def _detect_year(text: str) -> str | None:
    """Extract graduation year from text."""
    match = YEAR_PATTERN.search(text)
    return match.group() if match else None


def _detect_institution(text: str, year: str | None, degree: str | None) -> str | None:
    """
    Try to extract the institution name.

    Strategy: Remove known elements (degree, year, field) and what remains
    is likely the institution.
    """
    cleaned = text

    # Remove year
    if year:
        cleaned = cleaned.replace(year, "")

    # Remove degree keywords
    for keywords in DEGREE_LEVELS.values():
        for kw in keywords:
            cleaned = re.sub(re.escape(kw), "", cleaned, flags=re.IGNORECASE)

    # Remove field of study
    for field in FIELDS_OF_STUDY:
        cleaned = re.sub(re.escape(field), "", cleaned, flags=re.IGNORECASE)

    # Remove common separators and noise
    cleaned = re.sub(r"[,.\-–—|()/]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # What's left is likely the institution
    if cleaned and len(cleaned) > 2:
        return cleaned.title()
    return None


def extract_education(text: str) -> list[EducationEntry]:
    """
    Extract education entries from education section text.

    Handles formats like:
    "Bachelor of Science in Computer Science, MIT, 2018"
    "M.S. Computer Science — Stanford University (2020)"
    "University of California, Berkeley | BS Computer Science | 2018"
    """
    # Split into potential entries (by newline or bullet points)
    entries_raw = re.split(r"\n|•|●|▪|- ", text)
    results: list[EducationEntry] = []

    for entry_text in entries_raw:
        entry_text = entry_text.strip()
        if not entry_text or len(entry_text) < 3:
            continue

        # Check if this line contains education-related content
        entry_lower = entry_text.lower()
        has_degree = _detect_degree_level(entry_text) is not None
        has_field = _detect_field(entry_text) is not None
        has_year = _detect_year(entry_text) is not None
        has_institution_keyword = any(
            kw in entry_lower
            for kw in ["university", "college", "institute", "school", "academy"]
        )

        # Include if it has at least a degree or institution keyword
        if has_degree or has_field or has_institution_keyword or has_year:
            degree_level = _detect_degree_level(entry_text)
            field = _detect_field(entry_text)
            year = _detect_year(entry_text)

            # Build degree string
            degree = None
            if degree_level:
                degree = degree_level.title()

            institution = _detect_institution(entry_text, year, degree)

            results.append(EducationEntry(
                degree=degree,
                field_of_study=field,
                institution=institution,
                year=year,
                raw_text=entry_text,
            ))

    return results
