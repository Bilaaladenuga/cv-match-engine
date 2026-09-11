"""
Work Experience Extractor — extracts structured work history from CV text.

Extracts:
- Company names
- Job titles / roles
- Date ranges (start → end)
- Duration in months
"""

import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WorkExperience:
    """A single work experience entry."""
    company: str | None
    role: str | None
    start_date: str | None  # Original string, e.g., "Jan 2021"
    end_date: str | None  # Original string or "Present"
    duration_months: int | None  # Calculated if possible

    @property
    def duration_years(self) -> float | None:
        if self.duration_months is None:
            return None
        return round(self.duration_months / 12, 1)


# Date patterns
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Pattern: "Jan 2021", "January 2021", "01/2021", "2021-01"
DATE_PATTERN = re.compile(
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4}"
    r"|\d{1,2}/\d{4}"
    r"|\d{4}-\d{1,2}"
    r"|\d{4})",
    re.IGNORECASE,
)

# Pattern: "Jan 2021 - Present", "2019 - 2021", "Mar 2020 – Aug 2022"
DATE_RANGE_PATTERN = re.compile(
    r"((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4}"
    r"|\d{1,2}/\d{4}"
    r"|\d{4}-\d{1,2}"
    r"|\d{4})"
    r"\s*[-–—to]+\s*"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4}"
    r"|\d{1,2}/\d{4}"
    r"|\d{4}-\d{1,2}"
    r"|\d{4}"
    r"|present|current|now))",
    re.IGNORECASE,
)


def _parse_date(date_str: str) -> datetime | None:
    """Try to parse a date string into a datetime object."""
    date_str = date_str.strip().lower()

    # "present" / "current" / "now"
    if date_str in ("present", "current", "now"):
        return datetime.now()

    # Month Year format: "Jan 2021", "January 2021"
    match = re.match(r"(\w+)\.?\s+(\d{4})", date_str)
    if match:
        month_str, year_str = match.groups()
        month = MONTHS.get(month_str[:3].lower())
        if month:
            try:
                return datetime(int(year_str), month, 1)
            except ValueError:
                pass

    # MM/YYYY format: "01/2021"
    match = re.match(r"(\d{1,2})/(\d{4})", date_str)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            try:
                return datetime(year, month, 1)
            except ValueError:
                pass

    # YYYY-MM format: "2021-01"
    match = re.match(r"(\d{4})-(\d{1,2})", date_str)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            try:
                return datetime(year, month, 1)
            except ValueError:
                pass

    # Year only: "2021"
    match = re.match(r"(\d{4})$", date_str)
    if match:
        try:
            return datetime(int(match.group(1)), 1, 1)
        except ValueError:
            pass

    return None


MONTH_PRECISION_PATTERN = re.compile(
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4}"
    r"|\d{1,2}/\d{4}"
    r"|\d{4}-\d{1,2})",
    re.IGNORECASE,
)


def _has_month_precision(date_str: str) -> bool:
    """True if the date string specifies a month (not just a year)."""
    return bool(MONTH_PRECISION_PATTERN.match(date_str.strip()))


def _calculate_months(start: str, end: str) -> int | None:
    """
    Calculate months between two date strings.

    A month-precision end date (e.g. "Dec 2022") is treated as inclusive:
    the person worked through that month. Year-only end dates are treated
    as exclusive ("2019 - 2021" is 24 months).
    """
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if start_date and end_date:
        delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        # Employment ending in a named month runs through the end of it
        if _has_month_precision(end):
            delta_months += 1
        return max(0, delta_months)
    return None


def extract_experience(text: str) -> list[WorkExperience]:
    """
    Extract work experience entries from experience section text.

    Looks for patterns like:
    "Software Engineer | Google | Jan 2020 - Dec 2022"
    "Developer at Microsoft (2019 - 2021)"
    """
    lines = text.strip().split("\n")
    experiences: list[WorkExperience] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue

        # Look for date ranges in the line
        date_match = DATE_RANGE_PATTERN.search(stripped)

        if date_match:
            date_range = date_match.group(1)
            # Split on the separator
            parts = re.split(r"\s*[-–—to]+\s*", date_range, maxsplit=1, flags=re.IGNORECASE)
            start_str = parts[0].strip() if parts else None
            end_str = parts[1].strip() if len(parts) > 1 else None

            # Extract role and company from the text before/around the date
            before_date = stripped[:date_match.start()].strip()
            after_date = stripped[date_match.end():].strip()

            # Common separators: |, –, at, at Company
            role = None
            company = None

            # Pattern: "Role | Company" or "Role at Company"
            sep_match = re.split(r"\s*(?:\|)\s*", before_date)
            if len(sep_match) >= 2:
                role_part = sep_match[0].strip()
                company_part = sep_match[1].strip()
                # The role part might still contain "at Company"
                if re.search(r"\bat\b", role_part, re.IGNORECASE):
                    parts_at = re.split(r"\bat\b", role_part, flags=re.IGNORECASE)
                    role = parts_at[0].strip()
                    company = parts_at[1].strip()
                else:
                    role = role_part
                    company = company_part if company_part else None
            elif re.search(r"\bat\b", before_date, re.IGNORECASE):
                parts_at = re.split(r"\bat\b", before_date, flags=re.IGNORECASE)
                role = parts_at[0].strip()
                company = parts_at[1].strip()
            elif before_date:
                # Could be just the role; check after_date for company
                role = before_date
                if after_date:
                    company = after_date.strip("(|)")

            duration_months = _calculate_months(start_str, end_str) if start_str and end_str else None

            experiences.append(WorkExperience(
                company=company,
                role=role,
                start_date=start_str,
                end_date=end_str,
                duration_months=duration_months,
            ))

    return experiences


def calculate_total_years_experience(experiences: list[WorkExperience]) -> float | None:
    """Sum up total experience from all work entries."""
    total_months = 0
    has_data = False
    for exp in experiences:
        if exp.duration_months is not None:
            total_months += exp.duration_months
            has_data = True

    if not has_data:
        return None
    return round(total_months / 12, 1)


# ---------------------------------------------------------------------------
# Full-text fallback (CVs without a detectable experience section)
# ---------------------------------------------------------------------------

# Lines that look like education entries, not jobs — scanning full text, a
# date range next to "B.S." or "University" is a graduation date, not tenure.
_EDU_LINE_HINTS = re.compile(
    r"\b(b\.?s\.?c?\.?|b\.?a\.?|m\.?s\.?c?\.?|m\.?a\.?|m\.?b\.?a|ph\.?d\.?|"
    r"bachelor|master|associate degree|diploma|graduat(?:e|ed|ing)|degree|"
    r"university|college|institute|school of|education)\b",
    re.IGNORECASE,
)


def extract_experience_from_text(text: str) -> list[WorkExperience]:
    """
    Fallback for prose-style CVs with no detectable experience section.

    Scans every line for a date range accompanied by role/company text,
    skipping education-looking lines (their dates are graduation years).
    Reuses the per-line parsing of `extract_experience`.
    """
    experiences: list[WorkExperience] = []
    seen: set[tuple] = set()
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue
        if _EDU_LINE_HINTS.search(stripped):
            continue
        if not DATE_RANGE_PATTERN.search(stripped):
            continue
        # Require at least one alphabetic word besides the dates (a bare
        # "2019 - 2021" line carries no role/company evidence).
        if not re.search(r"[A-Za-z]{3,}", DATE_RANGE_PATTERN.sub(" ", stripped)):
            continue
        for exp in extract_experience(stripped):
            key = (exp.role, exp.company, exp.start_date, exp.end_date)
            if key not in seen:
                seen.add(key)
                experiences.append(exp)
    return experiences


# ---------------------------------------------------------------------------
# Prose years-of-experience claims ("thirteen years of experience")
# ---------------------------------------------------------------------------

_WORD_YEARS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

YEARS_CLAIM_PATTERN = re.compile(
    r"\b(\d{1,2}|" + "|".join(_WORD_YEARS) + r")\s*\+?\s*"
    r"years?'?(?:\s+of)?\s+(?:professional\s+|relevant\s+|hands-on\s+|"
    r"proven\s+|extensive\s+|valuable\s+|direct\s+|industry\s+)?experience\b",
    re.IGNORECASE,
)


def extract_total_years_claim(text: str) -> float | None:
    """
    Extract a stated total years-of-experience claim from prose.

    Handles "13 years of experience", "five+ years experience",
    "3 years' professional experience". Returns the maximum claim found
    (a CV may restate its tenure in several places), or None when no claim
    is present. This is the candidate's own statement, not verified history.
    """
    best: float | None = None
    for match in YEARS_CLAIM_PATTERN.finditer(text):
        token = match.group(1).lower()
        years = float(_WORD_YEARS.get(token, token))
        if best is None or years > best:
            best = years
    return best
