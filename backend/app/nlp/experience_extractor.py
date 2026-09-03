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
