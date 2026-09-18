"""
ATS (Applicant Tracking System) Friendliness Checker.

Analyzes CV text to determine how well it will pass through ATS filters.
Returns a score (0-100) with specific issues and recommendations.

Common ATS rejection reasons we check for:
1. Non-standard section headings
2. Missing critical sections
3. Poor keyword coverage
4. Inconsistent date formatting
5. Contact info issues
6. Length problems
7. Format hints (tables, columns, graphics)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ATSCheck:
    """One ATS check result."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    weight: float  # importance weight
    message: str
    suggestion: str | None = None


@dataclass
class ATSResult:
    """Complete ATS friendliness analysis."""
    overall_score: int  # 0-100
    pass_estimate: str  # "likely_pass", "borderline", "likely_fail"
    checks: list[ATSCheck]
    issues: list[str]  # list of problems found
    suggestions: list[str]  # list of fixes

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "pass_estimate": self.pass_estimate,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "score": round(c.score, 2),
                    "weight": c.weight,
                    "message": c.message,
                    "suggestion": c.suggestion,
                }
                for c in self.checks
            ],
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


# ---------------------------------------------------------------------------
# Standard ATS section headings
# ---------------------------------------------------------------------------

_STANDARD_HEADINGS = {
    "summary": ["summary", "professional summary", "career summary", "objective", "profile", "about me"],
    "experience": ["experience", "work experience", "employment", "work history", "professional experience"],
    "education": ["education", "academic", "qualifications", "degree", "university"],
    "skills": ["skills", "technical skills", "competencies", "technologies", "proficiencies"],
    "certifications": ["certifications", "licenses", "credentials", "certificates"],
    "projects": ["projects", "portfolio", "personal projects"],
}

# Contact info patterns
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")
_LINKEDIN_PATTERN = re.compile(r"linkedin\.com/in/[a-zA-Z0-9_-]+", re.IGNORECASE)

# Date patterns
_DATE_RANGE_PATTERN = re.compile(
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\.?\s+\d{4}"
    r"|\d{1,2}/\d{4}"
    r"|\d{4}-\d{1,2}"
    r"|\d{4}",
    re.IGNORECASE,
)

# Format red flags (hints of tables, columns, graphics)
_TABLE_HINTS = re.compile(
    r"(?:\|.*\|.*\||"  # pipe-separated columns
    r"\t{2,}|"  # multiple tabs (column alignment)
    r"(?:─|━|═|│|┌|┐|└|┘|├|┤|┬|┴|┼){3,})",  # box drawing characters
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_sections(text: str, sections_found: list[str]) -> ATSCheck:
    """Check for standard section headings."""
    standard_count = sum(1 for s in _STANDARD_HEADINGS if s in sections_found)
    total = len(_STANDARD_HEADINGS)
    score = standard_count / total

    missing = [s for s in _STANDARD_HEADINGS if s not in sections_found]

    if score >= 0.6:
        return ATSCheck(
            name="section_headings",
            passed=True,
            score=score,
            weight=0.20,
            message=f"Found {standard_count}/{total} standard sections",
            suggestion=None,
        )
    else:
        return ATSCheck(
            name="section_headings",
            passed=False,
            score=score,
            weight=0.20,
            message=f"Only {standard_count}/{total} standard sections found. Missing: {', '.join(missing[:3])}",
            suggestion=f"Add standard section headings: {', '.join(missing[:3])}",
        )


def _check_contact_info(text: str) -> ATSCheck:
    """Check for proper contact information."""
    has_email = bool(_EMAIL_PATTERN.search(text))
    has_phone = bool(_PHONE_PATTERN.search(text))
    has_linkedin = bool(_LINKEDIN_PATTERN.search(text))

    found = sum([has_email, has_phone, has_linkedin])
    score = found / 3

    issues = []
    if not has_email:
        issues.append("email")
    if not has_phone:
        issues.append("phone")
    if not has_linkedin:
        issues.append("LinkedIn")

    if score >= 0.67:
        return ATSCheck(
            name="contact_info",
            passed=True,
            score=score,
            weight=0.15,
            message=f"Found {found}/3 contact methods (email, phone, LinkedIn)",
            suggestion=None,
        )
    else:
        return ATSCheck(
            name="contact_info",
            passed=False,
            score=score,
            weight=0.15,
            message=f"Missing contact info: {', '.join(issues)}",
            suggestion=f"Add missing contact information: {', '.join(issues)}",
        )


def _check_skills_section(text: str, has_skills_section: bool) -> ATSCheck:
    """Check for a dedicated skills section."""
    if has_skills_section:
        return ATSCheck(
            name="skills_section",
            passed=True,
            score=1.0,
            weight=0.15,
            message="Dedicated skills section found",
            suggestion=None,
        )
    else:
        return ATSCheck(
            name="skills_section",
            passed=False,
            score=0.0,
            weight=0.15,
            message="No dedicated skills section found",
            suggestion="Add a clear 'Skills' or 'Technical Skills' section with a comma-separated list",
        )


def _check_dates(text: str) -> ATSCheck:
    """Check for consistent date formatting."""
    dates = _DATE_RANGE_PATTERN.findall(text)

    if not dates:
        return ATSCheck(
            name="date_format",
            passed=False,
            score=0.3,
            weight=0.10,
            message="No dates found in CV",
            suggestion="Add dates to your work experience and education (e.g., 'Jan 2020 - Present')",
        )

    # Check for consistency
    formats = set()
    for d in dates:
        d_lower = d.lower()
        if re.match(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", d_lower):
            formats.add("month_year")
        elif re.match(r"\d{1,2}/\d{4}", d):
            formats.add("mm_yyyy")
        elif re.match(r"\d{4}-\d{1,2}", d):
            formats.add("yyyy_mm")
        elif re.match(r"\d{4}$", d):
            formats.add("year_only")

    if len(formats) <= 2:
        return ATSCheck(
            name="date_format",
            passed=True,
            score=0.9,
            weight=0.10,
            message=f"Found {len(dates)} dates with consistent formatting",
            suggestion=None,
        )
    else:
        return ATSCheck(
            name="date_format",
            passed=False,
            score=0.5,
            weight=0.10,
            message=f"Mixed date formats detected ({len(formats)} different styles)",
            suggestion="Use consistent date formatting throughout (e.g., 'MMM YYYY - MMM YYYY')",
        )


def _check_length(text: str, word_count: int) -> ATSCheck:
    """Check if CV length is appropriate."""
    # General guidelines: 1 page per 10 years of experience
    # Ideal: 400-800 words for most levels
    if 300 <= word_count <= 1000:
        return ATSCheck(
            name="cv_length",
            passed=True,
            score=1.0,
            weight=0.10,
            message=f"Good length: {word_count} words",
            suggestion=None,
        )
    elif word_count < 300:
        return ATSCheck(
            name="cv_length",
            passed=False,
            score=0.5,
            weight=0.10,
            message=f"CV may be too short: {word_count} words",
            suggestion="Add more details about your experience and achievements",
        )
    else:
        return ATSCheck(
            name="cv_length",
            passed=False,
            score=0.6,
            weight=0.10,
            message=f"CV may be too long: {word_count} words",
            suggestion="Consider condensing to 1-2 pages for better ATS readability",
        )


def _check_format_hints(text: str) -> ATSCheck:
    """Check for format issues that confuse ATS."""
    table_issues = _TABLE_HINTS.findall(text)

    if not table_issues:
        return ATSCheck(
            name="formatting",
            passed=True,
            score=1.0,
            weight=0.15,
            message="No table/column formatting detected",
            suggestion=None,
        )
    else:
        return ATSCheck(
            name="formatting",
            passed=False,
            score=0.3,
            weight=0.15,
            message=f"Possible table/column formatting detected ({len(table_issues)} instances)",
            suggestion="Remove tables, columns, and graphics. Use simple bullet points and standard text",
        )


def _check_keywords(text: str, job_text: str | None = None) -> ATSCheck:
    """Check for keyword presence (especially from job description)."""
    if not job_text:
        # Without a job description, just check for general keyword density
        words = text.lower().split()
        unique_words = set(words)
        density = len(unique_words) / max(len(words), 1)

        if density > 0.5:
            return ATSCheck(
                name="keyword_density",
                passed=True,
                score=0.8,
                weight=0.15,
                message="Good keyword diversity",
                suggestion=None,
            )
        else:
            return ATSCheck(
                name="keyword_density",
                passed=False,
                score=0.5,
                weight=0.15,
                message="Low keyword diversity",
                suggestion="Include more industry-specific keywords and technical terms",
            )

    # With job description, check for keyword overlap
    job_words = set(re.findall(r"\b\w{4,}\b", job_text.lower()))
    cv_words = set(re.findall(r"\b\w{4,}\b", text.lower()))
    overlap = job_words & cv_words
    coverage = len(overlap) / max(len(job_words), 1)

    if coverage >= 0.4:
        return ATSCheck(
            name="keyword_match",
            passed=True,
            score=min(coverage * 1.5, 1.0),
            weight=0.15,
            message=f"Good keyword match: {len(overlap)}/{len(job_words)} job keywords found",
            suggestion=None,
        )
    else:
        missing_keywords = list(job_words - cv_words)[:5]
        return ATSCheck(
            name="keyword_match",
            passed=False,
            score=coverage,
            weight=0.15,
            message=f"Low keyword match: only {len(overlap)}/{len(job_words)} job keywords found",
            suggestion=f"Add these keywords from the job description: {', '.join(missing_keywords)}",
        )


def _check_bullet_points(text: str) -> ATSCheck:
    """Check for bullet point usage (ATS-friendly)."""
    lines = text.split("\n")
    bullet_lines = sum(1 for line in lines if re.match(r"^\s*[-•*▪▸→]\s", line.strip()))

    if bullet_lines >= 3:
        return ATSCheck(
            name="bullet_points",
            passed=True,
            score=1.0,
            weight=0.10,
            message=f"Good use of bullet points ({bullet_lines} found)",
            suggestion=None,
        )
    else:
        return ATSCheck(
            name="bullet_points",
            passed=False,
            score=0.4,
            weight=0.10,
            message=f"Few bullet points found ({bullet_lines})",
            suggestion="Use bullet points to list achievements and responsibilities",
        )


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

def check_ats_friendliness(
    cv_text: str,
    job_text: str | None = None,
    sections_found: list[str] | None = None,
    has_skills_section: bool = False,
) -> ATSResult:
    """
    Analyze CV text for ATS friendliness.

    Args:
        cv_text: The raw CV text.
        job_text: Optional job description for keyword matching.
        sections_found: List of detected section names.
        has_skills_section: Whether a skills section was detected.

    Returns:
        ATSResult with score, checks, and recommendations.
    """
    word_count = len(cv_text.split())
    issues = []
    suggestions = []

    # Run all checks
    checks = [
        _check_sections(cv_text, sections_found or []),
        _check_contact_info(cv_text),
        _check_skills_section(cv_text, has_skills_section),
        _check_dates(cv_text),
        _check_length(cv_text, word_count),
        _check_format_hints(cv_text),
        _check_keywords(cv_text, job_text),
        _check_bullet_points(cv_text),
    ]

    # Calculate weighted score
    total_weight = sum(c.weight for c in checks)
    weighted_score = sum(c.score * c.weight for c in checks) / total_weight
    overall_score = int(weighted_score * 100)

    # Collect issues and suggestions
    for check in checks:
        if not check.passed:
            issues.append(check.message)
            if check.suggestion:
                suggestions.append(check.suggestion)

    # Determine pass estimate
    if overall_score >= 75:
        pass_estimate = "likely_pass"
    elif overall_score >= 50:
        pass_estimate = "borderline"
    else:
        pass_estimate = "likely_fail"

    return ATSResult(
        overall_score=overall_score,
        pass_estimate=pass_estimate,
        checks=checks,
        issues=issues,
        suggestions=suggestions,
    )
