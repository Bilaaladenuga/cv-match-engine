"""
Job Title Extractor — identifies job titles and roles from CV text.
"""

import re

# Common job title patterns
TITLE_KEYWORDS = [
    # Software / Engineering
    "software engineer", "software developer", "full stack developer",
    "full-stack developer", "frontend developer", "front-end developer",
    "backend developer", "back-end developer", "web developer",
    "senior engineer", "junior engineer", "staff engineer", "principal engineer",
    "devops engineer", "platform engineer", "data engineer",
    "machine learning engineer", "ml engineer", "ai engineer",
    "cloud engineer", "infrastructure engineer",
    # Data
    "data scientist", "data analyst", "data engineer",
    "business intelligence", "analytics engineer",
    # Management
    "engineering manager", "tech lead", "technical lead",
    "team lead", "vp of engineering", "cto", "chief technology officer",
    "director of engineering",
    # Design
    "ux designer", "ui designer", "product designer", "ux engineer",
    # Product
    "product manager", "technical product manager",
    # Security
    "security engineer", "security analyst", "cybersecurity analyst",
    # Other IT
    "systems administrator", "network engineer", "database administrator",
    "qa engineer", "quality assurance engineer", "test engineer",
    "scrum master", "agile coach",
]

# Seniority indicators
SENIORITY_PATTERNS = re.compile(
    r"\b(intern|junior|jr\.?|senior|sr\.?|staff|principal|lead|head|director|vp|chief)\b",
    re.IGNORECASE,
)


def extract_titles(text: str) -> list[str]:
    """
    Extract job titles mentioned in CV text.

    Returns deduplicated list of detected titles, sorted by specificity.
    """
    text_lower = text.lower()
    found: list[str] = []

    for title in TITLE_KEYWORDS:
        if title.lower() in text_lower:
            found.append(title)

    # Also look for "Senior Developer" type patterns
    for match in SENIORITY_PATTERNS.finditer(text):
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        context = text[start:end].strip()
        # Check if context contains a role word
        role_words = ["engineer", "developer", "designer", "manager", "analyst",
                       "scientist", "architect", "administrator", "lead"]
        for word in role_words:
            if word in context.lower():
                # Extract the full title phrase
                title_pattern = re.compile(
                    rf"\b\w*\s*{re.escape(word)}\b",
                    re.IGNORECASE,
                )
                title_match = title_pattern.search(context)
                if title_match:
                    candidate = title_match.group().strip()
                    if candidate.lower() not in [t.lower() for t in found]:
                        found.append(candidate)

    return sorted(set(found), key=lambda t: len(t), reverse=True)
