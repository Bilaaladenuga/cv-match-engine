"""
Certification Extractor — extracts professional certifications from CV text.
"""

import re
from dataclasses import dataclass


@dataclass
class Certification:
    """A professional certification."""
    name: str
    issuer: str | None  # e.g., "AWS", "Google", "Microsoft"
    year: str | None
    raw_text: str


# Known certification issuers and their patterns
KNOWN_ISSUERS: dict[str, list[str]] = {
    "AWS": ["aws", "amazon web services"],
    "Google": ["google", "gcp"],
    "Microsoft": ["microsoft", "azure"],
    "Cisco": ["cisco"],
    "Oracle": ["oracle"],
    "CompTIA": ["comptia"],
    "PMI": ["pmi", "pmp"],
    "Scrum.org": ["scrum", "psm", "pspo"],
    "Kubernetes": ["cncf", "ckad", "cka", "cks"],
    "HashiCorp": ["hashicorp", "terraform associate"],
}

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


def _detect_issuer(text: str) -> str | None:
    """Detect certification issuer from text."""
    text_lower = text.lower()
    for issuer, keywords in KNOWN_ISSUERS.items():
        for kw in keywords:
            if kw in text_lower:
                return issuer
    return None


def extract_certifications(text: str) -> list[Certification]:
    """
    Extract certification entries from text.

    Handles formats like:
    "AWS Certified Solutions Architect (2022)"
    "PMP - Project Management Professional, 2021"
    "Certified Kubernetes Administrator (CKA)"
    """
    lines = text.strip().split("\n")
    results: list[Certification] = []

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # Skip if line looks like a section heading
        if line.isupper() and len(line.split()) <= 3:
            continue

        year = None
        year_match = YEAR_PATTERN.search(line)
        if year_match:
            year = year_match.group()

        issuer = _detect_issuer(line)

        # Clean up the certification name
        name = line
        if year:
            name = name.replace(f"({year})", "").replace(year, "")
        name = re.sub(r"\s+", " ", name).strip(" -,·•")

        if name and len(name) > 2:
            results.append(Certification(
                name=name,
                issuer=issuer,
                year=year,
                raw_text=line,
            ))

    return results
