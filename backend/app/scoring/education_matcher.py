"""
Education Matching — Phase 11

Deterministic comparison of the candidate's education against the job's
education requirements.

Design notes
------------
    - Degree levels form a strict hierarchy:
          high school < associate < bachelor < master < doctorate
      A candidate meeting or EXCEEDING the required level satisfies the
      requirement (a Master's holder satisfies a Bachelor's requirement).
    - Field comparison is relevance-based, not string equality:
          exact same field ..................... 1.0
          known related fields (CS ↔ SWE) ...... 0.8
          same broad stem/tech family .......... 0.6
          unrelated field ...................... 0.3 (still an earned degree)
    - Scores blend level and field:  level 0.6 / field 0.4.
      Level dominates because job posts filter on it first.
    - No requirement  →  neutral 0.85 (not 1.0: absence of a requirement is
      not evidence FOR education, it just removes the penalty).
    - No candidate education but a requirement exists  →  0.0 with UNKNOWN
      flag (the parser may simply have failed to see it — the report will
      phrase it that way rather than accusing the candidate).

Protected characteristics such as university prestige are deliberately NOT
used (see Phase 27, ethics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Degree level hierarchy
# ---------------------------------------------------------------------------

_LEVEL_RANK: dict[str, int] = {
    "high school": 1,
    "diploma": 1,
    "associate": 2,
    "associate's": 2,
    "bachelor": 3,
    "bachelor's": 3,
    "master": 4,
    "master's": 4,
    "doctorate": 5,
    "phd": 5,
}

# Canonical display names
_CANONICAL_LEVEL = {
    1: "High School",
    2: "Associate's",
    3: "Bachelor's",
    4: "Master's",
    5: "Doctorate",
}

# Maps degree keywords (as extracted from CVs) to a rank.
# Mirrors DEGREE_LEVELS in education_extractor but keyed for quick lookup.
# Keywords resolved AFTER punctuation normalisation ("m.s." -> "m s").
_KEYWORDS: dict[int, list[str]] = {
    1: ["high school", "diploma", "ged"],
    2: ["associate", "a s", "a a"],
    3: ["bachelor", "b s", "b a", "b sc", "b eng", "bs", "ba", "btech"],
    4: ["master", "m s", "m a", "m sc", "m eng", "mba", "m b a", "ms", "ma"],
    5: ["phd", "ph d", "doctorate", "doctoral", "dphil"],
}

_LEVEL_RANK_NORMALIZED: dict[str, int] = {}
for _rank, _keys in _KEYWORDS.items():
    for _k in _keys:
        _LEVEL_RANK_NORMALIZED[_k] = _rank

# Full degree-name prefixes, longest first ("bachelor of science" etc.)
_NAME_PREFIX_RANK: list[tuple[str, int]] = [
    ("doctor of philosophy", 5),
    ("doctorate", 5),
    ("master of", 4),
    ("bachelor of", 3),
    ("associate of", 2),
]

# ---------------------------------------------------------------------------
# Field relevance
# ---------------------------------------------------------------------------

# Exact-canonical groups: same field under different spellings
_FIELD_ALIASES: dict[str, str] = {
    "computer science": "computer science",
    "cs": "computer science",
    "computing": "computer science",
    "software engineering": "software engineering",
    "software development": "software engineering",
    "computer engineering": "computer engineering",
    "computer science and engineering": "computer engineering",
    "information technology": "information technology",
    "it": "information technology",
    "information systems": "information systems",
    "data science": "data science",
    "mathematics": "mathematics",
    "math": "mathematics",
    "applied mathematics": "mathematics",
    "statistics": "statistics",
    "stats": "statistics",
    "electrical engineering": "electrical engineering",
    "ee": "electrical engineering",
    "physics": "physics",
}

# Groups of fields considered "related" (0.8)
_RELATED_GROUPS: list[frozenset[str]] = [
    frozenset({"computer science", "software engineering", "computer engineering"}),
    frozenset({"information technology", "information systems"}),
    frozenset({"data science", "statistics", "mathematics"}),
    frozenset({"electrical engineering", "physics"}),
]

# Broad technical family (0.6): any two fields within this set
_TECH_FAMILY: frozenset[str] = frozenset(
    {
        "computer science", "software engineering", "computer engineering",
        "information technology", "information systems", "data science",
        "mathematics", "statistics", "electrical engineering", "physics",
    }
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EducationMatch:
    """Result of comparing candidate education against one requirement."""

    required_level: str | None
    candidate_level: str | None
    level_score: float
    required_field: str | None
    candidate_field: str | None
    field_score: float
    score: float                 # blended 0.0–1.0
    meets_requirement: bool
    evidence: str

    def to_dict(self) -> dict:
        return {
            "required_level": self.required_level,
            "candidate_level": self.candidate_level,
            "level_score": round(self.level_score, 4),
            "required_field": self.required_field,
            "candidate_field": self.candidate_field,
            "field_score": round(self.field_score, 4),
            "score": round(self.score, 4),
            "meets_requirement": self.meets_requirement,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _level_rank(level: str | None) -> int | None:
    """Resolve any degree-string form to its hierarchy rank."""
    if not level:
        return None
    # Normalise: lowercase, collapse dots/spacing ("M.S." -> "m s", "Ph.D" -> "ph d")
    key = re.sub(r"[^a-z0-9]+", " ", level.strip().lower()).strip()
    if not key:
        return None
    if key in _LEVEL_RANK_NORMALIZED:
        return _LEVEL_RANK_NORMALIZED[key]
    # Possessive forms: "Bachelor's" -> "bachelor s" -> "bachelor"
    if key.endswith(" s") and key[:-2] in _LEVEL_RANK_NORMALIZED:
        return _LEVEL_RANK_NORMALIZED[key[:-2]]
    # Full degree names: "Bachelor of Science", "Master of Engineering" ...
    for prefix, rank in _NAME_PREFIX_RANK:
        if key.startswith(prefix):
            return rank
    # Token-subsequence fallback: "b a mathematics" contains keyword "b a",
    # "master of business administration x" contains "mba"-style tokens.
    tokens = key.split()
    best: int | None = None
    for kw, rank in _LEVEL_RANK_NORMALIZED.items():
        kw_tokens = kw.split()
        n = len(kw_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == kw_tokens:
                if best is None or rank > best:
                    best = rank
                break
    return best


def _canonical_level(level: str | None) -> str | None:
    rank = _level_rank(level)
    return _CANONICAL_LEVEL.get(rank) if rank else None


def _canonical_field(field: str | None) -> str | None:
    if not field:
        return None
    return _FIELD_ALIASES.get(field.strip().lower(), field.strip().lower())


def field_relevance(candidate_field: str | None, required_field: str | None) -> float:
    """Score how relevant the candidate's field of study is to the requirement."""
    cand = _canonical_field(candidate_field)
    req = _canonical_field(required_field)

    if not req:
        return 1.0          # nothing specific demanded
    if not cand:
        return 0.5          # requirement exists, candidate field unknown
    if cand == req:
        return 1.0
    for group in _RELATED_GROUPS:
        if cand in group and req in group:
            return 0.8
    if cand in _TECH_FAMILY and req in _TECH_FAMILY:
        return 0.6
    return 0.3


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def match_education(
    candidate_education: list,          # list[EducationEntry]
    job_education: dict | None,         # {"level", "field", "required"} or None
) -> EducationMatch:
    """
    Compare the candidate's best education entry against the job requirement.

    A single EducationMatch is returned even when nothing can be compared —
    callers can rely on `.score` and `.evidence` unconditionally.
    """
    best_entry = None
    best_rank = 0
    for entry in candidate_education or []:
        rank = _level_rank(getattr(entry, "degree", None)) or 0
        if rank > best_rank:
            best_rank = rank
            best_entry = entry

    cand_rank = best_rank or None
    cand_level = _canonical_level(getattr(best_entry, "degree", None))
    cand_field = getattr(best_entry, "field_of_study", None)

    if not job_education:
        score = 0.85
        return EducationMatch(
            required_level=None,
            candidate_level=cand_level,
            level_score=score,
            required_field=None,
            candidate_field=cand_field,
            field_score=score,
            score=score,
            meets_requirement=True,
            evidence=(
                "No education requirement specified; neutral score applied."
            ),
        )

    req_level = job_education.get("level")
    req_field = job_education.get("field")
    req_rank = _level_rank(req_level)

    if req_rank is None and not req_field:
        score = 0.85
        return EducationMatch(
            required_level=req_level,
            candidate_level=cand_level,
            level_score=score,
            required_field=None,
            candidate_field=cand_field,
            field_score=score,
            score=score,
            meets_requirement=True,
            evidence="Education requirement could not be parsed; neutral score applied.",
        )

    # Level component
    if req_rank is None:
        level_score = 1.0
        level_evidence = "No specific degree level required."
    elif cand_rank is None:
        level_score = 0.0
        level_evidence = (
            f"Job requires {_CANONICAL_LEVEL.get(req_rank, req_level)} degree, "
            "but no degree was detected on the CV (it may be unlisted or "
            "unparsed rather than absent)."
        )
    elif cand_rank >= req_rank:
        level_score = 1.0
        level_evidence = (
            f"Candidate holds {_canonical_level(cand_level) or cand_level}, "
            f"meeting the {_CANONICAL_LEVEL.get(req_rank, req_level)} requirement."
        )
    else:
        # One level below is a partial (many roles accept "in progress or
        # equivalent experience"); two or more below scores near zero.
        gap = req_rank - cand_rank
        level_score = {1: 0.6}.get(gap, max(0.0, 0.3 - 0.15 * (gap - 2)))
        level_evidence = (
            f"Candidate holds {_canonical_level(cand_level) or cand_level}, "
            f"below the required {_CANONICAL_LEVEL.get(req_rank, req_level)} degree."
        )

    # Field component
    field_score = field_relevance(cand_field, req_field)
    if req_field and cand_field:
        field_evidence = (
            f"Field of study: {cand_field} vs required {req_field} "
            f"(relevance {field_score:.0%})."
        )
    elif req_field:
        field_evidence = f"Job prefers a {req_field} background; candidate field undetected."
    else:
        field_evidence = "No specific field of study required."

    blended = 0.6 * level_score + 0.4 * field_score
    meets = level_score >= 1.0 and (not req_field or field_score >= 0.6)

    evidence = "; ".join([level_evidence, field_evidence])

    return EducationMatch(
        required_level=_CANONICAL_LEVEL.get(req_rank) if req_rank else req_level,
        candidate_level=cand_level,
        level_score=level_score,
        required_field=req_field,
        candidate_field=cand_field,
        field_score=field_score,
        score=blended,
        meets_requirement=meets,
        evidence=evidence,
    )
