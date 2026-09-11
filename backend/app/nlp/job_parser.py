"""
Job Description Parser — extracts structured requirements from job posts.

Outputs a `JobProfile` with:
- Job title / company / location
- Required vs. preferred skills (canonical names from the skill taxonomy)
- Minimum experience in years (+ a textual seniority level)
- Education requirement (level + field)
- Certifications
- Responsibilities (bullet items under responsibility-style headings)

Deterministic, regex + heuristic based — mirroring the CV extractors so the
pipeline stays explainable and independently testable. Skills are classified
as *preferred* only when an explicit signal ("preferred", "nice to have",
"bonus", ...) is nearby; everything else that lists skills is treated as
required, matching how job posts actually read.

Example:
    "Looking for a frontend developer with 2+ years experience in React,
     TypeScript and Next.js. AWS experience is preferred."
    ->
    required_skills: React, TypeScript, Next.js
    preferred_skills: AWS
    minimum_experience_years: 2
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.nlp.certification_extractor import extract_certifications
from app.nlp.skill_extractor import ExtractedSkill, extract_skills, extract_skills_from_list

# ---------------------------------------------------------------------------
# Heading classification
# ---------------------------------------------------------------------------

_PREFERRED_KW = (
    "preferred", "nice to have", "good to have", "a plus", "bonus", "beneficial",
    "desirable", "would love", "what will help",
)
_RESPONSIBILITIES_KW = (
    "responsibilities", "duties", "what you'll do", "what you will do",
    "day-to-day", "day to day", "accountabilities", "the role involves",
    "your role will include", "you will:",
)
_EDUCATION_KW = ("education", "educational", "degree requirements")
_REQUIRED_KW = (
    "required", "requirements", "must have", "must-haves", "essential",
    "qualifications", "you should have", "what you'll need", "what you will need",
    "what we need", "key skills", "skills & experience", "skills and experience",
    "technical skills", "about you", "minimum qualifications",
)

_HEADING_STRIP = re.compile(r"^\s*(?:[\d\-•*]+[.)]?\s*)+")


def _classify_heading(heading: str) -> str | None:
    """Map a section heading to a mode: required|preferred|responsibilities|education."""
    h = _HEADING_STRIP.sub("", heading).strip(" :\t-").lower()
    if not h:
        return None
    if any(kw in h for kw in _PREFERRED_KW):
        return "preferred"
    if any(kw in h for kw in _RESPONSIBILITIES_KW):
        return "responsibilities"
    if any(kw in h for kw in _EDUCATION_KW):
        return "education"
    if any(kw in h for kw in _REQUIRED_KW):
        return "required"
    return None


def _is_heading_line(line: str) -> bool:
    """Heuristic: short ALL CAPS / Title Case line or a line ending with ':'."""
    stripped = line.strip()
    if not stripped or len(stripped) > 70:
        return False
    if stripped.endswith(":") and len(stripped) <= 60:
        return True
    if stripped.isupper() and len(stripped.split()) <= 6:
        return True
    return stripped.istitle() and len(stripped.split()) <= 6 and _classify_heading(stripped) is not None


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?:\r?\n)+")

_PREFERRED_SIGNAL = re.compile(
    r"\b(preferred|nice[- ]to[- ]have|good[- ]to[- ]have|a plus|is a plus|"
    r"would be (a )?(great|nice)|beneficial|desirable|bonus|plus:)\b",
    re.IGNORECASE,
)
_REQUIRED_SIGNAL = re.compile(
    r"\b(required|must[- ]have|must have|essential|require|requires|requiring|"
    r"you (will|'ll) need|we (need|require)|minimum)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Labeled lines (Company: / Location: / Job Title: ...)
# ---------------------------------------------------------------------------

_LABELED = re.compile(
    r"^\s*(?P<label>job\s*title|title|role|position|company|location|work\s*type)\s*[:|-]\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)

_REMOTE_PATTERN = re.compile(
    r"\b(remote|fully remote|remote-first|100%? remote|work from home|wfh)\b",
    re.IGNORECASE,
)

_CERT_MARKER = re.compile(
    r"\b(certified|certification|certificate|architect|associate|professional|practitioner)\b",
    re.IGNORECASE,
)
_CERT_STRONG_MARKER = re.compile(
    r"\b(certified|certification|certificate|pmp|cka|ckad|cks|psm|pspo)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Job title
# ---------------------------------------------------------------------------

_ROLE_WORDS = {
    "developer", "engineer", "scientist", "analyst", "architect", "designer",
    "manager", "consultant", "specialist", "administrator", "researcher",
    "intern", "programmer", "associate", "director",
}
_TITLE_STOPWORDS = {
    "with", "in", "at", "for", "and", "or", "to", "of", "on", "who", "join",
    "experience", "years", "year", "remote", "the", "a", "an", "our", "is",
    "be", "will", "us", "we", "you", "are", "this", "role", "position",
}
_SENIORITY = r"(?:entry[- ]level|junior|mid[- ]level|senior|lead|principal|staff|experienced)\s+"
_TITLE_PHRASE = re.compile(
    r"\b(?:we(?:'| a)?re looking for|we are looking for|looking for|hiring|"
    r"seeking|recruiting|need(?:ing)? an?|opening for|position for)\s+"
    r"(?:an?|the|someone|a highly)?\s*"
    rf"(?P<sen>{_SENIORITY})?"
    r"(?P<title>[A-Za-z][A-Za-z+.#&/-]*(?:\s+[A-Za-z+.#&/-]+){0,4})",
    re.IGNORECASE,
)


def _clean_title(candidate: str) -> str | None:
    """
    Trim stopwords and truncate the candidate at its last role word.

    "Principal Engineer at Rocket" becomes "Principal Engineer" — a title
    ends at the role (developer/engineer/analyst/...), not at the company or
    trailing prose.
    """
    tokens = candidate.split()
    meaningful = [
        t for t in tokens if t.lower().rstrip(".,;:!?") not in _TITLE_STOPWORDS
    ]
    if not meaningful:
        return None

    role_idx = None
    for i, token in enumerate(meaningful):
        if token.lower().rstrip(".,;:!?") in _ROLE_WORDS:
            role_idx = i
    if role_idx is None or role_idx > 4:
        return None

    title = " ".join(meaningful[: role_idx + 1]).strip(" .,;:!?")
    words = [w[:1].upper() + w[1:] if w[:1].isalpha() else w for w in title.split()]
    return " ".join(words)


def extract_job_title(text: str) -> str | None:
    """Best-effort job title extraction: labeled line, standalone title, phrase."""
    lines = text.splitlines()
    for line in lines:
        m = _LABELED.match(line.strip())
        if m and m.group("label").lower() in {"job title", "title", "role", "position"}:
            title = _clean_title(m.group("value").strip())
            if title:
                return title

    # Many JDs start with the title on its own line (no label). Only trust
    # short Title Case lines early in the document before falling back to
    # phrases later in the text.
    for line in lines[:8]:
        stripped = line.strip()
        if not stripped or _is_heading_line(stripped) or _LABELED.match(stripped):
            continue
        if stripped[-1:] in ".:;,!?-":
            continue
        if stripped.istitle() and len(stripped.split()) <= 6:
            title = _clean_title(stripped)
            if title:
                return title

    for m in _TITLE_PHRASE.finditer(text):
        candidate = (m.group("sen") or "") + m.group("title")
        title = _clean_title(candidate)
        if title:
            return title
    return None


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------

# Matches: "3+ years of experience", "at least 5 years experience",
# "minimum of 2 years", "5-7 years of professional experience"
_EXPERIENCE_PATTERN = re.compile(
    r"\b(?:minimum(?: of)?|at least|min\.?)\s+"
    r"(?P<n1>\d+(?:\.\d+)?)\s*[-–]?\s*(?P<n2>\d+(?:\.\d+)?)?\s*\+?\s*years?"
    r"|"
    r"\b(?P<m1>\d+(?:\.\d+)?)\s*[-–]\s*(?P<m2>\d+(?:\.\d+)?)\s*\+?\s*years?"
    r"|"
    r"\b(?P<g1>\d+(?:\.\d+)?)\s*\+?\s*years?\+?\s*(?:of\s+)?(?:relevant\s+|professional\s+)?"
    r"(?:work\s+|hands-on\s+)?experience\b",
    re.IGNORECASE,
)

_LEVEL_PATTERN = re.compile(
    r"\b(entry[- ]level|junior|mid[- ]level|senior|lead|principal|staff)\b",
    re.IGNORECASE,
)


def _is_preferred_sentence(sentence: str) -> bool:
    return bool(_PREFERRED_SIGNAL.search(sentence))


def extract_experience_requirement(text: str) -> tuple[float | None, str | None]:
    """
    Extract (minimum_years, seniority_level) from a job description.

    Numeric requirements in preferred-signal sentences ("5+ years preferred")
    are ignored — we want the *required* minimum, not an aspirational one.
    Explicit minimum/at-least/range phrasing wins over bare "N+ years".
    """
    numbers: list[float] = []
    explicit: list[float] = []

    for sentence in _SENTENCE_SPLIT.split(text):
        if _is_preferred_sentence(sentence):
            continue
        for m in _EXPERIENCE_PATTERN.finditer(sentence):
            if m.group("n1"):
                explicit.append(float(m.group("n1")))
            elif m.group("m1"):
                explicit.append(float(m.group("m1")))  # use the range floor
            elif m.group("g1"):
                numbers.append(float(m.group("g1")))

    minimum = max(explicit) if explicit else (max(numbers) if numbers else None)

    level_match = _LEVEL_PATTERN.search(text)
    level = level_match.group(1).lower() if level_match else None
    return minimum, level


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------

_EDUCATION_PATTERN = re.compile(
    r"\b(?P<level>ph\.?d\.?|doctorate|masters?|m\.s\.?|m\.a\.?|m\.b\.?a\.?|b\.s\.?|"
    r"b\.a\.?|bachelor'?s?|associate'?s?|high school)\b"
    r"(?:\s*(?:degree|of science|of arts|in science|in arts))?"
    r"(?:\s+in\s+(?P<field>[A-Za-z][A-Za-z &+-]{1,40}))?",
    re.IGNORECASE,
)

_EDUCATION_LEVEL_RANK = {
    "high school": 1, "associate": 2, "bachelor": 3, "b.s.": 3, "b.a.": 3,
    "master": 4, "m.s.": 4, "m.a.": 4, "m.b.a.": 4, "mba": 4,
    "doctorate": 5, "ph.d.": 5, "phd": 5,
}


def _normalize_education_level(raw: str) -> str:
    r = re.sub(r"\.", "", raw.strip().lower())
    if r in ("phd", "doctorate"):
        return "Doctorate"
    if r.startswith("master") or r in ("ms", "ma", "mba"):
        return "Master's"
    if r.startswith("bachelor") or r.startswith("b") and r in ("bs", "ba"):
        return "Bachelor's"
    if r == "associate" or r.startswith("associate"):
        return "Associate's"
    return raw.strip()


def extract_education_requirement(text: str) -> dict | None:
    """
    Extract an education requirement: {'level', 'field', 'required'} or None.

    Multiple mentions ("BS required, MS preferred") collapse to the highest
    level; `required` reflects whether it sits in a preferred-signal sentence.
    """
    best: tuple[int, dict] | None = None
    for sentence in _SENTENCE_SPLIT.split(text):
        m = _EDUCATION_PATTERN.search(sentence)
        if not m:
            continue
        raw_level = m.group("level").lower()
        rank = _EDUCATION_LEVEL_RANK.get(re.sub(r"\.", "", raw_level), 3)
        required = not _is_preferred_sentence(sentence)
        field = m.group("field")
        if field:
            field = _trim_education_field(field)
        entry = {
            "level": _normalize_education_level(raw_level),
            "field": field.strip() if field else None,
            "required": required,
        }
        if entry["field"] == "":
            entry["field"] = None
        if best is None or rank >= best[0]:
            best = (rank, entry)
    return best[1] if best else None


_FIELD_TRAILER_WORDS = {
    "required", "require", "preferred", "or", "and", "related", "field",
    "equivalent", "similar", "degree", "a", "an", "in", "of", "experience",
    "minimum", "must", "have", "is", "are",
}


def _trim_education_field(field: str) -> str:
    """
    Drop trailing qualifiers from a captured field.

    "Computer Science or related field required" -> "Computer Science".
    The regex field is char-capped, so trailing words can arrive truncated
    ("require") — trimming word by word handles that.
    """
    cleaned = field.strip().rstrip(".,;:")
    changed = True
    while changed and cleaned:
        changed = False
        parts = cleaned.rsplit(maxsplit=1)
        if parts[-1].lower() in _FIELD_TRAILER_WORDS:
            # len(parts) == 1 means the whole string is a trailer word;
            # emptying it ends the loop (rsplit would otherwise return the
            # same single word forever).
            cleaned = parts[0] if len(parts) > 1 else ""
            changed = True
    # Doubled preposition artifact ("degree in in X"): the regex consumes the
    # first "in" and the field capture starts at the second one.
    if cleaned[:3].lower() == "in " and len(cleaned) > 3:
        cleaned = cleaned[3:].strip()
    return cleaned


# ---------------------------------------------------------------------------
# Location / company / remote
# ---------------------------------------------------------------------------

def extract_company_location(text: str) -> tuple[str | None, str | None, bool]:
    """Extract (company, location, is_remote) from labeled lines and keywords."""
    company = None
    location = None
    for line in text.splitlines():
        m = _LABELED.match(line.strip())
        if not m:
            continue
        label, value = m.group("label").lower(), m.group("value").strip()
        if label == "company" and company is None:
            company = value
        elif label == "location" and location is None:
            location = value
    is_remote = bool(_REMOTE_PATTERN.search(text))
    return company, location, is_remote


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class JobProfile:
    """Structured requirements extracted from a job description."""

    job_title: str | None = None
    company: str | None = None
    location: str | None = None
    is_remote: bool = False
    required_skills: list[ExtractedSkill] = field(default_factory=list)
    preferred_skills: list[ExtractedSkill] = field(default_factory=list)
    minimum_experience_years: float | None = None
    experience_level: str | None = None
    education: dict | None = None
    certifications: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_title": self.job_title,
            "company": self.company,
            "location": self.location,
            "is_remote": self.is_remote,
            "required_skills": [
                {"name": s.name, "category": s.category, "confidence": s.confidence}
                for s in self.required_skills
            ],
            "preferred_skills": [
                {"name": s.name, "category": s.category, "confidence": s.confidence}
                for s in self.preferred_skills
            ],
            "minimum_experience_years": self.minimum_experience_years,
            "experience_level": self.experience_level,
            "education": self.education,
            "certifications": self.certifications,
            "responsibilities": self.responsibilities,
        }


_BULLET_LEAD = re.compile(r"^[\s•\-*]+")


def _strip_bullet(line: str) -> str:
    """Remove leading bullet/list markers and whitespace from a line."""
    return _BULLET_LEAD.sub("", line).strip()


def _merge_skills(accumulator: dict[str, ExtractedSkill], skills: list[ExtractedSkill]) -> None:
    for skill in skills:
        accumulator.setdefault(skill.name, skill)


def _extract_block_skills(block_text: str) -> list[ExtractedSkill]:
    """Skills from one content block (list formats and prose both handled)."""
    merged: dict[str, ExtractedSkill] = {}
    for line in block_text.splitlines():
        stripped = _strip_bullet(line)
        if not stripped:
            continue
        # Whole-line scan catches hyphenated aliases (scikit-learn) that a
        # delimiter split would break; the list pass handles "Python, React".
        for skill in extract_skills(stripped):
            merged.setdefault(skill.name, skill)
        for skill in extract_skills_from_list(stripped):
            merged.setdefault(skill.name, skill)
    return list(merged.values())


def parse_job_description(text: str) -> JobProfile:
    """
    Parse a job description into a structured JobProfile.

    Paragraphs under Preferred / Required / Responsibilities / Education
    headings are classified accordingly; unheaded text is scanned sentence by
    sentence, defaulting to required unless a preferred signal is present.
    """
    if not text or not text.strip():
        return JobProfile()

    # --- Pass 1: labeled lines -------------------------------------------------
    company, location, is_remote = extract_company_location(text)
    job_title = extract_job_title(text)

    # --- Pass 2: group lines into heading-classified blocks --------------------
    blocks: list[tuple[str | None, list[str]]] = []
    current_mode: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        mode = _classify_heading(line) if _is_heading_line(line) else None
        if mode is not None:
            if current_lines:
                blocks.append((current_mode, current_lines))
            current_mode = mode
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_mode, current_lines))

    required: dict[str, ExtractedSkill] = {}
    preferred: dict[str, ExtractedSkill] = {}
    responsibilities: list[str] = []

    for mode, lines in blocks:
        block_text = "\n".join(lines)
        if mode == "responsibilities":
            for line in lines:
                cleaned = _strip_bullet(line)
                if cleaned and cleaned not in responsibilities:
                    responsibilities.append(cleaned)
            continue
        if mode == "education":
            continue  # education handled globally below
        if mode == "preferred":
            _merge_skills(preferred, _extract_block_skills(block_text))
            continue
        if mode == "required":
            # Preferred signals can still appear inside a requirements block
            # ("Experience with X is a plus") and demote that line's skills
            for line in lines:
                if _PREFERRED_SIGNAL.search(line):
                    _merge_skills(preferred, _extract_block_skills(line))
                else:
                    _merge_skills(required, _extract_block_skills(line))
            continue

        # Unheaded block: sentence-level required/preferred classification
        for sentence in _SENTENCE_SPLIT.split(block_text):
            skills = _extract_block_skills(sentence)
            if not skills:
                continue
            if _is_preferred_sentence(sentence):
                _merge_skills(preferred, skills)
            else:
                _merge_skills(required, skills)

    # Skills explicitly listed as required win over preferred mentions
    for name in required:
        preferred.pop(name, None)

    minimum_experience_years, experience_level = extract_experience_requirement(text)
    education = extract_education_requirement(text)
    # The cert scanner (built for CV certification *sections*) tags any line;
    # on full JD text only keep cert-shaped entries and strip any leading
    # "Certifications:" label.
    certifications = []
    for cert in extract_certifications(text):
        name = re.sub(r"^\s*(?:certifications?|certificates?)\s*[:.-]?\s*", "", cert.name, flags=re.IGNORECASE)
        name = name.strip().rstrip(".").strip()
        if not name:
            continue
        name = re.sub(r"\s+(?:preferred|required|is a plus|a plus)$", "", name, flags=re.IGNORECASE)
        name = name.strip()
        has_marker = _CERT_MARKER.search(name)
        has_strong_marker = _CERT_STRONG_MARKER.search(name)
        keep = has_strong_marker is not None or (cert.issuer is not None and has_marker is not None)
        if keep and name not in certifications:
            certifications.append(name)

    return JobProfile(
        job_title=job_title,
        company=company,
        location=location or ("Remote" if is_remote else None),
        is_remote=is_remote,
        required_skills=list(required.values()),
        preferred_skills=list(preferred.values()),
        minimum_experience_years=minimum_experience_years,
        experience_level=experience_level,
        education=education,
        certifications=certifications,
        responsibilities=responsibilities[:30],
    )
