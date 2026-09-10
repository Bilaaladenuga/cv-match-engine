"""
Experience Matching Engine — Phase 9

Compares a candidate's work experience against a job's experience
requirements.

Unlike a naive "total years" comparison, this engine:
    1. Estimates per-skill experience from the CV's work history by
       scanning each job description/context for skill mentions.
    2. Compares required experience (years) against total AND per-skill
       candidate experience.
    3. Produces a gap analysis with a score and explanation.

Output classification:
    STRONG_MATCH   — candidate meets or exceeds requirement with margin
    MEETS          — candidate meets requirement exactly
    NEAR_MATCH     — candidate is within 1 year of requirement
    BELOW          — candidate is below requirement
    UNKNOWN        — insufficient data to judge
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.nlp.experience_extractor import WorkExperience
from app.nlp.skill_extractor import ExtractedSkill

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Margins for classification (in months)
_STRONG_MATCH_MARGIN = 12   # 1+ years above
_NEAR_MATCH_MARGIN = 12     # within 1 year below
# Everything below NEAR_MATCH_MARGIN = BELOW


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SkillExperience:
    """Estimated experience for one skill, derived from work history."""

    skill: str
    estimated_months: int
    confidence: float  # 0.0–1.0 (based on how many roles mention the skill)
    sources: list[str] = field(default_factory=list)  # Companies/roles


@dataclass
class ExperienceMatch:
    """Match result for one experience dimension."""

    dimension: str          # "total_years", "specific_skill", "level"
    required: str | float | None
    candidate: str | float | None
    gap_months: int | None  # negative = below, positive = above, 0 = exact
    score: float            # 0.0–1.0
    match_type: str         # "strong_match", "meets", "near_match", "below", "unknown"
    evidence: str


@dataclass
class ExperienceMatchResult:
    """Aggregate result of experience matching."""

    matches: list[ExperienceMatch]
    total_experience_years: float | None
    required_years: float | None
    per_skill_experience: list[SkillExperience]

    overall_score: float = 0.0
    experience_level: str = "unknown"  # "strong_match", "meets", "near_match", "below", "unknown"

    def to_dict(self) -> dict:
        """Serialize for API responses / database storage."""
        return {
            "matches": [
                {
                    "dimension": m.dimension,
                    "required": m.required,
                    "candidate": m.candidate,
                    "gap_months": m.gap_months,
                    "score": m.score,
                    "match_type": m.match_type,
                    "evidence": m.evidence,
                }
                for m in self.matches
            ],
            "summary": {
                "total_experience_years": self.total_experience_years,
                "required_years": self.required_years,
                "overall_score": round(self.overall_score, 4),
                "experience_level": self.experience_level,
                "per_skill_count": len(self.per_skill_experience),
            },
        }


# ---------------------------------------------------------------------------
# Per-skill experience estimation
# ---------------------------------------------------------------------------


def estimate_skill_experience(
    experiences: list[WorkExperience],
    skills: list[ExtractedSkill],
) -> list[SkillExperience]:
    """
    Estimate how many months of experience the candidate has per skill.

    Algorithm:
    - For each role, scan the role title and company for skill mentions.
    - Attribute the role's duration to any skill found in that context.
    - A skill mentioned in multiple roles accumulates total months.
    - For skills listed on the CV but not found in any role title,
      infer experience from total career duration (lower confidence).
    - Confidence is based on how many distinct roles mention the skill
      and whether the inference came from role titles or career duration.

    Args:
        experiences: List of WorkExperience from the CV.
        skills: List of ExtractedSkill from the CV.

    Returns:
        List of SkillExperience, one per skill found.
    """
    skill_map: dict[str, SkillExperience] = {}
    total_months = sum(
        e.duration_months for e in experiences
        if e.duration_months is not None and e.duration_months > 0
    )

    # --- Pass 1: attribute skills from role titles / company context ---
    for exp in experiences:
        if exp.duration_months is None or exp.duration_months <= 0:
            continue

        # Build a text blob from the role title and company
        context = " ".join(
            filter(None, [exp.role or "", exp.company or ""])
        ).lower()

        for skill in skills:
            if _skill_mentioned_in(skill.name, context):
                if skill.name not in skill_map:
                    skill_map[skill.name] = SkillExperience(
                        skill=skill.name,
                        estimated_months=0,
                        confidence=0.0,
                        sources=[],
                    )
                entry = skill_map[skill.name]
                entry.estimated_months += exp.duration_months
                source_label = (
                    f"{exp.role or 'Unknown'} at {exp.company or 'Unknown'}"
                )
                if source_label not in entry.sources:
                    entry.sources.append(source_label)

    # --- Pass 2: for CV-listed skills not found in any role title,
    #     infer from total career duration.  A skill on the CV means
    #     the candidate has used it, but without role context we can't
    #     pinpoint when — so we spread it across the full career span
    #     with lower confidence.
    # ---
    if total_months > 0:
        for skill in skills:
            if skill.name not in skill_map:
                skill_map[skill.name] = SkillExperience(
                    skill=skill.name,
                    estimated_months=total_months,
                    confidence=0.50,  # Inferred, not confirmed from roles
                    sources=["CV skills section (inferred)"],
                )
            elif skill_map[skill.name].estimated_months < total_months:
                # Skill was found in some roles but not all — bump up to
                # total career duration with adjusted confidence.
                entry = skill_map[skill.name]
                entry.estimated_months = total_months
                entry.confidence = min(entry.confidence + 0.10, 0.90)
                note = "CV skills section (inferred)"
                if note not in entry.sources:
                    entry.sources.append(note)

    # Compute confidence: based on number of roles mentioning the skill
    for entry in skill_map.values():
        # Only count actual role entries, not the inferred note
        actual_roles = [
            s for s in entry.sources
            if s != "CV skills section (inferred)"
        ]
        if len(actual_roles) >= 3:
            entry.confidence = 0.95
        elif len(actual_roles) == 2:
            entry.confidence = 0.85
        elif len(actual_roles) == 1:
            entry.confidence = 0.70
        # else: keep the inferred confidence (0.50)

    return sorted(skill_map.values(), key=lambda s: s.estimated_months, reverse=True)


def _skill_mentioned_in(skill_name: str, text: str) -> bool:
    """
    Check if a skill name is mentioned in a text context.

    Uses word-boundary-aware matching to avoid false positives.
    For multi-word skills, checks the full name; for short aliases,
    also checks common abbreviations.
    """
    skill_lower = skill_name.lower()
    # Direct substring match (word-boundary aware)
    pattern = rf"\b{re.escape(skill_lower)}\b"
    if re.search(pattern, text):
        return True

    # Handle compound skills: "Machine Learning" → "machine learning" or "ml"
    # Also check common abbreviations
    abbrev: dict[str, list[str]] = {
        "machine learning": ["ml"],
        "react": ["reactjs", "react.js"],
        "vue.js": ["vuejs", "vue"],
        "angular": ["angularjs"],
        "postgresql": ["postgres"],
        "javascript": ["js"],
        "typescript": ["ts"],
    }
    for canonical, abbrevs in abbrev.items():
        if skill_lower == canonical:
            for abbr in abbrevs:
                if re.search(rf"\b{re.escape(abbr)}\b", text):
                    return True

    return False


# ---------------------------------------------------------------------------
# Experience Matcher
# ---------------------------------------------------------------------------


class ExperienceMatcher:
    """
    Compares candidate experience against job requirements.

    Usage:
        matcher = ExperienceMatcher()
        result = matcher.match(
            experiences=candidate_experiences,
            skills=candidate_skills,
            required_years=3.0,
            seniority_level="senior",
        )
    """

    def match(
        self,
        experiences: list[WorkExperience],
        skills: list[ExtractedSkill],
        required_years: float | None = None,
        seniority_level: str | None = None,
        required_skills: list[str] | None = None,
    ) -> ExperienceMatchResult:
        """
        Match candidate experience against job requirements.

        Args:
            experiences: Candidate's work history from the CV.
            skills: Candidate's extracted skills.
            required_years: Minimum years required by the job (or None).
            seniority_level: Required seniority level (e.g., "senior").
            required_skills: Specific skills from the job to check
                             per-skill experience for.

        Returns:
            ExperienceMatchResult with matches and aggregates.
        """
        matches: list[ExperienceMatch] = []

        # --- Total experience ---
        total_months = sum(
            e.duration_months for e in experiences
            if e.duration_months is not None and e.duration_months > 0
        )
        total_years = round(total_months / 12, 1) if total_months > 0 else None

        if required_years is not None:
            total_match = self._match_total_years(total_years, required_years)
            matches.append(total_match)

        # --- Per-skill experience ---
        skill_experience = estimate_skill_experience(experiences, skills)

        if required_skills:
            for req_skill in required_skills:
                skill_match = self._match_skill_experience(
                    req_skill, skill_experience, required_years
                )
                matches.append(skill_match)

        # --- Seniority level ---
        if seniority_level:
            level_match = self._match_seniority(
                total_years, seniority_level, experiences
            )
            matches.append(level_match)

        # --- Aggregate ---
        return self._aggregate(
            matches=matches,
            total_years=total_years,
            required_years=required_years,
            skill_experience=skill_experience,
        )

    def _match_total_years(
        self, total_years: float | None, required_years: float
    ) -> ExperienceMatch:
        """Compare total experience against the requirement."""
        if total_years is None:
            return ExperienceMatch(
                dimension="total_years",
                required=required_years,
                candidate=None,
                gap_months=None,
                score=0.0,
                match_type="unknown",
                evidence="Could not determine total experience from CV.",
            )

        gap_months = int((total_years - required_years) * 12)
        score, match_type = self._score_gap(gap_months, required_years)

        if match_type == "strong_match":
            evidence = (
                f"Candidate has {total_years} years, exceeding the "
                f"{required_years}-year requirement by {abs(gap_months)//12} year(s)."
            )
        elif match_type == "meets":
            evidence = (
                f"Candidate has {total_years} years, meeting the "
                f"{required_years}-year requirement."
            )
        elif match_type == "near_match":
            deficit_months = abs(gap_months)
            evidence = (
                f"Candidate has {total_years} years, slightly below the "
                f"{required_years}-year requirement (deficit: {deficit_months} months)."
            )
        else:
            deficit_months = abs(gap_months)
            evidence = (
                f"Candidate has {total_years} years, below the "
                f"{required_years}-year requirement (deficit: {deficit_months} months)."
            )

        return ExperienceMatch(
            dimension="total_years",
            required=required_years,
            candidate=total_years,
            gap_months=gap_months,
            score=score,
            match_type=match_type,
            evidence=evidence,
        )

    def _match_skill_experience(
        self,
        required_skill: str,
        skill_experience: list[SkillExperience],
        required_years: float | None,
    ) -> ExperienceMatch:
        """Check if the candidate has enough experience for a specific skill."""
        # Find estimated experience for this skill
        se = next(
            (s for s in skill_experience if s.skill.lower() == required_skill.lower()),
            None,
        )

        if se is None or se.estimated_months == 0:
            return ExperienceMatch(
                dimension=f"skill:{required_skill}",
                required=required_years,
                candidate=0,
                gap_months=None,
                score=0.0,
                match_type="unknown",
                evidence=(
                    f"Could not estimate experience for '{required_skill}' "
                    f"from CV work history."
                ),
            )

        candidate_years = round(se.estimated_months / 12, 1)

        if required_years is not None:
            gap_months = se.estimated_months - int(required_years * 12)
            score, match_type = self._score_gap(gap_months, required_years)
            evidence = (
                f"Candidate has ~{candidate_years} years of '{required_skill}' "
                f"experience (from {len(se.sources)} role(s): "
                f"{', '.join(se.sources)})."
            )
        else:
            # No specific requirement — just report what we found
            score = min(1.0, candidate_years / 5.0)  # normalize to 5 years = 1.0
            match_type = "meets" if candidate_years >= 2 else "near_match"
            gap_months = None
            evidence = (
                f"Candidate has ~{candidate_years} years of '{required_skill}' "
                f"experience (from {len(se.sources)} role(s))."
            )

        return ExperienceMatch(
            dimension=f"skill:{required_skill}",
            required=required_years,
            candidate=candidate_years,
            gap_months=gap_months,
            score=score,
            match_type=match_type,
            evidence=evidence,
        )

    def _match_seniority(
        self,
        total_years: float | None,
        required_level: str,
        experiences: list[WorkExperience],
    ) -> ExperienceMatch:
        """Check if the candidate's experience level matches seniority."""
        level_thresholds: dict[str, float] = {
            "junior": 0.0,
            "mid": 2.0,
            "senior": 5.0,
            "lead": 8.0,
            "principal": 10.0,
            "staff": 10.0,
            "director": 12.0,
        }

        required_years = level_thresholds.get(required_level.lower(), 3.0)

        if total_years is None:
            return ExperienceMatch(
                dimension="seniority",
                required=required_level,
                candidate=None,
                gap_months=None,
                score=0.0,
                match_type="unknown",
                evidence="Could not determine seniority level from CV.",
            )

        gap_months = int((total_years - required_years) * 12)
        score, match_type = self._score_gap(gap_months, required_years)

        if match_type in ("strong_match", "meets"):
            evidence = (
                f"Candidate has {total_years} years of experience, "
                f"matching the {required_level} level requirement "
                f"(threshold: {required_years} years)."
            )
        elif match_type == "near_match":
            evidence = (
                f"Candidate has {total_years} years of experience, "
                f"close to the {required_level} level "
                f"(threshold: {required_years} years)."
            )
        else:
            evidence = (
                f"Candidate has {total_years} years of experience, "
                f"below the {required_level} level "
                f"(threshold: {required_years} years)."
            )

        return ExperienceMatch(
            dimension="seniority",
            required=required_level,
            candidate=total_years,
            gap_months=gap_months,
            score=score,
            match_type=match_type,
            evidence=evidence,
        )

    @staticmethod
    def _score_gap(gap_months: int, required_years: float) -> tuple[float, str]:
        """
        Score an experience gap.

        Args:
            gap_months: Positive = surplus, negative = deficit.
            required_years: The requirement in years.

        Returns:
            (score, match_type) tuple.
        """
        if gap_months >= _STRONG_MATCH_MARGIN:
            return 1.0, "strong_match"
        elif gap_months >= 0:
            return 0.9, "meets"
        elif abs(gap_months) <= _NEAR_MATCH_MARGIN:
            # Near match: score decreases linearly from 0.8 to 0.5
            ratio = 1.0 - (abs(gap_months) / _NEAR_MATCH_MARGIN)
            score = 0.5 + 0.3 * ratio
            return round(score, 4), "near_match"
        else:
            # Below: score decreases from 0.5 to 0.0
            max_gap = required_years * 12
            ratio = min(1.0, abs(gap_months) / max_gap) if max_gap > 0 else 1.0
            score = max(0.0, 0.5 * (1.0 - ratio))
            return round(score, 4), "below"

    @staticmethod
    def _aggregate(
        matches: list[ExperienceMatch],
        total_years: float | None,
        required_years: float | None,
        skill_experience: list[SkillExperience],
    ) -> ExperienceMatchResult:
        """Compute aggregate metrics."""
        if not matches:
            return ExperienceMatchResult(
                matches=[],
                total_experience_years=total_years,
                required_years=required_years,
                per_skill_experience=skill_experience,
            )

        # Overall score: average of all match scores
        overall_score = sum(m.score for m in matches) / len(matches)

        # Experience level: worst match determines the label
        type_order = ["strong_match", "meets", "near_match", "below", "unknown"]
        worst = "strong_match"
        for m in matches:
            if type_order.index(m.match_type) > type_order.index(worst):
                worst = m.match_type

        return ExperienceMatchResult(
            matches=matches,
            total_experience_years=total_years,
            required_years=required_years,
            per_skill_experience=skill_experience,
            overall_score=round(overall_score, 4),
            experience_level=worst,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def match_experience(
    experiences: list[WorkExperience],
    skills: list[ExtractedSkill],
    required_years: float | None = None,
    seniority_level: str | None = None,
    required_skills: list[str] | None = None,
) -> ExperienceMatchResult:
    """
    Quick entry point for experience matching.

    Args:
        experiences: Candidate's work history.
        skills: Candidate's extracted skills.
        required_years: Minimum years required by the job.
        seniority_level: Required seniority level.
        required_skills: Specific skills to check experience for.

    Returns:
        ExperienceMatchResult with matches and aggregates.
    """
    matcher = ExperienceMatcher()
    return matcher.match(
        experiences=experiences,
        skills=skills,
        required_years=required_years,
        seniority_level=seniority_level,
        required_skills=required_skills,
    )
