"""
CV Improvement Engine — Phase 16.

Turns the matching engines' raw outputs into *evidence-grounded* advice.
Every statement is derived from what the pipeline actually extracted from
the CV and the job description — the engine never invents experience.

Evidence model
--------------
For each job-required skill we assemble three independent signals:

1. **Skills-section presence**  — the candidate lists the skill in their
   skills block. Weak on its own: lists are easy to pad.
2. **Work-history mention**     — the skill (or a taxonomy alias) appears
   in the experience/projects sections. Behavioural evidence.
3. **Per-skill duration**       — the Phase 9 experience matcher estimated
   months of use from dated employment entries. Strongest signal.

These combine into a four-level strength grade:

    strong    ≥ 6 estimated months of use, or work-history mention
              *and* a skills-section listing
    moderate  some work-history evidence, but short or low-confidence
    weak      listed in the skills section only
    absent    no trace on the CV

Grades drive the report:

* matched + strong   → a positive "well supported" statement
* matched/partial + weak → the Phase 16 flagship warning: "X appears in
  your skills section but there is limited evidence of its use in your
  work experience."
* partial + moderate → "strengthen with quantified outcomes"
* missing            → a concrete build-the-skill action item

The graded block is also exposed verbatim as ``skill_evidence`` so the
frontend can render a per-skill evidence panel instead of re-inferring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.nlp.taxonomy import TAXONOMY

# ---------------------------------------------------------------------------
# Evidence grades
# ---------------------------------------------------------------------------

STRONG = "strong"
MODERATE = "moderate"
WEAK = "weak"
ABSENT = "absent"

# Months of estimated use above which evidence is considered strong.
STRONG_MONTHS_THRESHOLD = 6

MAX_RECOMMENDATIONS = 8

# Source note the Phase 9 estimator attaches when it infers months from the
# total career span instead of a named role (low-confidence fallback).
_INFERRED_SOURCE_NOTE = "inferred"

_STATUSES = ("matched", "partial", "missing")


@dataclass
class SkillEvidence:
    """Evidence-backed assessment of one job-required skill."""

    skill: str                     # As phrased in the job description
    status: str                    # "matched" | "partial" | "missing"
    candidate_skill: str | None    # Canonical CV skill it mapped to
    strength: str                  # STRONG / MODERATE / WEAK / ABSENT
    in_skills_section: bool        # Listed in the CV's skills block
    in_work_history: bool          # Mentioned in experience/projects text
    estimated_months: int | None   # Per-skill duration estimate (Phase 9)
    sources: list[str] = field(default_factory=list)  # Roles/companies

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "status": self.status,
            "candidate_skill": self.candidate_skill,
            "strength": self.strength,
            "in_skills_section": self.in_skills_section,
            "in_work_history": self.in_work_history,
            "estimated_months": self.estimated_months,
            "sources": list(self.sources),
        }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _alias_pattern(alias: str) -> str:
    """Word-boundary regex for one alias (mirrors the skill extractor)."""
    if len(alias) == 1:
        return rf"(?<!\S){re.escape(alias)}(?!\S)"
    return rf"(?<![\w.]){re.escape(alias)}(?!\w)"


def _mentions(text: str | None, canonical_skill: str) -> bool:
    """True when the skill or any of its taxonomy aliases appears in text."""
    if not text:
        return False
    aliases = [canonical_skill]
    skill_def = TAXONOMY.get_skill(canonical_skill)
    if skill_def is not None:
        aliases.extend(skill_def.aliases)
    lowered = text.lower()
    return any(re.search(_alias_pattern(a), lowered) for a in aliases)


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------


def _strength(evidence: SkillEvidence) -> str:
    """
    Combine the three signals into a strength grade.

    Per-skill month estimates only confirm *strong* evidence when the skill
    is actually mentioned in the work history. Without a direct mention the
    months come from the coarse whole-span fallback (Phase 9), so they
    corroborate to moderate only — a skill-listing padding trick ("Docker,
    Kubernetes, Terraform" under a 4-year admin role) must not manufacture
    strong evidence for every listed tool.
    """
    if evidence.candidate_skill is None:
        return ABSENT
    months = evidence.estimated_months or 0
    if evidence.in_work_history and months >= STRONG_MONTHS_THRESHOLD:
        return STRONG
    if evidence.in_work_history and evidence.in_skills_section:
        return STRONG
    if evidence.in_work_history or months > 0:
        return MODERATE
    if evidence.in_skills_section:
        return WEAK
    return ABSENT


def build_skill_evidence(
    candidate,
    skill_match,
    experience_match,
) -> list[SkillEvidence]:
    """
    Grade the evidence behind every job-required skill.

    Args:
        candidate: CandidateProfile from the Phase 4 builder (needs
            ``skills`` and ``sections``).
        skill_match: SkillMatchResult from the Phase 8 matcher.
        experience_match: ExperienceMatchResult from the Phase 9 matcher.
    """
    skills_text = ""
    work_text = ""
    sections = getattr(candidate, "sections", None)
    if sections is not None:
        skills_text = (sections.get_section_text("skills") or "").lower()
        work_text = " ".join(
            sections.get_section_text(name) or "" for name in ("experience", "projects")
        ).lower()

    # Candidate skills observed anywhere on the CV (canonical names)
    cv_skill_names = {s.name.lower() for s in getattr(candidate, "skills", [])}

    # Phase 9 per-skill duration estimates, keyed by canonical skill name
    months_by_skill: dict[str, object] = {}
    for se in getattr(experience_match, "per_skill_experience", []) or []:
        months_by_skill[str(se.skill).lower()] = se

    out: list[SkillEvidence] = []
    for m in getattr(skill_match, "matches", []):
        candidate_skill = m.candidate_skill or m.taxonomy_resolved
        cand_lower = (candidate_skill or "").lower()

        in_skills = bool(cand_lower) and (
            cand_lower in cv_skill_names or cand_lower in skills_text
        )
        in_work = _mentions(work_text, candidate_skill) if candidate_skill else False

        se = months_by_skill.get(cand_lower)
        months = int(getattr(se, "estimated_months", 0) or 0) if se else 0
        all_sources = list(getattr(se, "sources", []) or []) if se else []
        # The Phase 9 coarse fallback attaches the whole employment span to
        # every CV-listed skill it cannot find in a role, marked with an
        # "(inferred)" source note. Month estimates only corroborate evidence
        # when a *named role* attributes the skill or the work text mentions
        # it directly — otherwise the months are an artifact of the fallback,
        # not evidence of use.
        sources = [s for s in all_sources if _INFERRED_SOURCE_NOTE not in s.lower()]
        if not in_work and not sources:
            months = 0

        ev = SkillEvidence(
            skill=m.required_skill,
            status=str(m.match_class.value) if hasattr(m.match_class, "value") else str(m.match_class),
            candidate_skill=candidate_skill,
            strength=ABSENT,
            in_skills_section=in_skills,
            in_work_history=in_work,
            estimated_months=months if months > 0 else None,
            sources=sources,
        )
        ev.strength = _strength(ev)
        out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------


def build_recommendations(
    skill_evidence: list[SkillEvidence],
    *,
    experience_level: str | None = None,
    missing_certifications: list[str] | None = None,
) -> list[str]:
    """
    Produce actionable, evidence-grounded recommendations.

    Priority order: build missing skills → substantiate weak evidence →
    strengthen moderate evidence → experience gap → certifications.
    The list is capped at :data:`MAX_RECOMMENDATIONS` so reports stay useful.
    """
    recs: list[str] = []
    positives: list[str] = []

    for ev in skill_evidence:
        name = ev.candidate_skill or ev.skill

        if ev.status == "missing":
            recs.append(
                f"{ev.skill} is required by this job but there is no evidence "
                "of it on the CV. Consider building demonstrable experience "
                "through a focused project or practical deployment."
            )

        elif ev.status == "partial":
            if ev.strength in (WEAK, ABSENT):
                recs.append(
                    f"{name} is only a partial match for {ev.skill}. Add a "
                    "concrete project or measurable outcome to close the gap."
                )
            else:
                recs.append(
                    f"Strengthen the evidence for {name} — quantify outcomes "
                    f"(scale, impact, results) so {ev.skill} counts as a full match."
                )

        elif ev.status == "matched" and ev.strength == WEAK:
            # The Phase 16 flagship warning.
            recs.append(
                f"{name} appears in your skills section but there is limited "
                "evidence of its use in your work experience — add the "
                "project, role, or outcome where you applied it."
            )

        elif ev.status == "matched" and ev.strength == STRONG:
            months = ev.estimated_months
            if months and months >= STRONG_MONTHS_THRESHOLD:
                n_sources = len(ev.sources)
                via = f" across {n_sources} role{'s' if n_sources != 1 else ''}" if n_sources else ""
                positives.append(
                    f"Your {name} experience is well supported — an estimated "
                    f"{months} months of use{via}."
                )
            else:
                positives.append(
                    f"Your {name} experience is backed by both your skills "
                    "section and your work history."
                )

    recs.extend(positives)

    if experience_level in ("below", "near_match"):
        recs.append(
            "Highlight measurable achievements and scope of responsibility "
            "to compensate for being slightly under the experience requirement."
        )

    for cert in (missing_certifications or [])[:2]:
        recs.append(
            f"Obtaining the {cert} certification would directly satisfy a "
            "stated requirement."
        )

    return recs[:MAX_RECOMMENDATIONS]
