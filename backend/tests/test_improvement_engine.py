"""
Tests for the Phase 16 CV Improvement Engine.

Covers the evidence-grading model (strong/moderate/weak/absent), the
flagship "listed but not evidenced" recommendation, grounded positives,
the recommendation cap, and graceful behaviour on legacy inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.nlp.experience_matcher import SkillExperience
from app.nlp.skill_extractor import ExtractedSkill
from app.nlp.skill_matcher import MatchClass, MatchLevel, SkillMatchResult
from app.scoring.improvement_engine import (
    ABSENT,
    MAX_RECOMMENDATIONS,
    MODERATE,
    STRONG,
    WEAK,
    SkillEvidence,
    build_recommendations,
    build_skill_evidence,
)

# ---------------------------------------------------------------------------
# Fakes / builders
# ---------------------------------------------------------------------------


@dataclass
class FakeSections:
    """Stand-in for DetectedSections with just the getters we use."""

    texts: dict[str, str] = field(default_factory=dict)

    def get_section_text(self, name: str) -> str | None:
        return self.texts.get(name)


@dataclass
class FakeCandidate:
    skills: list[ExtractedSkill] = field(default_factory=list)
    sections: FakeSections | None = None


@dataclass
class FakeExperienceResult:
    """Stand-in for ExperienceMatchResult (only per_skill_experience used)."""

    per_skill_experience: list = field(default_factory=list)


@dataclass
class FakeMatch:
    """Stand-in for skill_matcher.SkillMatch."""

    required_skill: str
    candidate_skill: str | None
    match_level: MatchLevel = MatchLevel.EXACT
    match_class: MatchClass = MatchClass.MATCHED
    score: float = 1.0
    taxonomy_resolved: str | None = None


def make_result(matches: list[FakeMatch]) -> SkillMatchResult:
    matched = [m.required_skill for m in matches if m.match_class == MatchClass.MATCHED]
    partial = [m.required_skill for m in matches if m.match_class == MatchClass.PARTIAL]
    missing = [m.required_skill for m in matches if m.match_class == MatchClass.MISSING]
    return SkillMatchResult(
        matches=matches,  # type: ignore[arg-type]
        required_skills=[m.required_skill for m in matches],
        preferred_skills=[],
        matched_skills=matched,
        partial_skills=partial,
        missing_skills=missing,
    )


# ---------------------------------------------------------------------------
# Evidence grading
# ---------------------------------------------------------------------------


class TestEvidenceGrading:
    def test_strong_from_sustained_use(self):
        """≥6 months of per-skill use grades as strong."""
        candidate = FakeCandidate(
            skills=[ExtractedSkill("Python", "python", "programming", 1.0)],
            sections=FakeSections({"skills": "Python", "experience": "used python at Acme"}),
        )
        sm = make_result([FakeMatch("Python", "Python")])
        em = FakeExperienceResult([SkillExperience("Python", 18, 0.9, ["Acme"])])
        ev = build_skill_evidence(candidate, sm, em)
        assert len(ev) == 1
        assert ev[0].strength == STRONG
        assert ev[0].estimated_months == 18
        assert ev[0].sources == ["Acme"]

    def test_weak_when_only_listed(self):
        """A skill present only in the skills block grades as weak."""
        candidate = FakeCandidate(
            skills=[ExtractedSkill("Docker", "docker", "devops", 1.0)],
            sections=FakeSections({"skills": "Docker", "experience": "worked on APIs"}),
        )
        sm = make_result([FakeMatch("Docker", "Docker")])
        ev = build_skill_evidence(candidate, sm, [])
        assert ev[0].strength == WEAK
        assert ev[0].in_skills_section is True
        assert ev[0].in_work_history is False

    def test_strong_via_history_plus_listing(self):
        """Work-history mention + skills listing is strong without months."""
        candidate = FakeCandidate(
            skills=[ExtractedSkill("React", "react", "frontend", 1.0)],
            sections=FakeSections(
                {"skills": "React", "experience": "built interfaces with react.js at Globex"}
            ),
        )
        sm = make_result([FakeMatch("React", "React")])
        ev = build_skill_evidence(candidate, sm, [])
        assert ev[0].strength == STRONG

    def test_fallback_months_are_not_evidence(self):
        """Inferred fallback months (no named role) are zeroed — not evidence."""
        candidate = FakeCandidate(
            skills=[ExtractedSkill("Docker", "docker", "devops", 1.0)],
            sections=FakeSections({"skills": "Docker", "experience": "worked on APIs"}),
        )
        sm = make_result([FakeMatch("Docker", "Docker")])
        em = FakeExperienceResult(
            [SkillExperience("Docker", 56, 0.3, ["CV skills section (inferred)"])]
        )
        ev = build_skill_evidence(candidate, sm, em)
        assert ev[0].strength == WEAK
        assert ev[0].estimated_months is None
        assert ev[0].sources == []
        assert ev[0].in_work_history is False

    def test_named_role_months_without_mention_cap_at_moderate(self):
        """Role-attributed months without a work-text mention stay moderate."""
        candidate = FakeCandidate(
            skills=[ExtractedSkill("Docker", "docker", "devops", 1.0)],
            sections=FakeSections({"skills": "Docker", "experience": "worked on APIs"}),
        )
        sm = make_result([FakeMatch("Docker", "Docker")])
        em = FakeExperienceResult(
            [SkillExperience("Docker", 56, 0.7, ["Backend Dev at Acme"])]
        )
        ev = build_skill_evidence(candidate, sm, em)
        assert ev[0].strength == MODERATE
        assert ev[0].estimated_months == 56
        assert ev[0].sources == ["Backend Dev at Acme"]
        assert ev[0].in_work_history is False

    def test_moderate_when_history_only(self):
        """Mentioned at work but absent from the skills block → moderate."""
        candidate = FakeCandidate(
            skills=[],
            sections=FakeSections({"skills": "Excel", "experience": "deployed with k8s daily"}),
        )
        sm = make_result([FakeMatch("Kubernetes", "Kubernetes")])
        ev = build_skill_evidence(candidate, sm, [])
        assert ev[0].strength == MODERATE

    def test_absent_when_no_trace(self):
        candidate = FakeCandidate(
            skills=[],
            sections=FakeSections({"skills": "Excel", "experience": "nothing relevant"}),
        )
        sm = make_result([FakeMatch("AWS", None, MatchLevel.MISSING, MatchClass.MISSING, 0.0)])
        ev = build_skill_evidence(candidate, sm, [])
        assert ev[0].strength == ABSENT
        assert ev[0].status == "missing"

    def test_alias_resolution_in_work_history(self):
        """A taxonomy alias (k8s → Kubernetes) counts as a work-history mention."""
        candidate = FakeCandidate(
            skills=[],
            sections=FakeSections({"skills": "", "experience": "managed clusters with k8s"}),
        )
        sm = make_result([FakeMatch("Kubernetes", "Kubernetes")])
        ev = build_skill_evidence(candidate, sm, [])
        assert ev[0].in_work_history is True


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def _ev(self, **kwargs) -> SkillEvidence:
        defaults = dict(
            skill="Python",
            status="matched",
            candidate_skill="Python",
            strength=STRONG,
            in_skills_section=True,
            in_work_history=True,
            estimated_months=None,
            sources=[],
        )
        defaults.update(kwargs)
        return SkillEvidence(**defaults)

    def test_flagship_listed_but_not_evidenced(self):
        recs = build_recommendations([self._ev(strength=WEAK, in_work_history=False)])
        assert any(
            "limited" in r and "skills section" in r for r in recs
        ), recs

    def test_missing_skill_gets_build_action(self):
        recs = build_recommendations([self._ev(skill="AWS", status="missing", candidate_skill=None, strength=ABSENT)])
        assert any("AWS" in r and "no evidence" in r for r in recs), recs

    def test_partial_gets_strengthen_action(self):
        recs = build_recommendations([self._ev(status="partial", strength=MODERATE)])
        assert any("Strengthen" in r or "quantify" in r for r in recs), recs

    def test_strong_match_generates_positive(self):
        recs = build_recommendations(
            [self._ev(estimated_months=14, sources=["Acme", "Globex"])]
        )
        assert any("14 months" in r and "2 roles" in r for r in recs), recs

    def test_experience_and_certifications_appended(self):
        recs = build_recommendations(
            [],
            experience_level="below",
            missing_certifications=["AWS Certified Developer"],
        )
        assert any("experience requirement" in r for r in recs)
        assert any("AWS Certified Developer" in r for r in recs)

    def test_cap_on_total_recommendations(self):
        many = [
            self._ev(skill=f"Skill{i}", status="missing", candidate_skill=None, strength=ABSENT)
            for i in range(12)
        ]
        recs = build_recommendations(many)
        assert len(recs) <= MAX_RECOMMENDATIONS


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_round_trip(self):
        ev = SkillEvidence(
            skill="Python",
            status="matched",
            candidate_skill="Python",
            strength=STRONG,
            in_skills_section=True,
            in_work_history=True,
            estimated_months=12,
            sources=["Acme"],
        )
        d = ev.to_dict()
        assert d["skill"] == "Python"
        assert d["strength"] == "strong"
        assert d["estimated_months"] == 12
        assert d["sources"] == ["Acme"]
