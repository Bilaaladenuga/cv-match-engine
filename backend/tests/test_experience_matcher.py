"""
Tests for Phase 9 — Experience Matching Engine.

Covers:
    - Total years matching (strong, meets, near, below)
    - Per-skill experience estimation
    - Seniority level matching
    - Gap scoring (linear interpolation, thresholds)
    - Edge cases (no experience, no requirement, all roles)
"""


from app.nlp.experience_extractor import WorkExperience
from app.nlp.experience_matcher import (
    ExperienceMatch,
    ExperienceMatcher,
    ExperienceMatchResult,
    SkillExperience,
    match_experience,
)
from app.nlp.skill_extractor import ExtractedSkill

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _exp(
    company: str | None = "Acme",
    role: str | None = "Engineer",
    months: int | None = 24,
) -> WorkExperience:
    """Create a WorkExperience with minimal defaults."""
    return WorkExperience(
        company=company,
        role=role,
        start_date="Jan 2022",
        end_date="Jan 2024",
        duration_months=months,
    )


def _skill(name: str) -> ExtractedSkill:
    """Create an ExtractedSkill."""
    return ExtractedSkill(name=name, raw_text=name, category="programming", confidence=0.9)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestExperienceMatchDataClass:
    def test_creates_with_all_fields(self):
        m = ExperienceMatch(
            dimension="total_years",
            required=3.0,
            candidate=5.0,
            gap_months=24,
            score=1.0,
            match_type="strong_match",
            evidence="5 years vs 3 required.",
        )
        assert m.score == 1.0
        assert m.gap_months == 24


class TestExperienceMatchResultDataClass:
    def test_to_dict(self):
        m = ExperienceMatch(
            dimension="total_years",
            required=3.0,
            candidate=5.0,
            gap_months=24,
            score=1.0,
            match_type="strong_match",
            evidence="Good.",
        )
        result = ExperienceMatchResult(
            matches=[m],
            total_experience_years=5.0,
            required_years=3.0,
            per_skill_experience=[],
            overall_score=1.0,
            experience_level="strong_match",
        )
        d = result.to_dict()
        assert "matches" in d
        assert "summary" in d
        assert d["summary"]["total_experience_years"] == 5.0


# ---------------------------------------------------------------------------
# Total years matching
# ---------------------------------------------------------------------------


class TestTotalYearsMatching:
    """Test matching total experience against requirements."""

    def test_strong_match(self):
        result = match_experience(
            experiences=[_exp(months=60)],  # 5 years
            skills=[],
            required_years=3.0,
        )
        m = result.matches[0]
        assert m.match_type == "strong_match"
        assert m.score == 1.0

    def test_meets(self):
        result = match_experience(
            experiences=[_exp(months=36)],  # 3 years
            skills=[],
            required_years=3.0,
        )
        m = result.matches[0]
        assert m.match_type == "meets"
        assert m.score == 0.9

    def test_near_match(self):
        result = match_experience(
            experiences=[_exp(months=24)],  # 2 years, need 3
            skills=[],
            required_years=3.0,
        )
        m = result.matches[0]
        assert m.match_type == "near_match"
        assert 0.5 <= m.score <= 0.8

    def test_below(self):
        result = match_experience(
            experiences=[_exp(months=6)],  # 0.5 years, need 3
            skills=[],
            required_years=3.0,
        )
        m = result.matches[0]
        assert m.match_type == "below"
        assert m.score < 0.5

    def test_no_experience(self):
        result = match_experience(
            experiences=[],
            skills=[],
            required_years=3.0,
        )
        m = result.matches[0]
        assert m.match_type == "unknown"
        assert m.candidate is None

    def test_gap_months_correct(self):
        result = match_experience(
            experiences=[_exp(months=48)],  # 4 years
            skills=[],
            required_years=3.0,
        )
        m = result.matches[0]
        assert m.gap_months == 12  # 4 - 3 = 1 year = 12 months


# ---------------------------------------------------------------------------
# Per-skill experience
# ---------------------------------------------------------------------------


class TestPerSkillExperience:
    """Test skill-level experience estimation."""

    def test_estimates_python_from_role(self):
        exps = [_exp(role="Python Developer", company="Acme", months=24)]
        skills = [_skill("Python")]
        result = match_experience(
            experiences=exps,
            skills=skills,
            required_skills=["Python"],
        )
        # Should find Python experience
        python_match = next(m for m in result.matches if m.dimension == "skill:Python")
        assert python_match.candidate > 0
        assert python_match.match_type in ("strong_match", "meets", "near_match")

    def test_no_skill_mention_inferred_from_cv(self):
        """Skill listed on CV but not in role titles → inferred from total career."""
        exps = [_exp(role="Backend Engineer", months=24)]
        skills = [_skill("Python")]
        result = match_experience(
            experiences=exps,
            skills=skills,
            required_skills=["Python"],
        )
        python_match = next(m for m in result.matches if m.dimension == "skill:Python")
        # Should now be inferred, not unknown
        assert python_match.candidate == 2.0  # 24 months = 2 years
        assert python_match.match_type in ("near_match", "below", "meets")

    def test_multiple_roles_accumulate(self):
        exps = [
            _exp(role="Python Developer", company="Acme", months=24),
            _exp(role="Python Engineer", company="Beta", months=12),
        ]
        skills = [_skill("Python")]
        result = match_experience(
            experiences=exps,
            skills=skills,
            required_skills=["Python"],
        )
        python_match = next(m for m in result.matches if m.dimension == "skill:Python")
        assert python_match.candidate == 3.0  # 36 months = 3 years

    def test_skill_experience_confidence(self):
        exps = [
            _exp(role="Python Developer", company="Acme", months=24),
            _exp(role="Python Engineer", company="Beta", months=12),
            _exp(role="Python Lead", company="Gamma", months=12),
        ]
        skills = [_skill("Python")]
        result = match_experience(
            experiences=exps,
            skills=skills,
            required_skills=["Python"],
        )
        # 3 roles → confidence 0.95
        python_match = next(m for m in result.matches if m.dimension == "skill:Python")
        assert python_match.evidence.count("role(s)") == 1

    def test_inferred_skill_uses_total_duration(self):
        """Skill on CV but not in any role title gets total career duration."""
        exps = [
            _exp(role="Backend Dev", company="Acme", months=24),
            _exp(role="Backend Dev", company="Beta", months=12),
        ]
        skills = [_skill("Python")]
        result = match_experience(
            experiences=exps,
            skills=skills,
            required_skills=["Python"],
        )
        python_match = next(m for m in result.matches if m.dimension == "skill:Python")
        # 36 total months = 3 years (inferred from CV skills list)
        assert python_match.candidate == 3.0
        assert python_match.match_type in ("meets", "near_match")

    def test_partial_role_match_bumps_to_total(self):
        """Skill found in 1 of 3 roles gets bumped to total duration."""
        exps = [
            _exp(role="Python Developer", company="Acme", months=24),
            _exp(role="Backend Dev", company="Beta", months=12),
            _exp(role="Backend Dev", company="Gamma", months=12),
        ]
        skills = [_skill("Python")]
        result = match_experience(
            experiences=exps,
            skills=skills,
            required_skills=["Python"],
        )
        python_match = next(m for m in result.matches if m.dimension == "skill:Python")
        # Found in 1 role (24mo) but bumped to total (48mo)
        assert python_match.candidate == 4.0


# ---------------------------------------------------------------------------
# Seniority level
# ---------------------------------------------------------------------------


class TestSeniorityLevel:
    """Test seniority level matching."""

    def test_senior_match(self):
        result = match_experience(
            experiences=[_exp(months=72)],  # 6 years
            skills=[],
            seniority_level="senior",
        )
        m = next(m for m in result.matches if m.dimension == "seniority")
        assert m.match_type in ("strong_match", "meets")

    def test_senior_below(self):
        result = match_experience(
            experiences=[_exp(months=24)],  # 2 years
            skills=[],
            seniority_level="senior",
        )
        m = next(m for m in result.matches if m.dimension == "seniority")
        assert m.match_type in ("below", "near_match")

    def test_junior_level(self):
        result = match_experience(
            experiences=[_exp(months=6)],  # 0.5 years
            skills=[],
            seniority_level="junior",
        )
        m = next(m for m in result.matches if m.dimension == "seniority")
        assert m.match_type in ("strong_match", "meets")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_required_years(self):
        """Without required years, only seniority/skill matches."""
        result = match_experience(
            experiences=[_exp(months=24)],
            skills=[_skill("Python")],
        )
        # No total_years match, but skills might be estimated
        total_matches = [m for m in result.matches if m.dimension == "total_years"]
        assert len(total_matches) == 0

    def test_multiple_requirements(self):
        result = match_experience(
            experiences=[_exp(months=36)],
            skills=[_skill("Python")],
            required_years=3.0,
            seniority_level="mid",
            required_skills=["Python"],
        )
        # Should have total_years, skill:Python, and seniority matches
        dimensions = {m.dimension for m in result.matches}
        assert "total_years" in dimensions
        assert "skill:Python" in dimensions
        assert "seniority" in dimensions

    def test_empty_experiences_list(self):
        result = match_experience(
            experiences=[],
            skills=[],
            required_years=5.0,
            seniority_level="senior",
        )
        assert result.overall_score == 0.0
        assert result.experience_level == "unknown"

    def test_to_dict_serialization(self):
        result = match_experience(
            experiences=[_exp(months=36)],
            skills=[],
            required_years=3.0,
        )
        d = result.to_dict()
        assert "matches" in d
        assert "summary" in d
        assert d["summary"]["total_experience_years"] == 3.0


# ---------------------------------------------------------------------------
# SkillExperience data class
# ---------------------------------------------------------------------------


class TestSkillExperience:
    def test_skill_experience_properties(self):
        se = SkillExperience(
            skill="Python",
            estimated_months=36,
            confidence=0.9,
            sources=["Acme", "Beta"],
        )
        assert se.skill == "Python"
        assert se.estimated_months == 36
        assert len(se.sources) == 2


# ---------------------------------------------------------------------------
# Score gap function
# ---------------------------------------------------------------------------


class TestScoreGap:
    """Test the gap scoring logic directly."""

    def test_large_surplus(self):
        score, label = ExperienceMatcher._score_gap(24, 3.0)  # 2 years over
        assert score == 1.0
        assert label == "strong_match"

    def test_exact_match(self):
        score, label = ExperienceMatcher._score_gap(0, 3.0)
        assert score == 0.9
        assert label == "meets"

    def test_small_deficit(self):
        score, label = ExperienceMatcher._score_gap(-6, 3.0)  # 6 months under
        assert label == "near_match"
        assert 0.5 < score < 0.8

    def test_large_deficit(self):
        score, label = ExperienceMatcher._score_gap(-36, 3.0)  # 3 years under
        assert label == "below"
        assert score < 0.5
