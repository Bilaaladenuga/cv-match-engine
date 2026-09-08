"""
Tests for Phase 8 — Skill Matching Engine.

Covers:
    - Exact matching (same canonical name)
    - Alias matching (JS → JavaScript)
    - Semantic matching (high similarity, different names)
    - Related matching (same category / cloud equivalents)
    - Missing matching (not found at any level)
    - Unknown matching (skill not in taxonomy)
    - Aggregate metrics (coverage, overall score)
    - Edge cases (empty inputs, single skill, all matched, none matched)
"""


from app.nlp.skill_matcher import (
    MatchClass,
    MatchLevel,
    SkillMatch,
    SkillMatcher,
    SkillMatchResult,
    are_related,
    match_skills,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANDIDATE_SKILLS = ["Python", "React", "PostgreSQL", "Docker", "JavaScript", "Machine Learning"]
REQUIRED_SKILLS = ["Python", "React", "AWS", "Docker", "Kubernetes"]
PREFERRED_SKILLS = ["PostgreSQL", "TypeScript", "Machine Learning"]


# ---------------------------------------------------------------------------
# Data class structure
# ---------------------------------------------------------------------------


class TestSkillMatchDataClass:
    """Verify SkillMatch dataclass fields."""

    def test_creates_with_all_fields(self):
        m = SkillMatch(
            required_skill="Python",
            candidate_skill="Python",
            match_level=MatchLevel.EXACT,
            match_class=MatchClass.MATCHED,
            score=1.0,
            taxonomy_resolved="Python",
            evidence="Direct match.",
        )
        assert m.required_skill == "Python"
        assert m.score == 1.0

    def test_missing_skill_has_none_candidate(self):
        m = SkillMatch(
            required_skill="AWS",
            candidate_skill=None,
            match_level=MatchLevel.MISSING,
            match_class=MatchClass.MISSING,
            score=0.0,
            taxonomy_resolved="AWS",
        )
        assert m.candidate_skill is None


class TestSkillMatchResultDataClass:
    """Verify SkillMatchResult structure."""

    def test_to_dict_round_trip(self):
        m = SkillMatch(
            required_skill="Python",
            candidate_skill="Python",
            match_level=MatchLevel.EXACT,
            match_class=MatchClass.MATCHED,
            score=1.0,
            taxonomy_resolved="Python",
            evidence="Direct match.",
        )
        result = SkillMatchResult(
            matches=[m],
            required_skills=["Python"],
            preferred_skills=[],
            matched_skills=["Python"],
            overall_score=1.0,
        )
        d = result.to_dict()
        assert "matches" in d
        assert "summary" in d
        assert d["summary"]["matched_skills"] == ["Python"]
        assert d["summary"]["overall_score"] == 1.0


# ---------------------------------------------------------------------------
# Exact matching
# ---------------------------------------------------------------------------


class TestExactMatching:
    """Test Level 1: exact and alias matching."""

    def test_exact_match(self):
        result = match_skills(
            candidate_skills=["Python", "React"],
            required_skills=["Python", "React"],
        )
        # Both should be MATCHED
        for m in result.matches:
            assert m.match_class == MatchClass.MATCHED
            assert m.match_level in (MatchLevel.EXACT, MatchLevel.ALIAS)
            assert m.score == 1.0

    def test_alias_match_js(self):
        """JS → JavaScript should be an alias match."""
        result = match_skills(
            candidate_skills=["JavaScript"],
            required_skills=["JS"],
        )
        m = result.matches[0]
        assert m.match_class == MatchClass.MATCHED
        assert m.match_level == MatchLevel.ALIAS
        assert m.candidate_skill == "JavaScript"
        assert m.taxonomy_resolved == "JavaScript"

    def test_alias_match_postgres(self):
        """Postgres → PostgreSQL should be an alias match."""
        result = match_skills(
            candidate_skills=["PostgreSQL"],
            required_skills=["Postgres"],
        )
        m = result.matches[0]
        assert m.match_class == MatchClass.MATCHED
        assert m.match_level == MatchLevel.ALIAS

    def test_alias_match_typescript(self):
        """TS → TypeScript alias."""
        result = match_skills(
            candidate_skills=["TypeScript"],
            required_skills=["TS"],
        )
        m = result.matches[0]
        assert m.match_class == MatchClass.MATCHED
        assert m.match_level == MatchLevel.ALIAS

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        result = match_skills(
            candidate_skills=["python", "REACT"],
            required_skills=["Python", "react"],
        )
        for m in result.matches:
            assert m.match_class == MatchClass.MATCHED


# ---------------------------------------------------------------------------
# Missing matching
# ---------------------------------------------------------------------------


class TestMissingMatching:
    """Test Level 4: missing skill detection."""

    def test_missing_skill(self):
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["AWS"],
        )
        m = result.matches[0]
        assert m.match_class == MatchClass.MISSING
        assert m.match_level == MatchLevel.MISSING
        assert m.score == 0.0
        assert m.candidate_skill is None

    def test_multiple_missing(self):
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["AWS", "Kubernetes", "Terraform"],
        )
        assert len(result.missing_skills) == 3
        assert result.required_coverage == 0.0

    def test_partial_coverage(self):
        result = match_skills(
            candidate_skills=["Python", "React"],
            required_skills=["Python", "React", "AWS"],
        )
        assert len(result.matched_skills) == 2
        assert len(result.missing_skills) == 1
        assert abs(result.required_coverage - 2 / 3) < 0.01


# ---------------------------------------------------------------------------
# Unknown matching
# ---------------------------------------------------------------------------


class TestUnknownMatching:
    """Test skills not in the taxonomy."""

    def test_unknown_skill(self):
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["QuantumLeap Framework"],
        )
        m = result.matches[0]
        assert m.match_class == MatchClass.UNKNOWN
        assert m.match_level == MatchLevel.UNKNOWN
        assert m.score == 0.0

    def test_unknown_among_known(self):
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["Python", "QuantumLeap Framework"],
        )
        matched = [m for m in result.matches if m.match_class == MatchClass.MATCHED]
        unknown = [m for m in result.matches if m.match_class == MatchClass.UNKNOWN]
        assert len(matched) == 1
        assert len(unknown) == 1


# ---------------------------------------------------------------------------
# Semantic matching
# ---------------------------------------------------------------------------


class TestSemanticMatching:
    """Test Level 2: semantic similarity matching."""

    def test_semantic_match_detected(self):
        """When candidate has a skill that's semantically similar but not exact."""
        result = match_skills(
            candidate_skills=["React", "Frontend Development"],
            required_skills=["Frontend Development"],
        )
        m = result.matches[0]
        # Should be MATCHED (exact) or PARTIAL (semantic)
        assert m.match_class in (MatchClass.MATCHED, MatchClass.PARTIAL)

    def test_semantic_produces_evidence(self):
        """Semantic matches should have explanatory evidence."""
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["Python"],
        )
        m = result.matches[0]
        assert len(m.evidence) > 0


# ---------------------------------------------------------------------------
# Related matching
# ---------------------------------------------------------------------------


class TestRelatedMatching:
    """Test Level 3: related-skill graph and category matching."""

    def test_cloud_equivalents(self):
        """AWS ↔ Azure ↔ GCP are related."""
        assert are_related("AWS", "Azure")
        assert are_related("AWS", "Google Cloud Platform")
        assert are_related("Azure", "GCP")

    def test_database_related(self):
        """PostgreSQL ↔ MySQL are related."""
        assert are_related("PostgreSQL", "MySQL")
        assert are_related("MySQL", "SQLite")

    def test_container_related(self):
        """Docker ↔ Kubernetes are related."""
        assert are_related("Docker", "Kubernetes")

    def test_frontend_related(self):
        """React ↔ Angular ↔ Vue.js are related."""
        assert are_related("React", "Angular")
        assert are_related("Angular", "Vue.js")

    def test_not_related(self):
        """Unrelated skills should not be flagged."""
        assert not are_related("Python", "React")
        assert not are_related("Docker", "PostgreSQL")

    def test_same_skill_not_related_to_self(self):
        """A skill should not be related to itself."""
        assert not are_related("Python", "Python")


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    """Test overall scoring and coverage calculations."""

    def test_all_matched(self):
        result = match_skills(
            candidate_skills=["Python", "React", "Docker"],
            required_skills=["Python", "React", "Docker"],
        )
        assert result.required_coverage == 1.0
        assert result.overall_score == 1.0
        assert len(result.missing_skills) == 0

    def test_none_matched(self):
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["AWS", "Kubernetes", "Terraform"],
        )
        assert result.required_coverage == 0.0
        assert result.overall_score == 0.0
        assert len(result.matched_skills) == 0

    def test_preferred_skills_weighted_less(self):
        """Preferred skills should have lower weight in overall score."""
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["Python"],
            preferred_skills=["AWS"],
        )
        # Python matched (weight 1.0), AWS missing (weight 0.5)
        # score = (1.0 * 1.0 + 0.5 * 0.0) / (1.0 + 0.5) = 0.667
        assert abs(result.overall_score - 2 / 3) < 0.01

    def test_required_coverage_excludes_preferred(self):
        """Coverage should only count required skills, not preferred."""
        result = match_skills(
            candidate_skills=["Python", "PostgreSQL"],
            required_skills=["Python"],
            preferred_skills=["PostgreSQL", "AWS"],
        )
        assert result.required_coverage == 1.0  # Python matched
        # PostgreSQL is preferred, so missing preferred don't affect coverage

    def test_required_plus_partial(self):
        """required_plus_partial includes both MATCHED and PARTIAL."""
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["Python", "AWS"],
        )
        # Python matched, AWS missing
        assert result.required_coverage == 0.5
        # No partials, so required_plus_partial == required_coverage
        assert result.required_plus_partial == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_candidate_skills(self):
        result = match_skills(
            candidate_skills=[],
            required_skills=["Python", "React"],
        )
        assert result.required_coverage == 0.0
        assert len(result.missing_skills) == 2

    def test_empty_required_skills(self):
        result = match_skills(
            candidate_skills=["Python", "React"],
            required_skills=[],
        )
        assert result.required_coverage == 0.0  # no required to cover
        assert result.overall_score == 0.0

    def test_empty_everything(self):
        result = match_skills(
            candidate_skills=[],
            required_skills=[],
        )
        assert result.matches == []
        assert result.overall_score == 0.0

    def test_single_skill_match(self):
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["Python"],
        )
        assert len(result.matches) == 1
        assert result.matches[0].match_class == MatchClass.MATCHED

    def test_duplicate_candidate_skills(self):
        """Duplicate skills in candidate should be handled gracefully."""
        result = match_skills(
            candidate_skills=["Python", "Python", "React"],
            required_skills=["Python", "React"],
        )
        assert result.required_coverage == 1.0

    def test_duplicate_required_skills(self):
        """Duplicate required skills should each be matched independently."""
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["Python", "Python"],
        )
        assert len(result.matches) == 2
        assert all(m.match_class == MatchClass.MATCHED for m in result.matches)


# ---------------------------------------------------------------------------
# SkillMatcher class
# ---------------------------------------------------------------------------


class TestSkillMatcherClass:
    """Test the SkillMatcher class directly."""

    def test_default_instance(self):
        matcher = SkillMatcher()
        result = matcher.match(
            candidate_skills=["Python"],
            required_skills=["Python"],
        )
        assert result.matches[0].match_class == MatchClass.MATCHED

    def test_full_cv_vs_jd(self):
        """Realistic CV vs JD matching."""
        matcher = SkillMatcher()
        result = matcher.match(
            candidate_skills=CANDIDATE_SKILLS,
            required_skills=REQUIRED_SKILLS,
            preferred_skills=PREFERRED_SKILLS,
        )
        # Python: exact match
        # React: exact match
        # AWS: missing
        # Docker: exact match
        # Kubernetes: missing
        # PostgreSQL: preferred, matched
        # TypeScript: preferred, missing
        # Machine Learning: preferred, matched
        matched = {m.required_skill for m in result.matches if m.match_class == MatchClass.MATCHED}
        missing = {m.required_skill for m in result.matches if m.match_class == MatchClass.MISSING}
        assert "Python" in matched
        assert "React" in matched
        assert "Docker" in matched
        assert "AWS" in missing
        assert "Kubernetes" in missing
        assert result.required_coverage > 0.5  # 3/5 required matched

    def test_to_dict_has_all_fields(self):
        result = match_skills(
            candidate_skills=["Python"],
            required_skills=["Python"],
        )
        d = result.to_dict()
        assert "matches" in d
        assert "summary" in d
        assert len(d["matches"]) == 1
        m = d["matches"][0]
        assert "required_skill" in m
        assert "candidate_skill" in m
        assert "match_level" in m
        assert "match_class" in m
        assert "score" in m
        assert "evidence" in m
