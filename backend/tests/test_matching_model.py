"""Tests for Phase 11 — the hybrid matching model."""

from types import SimpleNamespace

import pytest

from app.scoring.matching_model import (
    MODEL_VERSION,
    MatcherInputs,
    compute_match_score,
    score_band,
)
from app.scoring.weights import WeightsError


def _inputs(
    skills_score=0.8,
    semantic_score=0.7,
    experience_score=0.9,
    education_score=0.85,
    cert_score=0.85,
    **extra,
):
    """Build a MatcherInputs from plain namespaces (no NLP stack needed)."""

    def _skill_match():
        base = SimpleNamespace(
            overall_score=skills_score,
            required_coverage=0.8,
            required_plus_partial=0.9,
            matched_skills=["Python", "React"],
            partial_skills=["ML"],
            missing_skills=["Docker"],
            unknown_skills=[],
        )
        return SimpleNamespace(**{**base.__dict__, **extra.get("skill_match", {})})

    def _cert_match():
        if cert_score is None:
            return SimpleNamespace(score=0.85, matched=[], missing=[])
        n_req = extra.get("cert_required", 2)
        n_match = round(cert_score * n_req)
        return SimpleNamespace(
            score=cert_score,
            matched=[f"cert{i}" for i in range(n_match)],
            missing=[f"cert{i}" for i in range(n_match, n_req)],
        )

    return MatcherInputs(
        skill_match=_skill_match(),
        experience_match=SimpleNamespace(
            overall_score=experience_score,
            experience_level=extra.get("experience_level", "strong_match"),
        ),
        semantic_match=SimpleNamespace(
            raw_score=semantic_score,
            label="Good",
        ),
        education_match=SimpleNamespace(score=education_score, evidence="B.S. meets B.S."),
        certification_match=_cert_match(),
        job_title=extra.get("job_title"),
    )


class TestScoring:
    def test_perfect_inputs_score_one(self):
        r = compute_match_score(
            _inputs(
                skills_score=1.0,
                semantic_score=1.0,
                experience_score=1.0,
                education_score=1.0,
                cert_score=1.0,
            )
        )
        assert r.overall_score == 1.0
        assert r.overall_percent == 100
        assert r.band == "Excellent match"

    def test_weighted_sum_math(self):
        r = compute_match_score(
            _inputs(
                skills_score=1.0,
                semantic_score=0.0,
                experience_score=0.0,
                education_score=0.0,
                cert_score=0.0,
            )
        )
        # only skills contributes: 1.0 * 0.40
        assert r.overall_score == pytest.approx(0.40)

    def test_components_present_with_weights(self):
        r = compute_match_score(_inputs())
        names = [c.name for c in r.components]
        assert names == ["skills", "semantic", "experience", "education", "certifications"]
        assert sum(c.weight for c in r.components) == pytest.approx(1.0)

    def test_scores_clamped(self):
        r = compute_match_score(
            _inputs(skills_score=2.0, semantic_score=-1.0)
        )
        skills = next(c for c in r.components if c.name == "skills")
        semantic = next(c for c in r.components if c.name == "semantic")
        assert skills.raw_score == 1.0
        assert semantic.raw_score == 0.0

    def test_model_version_stamped(self):
        r = compute_match_score(_inputs())
        assert r.model_version == MODEL_VERSION
        assert MODEL_VERSION.startswith("match-model-v")

    def test_custom_weights_change_result(self):
        base = compute_match_score(_inputs(skills_score=0.0, semantic_score=1.0))
        custom = compute_match_score(
            _inputs(skills_score=0.0, semantic_score=1.0),
            weights={
                "skills": 0.0, "semantic": 0.6, "experience": 0.2,
                "education": 0.1, "certifications": 0.1,
            },
        )
        assert custom.overall_score > base.overall_score

    def test_invalid_weights_raise(self):
        with pytest.raises(WeightsError):
            compute_match_score(_inputs(), weights={"skills": 0.4})


class TestExplainability:
    def test_positive_and_negative_factors(self):
        r = compute_match_score(
            _inputs(skills_score=1.0, semantic_score=0.5, education_score=0.4)
        )
        assert r.positive_factors, "strong components should be listed"
        assert r.negative_factors, "weak components should be listed"
        assert any(f.startswith("Skills") for f in r.positive_factors)
        assert any(f.startswith("Education") for f in r.negative_factors)

    def test_recommendations_generated_from_missing_skills(self):
        r = compute_match_score(_inputs())
        assert any("Docker" in rec for rec in r.recommendations)

    def test_recommendations_cap_at_reasonable_length(self):
        r = compute_match_score(_inputs())
        assert len(r.recommendations) <= 8

    def test_disclaimer_present(self):
        r = compute_match_score(_inputs())
        assert "decision-support" in r.disclaimer
        assert "not a prediction" in r.disclaimer

    def test_to_dict_round_trip_fields(self):
        d = compute_match_score(_inputs()).to_dict()
        for key in (
            "overall_score", "overall_percent", "band", "components",
            "weights", "model_version", "positive_factors",
            "negative_factors", "recommendations", "disclaimer",
        ):
            assert key in d


class TestBands:
    def test_band_boundaries(self):
        assert score_band(100) == "Excellent match"
        assert score_band(85) == "Excellent match"
        assert score_band(84.9) == "Good match"
        assert score_band(70) == "Good match"
        assert score_band(55) == "Moderate match"
        assert score_band(40) == "Fair match"
        assert score_band(0) == "Weak match"


class TestNeutralHandling:
    def test_no_certs_required_is_neutral(self):
        r = compute_match_score(_inputs(cert_score=None))
        certs = next(c for c in r.components if c.name == "certifications")
        assert certs.raw_score == pytest.approx(0.85)
        assert "No certifications required" in certs.evidence

    def test_semantic_accepts_normalised_score_input(self):
        """Engines that expose only normalised_score (0-100) still work."""
        inputs = _inputs()
        inputs.semantic_match = SimpleNamespace(normalised_score=72, label="Good")
        r = compute_match_score(inputs)
        semantic = next(c for c in r.components if c.name == "semantic")
        assert semantic.raw_score == pytest.approx(0.72)
