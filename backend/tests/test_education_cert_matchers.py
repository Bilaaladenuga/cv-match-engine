"""Tests for Phase 11 — education and certification matching."""

from types import SimpleNamespace

from app.scoring.certification_matcher import (
    CertificationMatchResult,
    match_certifications,
)
from app.scoring.education_matcher import (
    field_relevance,
    match_education,
)


def _edu(degree=None, field=None):
    return SimpleNamespace(
        degree=degree, field_of_study=field, institution=None, year=None, raw_text=""
    )


class TestEducationLevels:
    def test_no_requirement_is_neutral(self):
        m = match_education([_edu("B.S.", "Computer Science")], None)
        assert m.score == 0.85
        assert m.meets_requirement is True

    def test_exact_level_meets(self):
        m = match_education(
            [_edu("Bachelor of Science", "Computer Science")],
            {"level": "Bachelor's", "field": "Computer Science", "required": True},
        )
        assert m.level_score == 1.0
        assert m.score > 0.9
        assert m.meets_requirement is True

    def test_higher_level_meets_lower_requirement(self):
        m = match_education(
            [_edu("PhD", "Computer Science")],
            {"level": "Bachelor's", "field": None, "required": True},
        )
        assert m.level_score == 1.0
        assert m.meets_requirement is True

    def test_one_level_below_is_partial(self):
        m = match_education(
            [_edu("B.A. Mathematics")],
            {"level": "Master's", "field": None, "required": True},
        )
        assert m.level_score == 0.6
        assert m.meets_requirement is False

    def test_two_levels_below_scores_low(self):
        m = match_education(
            [_edu("High School Diploma")],
            {"level": "Master's", "field": None, "required": True},
        )
        assert m.level_score < 0.3

    def test_missing_degree_with_requirement_scores_zero_level(self):
        m = match_education(
            [],
            {"level": "Bachelor's", "field": None, "required": True},
        )
        assert m.level_score == 0.0
        assert "may be unlisted or unparsed" in m.evidence

    def test_unparseable_requirement_is_neutral(self):
        m = match_education(
            [_edu("BSc CS")],
            {"level": "some weird level", "field": None, "required": True},
        )
        assert m.score == 0.85

    def test_best_entry_wins(self):
        entries = [_edu("Associate", "General Studies"), _edu("M.S.", "Computer Science")]
        m = match_education(
            entries, {"level": "Bachelor's", "field": None, "required": True}
        )
        assert m.candidate_level == "Master's"


class TestEducationFields:
    def test_same_field_full_score(self):
        assert field_relevance("Computer Science", "computer science") == 1.0

    def test_alias_field_full_score(self):
        assert field_relevance("CS", "Computer Science") == 1.0

    def test_related_fields_high(self):
        assert field_relevance("Software Engineering", "Computer Science") == 0.8

    def test_tech_family_moderate(self):
        assert field_relevance("Mathematics", "Computer Science") == 0.6

    def test_unrelated_field_low(self):
        assert field_relevance("History", "Computer Science") == 0.3

    def test_unknown_required_field_is_neutral(self):
        assert field_relevance("Computer Science", None) == 1.0

    def test_unknown_candidate_field_half(self):
        assert field_relevance(None, "Computer Science") == 0.5

    def test_field_blends_into_score(self):
        m = match_education(
            [_edu("Bachelor", "History")],
            {"level": "Bachelor's", "field": "Computer Science", "required": True},
        )
        # level 1.0 * 0.6 + field 0.3 * 0.4 = 0.72
        assert m.score == 0.72


class TestCertificationMatching:
    def test_no_requirement_neutral(self):
        r = match_certifications(["AWS Certified Solutions Architect"], [])
        assert isinstance(r, CertificationMatchResult)
        assert r.score == 0.85

    def test_exact_match(self):
        r = match_certifications(
            ["AWS Certified Solutions Architect"], ["AWS Certified Solutions Architect"]
        )
        assert r.score == 1.0
        assert r.matched == ["AWS Certified Solutions Architect"]

    def test_alias_match(self):
        r = match_certifications(
            ["AWS Solutions Architect"], ["AWS Certified Solutions Architect"]
        )
        assert r.score == 1.0
        assert r.matches[0].match_type == "alias"

    def test_missing_cert_scores_zero(self):
        r = match_certifications(["AWS Certified Developer"], ["CKA"])
        assert r.score == 0.0
        assert r.missing == ["CKA"]

    def test_partial_coverage(self):
        r = match_certifications(
            ["AWS Certified Developer"],
            ["AWS Certified Developer", "CKA"],
        )
        assert r.score == 0.5
        assert r.matched == ["AWS Certified Developer"]
        assert r.missing == ["CKA"]

    def test_vague_requirement_treated_as_none(self):
        r = match_certifications([], ["certifications"])
        # single vague token becomes an unmatched requirement (score 0),
        # which is preferable to pretending neutrality
        assert r.score == 0.0
        assert r.missing == ["certifications"]

    def test_distinct_certs_do_not_semantically_match(self):
        # AWS SAA vs AWS Developer Associate are different credentials
        r = match_certifications(
            ["AWS Certified Developer Associate"],
            ["AWS Certified Solutions Architect"],
        )
        assert r.score == 0.0
        assert r.matches[0].match_type == "missing"

    def test_to_dict_shape(self):
        r = match_certifications(["CKA"], ["CKA"])
        d = r.to_dict()
        assert d["score"] == 1.0
        assert d["matches"][0]["match_type"] in ("exact", "alias", "semantic")
