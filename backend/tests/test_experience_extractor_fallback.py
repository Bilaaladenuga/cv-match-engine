"""
Tests for the Phase 12 experience-extractor fallbacks:

- extract_experience_from_text: dated job lines in CVs without a detectable
  experience section (prose-style resumes), skipping education lines.
- extract_total_years_claim: stated "N years of experience" claims,
  including word numbers and possessive forms.
"""

from __future__ import annotations

from app.nlp.experience_extractor import (
    extract_experience_from_text,
    extract_total_years_claim,
)


class TestExtractExperienceFromText:
    def test_finds_dated_lines_without_section_header(self):
        cv = (
            "SummaryHighly motivated developer with five years of experience.\n"
            "Warehouse Associate,07/2015-2016 ACME Distribution\n"
            "Forklift Operator 01/2013 - 06/2015 Other Corp\n"
            "B.S. in Mechanical Engineering, University of Somewhere, 2012\n"
        )
        exps = extract_experience_from_text(cv)
        assert len(exps) == 2
        assert exps[0].role == "Warehouse Associate,"
        assert exps[0].duration_months is not None and exps[0].duration_months > 0

    def test_skips_education_lines(self):
        cv = (
            "B.S. in Computer Science, 2015 - 2019 State University\n"
            "Software Developer 01/2020 - Present TechCo\n"
        )
        exps = extract_experience_from_text(cv)
        assert len(exps) == 1
        assert exps[0].role is not None and "Software Developer" in exps[0].role

    def test_skips_bare_date_ranges(self):
        assert extract_experience_from_text("2019 - 2021\n") == []

    def test_empty_and_headerless_text(self):
        assert extract_experience_from_text("") == []
        assert extract_experience_from_text("no dates anywhere in here") == []

    def test_present_end_date(self):
        cv = "Barista 03/2018 - Present Cafe Loca\n"
        exps = extract_experience_from_text(cv)
        assert len(exps) == 1
        assert exps[0].duration_months and exps[0].duration_months > 12


class TestExtractTotalYearsClaim:
    def test_digit_claims(self):
        assert extract_total_years_claim("8 years of experience in sales") == 8.0
        assert extract_total_years_claim("3+ years experience required") == 3.0
        assert extract_total_years_claim("10 years' professional experience") == 10.0

    def test_word_claims(self):
        assert extract_total_years_claim("thirteen years of experience") == 13.0
        assert extract_total_years_claim("with five years experience") == 5.0

    def test_multiple_claims_take_max(self):
        text = "Started as an apprentice. Fifteen years of experience overall."
        assert extract_total_years_claim(text) == 15.0

    def test_no_claim_returns_none(self):
        assert extract_total_years_claim("I enjoy long walks.") is None
        assert extract_total_years_claim("") is None

    def test_does_not_match_experience_in_other_words(self):
        # "experienced" must not count as a claim
        assert extract_total_years_claim("an experienced team of 3 people") is None
