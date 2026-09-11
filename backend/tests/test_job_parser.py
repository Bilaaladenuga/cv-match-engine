"""
Tests for Phase 6 — Job Description Parser.
"""

from pathlib import Path

import pytest

from app.nlp.job_parser import (
    JobProfile,
    _trim_education_field,
    extract_education_requirement,
    extract_experience_requirement,
    extract_job_title,
    parse_job_description,
)

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"


def names(profile: JobProfile, kind: str) -> set[str]:
    return {s.name for s in getattr(profile, kind)}


class TestSpecExample:
    """The canonical example from the Phase 6 spec."""

    TEXT = (
        "Looking for a frontend developer with 2+ years experience in React, "
        "TypeScript and Next.js. AWS experience is preferred."
    )

    def test_required_skills(self):
        profile = parse_job_description(self.TEXT)
        assert names(profile, "required_skills") == {"React", "TypeScript", "Next.js"}

    def test_preferred_skills(self):
        profile = parse_job_description(self.TEXT)
        assert names(profile, "preferred_skills") == {"AWS"}

    def test_experience(self):
        profile = parse_job_description(self.TEXT)
        assert profile.minimum_experience_years == 2

    def test_title(self):
        profile = parse_job_description(self.TEXT)
        assert profile.job_title == "Frontend Developer"


class TestExperience:
    def test_plus_years(self):
        assert extract_experience_requirement("3+ years of experience with Python.")[0] == 3

    def test_at_least(self):
        assert extract_experience_requirement("At least 5 years experience required.")[0] == 5

    def test_minimum_of(self):
        assert extract_experience_requirement("Minimum of 2 years.")[0] == 2

    def test_range_uses_floor(self):
        assert extract_experience_requirement("5-7 years of professional experience.")[0] == 5

    def test_preferred_number_ignored(self):
        # "preferred" years must not become the required minimum
        text = "At least 2 years of experience. 8+ years preferred for senior."
        assert extract_experience_requirement(text)[0] == 2

    def test_no_experience(self):
        assert extract_experience_requirement("No requirements stated here.")[0] is None

    def test_level_detection(self):
        minimum, level = extract_experience_requirement(
            "Senior Backend Engineer. 5+ years of experience."
        )
        assert minimum == 5
        assert level == "senior"


class TestJobTitle:
    def test_labeled(self):
        assert extract_job_title("Job Title: Senior Data Scientist\nCompany: Acme") == (
            "Senior Data Scientist"
        )

    def test_standalone_first_line(self):
        assert extract_job_title(
            "Senior Frontend Engineer\n\nCompany: Acme\nWe need an engineer with React."
        ) == "Senior Frontend Engineer"

    def test_looking_for_phrase(self):
        assert extract_job_title(
            "We are looking for a Senior Frontend Engineer to join our team."
        ) == "Senior Frontend Engineer"

    def test_plain_phrase(self):
        assert extract_job_title(
            "Looking for a frontend developer with 2+ years of experience."
        ) == "Frontend Developer"

    def test_hiring_phrase(self):
        assert extract_job_title("We're hiring a Principal Engineer at Rocket.") == (
            "Principal Engineer"
        )

    def test_prose_is_not_a_title(self):
        assert extract_job_title("We are looking for a candidate who loves APIs.") is None

    def test_no_title(self):
        assert extract_job_title("Python, React, PostgreSQL.") is None


class TestEducation:
    def test_bachelors(self):
        edu = extract_education_requirement(
            "Bachelor's degree in Computer Science required."
        )
        assert edu["level"] == "Bachelor's"
        assert edu["field"] == "Computer Science"
        assert edu["required"] is True

    def test_masters_preferred(self):
        edu = extract_education_requirement("Master's degree in Data Science preferred.")
        assert edu["level"] == "Master's"
        assert edu["required"] is False

    def test_phd(self):
        edu = extract_education_requirement("PhD in ML or related field.")
        assert edu["level"] == "Doctorate"

    def test_no_degree(self):
        assert extract_education_requirement("No education requirement.") is None


class TestStructuredSections:
    TEXT = """Senior Backend Engineer

Company: Contoso
Location: Berlin, Germany

Responsibilities
- Build REST APIs with FastAPI
- Operate PostgreSQL in production

Required Qualifications
- 3+ years of Python experience
- Strong FastAPI and PostgreSQL skills
- Working knowledge of Docker

Preferred Skills
- Kubernetes
- Terraform
"""

    def test_title_company_location(self):
        profile = parse_job_description(self.TEXT)
        assert profile.job_title == "Senior Backend Engineer"
        assert profile.company == "Contoso"
        assert profile.location == "Berlin, Germany"
        assert profile.is_remote is False

    def test_required_and_preferred_split(self):
        profile = parse_job_description(self.TEXT)
        assert {"Python", "FastAPI", "PostgreSQL", "Docker"} <= names(profile, "required_skills")
        assert names(profile, "preferred_skills") == {"Kubernetes", "Terraform"}

    def test_responsibilities_captured(self):
        profile = parse_job_description(self.TEXT)
        assert profile.responsibilities == [
            "Build REST APIs with FastAPI",
            "Operate PostgreSQL in production",
        ]

    def test_plus_sentence_in_required_block(self):
        text = (
            "Requirements\n"
            "- 3+ years Python\n"
            "- Experience with scikit-learn is a plus for our data team\n"
        )
        profile = parse_job_description(text)
        assert "Scikit-learn" in names(profile, "preferred_skills")
        assert "Scikit-learn" not in names(profile, "required_skills")

    def test_required_wins_over_preferred_mention(self):
        text = (
            "Required: React and TypeScript. "
            "Preferred: React Native and TypeScript tooling."
        )
        profile = parse_job_description(text)
        assert "React" in names(profile, "required_skills")
        assert "React" not in names(profile, "preferred_skills")
        assert "React Native" in names(profile, "preferred_skills")


class TestCertificationsAndRemote:
    def test_certifications(self):
        profile = parse_job_description(
            "Certifications: AWS Certified Solutions Architect preferred."
        )
        assert any("Solutions Architect" in c for c in profile.certifications)

    def test_remote_detection(self):
        profile = parse_job_description("Location: Remote\nSenior Dev wanted.")
        assert profile.is_remote is True
        assert profile.location == "Remote"


class TestSampleFile:
    @pytest.fixture
    def text(self):
        path = SAMPLE_DATA_DIR / "sample_job.txt"
        if not path.exists():
            pytest.skip("Sample job not found")
        return path.read_text(encoding="utf-8")

    def test_full_profile(self, text):
        profile = parse_job_description(text)
        assert profile.job_title == "Senior Frontend Engineer"
        assert profile.company == "Cloudly Systems"
        assert profile.is_remote is True
        assert profile.minimum_experience_years == 4
        assert profile.experience_level == "senior"
        assert {"React", "TypeScript", "Next.js", "HTML", "CSS", "JavaScript"} <= names(
            profile, "required_skills"
        )
        assert names(profile, "preferred_skills") >= {
            "AWS", "Docker", "GraphQL", "REST API", "PostgreSQL", "Scikit-learn",
            "Figma", "UI Design",
        }
        assert len(profile.responsibilities) == 5
        assert profile.education is not None
        assert profile.education["level"] == "Bachelor's"

    def test_to_dict(self, text):
        profile = parse_job_description(text)
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["job_title"] == "Senior Frontend Engineer"
        assert isinstance(d["required_skills"], list)
        assert {"name", "category", "confidence"} <= set(d["required_skills"][0])
        assert isinstance(d["responsibilities"], list)


class TestEdgeCases:
    def test_empty_text(self):
        profile = parse_job_description("")
        assert profile.job_title is None
        assert profile.required_skills == []
        assert profile.responsibilities == []

    def test_whitespace_only(self):
        profile = parse_job_description("   \n  ")
        assert profile.job_title is None

    def test_unheaded_prose_defaults_to_required(self):
        profile = parse_job_description(
            "We need a developer experienced with React, Node.js and MongoDB. "
            "Go is a nice to have."
        )
        assert {"React", "Node.js", "MongoDB"} <= names(profile, "required_skills")
        assert names(profile, "preferred_skills") == {"Go"}


class TestEducationFieldTrimRegressions:
    """Regression: _trim_education_field looped forever on single trailer words.

    Found via the Phase 12 dataset — a JD containing '...in in...' produced a
    field capture of one trailer word, and rsplit on a single word returned the
    same string, so the trim never terminated (the API would hang).
    """

    def test_trim_single_trailer_words(self):
        for word in ("in", "or", "required", "a", "and"):
            assert _trim_education_field(word) == ""

    def test_trim_still_strips_trailing_qualifiers(self):
        assert _trim_education_field("Computer Science or related field required") == "Computer Science"
        assert _trim_education_field("Mechanical Engineering") == "Mechanical Engineering"

    def test_education_requirement_lone_preposition_returns_fast(self):
        # "degree in in ..." -> field capture starts at "in required"; before the
        # fix the trim loop never terminated on it.
        req = extract_education_requirement("Bachelor s degree in in required")
        assert req is not None
        assert req["field"] in (None, "")

    def test_full_parse_with_repeated_preposition(self):
        profile = parse_job_description(
            "We need a Masters degree in in Computer Science or equivalent. "
            "The role is a software engineer building backend systems."
        )
        assert profile.education is not None
        # "in in Computer Science" -> first "in" collapses to "" but the real
        # field must survive the trim.
        assert profile.education["field"] == "Computer Science"
