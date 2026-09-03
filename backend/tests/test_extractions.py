"""
Tests for Phase 4 — CV Information Extraction modules.
"""

import os
from pathlib import Path

import pytest

from app.nlp.candidate_builder import build_candidate_profile
from app.nlp.certification_extractor import extract_certifications
from app.nlp.education_extractor import extract_education
from app.nlp.experience_extractor import (
    calculate_total_years_experience,
    extract_experience,
)
from app.nlp.skill_extractor import extract_skills, extract_skills_from_list
from app.nlp.title_extractor import extract_titles

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"


# --- Skill Extraction Tests ---

class TestSkillExtractor:
    def test_extract_python(self):
        skills = extract_skills("I know Python and JavaScript")
        names = [s.name for s in skills]
        assert "Python" in names
        assert "JavaScript" in names

    def test_extract_aliases(self):
        skills = extract_skills("Experience with js, postgres, and reactjs")
        names = [s.name for s in skills]
        assert "JavaScript" in names
        assert "PostgreSQL" in names
        assert "React" in names

    def test_extract_from_list(self):
        skills = extract_skills_from_list("Python, React, PostgreSQL, Docker, AWS")
        names = [s.name for s in skills]
        assert len(names) == 5
        assert "Python" in names
        assert "React" in names
        assert "AWS" in names

    def test_deduplication(self):
        skills = extract_skills("Python and python3 for scripting in Python")
        py_count = sum(1 for s in skills if s.name == "Python")
        assert py_count == 1

    def test_category_detection(self):
        skills = extract_skills("React, PostgreSQL, Docker, AWS")
        by_name = {s.name: s.category for s in skills}
        assert by_name["React"] == "frontend"
        assert by_name["PostgreSQL"] == "database"
        assert by_name["Docker"] == "devops"
        assert by_name["AWS"] == "cloud"

    def test_no_skills(self):
        skills = extract_skills("I like to cook and travel")
        assert len(skills) == 0

    def test_sample_cv(self):
        sample_path = str(SAMPLE_DATA_DIR / "sample_cv.txt")
        if not os.path.exists(sample_path):
            pytest.skip("Sample CV not found")
        with open(sample_path) as f:
            text = f.read()
        skills = extract_skills(text)
        names = [s.name for s in skills]
        assert "Python" in names
        assert "React" in names
        assert "PostgreSQL" in names
        assert "Docker" in names
        assert "AWS" in names


# --- Experience Extraction Tests ---

class TestExperienceExtractor:
    def test_basic_range(self):
        text = "Software Engineer at Google | Jan 2020 - Dec 2022"
        exps = extract_experience(text)
        assert len(exps) == 1
        assert exps[0].role == "Software Engineer"
        assert exps[0].company == "Google"
        assert exps[0].start_date == "Jan 2020"
        assert exps[0].end_date == "Dec 2022"
        assert exps[0].duration_months == 36

    def test_present_date(self):
        text = "Dev at Startup | Mar 2021 - Present"
        exps = extract_experience(text)
        assert len(exps) == 1
        assert exps[0].end_date == "Present"
        assert exps[0].duration_months is not None

    def test_year_only(self):
        text = "Developer | 2019 - 2021"
        exps = extract_experience(text)
        assert len(exps) == 1
        assert exps[0].duration_months == 24

    def test_multiple_entries(self):
        text = """Software Engineer | TechCorp | 2021 - 2023
Developer | StartupXYZ | 2019 - 2021
Junior Dev | Agency | 2018 - 2019"""
        exps = extract_experience(text)
        assert len(exps) == 3
        assert exps[0].company == "TechCorp"

    def test_total_years(self):
        exps = extract_experience(
            "Engineer | Co1 | 2019 - 2021\nEngineer | Co2 | 2021 - 2023"
        )
        total = calculate_total_years_experience(exps)
        assert total == 4.0

    def test_empty_text(self):
        exps = extract_experience("")
        assert len(exps) == 0


# --- Education Extraction Tests ---

class TestEducationExtractor:
    def test_basic_degree(self):
        text = "Bachelor of Science in Computer Science, MIT, 2018"
        edu = extract_education(text)
        assert len(edu) == 1
        assert edu[0].degree == "Bachelor"
        assert edu[0].field_of_study == "Computer Science"
        assert edu[0].year == "2018"

    def test_masters(self):
        text = "M.S. Computer Science — Stanford University (2020)"
        edu = extract_education(text)
        assert len(edu) >= 1
        assert edu[0].field_of_study == "Computer Science"

    def test_no_degree(self):
        text = "University of California, Berkeley"
        edu = extract_education(text)
        assert len(edu) >= 1
        assert edu[0].institution is not None

    def test_multiple_entries(self):
        text = """M.S. Computer Science, Stanford, 2020
B.S. Mathematics, Berkeley, 2018"""
        edu = extract_education(text)
        assert len(edu) == 2

    def test_empty_text(self):
        edu = extract_education("")
        assert len(edu) == 0


# --- Certification Extraction Tests ---

class TestCertificationExtractor:
    def test_aws_cert(self):
        text = "AWS Certified Solutions Architect (2022)"
        certs = extract_certifications(text)
        assert len(certs) == 1
        assert certs[0].issuer == "AWS"
        assert certs[0].year == "2022"

    def test_pmp(self):
        text = "PMP - Project Management Professional, 2021"
        certs = extract_certifications(text)
        assert len(certs) == 1
        assert certs[0].issuer == "PMI"

    def test_multiple(self):
        text = """AWS Solutions Architect (2022)
Certified Kubernetes Administrator (2023)"""
        certs = extract_certifications(text)
        assert len(certs) == 2

    def test_empty(self):
        certs = extract_certifications("")
        assert len(certs) == 0


# --- Title Extraction Tests ---

class TestTitleExtractor:
    def test_basic_titles(self):
        text = "I worked as a Software Engineer and later became a Tech Lead"
        titles = extract_titles(text)
        assert "software engineer" in [t.lower() for t in titles]
        assert "tech lead" in [t.lower() for t in titles]

    def test_seniority(self):
        text = "Senior Developer at Company for 3 years"
        titles = extract_titles(text)
        assert any("senior" in t.lower() for t in titles)


# --- Full Profile Builder Tests ---

class TestCandidateBuilder:
    @pytest.fixture
    def sample_text(self):
        sample_path = str(SAMPLE_DATA_DIR / "sample_cv.txt")
        if not os.path.exists(sample_path):
            pytest.skip("Sample CV not found")
        with open(sample_path) as f:
            return f.read()

    def test_full_pipeline(self, sample_text):
        profile = build_candidate_profile(sample_text)

        # Identity
        assert profile.name == "Sarah Johnson"
        assert profile.email == "sarah.johnson@email.com"
        assert profile.phone is not None
        assert profile.linkedin is not None

        # Summary
        assert profile.summary is not None
        assert "Senior software engineer" in profile.summary

        # Skills
        skill_names = [s.name for s in profile.skills]
        assert "Python" in skill_names
        assert "React" in skill_names
        assert "PostgreSQL" in skill_names
        assert "Docker" in skill_names
        assert "AWS" in skill_names
        assert len(profile.skills) >= 5

        # Experience
        assert len(profile.work_experience) >= 2
        assert profile.total_years_experience is not None
        assert profile.total_years_experience >= 3.0

        # Education
        assert len(profile.education) >= 1

        # Titles
        assert len(profile.job_titles) >= 1

    def test_to_dict(self, sample_text):
        profile = build_candidate_profile(sample_text)
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert "skills" in d
        assert "work_experience" in d
        assert "education" in d
        assert isinstance(d["skills"], list)

    def test_empty_input(self):
        profile = build_candidate_profile("")
        assert profile.name is None
        assert len(profile.skills) == 0
        assert len(profile.work_experience) == 0
