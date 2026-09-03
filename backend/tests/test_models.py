"""
Unit tests for database ORM models.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.models import (
    CandidateProfile,
    CandidateSkill,
    Job,
    JobRequirement,
    Match,
    MatchExplanation,
    Resume,
    Skill,
    User,
)


@pytest.fixture(scope="module")
def engine():
    """Create an in-memory SQLite engine for testing."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    """Provide a transactional session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    sess = Session(bind=connection)
    yield sess
    sess.close()
    transaction.rollback()
    connection.close()


class TestUserModel:
    def test_create_user(self, session):
        user = User(
            email="test@example.com",
            name="Test User",
            hashed_password="hashed_pw_123",
        )
        session.add(user)
        session.flush()
        assert user.id is not None
        assert user.email == "test@example.com"

    def test_user_repr(self, session):
        user = User(email="a@b.com", name="A", hashed_password="x")
        session.add(user)
        session.flush()
        assert "a@b.com" in repr(user)


class TestResumeModel:
    def test_create_resume(self, session):
        user = User(email="u@e.com", name="U", hashed_password="h")
        session.add(user)
        session.flush()

        resume = Resume(user_id=user.id, filename="cv.pdf", status="uploaded")
        session.add(resume)
        session.flush()
        assert resume.id is not None
        assert resume.status == "uploaded"

    def test_resume_default_status(self, session):
        user = User(email="u2@e.com", name="U2", hashed_password="h")
        session.add(user)
        session.flush()

        resume = Resume(user_id=user.id, filename="cv.docx")
        session.add(resume)
        session.flush()
        assert resume.status == "uploaded"


class TestCandidateProfile:
    def test_create_candidate(self, session):
        user = User(email="c@e.com", name="C", hashed_password="h")
        session.add(user)
        session.flush()

        resume = Resume(user_id=user.id, filename="cv.pdf", status="parsed")
        session.add(resume)
        session.flush()

        profile = CandidateProfile(
            resume_id=resume.id,
            name="John Doe",
            summary="Senior developer",
            years_experience=5.0,
            education={"degree": "BS CS"},
        )
        session.add(profile)
        session.flush()
        assert profile.id is not None
        assert profile.years_experience == 5.0


class TestSkillModel:
    def test_create_skill(self, session):
        skill = Skill(name="Python", category="programming")
        session.add(skill)
        session.flush()
        assert skill.id is not None
        assert skill.category == "programming"

    def test_candidate_skill(self, session):
        user = User(email="cs@e.com", name="CS", hashed_password="h")
        session.add(user)
        session.flush()

        resume = Resume(user_id=user.id, filename="cv.pdf")
        session.add(resume)
        session.flush()

        profile = CandidateProfile(resume_id=resume.id, name="Test")
        session.add(profile)
        session.flush()

        skill = Skill(name="React", category="frontend")
        session.add(skill)
        session.flush()

        cs = CandidateSkill(
            candidate_id=profile.id, skill_id=skill.id,
            confidence=0.9, years_experience=3.0,
        )
        session.add(cs)
        session.flush()
        assert cs.confidence == 0.9


class TestJobModel:
    def test_create_job(self, session):
        job = Job(title="Backend Developer", description="Looking for Python dev")
        session.add(job)
        session.flush()
        assert job.id is not None

    def test_job_requirement(self, session):
        skill = Skill(name="Docker", category="devops")
        session.add(skill)
        session.flush()

        job = Job(title="DevOps", description="Need Docker experience")
        session.add(job)
        session.flush()

        req = JobRequirement(
            job_id=job.id, skill_id=skill.id,
            required=True, importance=0.8, minimum_experience=2.0,
        )
        session.add(req)
        session.flush()
        assert req.required is True
        assert req.importance == 0.8


class TestMatchModel:
    def test_create_match(self, session):
        user = User(email="m@e.com", name="M", hashed_password="h")
        session.add(user)
        session.flush()

        resume = Resume(user_id=user.id, filename="cv.pdf")
        session.add(resume)
        session.flush()

        profile = CandidateProfile(resume_id=resume.id, name="Match Test")
        session.add(profile)
        session.flush()

        job = Job(title="Engineer", description="Build things")
        session.add(job)
        session.flush()

        match = Match(
            candidate_id=profile.id,
            job_id=job.id,
            overall_score=78.5,
            semantic_score=84.0,
            skills_score=80.0,
            experience_score=70.0,
            education_score=90.0,
            model_version="match-model-v1.0",
        )
        session.add(match)
        session.flush()
        assert match.overall_score == 78.5

    def test_match_explanation(self, session):
        user = User(email="me@e.com", name="ME", hashed_password="h")
        session.add(user)
        session.flush()

        resume = Resume(user_id=user.id, filename="cv.pdf")
        session.add(resume)
        session.flush()

        profile = CandidateProfile(resume_id=resume.id, name="Explain Test")
        session.add(profile)
        session.flush()

        job = Job(title="Role", description="Desc")
        session.add(job)
        session.flush()

        match = Match(
            candidate_id=profile.id,
            job_id=job.id,
            overall_score=82.0,
            semantic_score=85.0,
            skills_score=88.0,
            experience_score=72.0,
            education_score=90.0,
            model_version="v1.0",
        )
        session.add(match)
        session.flush()

        explanation = MatchExplanation(
            match_id=match.id,
            matched_skills=["Python", "React"],
            missing_skills=["Docker"],
            partial_skills=["AWS"],
            recommendations=["Learn Docker"],
            positive_factors=["Strong Python skills"],
            negative_factors=["Missing cloud experience"],
        )
        session.add(explanation)
        session.flush()
        assert len(explanation.matched_skills) == 2
        assert "Docker" in explanation.missing_skills
