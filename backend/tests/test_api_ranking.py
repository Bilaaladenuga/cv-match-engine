"""
Tests for the Phase 15 ranking endpoint (POST /api/jobs/{id}/rank-candidates).

Uses SQLite in-memory + the real pipeline. Embedding calls are exercised
with the real MiniLM model if available (these tests tolerate a missing
artifact/model only where the production path does).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models import candidate, job, resume, skill, user  # noqa: F401
from app.models.resume import Resume
from app.models.user import User

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CV_STRONG = """Sarah Strong
Senior Python developer with ten years of experience building backend systems.
Skills: Python, FastAPI, PostgreSQL, Docker, AWS, pytest.
Experience:
Senior Backend Engineer, Acme (2018-2026): led migration to FastAPI, scaled PostgreSQL.
Education: BSc Computer Science, 2014.
Certifications: AWS Certified Developer."""

CV_MEDIUM = """Sam Medium
Python developer with four years of experience.
Skills: Python, FastAPI, PostgreSQL.
Experience:
Backend Developer, Beta Ltd (2021-2025): REST APIs with FastAPI.
Education: BSc Computer Science, 2020."""

CV_WEAK = """Wendy Weak
Junior frontend developer, two years of experience.
Skills: React, CSS, Figma.
Education: BSc Design, 2022."""

JOB_TEXT = """Senior Backend Engineer
We need a Python developer with 5+ years experience.
Required: Python, FastAPI, PostgreSQL, Docker, AWS.
BSc in Computer Science required."""


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    test_session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def _override_get_db():
        """Yield a session per request (matches get_db's generator shape)."""
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db

    # Seed a user, job, and three resumes
    db = test_session_factory()
    try:
        user = User(email="recruiter@test.local", name="Recruiter", hashed_password="x")
        db.add(user)
        db.flush()
        from app.models.job import Job

        job_row = Job(title="Senior Backend Engineer", description=JOB_TEXT)
        db.add(job_row)
        db.flush()
        for i, text in enumerate((CV_STRONG, CV_MEDIUM, CV_WEAK), start=1):
            db.add(Resume(user_id=user.id, filename=f"cv{i}.txt", raw_text=text, status="parsed"))
        db.commit()
        job_id = job_row.id
    finally:
        db.close()

    from app.ml.model_scorer import reset_model_cache

    reset_model_cache()
    client = TestClient(app)
    client.job_id = job_id  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRankingEndpoint:
    def test_ranks_all_candidates_with_stamps(self, client):
        r = client.post(
            f"/api/jobs/{client.job_id}/rank-candidates",
            json={"resume_ids": [1, 2, 3]},
        )
        assert r.status_code == 200, r.text
        body = r.json()

        # Version stamp + reproducibility metadata
        assert body["model_version"].startswith("match-model-")
        assert body["weights"], "weights must be recorded for reproducibility"
        assert body["created_at"]
        assert body["ranking_run_id"]
        assert body["job_id"] == client.job_id

        ranked = body["ranked"]
        assert len(ranked) == 3
        assert [c["rank"] for c in ranked] == [1, 2, 3]

        # Ordering: score desc
        scores = [c["overall_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

        # The backend-heavy CV should beat the frontend CV for this job
        assert ranked[0]["candidate_name"] in ("Sarah Strong", "Sam Medium")
        assert ranked[-1]["candidate_name"] == "Wendy Weak"

        # Explainability surfaces per row
        for c in ranked:
            assert c["match_id"] > 0
            assert isinstance(c["matched_skills"], list)
            assert isinstance(c["missing_skills"], list)
            assert c["band"]

        # Decision-support framing
        assert "not hiring recommendations" in body["disclaimer"]

    def test_matches_are_persisted(self, client):
        r = client.post(
            f"/api/jobs/{client.job_id}/rank-candidates",
            json={"resume_ids": [1, 2, 3]},
        )
        body = r.json()
        match_ids = [c["match_id"] for c in body["ranked"]]
        # unique persisted matches
        assert len(set(match_ids)) == 3

    def test_deterministic_tiebreak(self, client):
        """Identical inputs -> identical ordering (no DB-order dependence)."""
        r1 = client.post(
            f"/api/jobs/{client.job_id}/rank-candidates",
            json={"resume_ids": [1, 2, 3]},
        ).json()
        r2 = client.post(
            f"/api/jobs/{client.job_id}/rank-candidates",
            json={"resume_ids": [3, 2, 1]},  # reversed request order
        ).json()
        names1 = [c["candidate_name"] for c in r1["ranked"]]
        names2 = [c["candidate_name"] for c in r2["ranked"]]
        assert names1 == names2
        assert [c["rank"] for c in r2["ranked"]] == [1, 2, 3]

    def test_failed_resumes_are_isolated(self, client):
        """A missing/empty resume must not kill the whole run."""
        r = client.post(
            f"/api/jobs/{client.job_id}/rank-candidates",
            json={"resume_ids": [1, 999]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["failed_resume_ids"] == [999]
        assert len(body["ranked"]) == 1


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestRankingErrors:
    def test_empty_resume_ids_rejected(self, client):
        # FastAPI/Pydantic reject min_length=1 at the schema layer with 422
        # before the handler (and its 400 mapping) can run.
        r = client.post(f"/api/jobs/{client.job_id}/rank-candidates", json={"resume_ids": []})
        assert r.status_code == 422

    def test_duplicate_resume_ids_400(self, client):
        r = client.post(
            f"/api/jobs/{client.job_id}/rank-candidates", json={"resume_ids": [1, 1]}
        )
        assert r.status_code == 400

    def test_unknown_job_404(self, client):
        r = client.post("/api/jobs/99999/rank-candidates", json={"resume_ids": [1]})
        assert r.status_code == 404

    def test_all_resumes_missing_still_200_with_failures(self, client):
        r = client.post(
            f"/api/jobs/{client.job_id}/rank-candidates", json={"resume_ids": [998, 999]}
        )
        assert r.status_code == 200
        assert r.json()["ranked"] == []
        assert len(r.json()["failed_resume_ids"]) == 2
