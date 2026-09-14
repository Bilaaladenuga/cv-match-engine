"""
Tests for history retrieval endpoints (GET /api/history, GET
/api/matches/{match_id}).

Seeds real persisted matches by running the actual pipeline once, then
exercises the read paths.
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
from app.models.job import Job
from app.models.resume import Resume
from app.models.user import User

CV = """Sarah Strong
Senior Python developer with ten years of experience building backend systems.
Skills: Python, FastAPI, PostgreSQL, Docker, AWS.
Experience:
Senior Backend Engineer, Acme (2018-2026): led migration to FastAPI.
Education: BSc Computer Science, 2014.
Certifications: AWS Certified Developer."""

JOB_TEXT = """Senior Backend Engineer
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
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db

    # Seed user + resume + job, then create one real match via the pipeline
    db = test_session_factory()
    match_id = None
    try:
        user = User(email="h@test.local", name="H", hashed_password="x")
        db.add(user)
        db.flush()
        db.add(Resume(user_id=user.id, filename="cv.txt", raw_text=CV, status="parsed"))
        db.add(Job(title="Senior Backend Engineer", description=JOB_TEXT))
        db.commit()

        from app.services.matching_service import create_match_from_texts

        match_row, _ = create_match_from_texts(db, CV, JOB_TEXT)
        match_id = match_row.id
    finally:
        db.close()

    from app.ml.model_scorer import reset_model_cache

    reset_model_cache()
    client = TestClient(app)
    client.match_id = match_id  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()


class TestHistoryList:
    def test_lists_recent_matches_newest_first(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200
        items = r.json()
        assert len(items) >= 1
        created = [i["created_at"] for i in items if i["created_at"]]
        assert created == sorted(created, reverse=True)

    def test_list_entries_have_display_context(self, client):
        items = client.get("/api/history").json()
        first = items[0]
        assert first["job_title"] == "Senior Backend Engineer"
        assert first["candidate_name"]
        assert first["model_version"].startswith("match-model-")
        assert isinstance(first["matched_skills"], list)

    def test_list_does_not_leak_raw_documents(self, client):
        body = client.get("/api/history").json()
        text = str(body)
        assert "Senior Backend Engineer, Acme (2018-2026)" not in text
        assert "raw_text" not in text and "description" not in text

    def test_pagination_params(self, client):
        r = client.get("/api/history", params={"limit": 1, "offset": 0})
        assert r.status_code == 200
        assert len(r.json()) <= 1
        bad = client.get("/api/history", params={"limit": 0})
        assert bad.status_code == 422


class TestMatchDetail:
    def test_round_trip_of_stored_analysis(self, client):
        r = client.get(f"/api/matches/{client.match_id}")
        assert r.status_code == 200
        body = r.json()

        assert body["match_id"] == client.match_id
        assert body["job_title"] == "Senior Backend Engineer"
        assert body["model_version"].startswith("match-model-")
        assert body["band"]

        # Component scores persisted (0-100 scale)
        comps = body["component_scores"]
        assert set(comps) == {"skills", "semantic", "experience", "education"}
        assert all(0.0 <= v <= 100.0 for v in comps.values())

        # Explanation lists round-trip
        exp = body["explanation"]
        assert "Python" in exp["matched_skills"]
        assert isinstance(exp["recommendations"], list)

    def test_unknown_match_404(self, client):
        r = client.get("/api/matches/999999")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()
