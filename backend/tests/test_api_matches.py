"""Tests for POST /api/matches (Phase 11-era API slice).

Uses SQLite in-memory with Base.metadata.create_all so tests run without
live PostgreSQL. JSON columns fall back to SQLAlchemy's generic JSON type,
which works on SQLite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models import candidate, job, resume, skill, user  # noqa: F401
from app.models.job import Job
from app.models.resume import Resume
from app.models.user import User

SAMPLE_CV = (
    "Sarah Johnson\nsarah.johnson@email.com\n\n"
    "SUMMARY\nSenior software engineer with 8 years of experience building "
    "web applications with Python, React and PostgreSQL.\n\n"
    "SKILLS\nPython, JavaScript, TypeScript, React, Next.js, HTML, CSS, "
    "PostgreSQL, Docker, AWS, Git\n\n"
    "EXPERIENCE\n\n"
    "Senior Software Engineer\nTechCorp Inc.\n2021 - Present\n"
    "Built frontend features with React and TypeScript.\n\n"
    "Software Developer\nStartupXYZ\n2019 - 2021\n"
    "Developed REST APIs with Python and PostgreSQL.\n\n"
    "EDUCATION\nB.S. Computer Science, State University, 2018\n"
)

SAMPLE_JD = (
    "Senior Frontend Engineer\n\n"
    "Cloudly Systems is looking for a senior frontend engineer.\n\n"
    "Required Qualifications:\n"
    "- 4+ years of experience\n"
    "- React\n- TypeScript\n- Next.js\n\n"
    "Preferred Skills:\n- AWS\n- Docker\n- GraphQL\n\n"
    "Education: Bachelor's degree in Computer Science.\n"
)


@pytest.fixture()
def client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def _override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    yield TestClient(app), session_factory
    app.dependency_overrides.clear()


def _seed(db):
    new_user = User(email="u@test.local", name="T", hashed_password="x")
    db.add(new_user)
    db.flush()
    new_resume = Resume(
        user_id=new_user.id, filename="cv.txt", raw_text=SAMPLE_CV, status="parsed"
    )
    db.add(new_resume)
    new_job = Job(title="Senior Frontend Engineer", description=SAMPLE_JD)
    db.add(new_job)
    db.commit()
    return new_resume.id, new_job.id


def test_text_mode_returns_full_result(client):
    http, _ = client
    resp = http.post(
        "/api/matches",
        json={"cv_text": SAMPLE_CV, "job_text": SAMPLE_JD},
    )
    assert resp.status_code == 201
    body = resp.json()
    # Stateless mode: no persistence, no match row.
    assert body["match_id"] is None
    assert body["model_version"].startswith("match-model-v")
    assert 0 <= body["overall_percent"] <= 100
    assert body["band"]
    expected_components = {
        "skills", "semantic", "experience", "education", "certifications",
    }
    if body["ml_details"] is not None:
        expected_components.add("ml_model")
    assert {c["name"] for c in body["components"]} == expected_components
    assert body["disclaimer"].startswith("This score is a model-estimated")
    # detail blocks are attached
    assert isinstance(body["skill_matches"], list) and body["skill_matches"]
    assert body["experience"]["summary"]["total_experience_years"] is not None
    # ml_details present only when the trained artifact exists on this machine
    if body["ml_details"] is not None:
        assert body["model_version"].startswith("match-model-v0.1+")
        assert "probabilities" in body["ml_details"]


def test_entity_mode_persists_and_scores(client):
    http, session_factory = client
    db = session_factory()
    resume_id, job_id = _seed(db)
    db.close()

    resp = http.post(
        "/api/matches", json={"resume_id": resume_id, "job_id": job_id}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["match_id"] is not None

    # persisted
    db = session_factory()
    from app.models.match import Match, MatchExplanation

    match_row = db.get(Match, body["match_id"])
    assert match_row is not None
    assert match_row.model_version == body["model_version"]
    # ML share (0.25) is applied only when the trained artifact exists;
    # otherwise the pure hybrid weights apply.
    expected_skills = 0.40 * (1 - 0.25) if body["ml_details"] else 0.40
    assert match_row.feature_values["weights"]["skills"] == pytest.approx(expected_skills)
    exp = db.query(MatchExplanation).filter_by(match_id=match_row.id).one()
    assert isinstance(exp.matched_skills, list)
    db.close()


def test_text_mode_is_stateless(client):
    """Raw-text mode persists NOTHING (privacy-first, no accounts).

    The demo-user persistence path exists only behind the opt-in flag for
    self-hosters; the default product stores no CV or report server-side.
    """
    http, session_factory = client
    resp = http.post(
        "/api/matches", json={"cv_text": SAMPLE_CV, "job_text": SAMPLE_JD}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["match_id"] is None

    db = session_factory()
    from app.models.job import Job as JobRow
    from app.models.match import Match
    from app.models.resume import Resume as ResumeRow

    assert db.query(Match).count() == 0
    assert db.query(ResumeRow).count() == 0
    assert db.query(JobRow).count() == 0
    assert db.query(User).filter(User.email == "demo@career-match.local").count() == 0
    db.close()


def test_both_modes_rejected(client):
    http, session_factory = client
    db = session_factory()
    resume_id, job_id = _seed(db)
    db.close()
    resp = http.post(
        "/api/matches",
        json={
            "resume_id": resume_id, "job_id": job_id,
            "cv_text": "x", "job_text": "y",
        },
    )
    assert resp.status_code == 400


def test_neither_mode_rejected(client):
    http, _ = client
    resp = http.post("/api/matches", json={})
    assert resp.status_code == 400


def test_half_mode_rejected(client):
    http, _ = client
    resp = http.post("/api/matches", json={"resume_id": 1})
    assert resp.status_code == 400
    resp = http.post("/api/matches", json={"job_id": 1})
    assert resp.status_code == 400


def test_missing_entities_return_404(client):
    http, _ = client
    resp = http.post("/api/matches", json={"resume_id": 999, "job_id": 999})
    assert resp.status_code == 404


def test_invalid_weights_rejected(client):
    http, session_factory = client
    db = session_factory()
    resume_id, job_id = _seed(db)
    db.close()
    resp = http.post(
        "/api/matches",
        json={
            "resume_id": resume_id, "job_id": job_id,
            "weights": {"skills": 0.4, "bogus": 0.6},
        },
    )
    assert resp.status_code == 422


class DeadSession:
    """Session stand-in: every DB touch raises OperationalError."""

    def _fail(self, *a, **k):
        raise OperationalError(
            "SELECT", {}, Exception("connection refused")
        )

    query = _fail
    get = _fail
    add = _fail
    flush = _fail
    commit = _fail
    refresh = _fail
    execute = _fail

    def rollback(self):  # must succeed: the route rolls back before degrading
        pass

    def close(self):
        pass


def test_text_mode_works_when_db_down(client):
    """Stateless mode never touches the DB: a dead database is irrelevant.

    Regression test for the Phase 17 discovery (raw-text analysis used to
    500 when PostgreSQL was unreachable). The analysis is self-contained.
    """
    http, _ = client

    def _dead_get_db():
        db = DeadSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _dead_get_db
    try:
        resp = http.post(
            "/api/matches", json={"cv_text": SAMPLE_CV, "job_text": SAMPLE_JD}
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 201
    body = resp.json()
    assert body["match_id"] is None
    assert body["overall_percent"] >= 0
    assert body["disclaimer"]
    # The pipeline result must be identical in structure to the persisted path.
    assert isinstance(body["skill_evidence"], list)


def test_entity_mode_fails_when_db_down(client):
    """Entity mode REQUIRES the database: 503, not a silent 500."""
    http, _ = client

    def _dead_get_db():
        db = DeadSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _dead_get_db
    try:
        resp = http.post("/api/matches", json={"resume_id": 1, "job_id": 1})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 503


def test_custom_weights_change_score(client):
    http, session_factory = client
    db = session_factory()
    resume_id, job_id = _seed(db)
    db.close()

    default = http.post(
        "/api/matches", json={"resume_id": resume_id, "job_id": job_id}
    ).json()
    resp2 = http.post(
        "/api/matches",
        json={
            "resume_id": resume_id, "job_id": job_id,
            "weights": {
                "skills": 0.0, "semantic": 0.7, "experience": 0.2,
                "education": 0.1, "certifications": 0.0,
            },
        },
    )
    assert resp2.status_code == 201
    custom = resp2.json()
    scale = 1 - 0.25 if custom["ml_details"] else 1
    assert custom["weights"]["semantic"] == pytest.approx(0.7 * scale)
    assert custom["overall_score"] != default["overall_score"]


def test_ml_failure_is_reported_when_scorer_raises(client, monkeypatch):
    """When ML scoring raises, the report still ships — and now says WHY.

    Regression guard for the production issue where ml_details came back
    null with no observable cause.
    """
    def _boom(features):
        raise RuntimeError("boom for test")

    monkeypatch.setattr(
        "app.services.matching_service.score_features", _boom
    )

    http, _ = client
    resp = http.post(
        "/api/matches",
        json={"cv_text": SAMPLE_CV, "job_text": SAMPLE_JD},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["ml_details"] is None
    assert body["ml_failure"] is not None
    assert "boom for test" in body["ml_failure"]
    # Hybrid-only weights: no ml_model component, no re-normalisation surprise
    assert "ml_model" not in {c["name"] for c in body["components"]}


def test_ml_failure_distinguishes_missing_artifact(client, monkeypatch):
    """A missing/unloadable artifact is reported differently from a scoring error."""
    monkeypatch.setattr(
        "app.services.matching_service.score_features", lambda features: None
    )

    http, _ = client
    resp = http.post(
        "/api/matches",
        json={"cv_text": SAMPLE_CV, "job_text": SAMPLE_JD},
    )
    body = resp.json()
    assert resp.status_code == 201
    assert body["ml_details"] is None
    # On this machine the artifact exists, so the reason must say loadable
    assert body["ml_failure"] is not None
    assert "artifact" in body["ml_failure"]
