"""
Matching Service — orchestrates the Phase 3-11 pipeline for the API layer.

Runs the full CV→Job pipeline and persists a Match + MatchExplanation when
inputs reference stored entities. Raw-text mode is stateless: no user auth
exists yet, so it never touches the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.ml.ats_checker import check_ats_friendliness
from app.ml.feature_extraction import build_feature_vector
from app.ml.model_scorer import score_features
from app.ml.semantic_matcher import compute_semantic_match
from app.models.candidate import CandidateProfile
from app.models.job import Job
from app.models.match import Match, MatchExplanation
from app.models.resume import Resume
from app.nlp.candidate_builder import build_candidate_profile
from app.nlp.experience_matcher import match_experience
from app.nlp.job_parser import parse_job_description
from app.nlp.skill_matcher import match_skills
from app.scoring.certification_matcher import match_certifications
from app.scoring.education_matcher import match_education
from app.scoring.improvement_engine import build_prioritized_improvements, build_skill_evidence
from app.scoring.matching_model import MatcherInputs, MatchResult, compute_match_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MatchInputError(ValueError):
    """Client provided incomplete or contradictory match inputs."""


class MatchNotFoundError(LookupError):
    """Referenced resume/job does not exist."""


# ---------------------------------------------------------------------------
# Pipeline output bundle
# ---------------------------------------------------------------------------


@dataclass
class PipelineOutput:
    """Everything the pipeline produced, including pre-score context."""

    result: MatchResult
    candidate_name: str | None
    job_title: str | None
    matched_skills: list[str]
    partial_skills: list[str]
    missing_skills: list[str]
    skill_matches: list[dict]
    experience: dict
    semantic: dict
    education: dict
    certifications: dict
    skill_evidence: list[dict] = field(default_factory=list)
    ml_details: dict | None = None
    prioritized_improvements: list[dict] = field(default_factory=list)
    ats_details: dict | None = None
    # Why ML scoring was skipped (model missing, feature-schema drift, ...).
    # Surfaced in the API response so degraded ML behaviour is observable
    # instead of a silent hybrid-only fallback.
    ml_failure: str | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    cv_text: str, job_text: str, weights: dict | None = None
) -> PipelineOutput:
    """Run the complete matching pipeline on raw text. Pure function of inputs."""
    candidate = build_candidate_profile(cv_text)
    job = parse_job_description(job_text)

    cand_names = [s.name for s in candidate.skills]
    req_names = [s.name for s in job.required_skills]
    pref_names = [s.name for s in job.preferred_skills]

    skill_match = match_skills(cand_names, req_names, pref_names)
    experience_match = match_experience(
        candidate.work_experience,
        candidate.skills,
        required_years=job.minimum_experience_years,
        seniority_level=job.experience_level,
        required_skills=req_names,
    )
    semantic_match = compute_semantic_match(
        cv_text,
        job_text,
        candidate_skills=cand_names,
        job_required_skills=req_names,
        job_preferred_skills=pref_names,
        cv_sections=_cv_sections(candidate),
        job_sections=_job_sections(job, req_names),
    )
    education_match = match_education(candidate.education, job.education)
    certification_match = match_certifications(
        [c.name for c in candidate.certifications], job.certifications
    )

    # Build the Phase 12 feature vector from the SAME engine outputs the
    # hybrid model consumes (train/serve consistency — the trained
    # classifier must see features identical to training-time ones).
    features = build_feature_vector(
        candidate=candidate,
        job=job,
        skill_match=skill_match,
        experience_match=experience_match,
        semantic_match=semantic_match,
        education_match=education_match,
        certification_match=certification_match,
        cv_text=cv_text,
    )
    ml_result = None
    ml_failure: str | None = None
    try:
        ml_result = score_features(features)
    except Exception as exc:  # noqa: BLE001 - ML is supplementary, never fatal
        logger.exception("ML scorer failed; falling back to hybrid-only")
        ml_failure = f"{type(exc).__name__}: {exc}"
    if ml_result is None and ml_failure is None:
        # score_features returns None (rather than raising) when the artifact
        # is absent or unloadable — distinguish that from a scoring error.
        from app.ml.model_scorer import MODEL_PATH

        if not MODEL_PATH.exists():
            ml_failure = f"model artifact missing at {MODEL_PATH}"
        else:
            ml_failure = f"model artifact failed to load from {MODEL_PATH}"

    # Phase 16 — evidence-graded assessment of each required skill.
    skill_evidence = build_skill_evidence(candidate, skill_match, experience_match)

    result = compute_match_score(
        MatcherInputs(
            skill_match=skill_match,
            experience_match=experience_match,
            semantic_match=semantic_match,
            education_match=education_match,
            certification_match=certification_match,
            ml_result=ml_result,
            job_title=job.job_title,
            skill_evidence=skill_evidence,
        ),
        weights=weights,
    )

    # Generate ML-powered prioritized improvements
    ml_explanation = None
    if ml_result and hasattr(ml_result, 'explanation') and ml_result.explanation:
        ml_explanation = ml_result.explanation.to_dict() if hasattr(ml_result.explanation, 'to_dict') else None
    prioritized_improvements = build_prioritized_improvements(
        components=[c.to_dict() for c in result.components],
        skill_evidence=skill_evidence,
        ml_explanation=ml_explanation,
        skill_match_result=skill_match,
    )

    # ATS friendliness check
    sections_found = []
    has_skills = False
    if candidate.sections:
        sections_found = [s.name for s in candidate.sections.sections]
        has_skills = "skills" in sections_found
    ats_result = check_ats_friendliness(
        cv_text=cv_text,
        job_text=job_text,
        sections_found=sections_found,
        has_skills_section=has_skills,
    )

    return PipelineOutput(
        result=result,
        candidate_name=candidate.name,
        job_title=job.job_title,
        matched_skills=skill_match.matched_skills,
        partial_skills=skill_match.partial_skills,
        missing_skills=skill_match.missing_skills,
        skill_matches=skill_match.to_dict()["matches"],
        experience=experience_match.to_dict(),
        semantic=semantic_match.to_dict(),
        education=education_match.to_dict(),
        certifications=certification_match.to_dict(),
        skill_evidence=[ev.to_dict() for ev in skill_evidence],
        ml_details=result.ml_details,
        ml_failure=ml_failure,
        prioritized_improvements=prioritized_improvements,
        ats_details=ats_result.to_dict(),
    )


def _cv_sections(candidate) -> dict[str, str]:
    """Rebuild coarse CV sections from the parsed candidate for semantic matching."""
    parts: dict[str, str] = {}
    if getattr(candidate, "summary", None):
        parts["summary"] = candidate.summary
    skills = getattr(candidate, "skills", None)
    if skills:
        parts["skills"] = ", ".join(s.name for s in skills)
    experience = getattr(candidate, "work_experience", None)
    if experience:
        parts["experience"] = " ".join(
            f"{w.role or ''} at {w.company or ''}".strip() for w in experience
        )
    return parts


def _job_sections(job, req_names: list[str]) -> dict[str, str]:
    """Rebuild coarse JD sections for semantic matching."""
    parts: dict[str, str] = {}
    responsibilities = getattr(job, "responsibilities", None)
    if responsibilities:
        parts["responsibilities"] = " ".join(responsibilities)
    if req_names:
        parts["requirements"] = ", ".join(req_names)
    return parts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def create_match_from_texts(
    db: Session, cv_text: str, job_text: str, weights: dict | None = None
) -> tuple[Match, PipelineOutput]:
    """
    Match raw texts AND persist the result.

    Creates a deterministic demo user plus Resume/Job/CandidateProfile rows
    so the FK graph is satisfied until authentication exists (Phase 20).
    """
    from app.models.skill import CandidateSkill, Skill
    from app.models.user import User

    output = run_pipeline(cv_text, job_text, weights=weights)

    user = db.query(User).filter(User.email == "demo@career-match.local").first()
    if user is None:
        user = User(
            email="demo@career-match.local",
            name="Demo User",
            hashed_password="!oauth-not-implemented",
        )
        db.add(user)
        db.flush()

    resume = Resume(
        user_id=user.id, filename="inline-cv.txt", raw_text=cv_text, status="parsed"
    )
    db.add(resume)
    db.flush()

    job_row = Job(title=output.job_title or "Untitled Job", description=job_text)
    db.add(job_row)
    db.flush()

    candidate = CandidateProfile(
        resume_id=resume.id,
        name=output.candidate_name,
        years_experience=_total_years(cv_text),
        extracted_data={"source": "api"},
    )
    db.add(candidate)
    db.flush()

    # Persist candidate skills (only those present in the seeded Skill table).
    parsed = build_candidate_profile(cv_text)
    for extracted in parsed.skills:
        skill_row = db.query(Skill).filter(Skill.name == extracted.name).first()
        if skill_row is None:
            continue
        db.add(
            CandidateSkill(
                candidate_id=candidate.id,
                skill_id=skill_row.id,
                confidence=extracted.confidence,
            )
        )

    match_row = _persist_match(db, candidate.id, job_row.id, output)
    db.commit()
    db.refresh(match_row)
    return match_row, output


def create_match_from_entities(
    db: Session, resume_id: int, job_id: int, weights: dict | None = None
) -> tuple[Match, PipelineOutput]:
    """Match a stored Resume against a stored Job and persist the result."""
    resume = db.get(Resume, resume_id)
    if resume is None:
        raise MatchNotFoundError(f"Resume {resume_id} not found")
    job_row = db.get(Job, job_id)
    if job_row is None:
        raise MatchNotFoundError(f"Job {job_id} not found")

    cv_text = resume.raw_text or ""
    if not cv_text.strip():
        raise MatchInputError(f"Resume {resume_id} has no extracted text")

    output = run_pipeline(cv_text, job_row.description, weights=weights)

    candidate = (
        db.query(CandidateProfile).filter(CandidateProfile.resume_id == resume.id).first()
    )
    if candidate is None:
        candidate = CandidateProfile(
            resume_id=resume.id,
            name=output.candidate_name,
            extracted_data={"source": "api"},
        )
        db.add(candidate)
        db.flush()

    match_row = _persist_match(db, candidate.id, job_row.id, output)
    db.commit()
    db.refresh(match_row)
    return match_row, output


def _total_years(cv_text: str) -> float | None:
    parsed = build_candidate_profile(cv_text)
    return parsed.total_years_experience


def _persist_match(db: Session, candidate_id: int, job_id: int, output: PipelineOutput) -> Match:
    result = output.result
    comp = {c.name: c.raw_score for c in result.components}
    match_row = Match(
        candidate_id=candidate_id,
        job_id=job_id,
        overall_score=result.overall_percent,
        semantic_score=comp.get("semantic", 0.0) * 100,
        skills_score=comp.get("skills", 0.0) * 100,
        experience_score=comp.get("experience", 0.0) * 100,
        education_score=comp.get("education", 0.0) * 100,
        model_version=result.model_version,
        feature_values={
            "components": {c.name: c.to_dict() for c in result.components},
            "weights": result.weights,
            "band": result.band,
            "ml_details": result.ml_details,
        },
    )
    db.add(match_row)
    db.flush()

    db.add(
        MatchExplanation(
            match_id=match_row.id,
            matched_skills=output.matched_skills,
            missing_skills=output.missing_skills,
            partial_skills=output.partial_skills,
            recommendations=result.recommendations,
            positive_factors=result.positive_factors,
            negative_factors=result.negative_factors,
        )
    )
    db.flush()
    return match_row
