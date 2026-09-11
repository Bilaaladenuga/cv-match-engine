"""
Feature Extraction — Phase 12

Canonical feature builder for the learned matching model.

CRITICAL DESIGN RULE (train/serve consistency):
    Features are computed by the SAME engines the API uses at inference
    (parsers, taxonomy, skill/experience/semantic/education/cert matchers).
    There is no separate "training-time" implementation, so the model can
    never drift from the features it sees in production.

    Two entry points:
        - extract_features(cv_text, job_text)  — batch/training path: parses
          raw text, then delegates to build_feature_vector.
        - build_feature_vector(...)            — inference path: the API
          already ran the engines; it hands over their outputs and gets the
          identical feature dict without re-parsing.

Feature vector (one CV–JD pair):

    skill_overlap_ratio          fraction of unique JD skills matched (any class)
    required_skill_coverage      fraction of REQUIRED skills fully matched
    required_plus_partial        fraction of required matched or partial
    preferred_skill_coverage     fraction of preferred skills matched/partial
    semantic_similarity          Phase 10 overall semantic score (0-1)
    experience_gap_years         candidate total years minus required years
    experience_score             Phase 9 overall experience score (0-1)
    experience_data_available    1.0 if any experience signal was extracted
    seniority_match              1.0 if candidate level >= required level
    education_level_score        Phase 11 education level component
    education_field_score        Phase 11 education field component
    certification_match_ratio    fraction of required certifications held
    job_title_similarity         cosine(job title, best candidate title)
    n_candidate_skills           meta: candidate skill count
    n_required_skills            meta: required skill count
    n_preferred_skills           meta: preferred skill count
"""

from __future__ import annotations

import logging

from app.ml.embeddings import cosine_similarity, embed_texts
from app.ml.semantic_matcher import compute_semantic_match
from app.nlp.candidate_builder import build_candidate_profile
from app.nlp.experience_matcher import match_experience
from app.nlp.job_parser import parse_job_description
from app.nlp.skill_matcher import match_skills
from app.scoring.certification_matcher import match_certifications
from app.scoring.education_matcher import match_education

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = [
    "skill_overlap_ratio",
    "required_skill_coverage",
    "required_plus_partial",
    "preferred_skill_coverage",
    "semantic_similarity",
    "experience_gap_years",
    "experience_score",
    "experience_data_available",
    "seniority_match",
    "education_level_score",
    "education_field_score",
    "certification_match_ratio",
    "job_title_similarity",
    "n_candidate_skills",
    "n_required_skills",
    "n_preferred_skills",
]

# Seniority ranking used for the seniority_match feature
_SENIORITY_RANK = {
    None: 0, "junior": 1, "mid": 2, "intermediate": 2,
    "senior": 3, "lead": 4, "principal": 5,
}


def extract_features(cv_text: str, job_text: str) -> dict[str, float]:
    """
    Compute the full feature vector for one (CV, job description) pair.

    Returns a flat dict keyed by FEATURE_NAMES. Numeric meta features are
    floats for direct use with scikit-learn.
    """
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
    )
    education_match = match_education(candidate.education, job.education)
    certification_match = match_certifications(
        [c.name for c in candidate.certifications], job.certifications
    )

    return build_feature_vector(
        candidate=candidate,
        job=job,
        skill_match=skill_match,
        experience_match=experience_match,
        semantic_match=semantic_match,
        education_match=education_match,
        certification_match=certification_match,
        cv_text=cv_text,
    )


def build_feature_vector(
    *,
    candidate,
    job,
    skill_match,
    experience_match,
    semantic_match,
    education_match,
    certification_match,
    cv_text: str = "",
) -> dict[str, float]:
    """
    Assemble the feature dict from already-computed engine outputs.

    This is the single source of truth for the feature schema — the batch
    trainer and the live API both land here, so inference features are
    byte-identical to training features by construction.
    """
    cand_names = [s.name for s in candidate.skills]
    req_names = [s.name for s in job.required_skills]
    pref_names = [s.name for s in job.preferred_skills]
    all_jd_skills = list(dict.fromkeys(req_names + pref_names))

    # --- Skill features ----------------------------------------------------
    matched_set = set(skill_match.matched_skills) | set(skill_match.partial_skills)
    skill_overlap = (
        len(matched_set) / len(all_jd_skills) if all_jd_skills else 0.0
    )
    preferred_cov = (
        len(matched_set & set(pref_names)) / len(pref_names) if pref_names else 0.0
    )

    # --- Experience features -------------------------------------------------
    cand_years = candidate.total_years_experience or 0.0
    required_years = job.minimum_experience_years or 0.0
    experience_known = bool(candidate.work_experience) or cand_years > 0
    # Unknown candidate years must NOT look like a large negative gap (that
    # would teach the model "missing data = bad candidate"); 0.0 is neutral
    # and `experience_data_available` lets the model gate on missingness.
    experience_gap = (cand_years - required_years) if experience_known else 0.0

    # Candidate's implied level from total years (mirrors experience matcher)
    if cand_years >= 10:
        cand_level_rank = 5
    elif cand_years >= 8:
        cand_level_rank = 4
    elif cand_years >= 5:
        cand_level_rank = 3
    elif cand_years >= 2:
        cand_level_rank = 2
    elif cand_years > 0:
        cand_level_rank = 1
    else:
        cand_level_rank = 0
    required_rank = _SENIORITY_RANK.get(job.experience_level, 0)
    seniority_match = 1.0 if cand_level_rank >= required_rank else 0.0

    # --- Job title similarity ------------------------------------------------
    title_sim = _job_title_similarity(candidate.job_titles, job.job_title, cv_text)

    return {
        "skill_overlap_ratio": round(skill_overlap, 6),
        "required_skill_coverage": round(skill_match.required_coverage, 6),
        "required_plus_partial": round(skill_match.required_plus_partial, 6),
        "preferred_skill_coverage": round(preferred_cov, 6),
        "semantic_similarity": round(semantic_match.raw_score, 6),
        "experience_gap_years": round(experience_gap, 4),
        "experience_score": round(experience_match.overall_score, 6),
        "experience_data_available": 1.0 if experience_known else 0.0,
        "seniority_match": seniority_match,
        "education_level_score": round(education_match.level_score, 6),
        "education_field_score": round(education_match.field_score, 6),
        "certification_match_ratio": round(certification_match.score, 6),
        "job_title_similarity": round(title_sim, 6),
        "n_candidate_skills": float(len(cand_names)),
        "n_required_skills": float(len(req_names)),
        "n_preferred_skills": float(len(pref_names)),
    }


def _job_title_similarity(
    candidate_titles: list[str], job_title: str | None, cv_text: str
) -> float:
    """Cosine between the JD title and the best candidate title (or the CV)."""
    if not job_title:
        return 0.0
    candidates = [t for t in (candidate_titles or []) if t]
    if not candidates:
        # Fall back to the first 200 chars of the CV (name/summary region)
        candidates = [cv_text[:200]] if cv_text else []
    if not candidates:
        return 0.0
    try:
        vecs = embed_texts([job_title] + candidates)
        job_vec, title_vecs = vecs[0], vecs[1:]
        return max(cosine_similarity(job_vec, t) for t in title_vecs)
    except Exception as exc:  # embeddings unavailable → neutral zero
        logger.warning("Title similarity unavailable: %s", exc)
        return 0.0
