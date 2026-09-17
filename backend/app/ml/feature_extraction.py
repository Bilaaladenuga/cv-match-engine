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

Per-category coverage (taxonomy-driven; breaks the aggregate-only ceiling):

    cov_<cat>_required           fraction of REQUIRED skills in taxonomy
                                 category <cat> that the candidate covers
                                 (full or partial). Categories: programming,
                                 frontend, backend, database, cloud, devops,
                                 data_science, machine_learning. 0.0 when the
                                 JD demands nothing from that category (no
                                 demand -> no credit, matching the convention
                                 of the aggregate coverage features).
    cov_other_required           same ratio over required skills in all
                                 remaining categories (gis, design, tools,
                                 soft_skills) or unknown to the taxonomy.
    cov_<cat>_n                  JD demand: count of required+preferred
                                 skills from category <cat>.

CV-length normalization (v0.4.0; Phase 13 finding: the model used raw
skill count as a 'long CV = good fit' proxy):

    cv_word_count                normalized whitespace token count of the CV
    skills_per_100_words         100 * n_candidate_skills / cv_word_count
                                 (density; 0.0 when the CV is empty)
    cv_length_bucket             0 <500 words, 1 <1200, 2 <2500, 3 >=2500
"""

from __future__ import annotations

import logging

import numpy as np

from app.ml.embeddings import cosine_similarity, embed_texts
from app.ml.semantic_matcher import compute_semantic_match
from app.nlp.candidate_builder import build_candidate_profile
from app.nlp.experience_matcher import match_experience
from app.nlp.job_parser import parse_job_description
from app.nlp.skill_matcher import match_skills
from app.nlp.taxonomy import load_taxonomy
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
    # per-category coverage (Phase 12.1)
    "cov_programming_required",
    "cov_frontend_required",
    "cov_backend_required",
    "cov_database_required",
    "cov_cloud_required",
    "cov_devops_required",
    "cov_data_science_required",
    "cov_machine_learning_required",
    "cov_other_required",
    "cov_programming_n",
    "cov_frontend_n",
    "cov_backend_n",
    "cov_database_n",
    "cov_cloud_n",
    "cov_devops_n",
    "cov_data_science_n",
    "cov_machine_learning_n",
    # CV-length normalization (v0.4.0)
    "cv_word_count",
    "skills_per_100_words",
    "cv_length_bucket",
    # Augmented features (v0.5.0) — ratios and alignments
    "total_skill_coverage",
    "skills_per_required",
    "skill_surplus_ratio",
    "skills_per_year",
    "mean_category_coverage",
    "min_category_coverage",
    "max_category_coverage",
    "coverage_variance",
    "semantic_skill_gap",
    "semantic_skill_product",
    "edu_seniority_alignment",
    "skill_density_squared",
    "title_skill_alignment",
]

# Taxonomy categories with their own coverage feature. Everything else
# (gis, design, tools, soft_skills) and unknown skills fall into "other".
_COVERAGE_CATEGORIES = (
    "programming", "frontend", "backend", "database",
    "cloud", "devops", "data_science", "machine_learning",
)

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

    # --- Per-category coverage (taxonomy-driven) -----------------------------
    cat_features = compute_category_coverage(req_names, pref_names, matched_set)

    # --- CV-length normalization (v0.4.0) ------------------------------------
    length_features = compute_cv_length_features(cv_text, len(cand_names))

    # --- Augmented features (v0.5.0) ----------------------------------------
    aug_features = compute_augmented_features(
        skill_overlap=skill_overlap,
        required_coverage=skill_match.required_coverage,
        preferred_cov=preferred_cov,
        n_candidate=len(cand_names),
        n_required=len(req_names),
        n_preferred=len(pref_names),
        experience_gap=experience_gap,
        semantic_sim=semantic_match.raw_score,
        education_level=education_match.level_score,
        seniority_match=seniority_match,
        title_sim=title_sim,
        skills_per_100_words=length_features["skills_per_100_words"],
        cat_features=cat_features,
    )

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
        **cat_features,
        **length_features,
        **aug_features,
    }


def compute_cv_length_features(cv_text: str, n_skills: int) -> dict[str, float]:
    """CV-length normalization features (v0.4.0).

    Phase 13's error analysis showed the model conflated 'long CV / many
    skills listed' with 'good fit' (overrated candidates: ~10.5 skills vs
    ~6.8; permutation importance ranks n_candidate_skills #1). These
    features give the model explicit access to CV length and skill DENSITY
    so raw count no longer has to act as a hidden length proxy:

        cv_word_count      whitespace token count of the CV text
        skills_per_100_words  100 * n_skills / cv_word_count (0.0 if empty)
        cv_length_bucket   ordinal length class (see module docstring)

    Shared by build_feature_vector and the table augmentation script so
    training and serving cannot diverge.
    """
    words = len(cv_text.split()) if cv_text else 0
    density = (100.0 * n_skills / words) if words > 0 else 0.0
    if words >= 2500:
        bucket = 3.0
    elif words >= 1200:
        bucket = 2.0
    elif words >= 500:
        bucket = 1.0
    else:
        bucket = 0.0
    return {
        "cv_word_count": float(words),
        "skills_per_100_words": round(density, 6),
        "cv_length_bucket": bucket,
    }


def compute_augmented_features(
    *,
    skill_overlap: float,
    required_coverage: float,
    preferred_cov: float,
    n_candidate: int,
    n_required: int,
    n_preferred: int,
    experience_gap: float,
    semantic_sim: float,
    education_level: float,
    seniority_match: float,
    title_sim: float,
    skills_per_100_words: float,
    cat_features: dict[str, float],
) -> dict[str, float]:
    """Augmented features (v0.5.0) to reduce volume-proxy bias.

    These engineered features focus on ratios, alignments, and quality
    signals rather than raw counts, addressing the finding that the model
    learned 'longer CV = better fit' as a shortcut.
    """
    features: dict[str, float] = {}

    # 1. Skill coverage ratios (weighted aggregate)
    features["total_skill_coverage"] = round(
        required_coverage * 0.7 + preferred_cov * 0.3, 6
    )

    # 2. Skill density normalized by JD complexity
    features["skills_per_required"] = round(n_candidate / (n_required + 1), 6)
    features["skill_surplus_ratio"] = round(
        (n_candidate - n_required) / (n_required + 1), 6
    )

    # 3. Experience-adjusted skill count
    features["skills_per_year"] = round(n_candidate / (experience_gap + 5), 6)

    # 4. Coverage quality scores (category-level aggregates)
    coverage_cols = [v for k, v in cat_features.items() if k.startswith("cov_") and k.endswith("_required")]
    if coverage_cols:
        features["mean_category_coverage"] = round(float(np.mean(coverage_cols)), 6)
        features["min_category_coverage"] = round(float(np.min(coverage_cols)), 6)
        features["max_category_coverage"] = round(float(np.max(coverage_cols)), 6)
        features["coverage_variance"] = round(float(np.var(coverage_cols)), 6)

    # 5. Semantic-skill alignment
    features["semantic_skill_gap"] = round(semantic_sim - skill_overlap, 6)
    features["semantic_skill_product"] = round(semantic_sim * skill_overlap, 6)

    # 6. Education-experience alignment
    features["edu_seniority_alignment"] = round(education_level * seniority_match, 6)

    # 7. CV quality signal (non-volume)
    features["skill_density_squared"] = round(skills_per_100_words ** 2, 6)

    # 8. Title alignment strength
    features["title_skill_alignment"] = round(title_sim * skill_overlap, 6)

    return features


def compute_category_coverage(
    req_names: list[str],
    pref_names: list[str],
    covered: set[str],
) -> dict[str, float]:
    """
    Per-taxonomy-category coverage features.

    Args:
        req_names: canonical required skill names from the JD.
        pref_names: canonical preferred skill names (demand counts only).
        covered: canonical names matched fully or partially, from the
            Phase 8 skill matcher.

    Returns the cov_* feature dict (see module docstring for semantics).
    Shared by build_feature_vector and the table augmentation script so
    training rows and live inference cannot diverge.
    """
    taxonomy = load_taxonomy()

    # Map each REQUIRED skill to its taxonomy category (None = unknown/other).
    req_cats: dict[str | None, list[str]] = {}
    for name in req_names:
        skill = taxonomy.get_skill(name)
        cat = skill.category if skill is not None else None
        req_cats.setdefault(cat, []).append(name)

    cat_features: dict[str, float] = {}
    for cat in _COVERAGE_CATEGORIES:
        demanded = req_cats.get(cat, [])
        cov = (
            len([s for s in demanded if s in covered]) / len(demanded)
            if demanded
            else 0.0
        )
        cat_features[f"cov_{cat}_required"] = round(cov, 6)
    other_demanded = req_cats.get(None, [])
    for cat in taxonomy.category_ids:
        if cat not in _COVERAGE_CATEGORIES:
            other_demanded.extend(req_cats.get(cat, []))
    other_cov = (
        len([s for s in other_demanded if s in covered]) / len(other_demanded)
        if other_demanded
        else 0.0
    )
    cat_features["cov_other_required"] = round(other_cov, 6)

    # Demand counts: how many required+preferred skills come from each
    # category (JD complexity profile; complements the coverage ratios).
    for cat in _COVERAGE_CATEGORIES:
        n = sum(
            1
            for name in req_names + pref_names
            if (skill := taxonomy.get_skill(name)) is not None
            and skill.category == cat
        )
        cat_features[f"cov_{cat}_n"] = float(n)
    return cat_features


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
