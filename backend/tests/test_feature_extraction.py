"""
Tests for Phase 12 — feature extraction (backend/app/ml/feature_extraction.py).

These are integration-flavoured: they run the real pipeline (parsers,
matchers, embedding model) so they also pin the train/serve contract —
the exact code paths used at inference time.
"""

from __future__ import annotations

import pytest

from app.ml.feature_extraction import FEATURE_NAMES, extract_features

GOOD_CV = """
Sarah Chen
sarah.chen@example.com | (415) 555-0123

SUMMARY
Senior backend engineer with 6 years of experience building Python services.

SKILLS
Python, Django, FastAPI, PostgreSQL, Redis, Docker, AWS, Git, REST APIs

EXPERIENCE
Senior Backend Engineer, DataWorks (2020-01 - Present)
Built FastAPI services serving 2M requests/day on AWS with PostgreSQL.

Backend Developer, AppCo (2018-03 - 2019-12)
Developed Django REST APIs and Redis caching layers.

EDUCATION
B.S. in Computer Science, State University, 2018
"""

GOOD_JOB = """
Job Title: Senior Backend Engineer
Company: CloudScale Inc.

Requirements:
- 5+ years of professional software engineering experience
- Strong Python skills; Django or FastAPI experience
- PostgreSQL and Redis in production
- AWS and Docker experience

Preferred:
- Kubernetes
- Terraform

Responsibilities:
- Design and operate backend services
- Mentor junior engineers

Education: Bachelor's degree in Computer Science
"""

POOR_JOB = """
Job Title: Registered Nurse - Emergency Department
City General Hospital

Requirements:
- Active RN license
- 3+ years emergency room experience
- BLS and ACLS certifications
- Bachelor of Science in Nursing

Responsibilities:
- Patient triage and emergency care
"""

NO_EXPERIENCE_CV = """
Alex Doe
alex.doe@example.com

SKILLS
Python, HTML

EDUCATION
High school diploma, 2024
"""


@pytest.fixture(scope="module")
def good_features() -> dict[str, float]:
    return extract_features(GOOD_CV, GOOD_JOB)


class TestFeatureContract:
    def test_keys_exactly_match_feature_names(self, good_features):
        assert set(good_features.keys()) == set(FEATURE_NAMES)

    def test_all_values_are_floats(self, good_features):
        assert all(isinstance(v, float) for v in good_features.values())

    def test_scores_within_unit_range(self, good_features):
        for name in FEATURE_NAMES:
            if name in ("experience_gap_years", "n_candidate_skills",
                        "n_required_skills", "n_preferred_skills"):
                continue
            if name.startswith("cov_") and name.endswith("_n"):
                continue  # unbounded demand counts, not ratios
            assert 0.0 <= good_features[name] <= 1.0, name

    def test_meta_counts_are_sane(self, good_features):
        assert good_features["n_candidate_skills"] > 0
        assert good_features["n_required_skills"] > 0


class TestDiscrimination:
    """The features must separate a good pair from a mismatched pair."""

    def test_good_pair_beats_mismatched_pair(self):
        good = extract_features(GOOD_CV, GOOD_JOB)
        poor = extract_features(GOOD_CV, POOR_JOB)
        assert good["required_skill_coverage"] > poor["required_skill_coverage"]
        assert good["semantic_similarity"] > poor["semantic_similarity"]
        assert good["job_title_similarity"] > poor["job_title_similarity"]

    def test_education_field_separates(self):
        good = extract_features(GOOD_CV, GOOD_JOB)
        poor = extract_features(GOOD_CV, POOR_JOB)
        assert good["education_field_score"] > poor["education_field_score"]

    def test_experience_gap_reflects_reality(self):
        feats = extract_features(GOOD_CV, GOOD_JOB)
        # 6+ years total vs 5 required -> non-negative gap
        assert feats["experience_gap_years"] >= 0


class TestDegenerateInputs:
    # Neutral-on-no-requirement defaults (documented in Phase 9/11): an empty
    # JD states no seniority/education/cert requirements, so those components
    # cannot be failed — they return their neutral scores, not zero.
    NEUTRAL = {"seniority_match": 1.0, "education_level_score": 0.85,
               "education_field_score": 0.85, "certification_match_ratio": 0.85}
    META = {"experience_gap_years", "n_candidate_skills",
            "n_required_skills", "n_preferred_skills"}

    def test_empty_inputs_return_neutral_without_crashing(self):
        feats = extract_features("", "")
        for name in FEATURE_NAMES:
            if name in self.META:
                continue
            assert feats[name] == self.NEUTRAL.get(name, 0.0), name

    def test_seniority_match_is_vacuously_true_without_requirement(self):
        # No stated seniority -> cannot fail the requirement (mirrors the
        # education matcher's neutral-on-no-requirement philosophy).
        feats = extract_features("", "")
        assert feats["seniority_match"] == 1.0
        # ...but a real requirement must not be met by an empty CV.
        senior_job = GOOD_JOB.replace("5+ years", "8+ years").replace(
            "Education:", "Senior level required. Education:"
        )
        feats = extract_features("", senior_job)
        assert feats["seniority_match"] == 0.0

    def test_cv_without_relevant_experience(self):
        feats = extract_features(NO_EXPERIENCE_CV, GOOD_JOB)
        assert 0.0 <= feats["required_skill_coverage"] < 0.5

    def test_deterministic(self, good_features):
        again = extract_features(GOOD_CV, GOOD_JOB)
        assert again == good_features


class TestPerCategoryCoverage:
    """Category coverage features (Phase 12.1) — stubbed engine outputs."""

    def _build(self, cand_skills, req, pref=(), matched=(), partial=()):
        from types import SimpleNamespace

        from app.ml.feature_extraction import build_feature_vector

        cand = SimpleNamespace(
            skills=[SimpleNamespace(name=s) for s in cand_skills],
            total_years_experience=3.0,
            work_experience=[object()],
            job_titles=["Developer"],
        )
        job = SimpleNamespace(
            required_skills=[SimpleNamespace(name=s) for s in req],
            preferred_skills=[SimpleNamespace(name=s) for s in pref],
            minimum_experience_years=2,
            experience_level="mid",
            education=None,
            certifications=[],
            job_title="Backend Engineer",
        )
        sm = SimpleNamespace(
            matched_skills=list(matched),
            partial_skills=list(partial),
            required_coverage=0.5,
            required_plus_partial=0.5,
        )
        return build_feature_vector(
            candidate=cand, job=job, skill_match=sm,
            experience_match=SimpleNamespace(overall_score=0.5),
            semantic_match=SimpleNamespace(raw_score=0.6),
            education_match=SimpleNamespace(level_score=1.0, field_score=1.0),
            certification_match=SimpleNamespace(score=0.0),
            cv_text="developer",
        )

    def test_per_category_coverage(self):
        # React matched (frontend); AWS, Docker demanded but uncovered;
        # Figma (design) and an unknown skill fall into "other".
        f = self._build(
            cand_skills=["React", "Python"],
            req=["React", "AWS", "Docker", "Figma", "Zorp-Lang"],
            matched=["React"],
        )
        assert f["cov_frontend_required"] == 1.0
        assert f["cov_cloud_required"] == 0.0      # AWS uncovered
        assert f["cov_devops_required"] == 0.0     # Docker uncovered
        assert f["cov_other_required"] == 0.0      # Figma + Zorp-Lang uncovered
        # No database/data-science/ML/programming demand -> 0.0 (no credit)
        assert f["cov_database_required"] == 0.0
        assert f["cov_machine_learning_required"] == 0.0

    def test_partial_counts_as_coverage(self):
        f = self._build(
            cand_skills=["Pandas"], req=["Pandas"], matched=[], partial=["Pandas"],
        )
        assert f["cov_data_science_required"] == 1.0

    def test_demand_counts_include_preferred(self):
        # Terraform is devops in the taxonomy, not cloud
        f = self._build(
            cand_skills=["Python"], req=["Python", "AWS"], pref=["Docker"],
            matched=["Python"],
        )
        assert f["cov_programming_n"] == 1.0
        assert f["cov_cloud_n"] == 1.0             # AWS only
        assert f["cov_devops_n"] == 1.0            # Docker
        assert f["cov_programming_required"] == 1.0
        assert f["cov_cloud_required"] == 0.0

    def test_unknown_skill_falls_into_other(self):
        f = self._build(cand_skills=[], req=["Totally-Made-Up-Framework"])
        assert f["cov_other_required"] == 0.0
        for cat in ("programming", "frontend", "cloud", "devops"):
            assert f[f"cov_{cat}_n"] == 0.0        # unknown != taxonomy category

    def test_alias_canonicalization_maps_categories(self):
        # "js" resolves through the taxonomy alias map to JavaScript
        # (programming category in this taxonomy)
        f = self._build(cand_skills=["JS"], req=["JavaScript"], matched=["JavaScript"])
        assert f["cov_programming_required"] == 1.0
        assert f["cov_programming_n"] == 1.0

    def test_schema_still_matches_feature_names(self):
        from app.ml.feature_extraction import build_feature_vector  # noqa: F401
        f = self._build(cand_skills=["Python"], req=["Python"], matched=["Python"])
        assert set(f.keys()) == set(FEATURE_NAMES)
