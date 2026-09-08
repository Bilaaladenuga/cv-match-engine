"""
Tests for Phase 7 — Embedding Engine.

Covers:
    - Embedding dimensions and normalization
    - Cosine similarity correctness
    - Strategy A: full document
    - Strategy B: section-level
    - Strategy C: skill-level
    - Strategy D: weighted hybrid
    - Edge cases (empty inputs, identical texts)
"""

import numpy as np

from app.ml.embeddings import (
    _EMBEDDING_DIM,
    HYBRID_WEIGHTS,
    cosine_similarity,
    cosine_similarity_matrix,
    embed_text,
    embed_texts,
    max_similarity,
    strategy_full_document,
    strategy_hybrid,
    strategy_section_level,
    strategy_skill_level,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CV = """
Sarah Johnson — Senior Software Engineer
Email: sarah.johnson@email.com

Summary:
Experienced software engineer with 7+ years of experience in full-stack
web development. Proficient in Python, React, PostgreSQL. Strong
background in machine learning and GIS applications.

Experience:
Senior Software Engineer | TechCorp Inc | Jan 2020 – Present
- Led development of data pipeline using Python and Apache Spark
- Built React-based dashboard for real-time analytics
- Implemented ML models for predictive analytics

Skills: Python, React, PostgreSQL, Machine Learning, TensorFlow, GIS, Docker
"""

SAMPLE_JD = """
Senior Software Engineer — Data Platform

Company: DataFlow Inc.
Location: Remote

About the Role:
We are looking for a Senior Software Engineer to join our data platform
team. You will build scalable data pipelines and machine learning systems.

Required Qualifications:
- 5+ years of software engineering experience
- Proficiency in Python and SQL
- Experience with cloud platforms (AWS or GCP)
- Strong background in machine learning

Preferred:
- React or modern frontend experience
- Docker and Kubernetes experience
- PostgreSQL expertise
"""

SAMPLE_CV_SECTIONS = {
    "summary": "Experienced software engineer with 7+ years in full-stack development. "
    "Python, React, PostgreSQL, machine learning.",
    "skills": "Python, React, PostgreSQL, Machine Learning, TensorFlow, Docker",
    "experience": "Senior Software Engineer at TechCorp. Led data pipeline development.",
}

SAMPLE_JD_SECTIONS = {
    "description": "We are looking for a Senior Software Engineer to join our data platform team.",
    "requirements": "5+ years Python, SQL, cloud platforms (AWS/GCP), machine learning",
    "responsibilities": "Build scalable data pipelines and machine learning systems.",
}

CANDIDATE_SKILLS = ["Python", "React", "PostgreSQL", "Machine Learning", "Docker", "GIS"]
JOB_REQUIRED = ["Python", "SQL", "AWS", "Machine Learning"]
JOB_PREFERRED = ["React", "Kubernetes", "PostgreSQL"]


# ---------------------------------------------------------------------------
# Embedding basics
# ---------------------------------------------------------------------------


class TestEmbeddingDimensions:
    """Verify embedding dimensions and shape."""

    def test_single_embedding_shape(self):
        emb = embed_text("Hello world")
        assert emb.shape == (_EMBEDDING_DIM,)

    def test_batch_embedding_shape(self):
        texts = ["Hello", "World", "Test"]
        embs = embed_texts(texts)
        assert embs.shape == (3, _EMBEDDING_DIM)

    def test_embedding_is_float32(self):
        emb = embed_text("Test embedding")
        assert emb.dtype == np.float32

    def test_embedding_is_l2_normalized(self):
        emb = embed_text("Normalize me")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01  # Should be ~1.0

    def test_empty_input(self):
        embs = embed_texts([])
        assert embs.shape == (0, _EMBEDDING_DIM)

    def test_empty_single_text(self):
        embs = embed_texts([""])
        assert embs.shape == (1, _EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        vec = embed_text("same text")
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01

    def test_similar_texts_high_similarity(self):
        a = embed_text("Python programming language")
        b = embed_text("Python development")
        sim = cosine_similarity(a, b)
        assert sim > 0.6  # Should be quite similar

    def test_dissimilar_texts_low_similarity(self):
        a = embed_text("Python programming language")
        b = embed_text("Cooking Italian pasta recipe")
        sim = cosine_similarity(a, b)
        assert sim < 0.5  # Should be quite different

    def test_symmetry(self):
        a = embed_text("Machine learning")
        b = embed_text("Artificial intelligence")
        sim_ab = cosine_similarity(a, b)
        sim_ba = cosine_similarity(b, a)
        assert abs(sim_ab - sim_ba) < 0.001

    def test_zero_vector(self):
        vec = embed_text("Something")
        zero = np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        sim = cosine_similarity(vec, zero)
        assert sim == 0.0

    def test_range_0_to_1(self):
        a = embed_text("Apple")
        b = embed_text("Banana")
        sim = cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0


class TestCosineSimilarityMatrix:
    """Test pairwise similarity matrix."""

    def test_matrix_shape(self):
        a = embed_texts(["A", "B"])
        b = embed_texts(["X", "Y", "Z"])
        matrix = cosine_similarity_matrix(a, b)
        assert matrix.shape == (2, 3)

    def test_diagonal_of_identical(self):
        texts = ["Python", "JavaScript"]
        embs = embed_texts(texts)
        matrix = cosine_similarity_matrix(embs, embs)
        # Diagonal should be ~1.0
        for i in range(len(texts)):
            assert abs(matrix[i, i] - 1.0) < 0.01

    def test_empty_input(self):
        a = np.zeros((0, _EMBEDDING_DIM))
        b = embed_texts(["Test"])
        matrix = cosine_similarity_matrix(a, b)
        assert matrix.shape == (0, 1)


class TestMaxSimilarity:
    """Test max-similarity computation."""

    def test_perfect_match(self):
        source = embed_texts(["Python"])
        target = embed_texts(["Python", "Java", "C++"])
        sim = max_similarity(source, target)
        assert sim > 0.9  # Python ↔ Python should be very high

    def test_no_match(self):
        source = embed_texts(["Python"])
        target = embed_texts(["Cooking", "Painting"])
        sim = max_similarity(source, target)
        assert sim < 0.6

    def test_empty_source(self):
        source = np.zeros((0, _EMBEDDING_DIM))
        target = embed_texts(["Python"])
        sim = max_similarity(source, target)
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Strategy A: Full Document
# ---------------------------------------------------------------------------


class TestStrategyFullDocument:
    """Test full-document embedding strategy."""

    def test_returns_valid_score(self):
        result = strategy_full_document(SAMPLE_CV, SAMPLE_JD)
        assert 0.0 <= result["semantic_score"] <= 1.0

    def test_similar_documents_high_score(self):
        result = strategy_full_document(SAMPLE_CV, SAMPLE_JD)
        assert result["semantic_score"] > 0.5  # CV and JD are related

    def test_same_text_perfect_score(self):
        text = "Python developer with 5 years experience"
        result = strategy_full_document(text, text)
        assert result["semantic_score"] > 0.95

    def test_contains_embeddings(self):
        result = strategy_full_document(SAMPLE_CV, SAMPLE_JD)
        assert "cv_embedding" in result
        assert "job_embedding" in result
        assert result["cv_embedding"].shape == (_EMBEDDING_DIM,)
        assert result["job_embedding"].shape == (_EMBEDDING_DIM,)


# ---------------------------------------------------------------------------
# Strategy B: Section-Level
# ---------------------------------------------------------------------------


class TestStrategySectionLevel:
    """Test section-level embedding strategy."""

    def test_returns_valid_score(self):
        result = strategy_section_level(SAMPLE_CV_SECTIONS, SAMPLE_JD_SECTIONS)
        assert 0.0 <= result["semantic_score"] <= 1.0

    def test_contains_section_scores(self):
        result = strategy_section_level(SAMPLE_CV_SECTIONS, SAMPLE_JD_SECTIONS)
        assert "section_scores" in result
        assert isinstance(result["section_scores"], dict)
        # Should have at least one section scored
        assert len(result["section_scores"]) > 0

    def test_empty_sections(self):
        result = strategy_section_level({}, {})
        assert result["semantic_score"] == 0.0

    def test_partial_sections(self):
        cv = {"summary": "Python developer"}
        jd = {"description": "Looking for Python developer"}
        result = strategy_section_level(cv, jd)
        assert 0.0 <= result["semantic_score"] <= 1.0


# ---------------------------------------------------------------------------
# Strategy C: Skill-Level
# ---------------------------------------------------------------------------


class TestStrategySkillLevel:
    """Test skill-level embedding strategy."""

    def test_returns_valid_score(self):
        result = strategy_skill_level(
            CANDIDATE_SKILLS, JOB_REQUIRED, JOB_PREFERRED
        )
        assert 0.0 <= result["semantic_score"] <= 1.0

    def test_identical_skills_high_score(self):
        skills = ["Python", "React", "PostgreSQL"]
        result = strategy_skill_level(skills, skills)
        assert result["semantic_score"] > 0.9

    def test_contains_best_matches(self):
        result = strategy_skill_level(
            CANDIDATE_SKILLS, JOB_REQUIRED, JOB_PREFERRED
        )
        assert "best_matches" in result
        assert len(result["best_matches"]) == len(CANDIDATE_SKILLS)

    def test_best_match_structure(self):
        result = strategy_skill_level(
            CANDIDATE_SKILLS, JOB_REQUIRED, JOB_PREFERRED
        )
        match = result["best_matches"][0]
        assert "candidate_skill" in match
        assert "matched_job_skill" in match
        assert "similarity" in match
        assert 0.0 <= match["similarity"] <= 1.0

    def test_empty_skills(self):
        result = strategy_skill_level([], [])
        assert result["semantic_score"] == 0.0
        assert result["best_matches"] == []


# ---------------------------------------------------------------------------
# Strategy D: Weighted Hybrid
# ---------------------------------------------------------------------------


class TestStrategyHybrid:
    """Test weighted hybrid strategy."""

    def test_returns_valid_score(self):
        result = strategy_hybrid(
            SAMPLE_CV,
            SAMPLE_JD,
            SAMPLE_CV_SECTIONS,
            SAMPLE_JD_SECTIONS,
            CANDIDATE_SKILLS,
            JOB_REQUIRED,
            JOB_PREFERRED,
        )
        assert 0.0 <= result["semantic_score"] <= 1.0

    def test_contains_strategy_scores(self):
        result = strategy_hybrid(
            SAMPLE_CV,
            SAMPLE_JD,
            SAMPLE_CV_SECTIONS,
            SAMPLE_JD_SECTIONS,
            CANDIDATE_SKILLS,
            JOB_REQUIRED,
            JOB_PREFERRED,
        )
        assert "strategy_scores" in result
        scores = result["strategy_scores"]
        assert "full_document" in scores
        assert "section_level" in scores
        assert "skill_level" in scores

    def test_weights_sum_to_one(self):
        assert abs(sum(HYBRID_WEIGHTS.values()) - 1.0) < 0.001

    def test_custom_weights(self):
        custom = {"full_document": 1.0, "section_level": 0.0, "skill_level": 0.0}
        result = strategy_hybrid(
            SAMPLE_CV,
            SAMPLE_JD,
            SAMPLE_CV_SECTIONS,
            SAMPLE_JD_SECTIONS,
            CANDIDATE_SKILLS,
            JOB_REQUIRED,
            JOB_PREFERRED,
            weights=custom,
        )
        # With full_document weight=1.0, result should match strategy_a
        a_result = strategy_full_document(SAMPLE_CV, SAMPLE_JD)
        assert abs(result["semantic_score"] - a_result["semantic_score"]) < 0.01

    def test_contains_all_details(self):
        result = strategy_hybrid(
            SAMPLE_CV,
            SAMPLE_JD,
            SAMPLE_CV_SECTIONS,
            SAMPLE_JD_SECTIONS,
            CANDIDATE_SKILLS,
            JOB_REQUIRED,
            JOB_PREFERRED,
        )
        assert "details" in result
        assert "full_document" in result["details"]
        assert "section_level" in result["details"]
        assert "skill_level" in result["details"]
