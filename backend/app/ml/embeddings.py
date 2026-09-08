"""
Embedding Engine — Phase 7

Provides semantic embedding generation and similarity computation for the
CV–Job matching pipeline.

Model: all-MiniLM-L6-v2 (384-dim, fast, accurate)
- Sentence-transformer from HuggingFace
- Optimized for semantic similarity tasks
- Runs on CPU (no GPU required)

Embedding strategies:
    A. Full document embeddings (CV ↔ Job)
    B. Section-level embeddings (summary, experience, skills)
    C. Skill-level embeddings (individual skill comparison)
    D. Weighted hybrid (combines all strategies)

All strategies are independently testable and produce comparable outputs.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model singleton (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


def get_model() -> SentenceTransformer:
    """Return the global SentenceTransformer model (lazy-loaded singleton)."""
    global _model  # noqa: PLW0603
    if _model is None:
        logger.info("Loading embedding model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded (dim=%d)", _EMBEDDING_DIM)
    return _model


def reset_model() -> None:
    """Unload the model to free memory (useful in tests)."""
    global _model  # noqa: PLW0603
    _model = None


# ---------------------------------------------------------------------------
# Core embedding functions
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts into dense vectors.

    Args:
        texts: List of strings to embed.
        batch_size: Batch size for encoding (controls memory vs speed).

    Returns:
        np.ndarray of shape (len(texts), 384) with L2-normalized embeddings.
    """
    if not texts:
        return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalize for cosine via dot product
    )
    return embeddings.astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed a single text. Returns shape (384,)."""
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Assumes vectors are L2-normalized; otherwise normalizes first.

    Returns:
        Float in [0.0, 1.0].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a / norm_a, vec_b / norm_b))


def cosine_similarity_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix between two sets of embeddings.

    Args:
        embeddings_a: shape (N, dim)
        embeddings_b: shape (M, dim)

    Returns:
        np.ndarray of shape (N, M) with similarity scores in [0, 1].
    """
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        return np.zeros((len(embeddings_a), len(embeddings_b)), dtype=np.float32)

    # L2-normalize
    a_norm = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    b_norm = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)
    # Dot product = cosine similarity for normalized vectors
    return np.clip(a_norm @ b_norm.T, 0.0, 1.0).astype(np.float32)


def max_similarity(
    source_emb: np.ndarray,
    target_embs: np.ndarray,
) -> float:
    """
    For each source embedding, find the max similarity to any target embedding,
    then average across source embeddings.

    Args:
        source_emb: shape (N, dim)
        target_embs: shape (M, dim)

    Returns:
        Float in [0.0, 1.0] — the averaged best-match similarity.
    """
    if len(source_emb) == 0 or len(target_embs) == 0:
        return 0.0
    sim_matrix = cosine_similarity_matrix(source_emb, target_embs)
    return float(np.mean(np.max(sim_matrix, axis=1)))


# ---------------------------------------------------------------------------
# Strategy A: Full document embedding
# ---------------------------------------------------------------------------


def strategy_full_document(
    cv_text: str,
    job_text: str,
) -> dict:
    """
    Strategy A — Full document embedding comparison.

    Embeds the entire CV and job description, then computes cosine similarity.
    Simplest approach; works as a strong baseline.

    Returns:
        dict with 'semantic_score' (0-1 float), 'embeddings' (for reuse).
    """
    cv_emb = embed_text(cv_text)
    job_emb = embed_text(job_text)
    score = cosine_similarity(cv_emb, job_emb)
    return {
        "semantic_score": round(score, 4),
        "cv_embedding": cv_emb,
        "job_embedding": job_emb,
    }


# ---------------------------------------------------------------------------
# Strategy B: Section-level embedding
# ---------------------------------------------------------------------------


def strategy_section_level(
    cv_sections: dict[str, str],
    job_sections: dict[str, str],
) -> dict:
    """
    Strategy B — Section-level embedding comparison.

    Embeds matching sections (summary ↔ description, skills ↔ requirements,
    experience ↔ responsibilities), computes per-section similarity, and
    returns a weighted average.

    Args:
        cv_sections: dict with keys like 'summary', 'skills', 'experience'.
        job_sections: dict with keys like 'description', 'requirements',
                      'responsibilities'.

    Returns:
        dict with 'semantic_score', 'section_scores', 'embeddings'.
    """
    # Mapping: cv_section -> job_section to compare
    section_pairs = {
        "summary": "description",
        "skills": "requirements",
        "experience": "responsibilities",
    }

    # Default weights: summary most important, skills/requirements second
    weights = {
        "summary": 0.40,
        "skills": 0.35,
        "experience": 0.25,
    }

    section_scores = {}
    cv_embs = {}
    job_embs = {}

    total_weight = 0.0
    weighted_sum = 0.0

    for section, weight in weights.items():
        cv_text = cv_sections.get(section, "")
        job_section_key = section_pairs.get(section, section)
        job_text = job_sections.get(job_section_key, "")

        if not cv_text.strip() or not job_text.strip():
            continue

        cv_emb = embed_text(cv_text)
        job_emb = embed_text(job_text)
        score = cosine_similarity(cv_emb, job_emb)

        section_scores[section] = round(score, 4)
        cv_embs[section] = cv_emb
        job_embs[section] = job_emb
        weighted_sum += score * weight
        total_weight += weight

    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        "semantic_score": round(final_score, 4),
        "section_scores": section_scores,
        "cv_embeddings": cv_embs,
        "job_embeddings": job_embs,
    }


# ---------------------------------------------------------------------------
# Strategy C: Skill-level embedding
# ---------------------------------------------------------------------------


def strategy_skill_level(
    candidate_skills: list[str],
    job_required_skills: list[str],
    job_preferred_skills: list[str] | None = None,
) -> dict:
    """
    Strategy C — Skill-level semantic comparison.

    Embeds each skill individually and computes semantic similarity between
    candidate skills and job skills. Useful for finding "related" skills
    (e.g., "React" ↔ "Frontend Development").

    Args:
        candidate_skills: List of skills from the CV.
        job_required_skills: Required skills from the job description.
        job_preferred_skills: Preferred/nice-to-have skills.

    Returns:
        dict with 'semantic_score', 'best_matches', 'skill_embeddings'.
    """
    preferred = job_preferred_skills or []
    all_job_skills = job_required_skills + preferred

    if not candidate_skills or not all_job_skills:
        return {
            "semantic_score": 0.0,
            "best_matches": [],
            "candidate_embeddings": np.zeros((0, _EMBEDDING_DIM)),
            "job_embeddings": np.zeros((0, _EMBEDDING_DIM)),
        }

    cv_embs = embed_texts(candidate_skills)
    job_embs = embed_texts(all_job_skills)
    sim_matrix = cosine_similarity_matrix(cv_embs, job_embs)

    # For each candidate skill, find best matching job skill
    best_matches = []
    for i, skill in enumerate(candidate_skills):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_j])
        best_matches.append(
            {
                "candidate_skill": skill,
                "matched_job_skill": all_job_skills[best_j],
                "similarity": round(best_score, 4),
            }
        )

    # Overall score = average of best-match similarities
    overall = float(np.mean(np.max(sim_matrix, axis=1))) if len(candidate_skills) > 0 else 0.0

    return {
        "semantic_score": round(overall, 4),
        "best_matches": best_matches,
        "candidate_embeddings": cv_embs,
        "job_embeddings": job_embs,
    }


# ---------------------------------------------------------------------------
# Strategy D: Weighted hybrid
# ---------------------------------------------------------------------------


HYBRID_WEIGHTS = {
    "full_document": 0.30,
    "section_level": 0.40,
    "skill_level": 0.30,
}


def strategy_hybrid(
    cv_text: str,
    job_text: str,
    cv_sections: dict[str, str],
    job_sections: dict[str, str],
    candidate_skills: list[str],
    job_required_skills: list[str],
    job_preferred_skills: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Strategy D — Weighted hybrid of all strategies.

    Combines full-document, section-level, and skill-level embeddings
    into a single semantic score with configurable weights.

    Args:
        cv_text: Full CV text.
        job_text: Full job description text.
        cv_sections: CV sections dict.
        job_sections: JD sections dict.
        candidate_skills: Candidate's extracted skills.
        job_required_skills: Job required skills.
        job_preferred_skills: Job preferred skills.
        weights: Strategy weights (defaults to HYBRID_WEIGHTS).

    Returns:
        dict with 'semantic_score', 'strategy_scores', 'weights'.
    """
    w = weights or HYBRID_WEIGHTS

    result_a = strategy_full_document(cv_text, job_text)
    result_b = strategy_section_level(cv_sections, job_sections)
    result_c = strategy_skill_level(
        candidate_skills,
        job_required_skills,
        job_preferred_skills,
    )

    combined_score = (
        w["full_document"] * result_a["semantic_score"]
        + w["section_level"] * result_b["semantic_score"]
        + w["skill_level"] * result_c["semantic_score"]
    )

    return {
        "semantic_score": round(combined_score, 4),
        "strategy_scores": {
            "full_document": result_a["semantic_score"],
            "section_level": result_b["semantic_score"],
            "skill_level": result_c["semantic_score"],
        },
        "weights": w,
        "details": {
            "full_document": result_a,
            "section_level": result_b,
            "skill_level": result_c,
        },
    }


# ---------------------------------------------------------------------------
# Convenience: top-level embed & compare
# ---------------------------------------------------------------------------


def compute_semantic_similarity(
    cv_text: str,
    job_text: str,
    strategy: str = "full_document",
    **kwargs,
) -> float:
    """
    Compute semantic similarity using the specified strategy.

    Args:
        cv_text: Full CV text.
        job_text: Full job description text.
        strategy: One of 'full_document', 'section_level', 'skill_level', 'hybrid'.
        **kwargs: Additional arguments passed to the strategy function.

    Returns:
        Float in [0.0, 1.0].
    """
    strategies = {
        "full_document": strategy_full_document,
        "section_level": strategy_section_level,
        "skill_level": strategy_skill_level,
        "hybrid": strategy_hybrid,
    }
    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(strategies.keys())}"
        )

    if strategy == "full_document":
        return strategy_full_document(cv_text, job_text)["semantic_score"]
    elif strategy == "section_level":
        return strategy_section_level(
            kwargs.get("cv_sections", {}), kwargs.get("job_sections", {})
        )["semantic_score"]
    elif strategy == "skill_level":
        return strategy_skill_level(
            kwargs.get("candidate_skills", []),
            kwargs.get("job_required_skills", []),
            kwargs.get("job_preferred_skills"),
        )["semantic_score"]
    elif strategy == "hybrid":
        return strategy_hybrid(cv_text, job_text, **kwargs)["semantic_score"]
    return 0.0
