"""
Semantic Matching — Phase 10

Computes an overall semantic compatibility score between a candidate's CV
and a job description.

Unlike raw embedding similarity (Phase 7), this module:
    1. Combines multiple similarity signals into a single score.
    2. Normalises the score to a [0–100] range for display.
    3. Provides an interpretable label ("Strong", "Good", "Moderate", etc.).
    4. Separates the raw model score from any "probability of hiring"
       claim — the score represents *compatibility*, not hiring odds.

Scoring pipeline:
    CV text + JD text
        → full-document cosine similarity  (weight: 35%)
        → section-level similarity         (weight: 30%)
        → skill-level similarity           (weight: 35%)
        → weighted combination
        → normalised to [0–100]

All weights are configurable; the defaults were chosen after testing on
the sample CV/JD pairs to balance global relevance with skill specificity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.ml.embeddings import (
    EmbeddingUnavailableError,
    cosine_similarity,
    embed_texts,
    strategy_full_document,
    strategy_section_level,
    strategy_skill_level,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights & thresholds
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "full_document": 0.35,
    "section_level": 0.30,
    "skill_level": 0.35,
}

# Score → human-readable label (on a 0–100 scale)
_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (85, "Strong"),
    (70, "Good"),
    (55, "Moderate"),
    (40, "Fair"),
    (0, "Weak"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SemanticResult:
    """Full result of a semantic matching computation."""

    raw_score: float          # 0.0 – 1.0 (model output)
    normalised_score: int     # 0 – 100 (for display)
    label: str                # Human-readable (Strong/Good/Moderate/Fair/Weak)
    component_scores: dict    # Per-strategy breakdown
    weights: dict             # Weights used
    explanation: str          # One-paragraph interpretation
    available: bool = True    # False when the embedding model could not load

    def to_dict(self) -> dict:
        return {
            "raw_score": round(self.raw_score, 4),
            "normalised_score": self.normalised_score,
            "label": self.label,
            "component_scores": {
                k: round(v, 4) for k, v in self.component_scores.items()
            },
            "weights": self.weights,
            "explanation": self.explanation,
            "available": self.available,
        }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_semantic_match(
    cv_text: str,
    job_text: str,
    cv_sections: dict[str, str] | None = None,
    job_sections: dict[str, str] | None = None,
    candidate_skills: list[str] | None = None,
    job_required_skills: list[str] | None = None,
    job_preferred_skills: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> SemanticResult:
    """
    Compute overall semantic compatibility between a CV and a job description.

    Args:
        cv_text: Full CV text.
        job_text: Full JD text.
        cv_sections: CV sections (optional, for section-level strategy).
        job_sections: JD sections (optional, for section-level strategy).
        candidate_skills: Candidate's skill names (optional, for skill-level).
        job_required_skills: Required skills from the JD (optional).
        job_preferred_skills: Preferred skills from the JD (optional).
        weights: Strategy weights (defaults to DEFAULT_WEIGHTS).

    Returns:
        SemanticResult with score, label, and explanation.
    """
    w = weights or DEFAULT_WEIGHTS

    # Edge case: no text to compare
    if not cv_text.strip() or not job_text.strip():
        return SemanticResult(
            raw_score=0.0,
            normalised_score=0,
            label="Weak",
            component_scores={"full_document": 0.0, "section_level": 0.0, "skill_level": 0.0},
            weights={},
            explanation="No semantic comparison possible: one or both inputs are empty.",
        )

    # Strategy A: full document
    # The three strategies all need the embedding model, so a load failure is
    # caught once here and reported as "unavailable" rather than crashing the
    # whole match. The scoring engine then drops this component and
    # re-normalises the remaining weights (see scoring/matching_model.py).
    try:
        full_doc = strategy_full_document(cv_text, job_text)
        score_a = full_doc["semantic_score"]

        # Strategy B: section-level
        score_b = 0.0
        if cv_sections and job_sections:
            section = strategy_section_level(cv_sections, job_sections)
            score_b = section["semantic_score"]

        # Strategy C: skill-level
        score_c = 0.0
        if candidate_skills and job_required_skills:
            skill = strategy_skill_level(
                candidate_skills,
                job_required_skills,
                job_preferred_skills,
            )
            score_c = skill["semantic_score"]
    except EmbeddingUnavailableError as exc:
        logger.warning("Semantic matching unavailable: %s", exc)
        return SemanticResult(
            raw_score=0.0,
            normalised_score=0,
            label="Unavailable",
            component_scores={
                "full_document": 0.0,
                "section_level": 0.0,
                "skill_level": 0.0,
            },
            weights={},
            explanation=(
                "Semantic similarity could not be computed: the embedding "
                "model is unavailable in this environment."
            ),
            available=False,
        )

    # Weighted combination — only include strategies where data was provided
    has_section = cv_sections is not None and job_sections is not None
    has_skills = (
        candidate_skills is not None
        and job_required_skills is not None
        and len(candidate_skills) > 0
        and len(job_required_skills) > 0
    )

    active_weights: dict[str, float] = {}
    # full_document always has data (we always have the texts)
    active_weights["full_document"] = w.get("full_document", 0)
    if has_section:
        active_weights["section_level"] = w.get("section_level", 0)
    if has_skills:
        active_weights["skill_level"] = w.get("skill_level", 0)

    # Re-normalise active weights to sum to 1
    total_w = sum(active_weights.values())
    if total_w > 0:
        active_weights = {k: v / total_w for k, v in active_weights.items()}

    raw_score = (
        active_weights.get("full_document", 0) * score_a
        + active_weights.get("section_level", 0) * score_b
        + active_weights.get("skill_level", 0) * score_c
    )

    normalised = int(round(raw_score * 100))
    label = _score_to_label(normalised)
    explanation = _build_explanation(
        normalised, label, score_a, score_b, score_c, active_weights
    )

    return SemanticResult(
        raw_score=round(raw_score, 4),
        normalised_score=normalised,
        label=label,
        component_scores={
            "full_document": score_a,
            "section_level": score_b,
            "skill_level": score_c,
        },
        weights=active_weights,
        explanation=explanation,
    )


def compute_direct_skill_similarity(
    skill_a: str,
    skill_b: str,
) -> float:
    """
    Compute cosine similarity between two individual skill names.

    Useful for comparing a candidate skill against a job requirement
    when you need a quick pairwise score without the full pipeline.

    Returns:
        Float in [0.0, 1.0].
    """
    embs = embed_texts([skill_a, skill_b])
    return cosine_similarity(embs[0], embs[1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_to_label(score_100: int) -> str:
    """Map a 0–100 score to a human-readable label."""
    for threshold, label in _LABEL_THRESHOLDS:
        if score_100 >= threshold:
            return label
    return "Weak"


def _build_explanation(
    normalised: int,
    label: str,
    score_a: float,
    score_b: float,
    score_c: float,
    weights: dict[str, float],
) -> str:
    """Build a one-paragraph explanation of the semantic match."""
    parts = [
        f"Model-estimated semantic compatibility: {normalised}/100 ({label}).",
    ]

    # Strongest signal
    scores = {
        "full document": score_a,
        "section-level": score_b,
        "skill-level": score_c,
    }
    active = {k: v for k, v in scores.items() if v > 0}
    if active:
        best = max(active, key=active.get)
        parts.append(f"Strongest signal: {best} similarity ({active[best]:.2f}).")

    # Weakest signal (if more than one active)
    if len(active) > 1:
        worst = min(active, key=active.get)
        parts.append(f"Lowest signal: {worst} similarity ({active[worst]:.2f}).")

    parts.append(
        "This score represents semantic compatibility between the CV and "
        "job description as estimated by the embedding model; it is not a "
        "prediction of hiring outcome."
    )

    return " ".join(parts)
