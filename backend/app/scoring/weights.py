"""
Matching Weights Configuration — Phase 11

Central, validated configuration for how the hybrid matching model combines
its component scores.

Why these defaults?
-------------------
    skills          0.40   Hard evidence of capability. Skill overlap between
                           the CV and the JD is the single strongest proxy for
                           day-one productivity, and it is the component least
                           likely to be fooled by stylistic wording.
    semantic        0.25   Embedding-based similarity captures relevance that
                           discrete skill lists miss (adjacent experience,
                           domain familiarity). Noisy at high values, so it
                           complements rather than dominates skills.
    experience      0.20   Years and seniority matter but are weak signals on
                           their own (title inflation, varied career paths),
                           hence a moderate weight below skills/semantic.
    education       0.10   Often a filter rather than a predictor. Scored, but
                           deliberately low so it never dominates a match.
    certifications  0.05   Required only for a minority of roles; when no
                           certs are requested the component is neutral, so
                           its weight stays small.

These weights are a *starting configuration*, not a conclusion: Phase 12/13
will learn data-driven weights and compare them against this baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MATCHING_WEIGHTS: dict[str, float] = {
    "skills": 0.40,
    "semantic": 0.25,
    "experience": 0.20,
    "education": 0.10,
    "certifications": 0.05,
}

WEIGHT_KEYS = frozenset(DEFAULT_MATCHING_WEIGHTS)

# Tolerance for floating-point sums (0.4 + 0.25 + 0.2 + 0.1 + 0.05)
_SUM_TOLERANCE = 1e-6


class WeightsError(ValueError):
    """Raised when a weights mapping is invalid."""


def validate_weights(
    weights: dict[str, float], check_sum: bool = True
) -> dict[str, float]:
    """
    Validate a weights mapping.

    Rules:
        - Every default key must be present (extra keys are rejected too —
          typos like "certs" should fail loudly, not silently).
        - Values must be numeric and within [0.0, 1.0].
        - Values must sum to 1.0 (within tolerance) unless ``check_sum`` is
          False (used by :func:`normalize_weights`, which rescales itself).

    Returns the validated mapping (same values, ordered by canonical keys).

    Raises:
        WeightsError: with a human-readable message describing all problems.
    """
    if not isinstance(weights, dict):
        raise WeightsError(f"Weights must be a dict, got {type(weights).__name__}")

    problems: list[str] = []

    missing = sorted(WEIGHT_KEYS - weights.keys())
    if missing:
        problems.append(f"missing keys: {', '.join(missing)}")

    unknown = sorted(weights.keys() - WEIGHT_KEYS)
    if unknown:
        problems.append(f"unknown keys: {', '.join(unknown)}")

    for key, value in weights.items():
        if key not in WEIGHT_KEYS:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"'{key}' must be a number, got {type(value).__name__}")
        elif not 0.0 <= float(value) <= 1.0:
            problems.append(f"'{key}' must be within [0.0, 1.0], got {value}")

    if not problems and check_sum:
        total = sum(float(weights[k]) for k in DEFAULT_MATCHING_WEIGHTS)
        if abs(total - 1.0) > _SUM_TOLERANCE:
            problems.append(f"weights must sum to 1.0, got {total:.6f}")

    if problems:
        raise WeightsError("; ".join(problems))

    return {k: float(weights[k]) for k in DEFAULT_MATCHING_WEIGHTS}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Scale an arbitrary positive weights mapping so it sums to exactly 1.0.

    Useful after partial re-weighting (e.g. bumping "skills" without
    re-balancing the rest by hand). Zero-valued keys are preserved as zero.
    """
    # Shape/range validation only — normalization itself fixes the sum.
    validated = validate_weights(
        {**DEFAULT_MATCHING_WEIGHTS, **weights}, check_sum=False
    )
    total = sum(validated.values())
    if total <= 0:
        raise WeightsError("Cannot normalize: total weight is zero")
    return {k: v / total for k, v in validated.items()}


def load_weights(path: str | Path) -> dict[str, float]:
    """Load and validate a weights JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return validate_weights(raw)
    except WeightsError as exc:
        raise WeightsError(f"{path}: {exc}") from exc


def save_weights(weights: dict[str, float], path: str | Path) -> None:
    """Validate then persist a weights mapping as JSON."""
    validated = validate_weights(weights)
    Path(path).write_text(
        json.dumps(validated, indent=2) + "\n", encoding="utf-8"
    )
