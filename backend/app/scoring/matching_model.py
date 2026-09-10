"""
Hybrid Matching Model — Phase 11

Combines the independent matching engines into a single, explainable
compatibility score:

    overall = w_skills     * skills_score
            + w_semantic   * semantic_score
            + w_experience * experience_score
            + w_education  * education_score
            + w_certs      * certification_score

Design decisions
----------------
    - Weights live in app/scoring/weights.py and are validated, never
      hard-coded at call sites (Phase 12+ will learn them from data).
    - Every result carries a model_version so historical reports stay
      reproducible (Phase 23).
    - The model NEVER claims hiring probability: the score is "model-
      estimated compatibility" and the output text says so explicitly
      (ethics requirement, Phase 27).
    - Inputs are the *outputs of phases 8-10 plus the deterministic
      education/cert matchers* — this module contains no NLP of its own,
      so it is trivially unit-testable.
    - Explanations rank the strongest positive and negative contributors
      by their weighted impact on the final score, not by raw component
      score — that is what actually moved the number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.scoring.weights import DEFAULT_MATCHING_WEIGHTS, validate_weights

# Bump when scoring behaviour changes in a way that alters results.
# Format: match-model-v<MAJOR>.<MINOR>  (minor = tuning, major = redesign)
MODEL_VERSION = "match-model-v0.1"

# Score → human-readable band (0–100 scale)
_BANDS: list[tuple[float, str]] = [
    (85, "Excellent match"),
    (70, "Good match"),
    (55, "Moderate match"),
    (40, "Fair match"),
    (0, "Weak match"),
]

_ETHICS_DISCLAIMER = (
    "This score is a model-estimated compatibility between the CV and the "
    "job description. It is a decision-support signal, not a prediction of "
    "hiring outcomes and not a substitute for human judgement."
)


def score_band(score_0_100: float) -> str:
    """Map a 0-100 score to its human-readable band."""
    for threshold, label in _BANDS:
        if score_0_100 >= threshold:
            return label
    return _BANDS[-1][1]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ComponentScore:
    """One weighted component of the overall score."""

    name: str                 # "skills", "semantic", ...
    raw_score: float          # 0.0–1.0 from the underlying engine
    weight: float
    weighted: float           # raw_score * weight
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw_score": round(self.raw_score, 4),
            "weight": self.weight,
            "weighted": round(self.weighted, 4),
            "evidence": self.evidence,
        }


@dataclass
class MatchResult:
    """Full, explainable output of the hybrid matching model."""

    overall_score: float            # 0.0–1.0
    overall_percent: int            # 0–100 (display)
    band: str                       # e.g. "Good match"
    components: list[ComponentScore]
    weights: dict[str, float]
    model_version: str
    positive_factors: list[str] = field(default_factory=list)
    negative_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    disclaimer: str = _ETHICS_DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "overall_percent": self.overall_percent,
            "band": self.band,
            "components": [c.to_dict() for c in self.components],
            "weights": self.weights,
            "model_version": self.model_version,
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "recommendations": self.recommendations,
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Input bundle
# ---------------------------------------------------------------------------


@dataclass
class MatcherInputs:
    """
    Everything the hybrid model needs, produced by earlier phases.

    Attributes are deliberately plain types so the model can be exercised
    from tests, the API layer, or a future batch pipeline without touching
    the NLP stack.
    """

    # Phase 8 — SkillMatchResult (or anything exposing overall_score and the
    # matched/partial/missing/unknown lists)
    skill_match: object

    # Phase 9 — ExperienceMatchResult (exposes overall_score)
    experience_match: object

    # Phase 10 — SemanticResult (exposes raw_score 0-1)
    semantic_match: object

    # Phase 11 deterministic matchers
    education_match: object                 # EducationMatch (exposes score)
    certification_match: object             # CertificationMatchResult

    # Free-text context used to build recommendations
    job_title: str | None = None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _raw(component: str, inputs: MatcherInputs) -> tuple[float, str]:
    """Extract (raw_score, evidence) for a component from its engine result."""
    if component == "skills":
        r = inputs.skill_match
        pct = getattr(r, "required_coverage", 0.0)
        partial = getattr(r, "required_plus_partial", 0.0)
        evidence = (
            f"{pct:.0%} of required skills matched"
            + (f" ({partial:.0%} including partial matches)" if partial > pct else "")
        )
        return float(getattr(r, "overall_score", 0.0)), evidence

    if component == "semantic":
        r = inputs.semantic_match
        raw = float(getattr(r, "raw_score", getattr(r, "normalised_score", 0.0)))
        if raw > 1.0:  # normalised 0-100 was passed
            raw = raw / 100.0
        label = getattr(r, "label", "")
        evidence = f"Semantic compatibility {raw:.2f}" + (f" ({label})" if label else "")
        return raw, evidence

    if component == "experience":
        r = inputs.experience_match
        level = getattr(r, "experience_level", "")
        evidence = f"Experience fit: {level}" if level else "Experience fit"
        return float(getattr(r, "overall_score", 0.0)), evidence

    if component == "education":
        r = inputs.education_match
        return float(getattr(r, "score", 0.85)), getattr(r, "evidence", "")

    if component == "certifications":
        r = inputs.certification_match
        missing = getattr(r, "missing", [])
        matched = getattr(r, "matched", [])
        if not matched and not missing:
            return float(getattr(r, "score", 0.85)), "No certifications required."
        evidence = f"{len(matched)}/{len(matched) + len(missing)} required certifications held"
        return float(getattr(r, "score", 0.0)), evidence

    raise ValueError(f"Unknown component: {component}")


def _recommendations(inputs: MatcherInputs) -> list[str]:
    """Actionable, grounded recommendations derived from the matcher outputs."""
    recs: list[str] = []

    sk = inputs.skill_match
    for skill in getattr(sk, "missing_skills", [])[:3]:
        recs.append(
            f"Consider building demonstrable experience with {skill} — it is "
            "required by this job but not evident on the CV."
        )
    for skill in getattr(sk, "partial_skills", [])[:2]:
        recs.append(
            f"Strengthen the evidence for {skill} (projects, measurable "
            "outcomes) so it counts as a full match."
        )

    em = inputs.experience_match
    level = getattr(em, "experience_level", "")
    if level in ("below", "near_match"):
        recs.append(
            "Highlight measurable achievements and scope of responsibility to "
            "compensate for being slightly under the experience requirement."
        )

    cm = inputs.certification_match
    for cert in getattr(cm, "missing", [])[:2]:
        recs.append(f"Obtaining the {cert} certification would directly satisfy a stated requirement.")

    return recs


def compute_match_score(
    inputs: MatcherInputs,
    weights: dict[str, float] | None = None,
) -> MatchResult:
    """
    Compute the hybrid compatibility score.

    Args:
        inputs: outputs of the phase 8-11 engines.
        weights: optional override; validated against the canonical keys.
    """
    w = validate_weights(weights or DEFAULT_MATCHING_WEIGHTS)

    components: list[ComponentScore] = []
    for name in ("skills", "semantic", "experience", "education", "certifications"):
        raw, evidence = _raw(name, inputs)
        components.append(
            ComponentScore(
                name=name,
                raw_score=max(0.0, min(1.0, raw)),
                weight=w[name],
                weighted=0.0,
                evidence=evidence,
            )
        )

    overall = sum(c.weight * c.raw_score for c in components)
    for c in components:
        c.weighted = c.weight * c.raw_score

    overall = max(0.0, min(1.0, overall))
    percent = round(overall * 100)

    # --- Explainability: rank by weighted impact ---------------------------
    ranked = sorted(components, key=lambda c: c.weighted, reverse=True)
    positive = [
        f"{c.name.capitalize()}: {c.evidence} (+{c.weighted * 100:.0f} pts)"
        for c in ranked
        if c.raw_score >= 0.75 and c.weighted > 0
    ]
    negative = [
        f"{c.name.capitalize()}: {c.evidence} (+{c.weighted * 100:.0f} of "
        f"{c.weight * 100:.0f} possible pts)"
        for c in sorted(components, key=lambda c: c.weighted)
        if c.raw_score < 0.75
    ]

    return MatchResult(
        overall_score=overall,
        overall_percent=percent,
        band=score_band(percent),
        components=components,
        weights=w,
        model_version=MODEL_VERSION,
        positive_factors=positive,
        negative_factors=negative,
        recommendations=_recommendations(inputs),
    )
