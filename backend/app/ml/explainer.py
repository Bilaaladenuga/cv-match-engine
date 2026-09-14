"""
Explainability engine (Phase 14) — model-driven factor explanations.

Translates the Phase 13 error-analysis findings into per-match,
plain-language explanations:

    - LOCAL: per-feature contributions to the fit score via reference
      substitution (a Shapley-style single-reference approximation; for a
      linear model it equals the exact per-coefficient decomposition, which
      the unit tests pin). Contributions are computed on the CALIBRATED
      fit score so the explanation matches what the user is shown.
    - GLOBAL: permutation importance (ml/evaluation/perm_importance.py)
      ranks features for the docs; the local view is what ships per match.

Plain-language layer (grounded in the Phase 13 findings):

    - Findings drive CAUTIONS, not decoration: a strong volume-proxy
      signature (candidate lists many more skills than typical while core
      coverage is weak) triggers an explicit caveat instead of a naive
      "long CV = good" boost (Finding: volume-proxy shortcut).
    - Close grade probabilities are reported as uncertainty, not converted
      into a verdict (Finding: ordinal structure / calibration).
    - Probabilities are framed as compatibility estimates, never hiring
      chances (Ethics, docs/model-card.md).

Degradation rules: the explainer is advisory. Missing reference stats,
missing model, or schema mismatch produce empty explanations with a reason
— never an exception that could fail a match request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Volume-proxy guardrail (Phase 13 finding): the model conflates "long CV"
# with "good fit". If the candidate's skill count is well above the
# training median while required-skill coverage is mediocre, flag it.
VOLUME_RATIO_THRESHOLD = 1.5      # n_candidate_skills vs training median
VOLUME_COVERAGE_MAX = 0.60        # required_skill_coverage considered weak

# Grade-uncertainty framing (Phase 13 finding): adjacent grades with close
# probability are a "borderline" case, not a verdict.
GRADE_UNCERTAINTY_MARGIN = 0.15

# Disclaimer attached to every model explanation.
MODEL_DISCLAIMER = (
    "Model-estimated compatibility only — not a prediction of hiring "
    "outcomes. See docs/model-card.md for known limitations."
)


@dataclass
class Factor:
    """One explained factor, ready for plain-language rendering."""

    feature: str            # raw feature name (traceability)
    label: str              # human-readable label
    contribution: float     # signed contribution to fit_score (calibrated)
    value: float            # candidate's value
    reference: float        # reference (training median) value
    direction: str          # "helps" | "hurts" | "neutral"
    detail: str             # plain-language sentence


@dataclass
class ModelExplanation:
    """Result of explaining one scored match."""

    factors: list[Factor] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    grade_note: str | None = None
    disclaimer: str = MODEL_DISCLAIMER
    method: str = "reference_substitution"
    degraded_reason: str | None = None

    def to_dict(self) -> dict:
        out = {
            "method": self.method,
            "factors": [
                {
                    "feature": f.feature,
                    "label": f.label,
                    "contribution": round(f.contribution, 4),
                    "direction": f.direction,
                    "value": round(f.value, 4),
                    "reference": round(f.reference, 4),
                    "detail": f.detail,
                }
                for f in self.factors
            ],
            "cautions": list(self.cautions),
            "disclaimer": self.disclaimer,
        }
        if self.grade_note:
            out["grade_note"] = self.grade_note
        if self.degraded_reason:
            out["degraded_reason"] = self.degraded_reason
        return out


# Human-readable labels for the features users actually reason about.
# Features outside this map fall back to their raw name.
FEATURE_LABELS: dict[str, str] = {
    "skill_overlap_ratio": "Overall skill overlap",
    "required_skill_coverage": "Required-skills coverage",
    "required_plus_partial": "Required skills met or partially met",
    "preferred_skill_coverage": "Preferred-skills coverage",
    "semantic_similarity": "Resume-to-job semantic similarity",
    "experience_score": "Experience fit",
    "experience_gap_years": "Years-of-experience gap",
    "experience_data_available": "Experience evidence found in resume",
    "seniority_match": "Seniority level fit",
    "education_level_score": "Education level fit",
    "education_field_score": "Education field relevance",
    "certification_match_ratio": "Certifications coverage",
    "job_title_similarity": "Job-title similarity",
    "n_candidate_skills": "Number of skills listed",
    "n_required_skills": "Number of skills the job requires",
    "n_preferred_skills": "Number of preferred skills",
}

# Direction semantics: for these features a HIGHER value than the
# reference is a good thing. Everything else is inverted (counts, gaps).
HIGHER_IS_BETTER: set[str] = {
    "skill_overlap_ratio",
    "required_skill_coverage",
    "required_plus_partial",
    "preferred_skill_coverage",
    "semantic_similarity",
    "experience_score",
    "experience_data_available",
    "seniority_match",
    "education_level_score",
    "education_field_score",
    "certification_match_ratio",
    "job_title_similarity",
}


def _direction(feature: str, contribution: float) -> str:
    if abs(contribution) < 1e-9:
        return "neutral"
    if feature in HIGHER_IS_BETTER:
        return "helps" if contribution > 0 else "hurts"
    # Inverted features: a positive contribution means the value's
    # deviation (e.g. a gap) pushed the score up, which reads as "helps"
    # only if the deviation itself is beneficial; keep it simple and
    # direction-accurate on the score, wording handled in the template.
    return "helps" if contribution > 0 else "hurts"


def _detail(feature: str, value: float, reference: float, contribution: float) -> str:
    label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
    arrow = "raises" if contribution > 0 else "lowers"
    if feature == "experience_gap_years":
        return (
            f"Your experience is {value:.1f} years {'above' if value >= 0 else 'below'} "
            f"what the role asks for; compared with a typical applicant this "
            f"{arrow} your fit score."
        )
    if feature == "n_candidate_skills":
        return (
            f"You list {value:.0f} skills vs {reference:.0f} for a typical applicant; "
            f"this {arrow} the score, but listing more skills alone does not "
            f"demonstrate depth."
        )
    return (
        f"{label}: your value ({value:.2f}) vs a typical applicant "
        f"({reference:.2f}) — this {arrow} the fit score."
    )


def explain_score(
    model,
    features: dict[str, float],
    probabilities: dict[str, float],
    fit_score: float,
) -> ModelExplanation:
    """Explain one scored match via reference-substitution contributions.

    ``probabilities``/``fit_score`` are the CALIBRATED values already shown
    to the user; contributions are derived on top of them so explanations
    always match the displayed score.
    """
    explanation = ModelExplanation()

    reference_stats = getattr(model, "reference_stats_", None)
    medians = (reference_stats or {}).get("medians")
    if not medians:
        explanation.degraded_reason = "model lacks reference statistics; retrain with ml/training/train_baseline.py"
        return explanation

    # Model's expected column order (feature_names_in_) — required by
    # sklearn's feature-name validation; explanations must not reorder it.
    # (numpy arrays have no unambiguous truth value, hence explicit None check.)
    names_attr = getattr(model, "feature_names_in_", None)
    model_order = list(names_attr) if names_attr is not None else []
    if not model_order:
        explanation.degraded_reason = "model lacks feature_names_in_"
        return explanation
    missing = [name for name in model_order if name not in features or name not in medians]
    if missing:
        explanation.degraded_reason = f"no value or training reference for: {missing[:5]}"
        return explanation

    # --- Reference-substitution contributions ------------------------------
    # For each feature: contribution = f(x) - f(x with feature->median),
    # i.e. how much the candidate's actual value adds over the typical
    # applicant's value. Batched: n+1 forward passes on one matrix.
    import pandas as pd

    x = {name: float(features[name]) for name in model_order}
    base_frame = pd.DataFrame([x])[model_order]
    base_proba = model.predict_proba(base_frame)[0]
    base_fit = _fit_from_proba(base_proba, model.classes_)

    rows: list[dict[str, float]] = []
    for name in model_order:
        swapped = dict(x)
        swapped[name] = float(medians[name])
        rows.append(swapped)
    swap_frame = pd.DataFrame(rows)[model_order]
    swap_proba = model.predict_proba(swap_frame)
    swap_fits = np.array([_fit_from_proba(p, model.classes_) for p in swap_proba])

    contributions: dict[str, float] = {
        name: float(base_fit - swap_fits[i]) for i, name in enumerate(model_order)
    }

    factors = [
        Factor(
            feature=name,
            label=FEATURE_LABELS.get(name, name.replace("_", " ")),
            contribution=contributions[name],
            value=float(x[name]),
            reference=float(medians[name]),
            direction=_direction(name, contributions[name]),
            detail=_detail(name, float(x[name]), float(medians[name]), contributions[name]),
        )
        for name in sorted(model_order, key=lambda n: -abs(contributions[n]))
        if abs(contributions[name]) > 1e-4 or name in HIGHER_IS_BETTER
    ]
    explanation.factors = factors

    # --- Cautions grounded in the Phase 13 findings ------------------------
    n_skills = float(x.get("n_candidate_skills", 0.0))
    median_skills = float(medians.get("n_candidate_skills", n_skills))
    coverage = float(x.get("required_skill_coverage", 0.0))
    if (
        median_skills > 0
        and n_skills >= VOLUME_RATIO_THRESHOLD * median_skills
        and coverage <= VOLUME_COVERAGE_MAX
    ):
        explanation.cautions.append(
            "Your resume lists well above the typical number of skills while "
            "required-skill coverage is limited. Long skill lists are a weak "
            "fit signal on their own — depth on the job's core requirements "
            "matters more."
        )

    gap_available = float(x.get("experience_data_available", 1.0))
    if gap_available < 0.5:
        explanation.cautions.append(
            "We could not reliably extract years of experience from your "
            "resume, so the experience factor is less certain than usual."
        )

    # --- Grade uncertainty (calibrated probabilities) ----------------------
    explanation.grade_note = _grade_note(probabilities)
    return explanation


def _fit_from_proba(proba_row: np.ndarray, classes) -> float:
    """P(Good) + 0.5*P(Potential) under the artifact's class ordering."""
    p = {int(c): float(v) for c, v in zip(classes, proba_row, strict=True)}
    return p.get(2, 0.0) + 0.5 * p.get(1, 0.0)


def _grade_note(probabilities: dict[str, float]) -> str | None:
    good = probabilities.get("Good Fit", 0.0)
    potential = probabilities.get("Potential Fit", 0.0)
    no_fit = probabilities.get("No Fit", 0.0)
    ranked = sorted(
        [("Good Fit", good), ("Potential Fit", potential), ("No Fit", no_fit)],
        key=lambda kv: -kv[1],
    )
    (top_name, top_p), (second_name, second_p) = ranked[0], ranked[1]
    if top_p - second_p <= GRADE_UNCERTAINTY_MARGIN:
        return (
            f"The model sees this as a borderline case: {top_name.replace(' Fit', '').lower()} "
            f"({top_p:.0%}) and {second_name.replace(' Fit', '').lower()} ({second_p:.0%}) "
            "are close. Treat the grade as indicative, not decisive."
        )
    return None
