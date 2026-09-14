"""
Probability calibration for the trained fit classifier (Phase 13 finding).

The training feature table is stratified (1/3 per class by construction),
but the source data production input follows is ~50/25/25. A classifier
trained on the stratified table inherits the uniform prior and is
systematically overconfident on Good Fit (top calibration bin: predicted
~0.88 vs ~0.36 empirical — see ml/evaluation/classification_eval.md).

Saerens-style prior correction (Saerens et al. 2002): reweight posterior
probabilities from the training prior rho to the natural prior pi,

    p'_c = p_c * (pi_c / rho_c) / Z,    Z = sum_c p_c * (pi_c / rho_c)

This is a post-hoc threshold fix: it recovers accuracy on the natural
distribution but does NOT change discrimination (AUC) — the Phase 13
evaluation both predicted and verified that. The correction is applied at
serving time in model_scorer.py using priors stored in the artifact's
``calibration_`` metadata (written by ml/training/train_baseline.py);
the audited constants below are the fallback for older artifacts.

This module is the single source of truth for the correction math —
ml/evaluation/run_classification_eval.py imports from here too.
"""

from __future__ import annotations

import numpy as np

# Audited class distribution of the unstratified source split
# (data/raw/train.csv, 6,241 rows). Serving fallback when an artifact
# predates calibration metadata. A unit test re-derives these numbers from
# the raw CSV and fails if they drift.
NATURAL_PRIOR: dict[str, float] = {
    "No Fit": 0.5036,
    "Potential Fit": 0.2493,
    "Good Fit": 0.2471,
}

# Prior the model was trained on (stratified feature table, by construction).
TRAINING_PRIOR: dict[str, float] = {
    "No Fit": 1.0 / 3.0,
    "Potential Fit": 1.0 / 3.0,
    "Good Fit": 1.0 / 3.0,
}

CALIBRATION_METHOD = "saerens_prior_correction"


def prior_correct(
    proba_row: np.ndarray | list[float],
    class_names: list[str],
    natural_prior: dict[str, float] | None = None,
    training_prior: dict[str, float] | None = None,
) -> np.ndarray:
    """Rewrite one row of class probabilities from training prior to natural prior.

    ``class_names`` fixes the column order; lookups are by name, so any
    column order works. Falls back to the audited module constants when a
    prior mapping is not supplied. Raises KeyError if a class is missing
    from a supplied mapping — silently skipping a class would corrupt the
    output distribution.
    """
    pi_map = natural_prior if natural_prior is not None else NATURAL_PRIOR
    rho_map = training_prior if training_prior is not None else TRAINING_PRIOR

    p = np.asarray(proba_row, dtype=float)
    scale = np.array([pi_map[name] / rho_map[name] for name in class_names])
    scaled = p * scale
    z = scaled.sum()
    if z <= 0.0:  # degenerate input; return unchanged rather than NaN
        return p
    return scaled / z
