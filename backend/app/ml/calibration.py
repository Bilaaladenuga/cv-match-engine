"""
Probability calibration for the trained fit classifier (Phase 13 finding).

The training feature table is stratified (1/3 per class by construction),
but the source data production input follows is ~50/25/25. A classifier
trained on the stratified table inherits the uniform prior and is
systematically overconfident on Good Fit (top calibration bin: predicted
~0.88 vs ~0.36 empirical — see ml/evaluation/classification_eval.md).

Calibration methods:
    1. Saerens-style prior correction (default): reweight posteriors from
       training prior to natural prior. Post-hoc threshold fix that recovers
       accuracy but does NOT change discrimination (AUC).
    2. Platt scaling: fits a logistic regression on log(p/(1-p)) to produce
       better-calibrated probabilities. Used when the model is retrained
       with calibration awareness.

This module is the single source of truth for calibration math —
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


def platt_scale(
    proba_row: np.ndarray | list[float],
    class_names: list[str],
    platt_params: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Apply Platt scaling to probabilities.

    Platt scaling fits a logistic regression on the log-odds of each class.
    The parameters (A, B) are learned on a validation set. For each class:
        p_calibrated = sigmoid(A * log(p/(1-p)) + B)

    ``platt_params`` maps class names to (A, B) tuples. If not provided,
    returns probabilities unchanged.
    """
    if platt_params is None:
        return np.asarray(proba_row, dtype=float)

    p = np.asarray(proba_row, dtype=float)
    calibrated = np.zeros_like(p)

    for i, name in enumerate(class_names):
        if name in platt_params:
            A, B = platt_params[name]
            # Avoid log(0) and log(1) by clipping
            p_clipped = np.clip(p[i], 1e-7, 1 - 1e-7)
            log_odds = np.log(p_clipped / (1 - p_clipped))
            calibrated[i] = 1.0 / (1.0 + np.exp(-(A * log_odds + B)))
        else:
            calibrated[i] = p[i]

    # Normalize to sum to 1
    z = calibrated.sum()
    if z > 0:
        calibrated = calibrated / z
    else:
        calibrated = p

    return calibrated


def compute_calibration_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
    n_bins: int = 5,
) -> dict:
    """Compute calibration metrics for monitoring.

    Returns:
        - brier_score: mean squared error between predicted and actual
        - ece: expected calibration error (weighted average of bin gaps)
        - calibration_bins: per-bin analysis for Good-vs-rest
    """
    y_good = (y_true == 2).astype(int)
    prob_good = proba[:, 2]

    # Brier score (lower is better, 0 is perfect)
    brier = float(np.mean((prob_good - y_good) ** 2))

    # ECE (Expected Calibration Error)
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    total_samples = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        mask = (prob_good >= edges[i]) & (prob_good < edges[i + 1])
        if i == n_bins - 1:
            mask = (prob_good >= edges[i]) & (prob_good <= edges[i + 1])
        n = mask.sum()
        if n > 0:
            predicted = float(prob_good[mask].mean())
            empirical = float(y_good[mask].mean())
            gap = abs(predicted - empirical)
            ece += (n / total_samples) * gap
            bins.append({
                "bin": f"[{edges[i]:.1f},{edges[i+1]:.1f})",
                "n": int(n),
                "predicted": round(predicted, 4),
                "empirical": round(empirical, 4),
                "gap": round(gap, 4),
            })

    return {
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "calibration_bins": bins,
    }
