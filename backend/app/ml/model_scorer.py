"""
Model Scorer — applies the Phase 12 trained classifier alongside the
deterministic hybrid engine.

The model is a 3-class fit classifier (No / Potential / Good Fit) over the
16 Phase 12 features. At inference it contributes two things to a match
report:

    - ml_fit_score:  P(Good Fit) + 0.5 * P(Potential Fit)  ->  0..1
    - ml_label + per-class probabilities for explainability

Design rules
------------
    - Lazy, cached loading from ml/models/baseline_gradient_boosting.joblib
      (the strongest baseline in training_report.json).
    - Artifacts are gitignored and may be absent (fresh clone, slim deploy):
      raise ModelNotAvailableError so callers can degrade to the pure hybrid
      engine instead of crashing. Availability is checked ONCE per process.
    - The artifact's stored feature_names_in_ is the schema contract; if the
      live feature dict diverges from training, scoring fails loudly rather
      than silently producing garbage.
    - The expected-value mapping (1.0 * P(Good) + 0.5 * P(Potential)) treats
      the classes as ordinal and lands "Potential" mid-scale — the same
      convention used to build the training labels' target meaning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ml.explainer import ModelExplanation  # noqa: F401 (type re-export)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "ml" / "models" / "v0.5_ensemble.joblib"

# Label index -> (name, value contribution). Mirrors train_baseline.py's
# LABEL_ORDER = ["No Fit", "Potential Fit", "Good Fit"].
_LABEL_NAMES = ["No Fit", "Potential Fit", "Good Fit"]
_LABEL_VALUES = [0.0, 0.5, 1.0]


class ModelNotAvailableError(RuntimeError):
    """The trained model artifact is missing or cannot be loaded."""


@dataclass
class MLScorerResult:
    """Outcome of applying the trained classifier to one feature vector."""

    fit_score: float                 # expected-value score, 0..1 (calibrated)
    label: str                       # argmax class name (calibrated)
    probabilities: dict[str, float]  # per-class probabilities (calibrated)
    model_version: str               # stamped by the training script
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    calibration_method: str | None = None
    explanation: ModelExplanation | None = None

    def to_dict(self) -> dict:
        out = {
            "fit_score": round(self.fit_score, 4),
            "label": self.label,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "model_version": self.model_version,
        }
        if self.raw_probabilities:
            out["raw_probabilities"] = {
                k: round(v, 4) for k, v in self.raw_probabilities.items()
            }
        if self.calibration_method:
            out["calibration_method"] = self.calibration_method
        if self.explanation is not None:
            out["explanation"] = self.explanation.to_dict()
        return out


_model = None
_load_attempted = False


def _load_model():
    """Load (once) the trained pipeline. Raises ModelNotAvailableError."""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True

    import joblib  # local import: only needed when an artifact exists

    if not MODEL_PATH.exists():
        logger.info("ML model artifact not found at %s; ML scorer disabled", MODEL_PATH)
        return None
    try:
        _model = joblib.load(MODEL_PATH)
        logger.info("ML model loaded from %s", MODEL_PATH)
    except Exception as exc:  # corrupted artifact etc.
        logger.warning("Failed to load ML model artifact: %s", exc)
        _model = None
    return _model


def reset_model_cache() -> None:
    """Forget the cached model (used by tests)."""
    global _model, _load_attempted
    _model = None
    _load_attempted = False


def ml_model_available() -> bool:
    """True if the trained model is loadable in this process."""
    return _load_model() is not None


def score_features(features: dict[str, float]) -> MLScorerResult | None:
    """
    Score one feature dict with the trained classifier.

    Returns None when the model artifact is unavailable — callers are
    expected to fall back to the hybrid engine alone. Raises ValueError if
    the feature schema diverges from training (a bug, not a degradation).
    """
    model = _load_model()
    if model is None:
        return None

    import pandas as pd

    expected = list(getattr(model, "feature_names_in_", []))
    if expected:
        missing = [name for name in expected if name not in features]
        extra = [name for name in features if name not in expected]
        if missing or extra:
            raise ValueError(
                "Feature schema mismatch with trained model: "
                f"missing={missing}, extra={extra}"
            )
        feature_frame = pd.DataFrame([features])[expected]
    else:  # pragma: no cover - artifacts always carry feature_names_in_
        feature_frame = pd.DataFrame([features])

    proba = model.predict_proba(feature_frame)[0]
    raw_classes = [str(c) for c in model.classes_]
    # The artifact is trained on integer-encoded labels (0/1/2 per
    # train_baseline.py's LABEL_ORDER). Map index positions onto the canonical
    # label names; if the model was ever retrained on raw string labels,
    # they pass through unchanged.
    if set(raw_classes) <= {"0", "1", "2"}:
        classes = [_LABEL_NAMES[int(c)] for c in raw_classes]
    else:
        classes = raw_classes

    # --- Prior calibration (Phase 13 finding) ------------------------------
    # The stratified training table teaches a uniform prior; production
    # input follows the natural ~50/25/25 distribution. Correct the raw
    # posteriors to the natural prior (metadata stored in the artifact by
    # the training script; audited constants as fallback). The corrected
    # probabilities drive label, fit_score, and ml_details; raw ones are
    # kept alongside for transparency.
    calibration_meta = getattr(model, "calibration_", None) or {}
    natural_prior = calibration_meta.get("natural_prior")
    training_prior = calibration_meta.get("training_prior")
    # Load Platt params from JSON if not in model artifact
    platt_params = calibration_meta.get("platt_params")
    if platt_params is None:
        try:
            platt_file = REPO_ROOT / "ml" / "models" / "v0.5_platt_params.json"
            if platt_file.exists():
                import json as _json
                platt_data = _json.loads(platt_file.read_text(encoding="utf-8"))
                # Convert list format [A, B] back to tuple format
                raw_params = platt_data.get("platt_params", {})
                platt_params = {k: tuple(v) for k, v in raw_params.items()}
        except Exception:  # noqa: BLE001
            pass
    try:
        from app.ml.calibration import CALIBRATION_METHOD, prior_correct, platt_scale

        # Apply prior correction first
        corrected = prior_correct(proba, classes, natural_prior, training_prior)
        # Then apply Platt scaling if available
        if platt_params:
            corrected = platt_scale(corrected, classes, platt_params)
            applied_method = "saerens_prior_correction+platt_scaling"
        else:
            applied_method = calibration_meta.get("method", CALIBRATION_METHOD)
    except (KeyError, TypeError) as exc:
        # Unknown class names etc. — degrade to raw probabilities rather
        # than fail a match request over calibration metadata.
        logger.warning("Prior calibration unavailable (%s); using raw probabilities", exc)
        corrected = np.asarray(proba, dtype=float)
        applied_method = None

    probabilities = dict(zip(classes, (float(p) for p in corrected), strict=True))
    raw_probabilities = dict(zip(classes, (float(p) for p in proba), strict=True))

    fit_score = sum(
        probabilities.get(name, 0.0) * value
        for name, value in zip(_LABEL_NAMES, _LABEL_VALUES, strict=True)
    )
    label = max(probabilities, key=probabilities.get) if probabilities else _LABEL_NAMES[0]

    version = "match-model-v0.5.0-ensemble"
    try:  # read the authoritative version from the training report
        report = REPO_ROOT / "ml" / "models" / "v0.5_ensemble_report.json"
        if report.exists():
            import json

            version = json.loads(report.read_text(encoding="utf-8")).get(
                "model_version", version
            )
    except Exception:  # noqa: BLE001 - version stamping must never break scoring
        pass

    # --- Explainability (Phase 14) -----------------------------------------
    # Advisory: any failure inside the explainer degrades to no explanation
    # rather than failing the match. Contributions are computed against the
    # calibrated probabilities shown to the user.
    explanation = None
    try:
        from app.ml.explainer import explain_score

        explanation = explain_score(model, features, probabilities, fit_score)
    except Exception:  # noqa: BLE001 - explanations must never break scoring
        logger.exception("Explainer failed; continuing without model explanation")

    return MLScorerResult(
        fit_score=max(0.0, min(1.0, fit_score)),
        label=label,
        probabilities=probabilities,
        model_version=version,
        raw_probabilities=raw_probabilities,
        calibration_method=applied_method,
        explanation=explanation,
    )
