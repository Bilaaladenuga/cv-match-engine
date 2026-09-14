"""
Tests for prior calibration (app/ml/calibration.py) and its integration
into the serving scorer (app/ml/model_scorer.py).

Covers the correction math (hand-computed), the audited-prior regression
guard against data/raw/train.csv, metadata-driven serving behavior, and
graceful degradation on bad metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.ml import model_scorer  # noqa: E402
from app.ml.calibration import (  # noqa: E402
    NATURAL_PRIOR,
    TRAINING_PRIOR,
    prior_correct,
)
from app.ml.model_scorer import reset_model_cache, score_features  # noqa: E402

CLASSES = ["No Fit", "Potential Fit", "Good Fit"]


class TestCorrectionMath:
    def test_identity_when_priors_match(self):
        same = {"No Fit": 0.5, "Potential Fit": 0.25, "Good Fit": 0.25}
        p = np.array([0.2, 0.3, 0.5])
        out = prior_correct(p, CLASSES, natural_prior=same, training_prior=same)
        assert np.allclose(out, p)

    def test_hand_computed_scaling(self):
        # natural 2x training prior for Good: corrected doubles then normalizes
        p = np.array([0.25, 0.25, 0.5])
        out = prior_correct(
            p,
            CLASSES,
            natural_prior={"No Fit": 0.5, "Potential Fit": 0.25, "Good Fit": 0.5},
            training_prior={"No Fit": 0.5, "Potential Fit": 0.25, "Good Fit": 0.25},
        )
        # scaled = [0.25, 0.25, 1.0]; z = 1.5
        assert np.allclose(out, [0.25 / 1.5, 0.25 / 1.5, 1.0 / 1.5])

    def test_output_renormalizes_to_one(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            p = rng.dirichlet([1, 1, 1])
            out = prior_correct(p, CLASSES)
            assert out.sum() == pytest.approx(1.0)
            assert (out >= 0).all()

    def test_boosts_no_fit_under_default_priors(self):
        # Default audited priors: No Fit is under-predicted by a
        # uniform-prior model, so correction must RAISE its share.
        p = np.array([1 / 3, 1 / 3, 1 / 3])
        out = prior_correct(p, CLASSES)
        assert out[0] > 1 / 3  # No Fit boosted
        assert out[1] < 1 / 3 and out[2] < 1 / 3

    def test_preserves_within_class_ordering(self):
        p = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
        out = np.array([prior_correct(row, CLASSES) for row in p])
        assert out[0, 2] > out[1, 2]  # A has more Good than B, still true
        assert out[0, 0] < out[1, 0]

    def test_column_order_independence(self):
        # Same class->probability assignment expressed in two column orders:
        # No=0.2, Potential=0.3, Good=0.5.
        canonical = prior_correct([0.2, 0.3, 0.5], CLASSES)
        shuffled = prior_correct(
            [0.5, 0.2, 0.3], ["Good Fit", "No Fit", "Potential Fit"]
        )
        assert shuffled[0] == pytest.approx(canonical[2])  # Good
        assert shuffled[1] == pytest.approx(canonical[0])  # No
        assert shuffled[2] == pytest.approx(canonical[1])  # Potential

    def test_missing_class_name_raises(self):
        with pytest.raises(KeyError):
            prior_correct([0.3, 0.3, 0.4], ["No Fit", "Potential", "Good Fit"])

    def test_degenerate_zero_row_returned_unchanged(self):
        out = prior_correct([0.0, 0.0, 0.0], CLASSES)
        assert np.allclose(out, [0.0, 0.0, 0.0])


class TestAuditedPriors:
    def test_natural_prior_matches_source_csv(self):
        """Regression guard: constants must match data/raw/train.csv."""
        raw = REPO_ROOT / "data" / "raw" / "train.csv"
        if not raw.exists():
            pytest.skip("data/raw/train.csv not present on this machine")
        import pandas as pd

        counts = pd.read_csv(raw, usecols=["label"])["label"].value_counts()
        total = counts.sum()
        assert NATURAL_PRIOR["No Fit"] == pytest.approx(counts["No Fit"] / total, abs=5e-4)
        assert NATURAL_PRIOR["Potential Fit"] == pytest.approx(
            counts["Potential Fit"] / total, abs=5e-4
        )
        assert NATURAL_PRIOR["Good Fit"] == pytest.approx(counts["Good Fit"] / total, abs=5e-4)

    def test_priors_sum_to_one(self):
        assert sum(NATURAL_PRIOR.values()) == pytest.approx(1.0, abs=1e-3)
        assert sum(TRAINING_PRIOR.values()) == pytest.approx(1.0, abs=1e-9)


class TestScorerIntegration:
    def _stub(self, calibration_):
        class StubModel:
            feature_names_in_ = ["f1"]
            classes_ = [0, 1, 2]

            @staticmethod
            def predict_proba(features_frame):
                return [[1 / 3, 1 / 3, 1 / 3]]

        StubModel.calibration_ = calibration_
        return StubModel

    def test_metadata_drives_correction(self, monkeypatch):
        # Natural = training => no-op; result equals raw probabilities.
        third = {c: 1 / 3 for c in CLASSES}
        monkeypatch.setattr(
            model_scorer, "_load_model", lambda: self._stub(
                {"method": "saerens_prior_correction", "natural_prior": third, "training_prior": third}
            )
        )
        reset_model_cache()
        result = score_features({"f1": 1.0})
        assert result.calibration_method == "saerens_prior_correction"
        assert result.probabilities["No Fit"] == pytest.approx(1 / 3)
        assert result.raw_probabilities["No Fit"] == pytest.approx(1 / 3)

    def test_default_priors_boost_no_fit(self, monkeypatch):
        # Stub WITHOUT calibration_ -> fallback to audited module constants.
        class BareModel:
            feature_names_in_ = ["f1"]
            classes_ = [0, 1, 2]

            @staticmethod
            def predict_proba(features_frame):
                return [[1 / 3, 1 / 3, 1 / 3]]

        monkeypatch.setattr(model_scorer, "_load_model", lambda: BareModel())
        reset_model_cache()
        result = score_features({"f1": 1.0})
        assert result.calibration_method == "saerens_prior_correction"
        assert result.probabilities["No Fit"] > result.raw_probabilities["No Fit"]

    def test_bad_metadata_degrades_to_raw(self, monkeypatch):
        bad = {"natural_prior": {"Bogus": 1.0}, "training_prior": {c: 1 / 3 for c in CLASSES}}
        monkeypatch.setattr(model_scorer, "_load_model", lambda: self._stub(bad))
        reset_model_cache()
        result = score_features({"f1": 1.0})
        assert result.calibration_method is None
        assert result.probabilities == result.raw_probabilities

    def test_noop_calibration_method_stamped(self, monkeypatch):
        third = {c: 1 / 3 for c in CLASSES}
        monkeypatch.setattr(
            model_scorer, "_load_model", lambda: self._stub(
                {"method": "none", "natural_prior": third, "training_prior": third}
            )
        )
        reset_model_cache()
        result = score_features({"f1": 1.0})
        assert result.calibration_method == "none"

    def test_real_artifact_metadata_is_loadable(self):
        """The committed training script must embed calibration metadata."""
        artifact = REPO_ROOT / "ml" / "models" / "baseline_gradient_boosting.joblib"
        if not artifact.exists():
            pytest.skip("artifact not present (gitignored; reproducible from ml/training)")
        import joblib

        model = joblib.load(artifact)
        meta = getattr(model, "calibration_", None)
        assert meta, "artifact lacks calibration_ metadata; retrain with ml/training/train_baseline.py"
        assert meta["method"] == "saerens_prior_correction"
        assert set(meta["natural_prior"]) == set(CLASSES)
        assert abs(sum(meta["natural_prior"].values()) - 1.0) < 1e-6
