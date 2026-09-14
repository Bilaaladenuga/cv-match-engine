"""
Tests for Phase 13 classification evaluation helpers
(ml/evaluation/run_classification_eval.py).

Focuses on the pure logic pieces — prior correction, calibration bins, and
error-direction classification — with hand-computed expectations. The
module lives outside the backend package, so the repo root is added to
sys.path the same way the evaluation scripts do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.run_classification_eval import (  # noqa: E402
    LABELS,
    _calibration_bins,
    _error_analysis,
    prior_correct,
)

NATURAL_PRIOR = {"No Fit": 0.5, "Potential Fit": 0.25, "Good Fit": 0.25}


class TestPriorCorrect:
    def test_reweights_toward_natural_prior(self):
        # Uniform trained probs; natural prior doubles No Fit weight.
        proba = np.array([[1 / 3, 1 / 3, 1 / 3]])
        classes = [0, 1, 2]
        out = prior_correct(proba, classes, NATURAL_PRIOR)
        # scaled = [1/3,1/3,1/3] * [1.5, 0.75, 0.75] -> normalize
        raw = np.array([1 / 3 * 1.5, 1 / 3 * 0.75, 1 / 3 * 0.75])
        expected = raw / raw.sum()
        assert np.allclose(out, expected)
        assert out[0, 0] == pytest.approx(0.5)

    def test_rows_renormalize_to_one(self):
        rng = np.random.default_rng(7)
        proba = rng.dirichlet([1, 1, 1], size=20)
        out = prior_correct(proba, [0, 1, 2], NATURAL_PRIOR)
        assert np.allclose(out.sum(axis=1), 1.0)
        assert (out >= 0).all()

    def test_monotone_within_class(self):
        # If candidate A has higher P(Good) than B before correction, the
        # corrected order within that class cannot invert (same scaling).
        proba = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        out = prior_correct(proba, [0, 1, 2], NATURAL_PRIOR)
        assert out[0, 2] > out[1, 2]


class TestCalibrationBins:
    def test_bin_membership_and_counts(self):
        y = np.array([2, 2, 0, 0])  # two Goods, two non-Goods
        proba = np.array(
            [
                [0.1, 0.1, 0.8],
                [0.2, 0.2, 0.6],
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
            ]
        )
        bins = _calibration_bins(y, proba, [0, 1, 2], n_bins=4)["good_vs_rest"]
        top = bins[-1]  # [0.75, 1.0]
        assert top["n"] == 1
        assert top["mean_predicted"] == pytest.approx(0.8)
        assert top["empirical_good_rate"] == 1.0
        low = bins[0]  # [0.0, 0.25) contains both p_good=0.1 rows
        assert low["n"] == 2
        assert low["mean_predicted"] == pytest.approx(0.1)
        assert low["empirical_good_rate"] == 0.0

    def test_empty_bin_is_none_filled(self):
        y = np.array([0, 0])
        proba = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
        bins = _calibration_bins(y, proba, [0, 1, 2], n_bins=5)["good_vs_rest"]
        empty = [b for b in bins if b["n"] == 0]
        assert empty, "expected at least one empty bin"
        for b in empty:
            assert b["mean_predicted"] is None
            assert b["empirical_good_rate"] is None


class TestErrorAnalysis:
    def _frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "label": [
                    "No Fit",
                    "Potential Fit",
                    "Good Fit",
                    "No Fit",
                    "Potential Fit",
                ],
                "label_int": [0, 1, 2, 0, 1],
                "split_row": [0, 1, 2, 3, 4],
                "skill_overlap_ratio": [0.1, 0.4, 0.9, 0.2, 0.5],
                "semantic_similarity": [0.2, 0.5, 0.8, 0.3, 0.6],
            }
        )

    def test_direction_and_distance_counts(self):
        feat = self._frame()
        y_true = feat["label"].to_numpy()
        # predictions: No ok, Potential->Good (over, adjacent), Good->No
        # (under, distant), No->Good (over, distant), Potential ok
        y_pred = np.array(["No Fit", "Good Fit", "No Fit", "Good Fit", "Potential Fit"])
        proba = np.full((5, 3), 1 / 3)
        proba[1, 2] = 0.9  # confident error
        proba[3, 2] = 0.8  # confident error
        res = _error_analysis(feat, y_true, y_pred, proba, [0, 1, 2])
        assert res["n_errors"] == 3
        assert res["overrated"] == 2
        assert res["underrated"] == 1
        assert res["adjacent_errors"] == 1
        assert res["distant_errors"] == 2
        assert res["confusion_cells"]["Potential Fit -> Good Fit"] == 1
        assert res["confusion_cells"]["Good Fit -> No Fit"] == 1
        assert res["confusion_cells"]["No Fit -> Good Fit"] == 1

    def test_top_confident_errors_are_errors_only(self):
        feat = self._frame()
        y_true = feat["label"].to_numpy()
        y_pred = np.array(["No Fit", "Good Fit", "Good Fit", "No Fit", "Potential Fit"])
        proba = np.full((5, 3), 0.2)
        proba[0, 0] = 0.99  # confident CORRECT
        proba[1, 2] = 0.9  # confident error
        res = _error_analysis(feat, y_true, y_pred, proba, [0, 1, 2])
        rows = res["top_confident_errors"]
        assert len(rows) >= 1
        assert all(r["true"] != r["predicted"] for r in rows)
        assert rows[0]["confidence"] == pytest.approx(0.9)

    def test_feature_delta_profile_controls_true_class(self):
        feat = self._frame()
        y_true = feat["label"].to_numpy()
        # Potential Fit: one predicted Good (error), one correct Potential
        y_pred = np.array(["No Fit", "Good Fit", "Good Fit", "No Fit", "Potential Fit"])
        proba = np.full((5, 3), 1 / 3)
        res = _error_analysis(feat, y_true, y_pred, proba, [0, 1, 2])
        prof = res["profile_potential_to_good"]
        assert prof["n_error_rows"] == 1
        assert prof["n_correct_rows"] == 1
        # error row has skill_overlap 0.4, correct row 0.5
        assert prof["skill_overlap_ratio"]["error_rows"] == pytest.approx(0.4)
        assert prof["skill_overlap_ratio"]["correct_rows"] == pytest.approx(0.5)


class TestLabels:
    def test_label_order_is_ordinal(self):
        assert LABELS == ["No Fit", "Potential Fit", "Good Fit"]
