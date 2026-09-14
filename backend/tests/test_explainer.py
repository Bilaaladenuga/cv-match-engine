"""
Tests for the Phase 14 explainability engine (app/ml/explainer.py).

Key property: for a LINEAR model the reference-substitution contributions
must sum exactly to (fit_score - fit_at_reference), i.e. the explanation is
an exact decomposition, not an approximation. Tree models are approximate;
the serving path never promises additivity for them.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.explainer import (
    FEATURE_LABELS,
    GRADE_UNCERTAINTY_MARGIN,
    MODEL_DISCLAIMER,
    VOLUME_COVERAGE_MAX,
    VOLUME_RATIO_THRESHOLD,
    explain_score,
)

CLASSES = np.array([0, 1, 2])
LABELS = ["No Fit", "Potential Fit", "Good Fit"]
FEATURES = ["required_skill_coverage", "semantic_similarity", "n_candidate_skills"]


class LinearStubModel:
    """Affine probability model: fit = 0.5*coverage + 0.3*semantic - 0.01*count.

    Probabilities are affine in the features (not softmax), which is the
    regime where reference-substitution contributions decompose EXACTLY —
    the property this fixture pins. Real tree models are nonlinear, so
    their explanations are approximate (documented, not tested for exactness).
    Implements the sklearn surface the explainer needs: feature_names_in_,
    classes_, predict_proba on DataFrames or ndarrays.
    """

    feature_names_in_ = np.array(FEATURES)
    classes_ = CLASSES
    reference_stats_ = {
        "medians": {
            "required_skill_coverage": 0.5,
            "semantic_similarity": 0.4,
            "n_candidate_skills": 8.0,
        },
        "n_train": 100,
    }

    # coefficients chosen so proba is a softmax over (raw, 0, 0)
    W = {"required_skill_coverage": 0.5, "semantic_similarity": 0.3, "n_candidate_skills": -0.01}

    def _raw(self, matrix):
        return np.array(
            [sum(self.W[name] * row[i] for i, name in enumerate(FEATURES)) for row in matrix]
        )

    def predict_proba(self, frame):
        matrix = frame.to_numpy(dtype=float) if hasattr(frame, "to_numpy") else np.asarray(frame, dtype=float)
        raw = self._raw(matrix)
        # Affine (not softmax) so the fit metric stays linear in features:
        # p_good = 0.2 + 0.5*raw, p_pot = 0.3 - 0.2*raw, rest -> No Fit.
        # Test-domain raw values keep all probabilities in (0, 1).
        p_good = 0.2 + 0.5 * raw
        p_pot = 0.3 - 0.2 * raw
        p_no = 1.0 - p_good - p_pot
        return np.column_stack([p_no, p_pot, p_good])


def _features(**overrides) -> dict[str, float]:
    base = {
        "required_skill_coverage": 0.8,
        "semantic_similarity": 0.7,
        "n_candidate_skills": 6.0,
    }
    base.update(overrides)
    return base


def _probs(good: float, potential: float) -> dict[str, float]:
    return {"No Fit": 1 - good - potential, "Potential Fit": potential, "Good Fit": good}


class TestReferenceSubstitution:
    def test_linear_decomposition_is_exact(self):
        model = LinearStubModel()
        feats = _features()
        exp = explain_score(model, feats, _probs(0.6, 0.3), fit_score=0.75)
        assert not exp.degraded_reason

        # Reference point = all medians; fit(reference) + sum(contributions)
        # must equal fit(x) EXACTLY for a linear model.
        ref = model.reference_stats_["medians"]
        ref_row = np.array([[ref[f] for f in FEATURES]])
        ref_fit = model.predict_proba(ref_row)[0]
        from app.ml.explainer import _fit_from_proba

        fit_at_ref = _fit_from_proba(ref_fit, model.classes_)
        fit_at_x = _fit_from_proba(model.predict_proba(np.array([list(feats.values())]))[0], model.classes_)
        total_contribution = sum(f.contribution for f in exp.factors)
        assert fit_at_ref + total_contribution == pytest.approx(fit_at_x, abs=1e-9)

    def test_factors_sorted_by_absolute_contribution(self):
        model = LinearStubModel()
        exp = explain_score(model, _features(), _probs(0.6, 0.3), fit_score=0.75)
        contribs = [abs(f.contribution) for f in exp.factors]
        assert contribs == sorted(contribs, reverse=True)

    def test_higher_is_better_directions(self):
        model = LinearStubModel()
        exp = explain_score(model, _features(), _probs(0.6, 0.3), fit_score=0.75)
        by_name = {f.feature: f for f in exp.factors}
        # 0.8 coverage vs 0.5 median: removing it lowers fit -> positive contribution
        assert by_name["required_skill_coverage"].direction == "helps"
        assert by_name["semantic_similarity"].direction == "helps"

    def test_inverted_feature_low_count_helps(self):
        # 6 skills vs median 8: fewer skills is treated as fine by the model's
        # negative coefficient (-0.01 * 6 > -0.01 * 8), so removal to median
        # should LOWER the fit -> contribution positive -> "helps".
        model = LinearStubModel()
        exp = explain_score(model, _features(), _probs(0.6, 0.3), fit_score=0.75)
        by_name = {f.feature: f for f in exp.factors}
        assert by_name["n_candidate_skills"].contribution > 0


class TestPlainLanguage:
    def test_every_factor_has_detail_and_known_labels(self):
        model = LinearStubModel()
        exp = explain_score(model, _features(), _probs(0.6, 0.3), fit_score=0.75)
        for f in exp.factors:
            assert f.detail, f"missing detail for {f.feature}"
            assert f.label == FEATURE_LABELS.get(f.feature, f.feature.replace("_", " "))

    def test_disclaimer_always_present(self):
        model = LinearStubModel()
        exp = explain_score(model, _features(), _probs(0.6, 0.3), fit_score=0.75)
        assert exp.disclaimer == MODEL_DISCLAIMER
        assert "not a prediction of hiring" in exp.disclaimer

    def test_volume_proxy_caution_fires(self):
        model = LinearStubModel()
        feats = _features(n_candidate_skills=VOLUME_RATIO_THRESHOLD * 8 + 1, required_skill_coverage=0.3)
        exp = explain_score(model, feats, _probs(0.2, 0.3), fit_score=0.3)
        assert any("skill lists" in c or "Long skill lists" in c for c in exp.cautions)

    def test_volume_caution_does_not_fire_for_strong_coverage(self):
        model = LinearStubModel()
        feats = _features(n_candidate_skills=VOLUME_RATIO_THRESHOLD * 8 + 1, required_skill_coverage=0.95)
        exp = explain_score(model, feats, _probs(0.7, 0.2), fit_score=0.8)
        assert not any("skill lists" in c for c in exp.cautions)

    def test_missing_experience_evidence_caution(self):
        model = LinearStubModel()
        feats = _features()
        feats["experience_data_available"] = 0.0
        model.feature_names_in_ = np.array(FEATURES + ["experience_data_available"])
        model.reference_stats_["medians"]["experience_data_available"] = 1.0
        exp = explain_score(model, feats, _probs(0.5, 0.3), fit_score=0.6)
        assert any("experience" in c.lower() for c in exp.cautions)

    def test_grade_uncertainty_note_for_close_probs(self):
        model = LinearStubModel()
        close = _probs(0.40, 0.35)  # top two within margin
        exp = explain_score(model, _features(), close, fit_score=0.55)
        assert exp.grade_note is not None
        assert "borderline" in exp.grade_note

    def test_no_grade_note_when_clear(self):
        model = LinearStubModel()
        clear = _probs(0.8, 0.05)  # gap far above margin
        exp = explain_score(model, _features(), clear, fit_score=0.85)
        assert exp.grade_note is None or "borderline" not in exp.grade_note


class TestDegradation:
    def test_missing_reference_stats_degrades(self):
        class BareModel:
            feature_names_in_ = np.array(FEATURES)
            classes_ = CLASSES

            def predict_proba(self, frame):  # pragma: no cover
                return np.array([[0.3, 0.3, 0.4]])

        exp = explain_score(BareModel(), _features(), _probs(0.4, 0.3), fit_score=0.55)
        assert exp.degraded_reason is not None
        assert exp.factors == []

    def test_unknown_feature_degrades(self):
        model = LinearStubModel()
        exp = explain_score(model, {"unknown_feature": 1.0}, _probs(0.4, 0.3), fit_score=0.55)
        assert exp.degraded_reason is not None

    def test_to_dict_shape(self):
        model = LinearStubModel()
        exp = explain_score(model, _features(), _probs(0.6, 0.3), fit_score=0.75)
        d = exp.to_dict()
        assert d["method"] == "reference_substitution"
        assert d["disclaimer"] == MODEL_DISCLAIMER
        assert set(d["factors"][0]) == {
            "feature", "label", "contribution", "direction", "value", "reference", "detail"
        }
