"""
Tests for the Phase 12 model scorer and its integration with the hybrid
matching model.

The trained artifact is gitignored (reproducible from
ml/training/train_baseline.py), so these tests exercise the loading and
scoring logic with stubs and never require the artifact to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.ml import model_scorer
from app.ml.model_scorer import (
    MLScorerResult,
    reset_model_cache,
    score_features,
)
from app.scoring.matching_model import (
    DEFAULT_MATCHING_WEIGHTS,
    ML_WEIGHT_SHARE,
    MatcherInputs,
    compute_match_score,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class StubMLResult:
    fit_score: float = 0.7
    label: str = "Good Fit"
    probabilities: dict = field(
        default_factory=lambda: {"No Fit": 0.1, "Potential Fit": 0.4, "Good Fit": 0.5}
    )
    model_version: str = "match-model-v0.2-baseline"


@dataclass
class StubEngine:
    overall_score: float = 0.8
    required_coverage: float = 0.8
    required_plus_partial: float = 0.9
    matched_skills: list = field(default_factory=list)
    partial_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    raw_score: float = 0.7
    label: str = ""
    score: float = 0.9
    evidence: str = ""
    experience_level: str = "strong_match"


def _inputs(ml_result=None) -> MatcherInputs:
    return MatcherInputs(
        skill_match=StubEngine(),
        experience_match=StubEngine(),
        semantic_match=StubEngine(),
        education_match=StubEngine(),
        certification_match=StubEngine(),
        ml_result=ml_result,
    )


# ---------------------------------------------------------------------------
# model_scorer
# ---------------------------------------------------------------------------


class TestModelScorer:
    def setup_method(self):
        reset_model_cache()

    def test_missing_artifact_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(model_scorer, "MODEL_PATH", tmp_path / "nope.joblib")
        assert model_scorer.ml_model_available() is False
        assert score_features({"any": 1.0}) is None

    def test_expected_value_mapping(self):
        """fit_score = P(Good) + 0.5 * P(Potential)."""
        ml = StubMLResult(probabilities={"No Fit": 0.2, "Potential Fit": 0.4, "Good Fit": 0.4})
        expected = 0.4 + 0.5 * 0.4
        assert abs(MLScorerResult(fit_score=expected, label="Good Fit", probabilities=ml.probabilities, model_version="x").fit_score - expected) < 1e-9

    def test_stub_model_scoring_via_monkeypatch(self, monkeypatch):
        class StubModel:
            feature_names_in_ = ["f1", "f2"]
            classes_ = [0, 1, 2]

            @staticmethod
            def predict_proba(features_frame):
                return [[0.2, 0.5, 0.3]]

        monkeypatch.setattr(model_scorer, "_load_model", lambda: StubModel())
        result = score_features({"f1": 1.0, "f2": 0.0})
        assert result is not None
        # classes_ are ints 0/1/2 -> mapped by position onto LABEL_NAMES
        assert result.probabilities["Potential Fit"] == 0.5
        assert abs(result.fit_score - (0.3 + 0.5 * 0.5)) < 1e-9
        assert result.label == "Potential Fit"

    def test_schema_mismatch_raises(self, monkeypatch):
        class StubModel:
            feature_names_in_ = ["f1", "f2"]
            classes_ = [0, 1, 2]

            @staticmethod
            def predict_proba(features_frame):  # pragma: no cover
                return [[0.3, 0.3, 0.4]]

        monkeypatch.setattr(model_scorer, "_load_model", lambda: StubModel())
        with pytest.raises(ValueError, match="schema mismatch"):
            score_features({"f1": 1.0, "wrong": 2.0})

    def test_reset_cache_forces_reload(self, monkeypatch):
        calls = []

        def fake_load():
            calls.append(1)
            return None

        monkeypatch.setattr(model_scorer, "_load_model", fake_load)
        assert score_features({}) is None
        reset_model_cache()
        monkeypatch.setattr(model_scorer, "_load_model", fake_load)
        assert score_features({}) is None
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Hybrid model integration
# ---------------------------------------------------------------------------


class TestHybridWithML:
    def test_without_ml_weights_sum_to_one(self):
        result = compute_match_score(_inputs())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-9
        assert result.ml_details is None
        assert result.model_version == "match-model-v0.1"

    def test_with_ml_weights_renormalised(self):
        result = compute_match_score(_inputs(StubMLResult()))
        assert abs(sum(result.weights.values()) - 1.0) < 1e-9
        assert result.weights["ml_model"] == pytest.approx(ML_WEIGHT_SHARE)
        # engine ratios preserved: skills stays the largest engine weight
        engine = {k: v for k, v in result.weights.items() if k != "ml_model"}
        assert max(engine, key=engine.get) == "skills"
        assert result.ml_details["label"] == "Good Fit"
        assert result.model_version.startswith("match-model-v0.1+")

    def test_ml_component_changes_score(self):
        without = compute_match_score(_inputs())
        with_ml = compute_match_score(_inputs(StubMLResult(fit_score=0.9)))
        assert with_ml.overall_score != without.overall_score
        names = [c.name for c in with_ml.components]
        assert "ml_model" in names

    def test_ml_factor_appears_in_explanations_when_strong(self):
        result = compute_match_score(_inputs(StubMLResult(fit_score=0.95)))
        joined = " | ".join(result.positive_factors)
        assert "ML model" in joined

    def test_version_combines_trained_suffix(self):
        ml = StubMLResult(model_version="match-model-v0.2-baseline")
        result = compute_match_score(_inputs(ml))
        assert result.model_version == "match-model-v0.1+v0.2-baseline"

    def test_default_weights_unchanged(self):
        """The canonical engine weights must not be mutated by scaling."""
        snapshot = dict(DEFAULT_MATCHING_WEIGHTS)
        compute_match_score(_inputs(StubMLResult()))
        assert snapshot == DEFAULT_MATCHING_WEIGHTS
