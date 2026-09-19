"""
Tests for graceful degradation when the embedding model is unavailable.

The ONNX embedding model is the heaviest component in the pipeline. If it
cannot load (missing files, memory pressure on a small instance), the product
must still return a match report rather than crash. These tests pin that
contract:

    - compute_semantic_match reports unavailability instead of raising
    - compute_match_score drops the semantic component and re-normalises the
      remaining weights (a zero would blame the candidate for an infra fault)
    - get_model raises a typed error and memoises the failure
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ml import embeddings
from app.ml.embeddings import EmbeddingUnavailableError
from app.ml.semantic_matcher import compute_semantic_match
from app.scoring.matching_model import MatcherInputs, compute_match_score


@pytest.fixture(autouse=True)
def _clean_model_state():
    embeddings.reset_model()
    yield
    embeddings.reset_model()


# ---------------------------------------------------------------------------
# semantic_matcher
# ---------------------------------------------------------------------------


def _raise_unavailable(*_args, **_kwargs):
    raise EmbeddingUnavailableError("model files missing")


class TestSemanticMatcherDegrades:
    def test_unavailable_returns_flagged_result(self, monkeypatch):
        monkeypatch.setattr(
            "app.ml.semantic_matcher.strategy_full_document", _raise_unavailable
        )
        result = compute_semantic_match("Python developer", "Python engineer role")

        assert result.available is False
        assert result.normalised_score == 0
        assert result.label == "Unavailable"
        assert "unavailable" in result.explanation.lower()

    def test_unavailable_is_exposed_in_to_dict(self, monkeypatch):
        monkeypatch.setattr(
            "app.ml.semantic_matcher.strategy_full_document", _raise_unavailable
        )
        assert compute_semantic_match("a", "b").to_dict()["available"] is False

    def test_healthy_result_is_flagged_available(self):
        result = compute_semantic_match(
            "Python developer with SQL experience", "Python engineer needed"
        )
        assert result.available is True


# ---------------------------------------------------------------------------
# scoring engine
# ---------------------------------------------------------------------------


def _inputs(semantic_available: bool = True) -> MatcherInputs:
    return MatcherInputs(
        skill_match=SimpleNamespace(
            overall_score=0.80,
            required_coverage=0.8,
            required_plus_partial=0.9,
            matched_skills=["Python"],
            partial_skills=[],
            missing_skills=[],
            unknown_skills=[],
        ),
        experience_match=SimpleNamespace(
            overall_score=0.90, experience_level="strong_match"
        ),
        semantic_match=SimpleNamespace(
            raw_score=0.70, label="Good", available=semantic_available
        ),
        education_match=SimpleNamespace(score=0.85, evidence="B.S. meets B.S."),
        certification_match=SimpleNamespace(score=0.85, matched=[], missing=[]),
    )


class TestScoringDropsUnavailableSemantic:
    def test_semantic_component_removed(self):
        result = compute_match_score(_inputs(semantic_available=False))
        assert "semantic" not in [c.name for c in result.components]
        assert "semantic" not in result.weights

    def test_remaining_weights_renormalised_to_one(self):
        result = compute_match_score(_inputs(semantic_available=False))
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_overall_uses_only_available_components(self):
        result = compute_match_score(_inputs(semantic_available=False))
        expected = sum(c.weight * c.raw_score for c in result.components)
        assert result.overall_score == pytest.approx(expected)

    def test_dropping_is_less_punitive_than_a_zero_semantic(self):
        dropped = compute_match_score(_inputs(semantic_available=False))

        # Same inputs, but semantic present and scored 0 (the wrong way to
        # handle an infra failure).
        zeroed_inputs = _inputs(semantic_available=True)
        zeroed_inputs.semantic_match = SimpleNamespace(
            raw_score=0.0, label="Weak", available=True
        )
        zeroed = compute_match_score(zeroed_inputs)

        assert dropped.overall_score > zeroed.overall_score

    def test_available_path_unchanged(self):
        result = compute_match_score(_inputs(semantic_available=True))
        assert [c.name for c in result.components] == [
            "skills",
            "semantic",
            "experience",
            "education",
            "certifications",
        ]


# ---------------------------------------------------------------------------
# embeddings.get_model
# ---------------------------------------------------------------------------


class TestGetModelRaisesTypedError:
    def test_load_failure_raises_and_is_memoised(self, monkeypatch):
        calls = {"n": 0}

        def _failing(*_args, **_kwargs):
            calls["n"] += 1
            raise OSError("could not find model.onnx")

        monkeypatch.setattr(embeddings, "_local_model_dir", lambda: None)
        monkeypatch.setattr(embeddings, "TextEmbedding", _failing)

        with pytest.raises(EmbeddingUnavailableError):
            embeddings.get_model()
        with pytest.raises(EmbeddingUnavailableError):
            embeddings.get_model()

        assert calls["n"] == 1, "a failed load must not be retried per request"

    def test_reset_clears_the_memoised_failure(self, monkeypatch):
        def _failing(*_args, **_kwargs):
            raise OSError("boom")

        monkeypatch.setattr(embeddings, "_local_model_dir", lambda: None)
        monkeypatch.setattr(embeddings, "TextEmbedding", _failing)

        with pytest.raises(EmbeddingUnavailableError):
            embeddings.get_model()

        embeddings.reset_model()
        assert embeddings._load_error is None

        with pytest.raises(EmbeddingUnavailableError):
            embeddings.get_model()


class TestCacheDirResolution:
    def test_explicit_override_is_respected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FASTEMBED_CACHE_DIR", str(tmp_path / "custom"))
        assert embeddings._model_cache_dir() == str(tmp_path / "custom")

    def test_default_location_is_created_and_returned(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FASTEMBED_CACHE_DIR", raising=False)
        monkeypatch.setattr(embeddings, "_backend_root", lambda: tmp_path)
        result = embeddings._model_cache_dir()
        assert Path(result) == tmp_path / ".fastembed_cache"
        assert Path(result).is_dir(), "the default cache dir should be created"

    def test_unwritable_location_falls_back_to_temp(self, monkeypatch, tmp_path):
        # Simulate the container failure (root-owned backend dir) portably:
        # a *file* where the cache dir must be created makes mkdir raise even
        # with exist_ok=True, exercising the same fallback path.
        blocker = tmp_path / ".fastembed_cache"
        blocker.write_text("not a directory")
        monkeypatch.delenv("FASTEMBED_CACHE_DIR", raising=False)
        monkeypatch.setattr(embeddings, "_backend_root", lambda: tmp_path)
        result = embeddings._model_cache_dir()
        assert "fastembed_cache" in result
        assert Path(result) != blocker
