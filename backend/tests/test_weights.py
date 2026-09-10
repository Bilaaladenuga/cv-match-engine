"""Tests for Phase 11 — matching weights configuration."""

import json

import pytest

from app.scoring.weights import (
    DEFAULT_MATCHING_WEIGHTS,
    WeightsError,
    load_weights,
    normalize_weights,
    save_weights,
    validate_weights,
)


class TestValidateWeights:
    def test_defaults_are_valid(self):
        assert validate_weights(DEFAULT_MATCHING_WEIGHTS) == DEFAULT_MATCHING_WEIGHTS

    def test_sums_to_one(self):
        assert abs(sum(DEFAULT_MATCHING_WEIGHTS.values()) - 1.0) < 1e-9

    def test_valid_override_passes(self):
        w = {
            "skills": 0.5, "semantic": 0.2, "experience": 0.15,
            "education": 0.1, "certifications": 0.05,
        }
        assert validate_weights(w) == w

    def test_missing_key_rejected(self):
        bad = {k: v for k, v in DEFAULT_MATCHING_WEIGHTS.items() if k != "education"}
        with pytest.raises(WeightsError, match="missing key"):
            validate_weights(bad)

    def test_unknown_key_rejected(self):
        bad = {**DEFAULT_MATCHING_WEIGHTS, "certs": 0.1}
        with pytest.raises(WeightsError, match="unknown key"):
            validate_weights(bad)

    def test_out_of_range_rejected(self):
        bad = {**DEFAULT_MATCHING_WEIGHTS, "skills": 1.5}
        with pytest.raises(WeightsError, match=r"skills.*within"):
            validate_weights(bad)

    def test_negative_rejected(self):
        bad = {**DEFAULT_MATCHING_WEIGHTS, "semantic": -0.1}
        with pytest.raises(WeightsError, match="semantic"):
            validate_weights(bad)

    def test_non_numeric_rejected(self):
        bad = {**DEFAULT_MATCHING_WEIGHTS, "skills": "0.4"}
        with pytest.raises(WeightsError, match="number"):
            validate_weights(bad)

    def test_bool_rejected(self):
        bad = {**DEFAULT_MATCHING_WEIGHTS, "skills": True}
        with pytest.raises(WeightsError, match="number"):
            validate_weights(bad)

    def test_sum_not_one_rejected(self):
        bad = {**DEFAULT_MATCHING_WEIGHTS, "skills": 0.45}  # sums to 1.05
        with pytest.raises(WeightsError, match="sum"):
            validate_weights(bad)

    def test_error_lists_all_problems(self):
        bad = {"skills": 0.4, "foo": 0.6}
        with pytest.raises(WeightsError) as exc:
            validate_weights(bad)
        msg = str(exc.value)
        assert "missing" in msg and "unknown" in msg

    def test_non_dict_rejected(self):
        with pytest.raises(WeightsError, match="dict"):
            validate_weights([0.4, 0.25, 0.2, 0.1, 0.05])


class TestNormalizeWeights:
    def test_already_normalized_is_unchanged(self):
        assert normalize_weights(DEFAULT_MATCHING_WEIGHTS) == pytest.approx(
            DEFAULT_MATCHING_WEIGHTS
        )

    def test_scales_partial_to_full(self):
        # Only 0.9 total -> scaled up to 1.0
        w = {**DEFAULT_MATCHING_WEIGHTS, "skills": 0.35}  # 0.95 total
        out = normalize_weights(w)
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert out["skills"] == pytest.approx(0.35 / 0.95)

    def test_zero_total_raises(self):
        zero = {k: 0.0 for k in DEFAULT_MATCHING_WEIGHTS}
        with pytest.raises(WeightsError, match="zero"):
            normalize_weights(zero)


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "weights.json"
        save_weights(DEFAULT_MATCHING_WEIGHTS, path)
        assert json.loads(path.read_text(encoding="utf-8")) == DEFAULT_MATCHING_WEIGHTS
        assert load_weights(path) == DEFAULT_MATCHING_WEIGHTS

    def test_load_invalid_file_raises_with_path(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"skills": 0.4}), encoding="utf-8")
        with pytest.raises(WeightsError, match="bad.json"):
            load_weights(path)
