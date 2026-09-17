"""Tests for the v0.5 dataset audit helpers (ml/datasets/audit_multi_domain.py).

Covers the two pure-logic pieces with hand-computed expectations:
project_field_support (sample→full-scale projection gate) and classify_field
(dominant-field attribution). The module lives outside the backend package,
so the repo root goes on sys.path the same way the ml scripts do.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.audit_multi_domain import classify_field, project_field_support  # noqa: E402


# --- project_field_support ----------------------------------------------------
class TestProjectFieldSupport:
    def test_projection_scales_sample_share_to_full_dataset(self):
        # 15% of a 200-row sample, 80k total → 12,000 projected rows.
        counts = {"finance": 30, "programming": 10}
        projected = project_field_support(counts, sampled_rows=200, dataset_rows_total=80_000)
        assert projected["finance"] == (12_000, 30)
        assert projected["programming"] == (4_000, 10)

    def test_rare_field_projects_to_viable_count(self):
        # The motivating case: law at 1.5% of 80k ≈ 1,200 rows — viable at
        # full scale even though 3 sample rows would fail a raw-count gate.
        counts = {"law": 3}
        projected = project_field_support(counts, sampled_rows=200, dataset_rows_total=80_000)
        assert projected["law"][0] == 1_200
        assert projected["law"][1] == 3

    def test_zero_sample_rows_does_not_divide_by_zero(self):
        assert project_field_support({"finance": 0}, 0, 10_000) == {}

    def test_unknown_field_excluded(self):
        counts = {"unknown": 50, "finance": 10}
        projected = project_field_support(counts, 100, 1_000)
        assert set(projected) == {"finance"}

    def test_zero_count_field_excluded(self):
        assert project_field_support({"finance": 0}, 100, 1_000) == {}

    def test_rounding_is_stable(self):
        # 1/3 of 100 rows against a 1000-row dataset → 333 (not 333.33…).
        projected = project_field_support({"design": 34}, 100, 1_000)
        assert projected["design"][0] == 340


# --- classify_field -----------------------------------------------------------
@dataclass
class _Skill:
    name: str
    category: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class _Taxonomy:
    skills: list
    _by_name: dict = field(default_factory=dict)

    def __post_init__(self):
        self._by_name = {s.name: s for s in self.skills}

    def get_skill(self, name: str):
        return self._by_name.get(name)


@pytest.fixture()
def taxonomy():
    return _Taxonomy(
        [
            _Skill("Python", "programming", ["python3"]),
            _Skill("AWS", "devops", ["amazon web services"]),
            _Skill("PostgreSQL", "database", ["postgres"]),
            _Skill("TensorFlow", "machine_learning"),
            _Skill("Excel", "tools", ["microsoft excel"]),
        ]
    )


@pytest.fixture()
def alias_index(taxonomy):
    from ml.datasets.audit_multi_domain import build_alias_index

    return build_alias_index(taxonomy)


class TestClassifyField:
    def test_dominant_field_attributed(self, taxonomy, alias_index):
        text = "python developer with python3 and tensorflow experience"
        # Hits: programming x2 (python, python3), machine_learning x1.
        result, hits = classify_field(text, alias_index, taxonomy, min_hits=2)
        assert result == "programming"
        assert hits == {"programming": 2, "machine_learning": 1}

    def test_below_min_hits_stays_unknown(self, taxonomy, alias_index):
        # Only 1 programming hit, min_hits=2 → unknown.
        result, hits = classify_field("knows python", alias_index, taxonomy, min_hits=2)
        assert result is None
        assert hits == {"programming": 1}

    def test_no_dominance_stays_unknown(self, taxonomy, alias_index):
        # 4 hits across 4 fields: top field is 25% of hits, below the 40%
        # dominance bar → unknown rather than a coin flip.
        text = "python aws postgres tensorflow"
        result, hits = classify_field(text, alias_index, taxonomy, min_hits=2)
        assert result is None
        assert hits == {"programming": 1, "devops": 1, "database": 1, "machine_learning": 1}

    def test_matches_are_case_insensitive(self, taxonomy, alias_index):
        # Contract: the CALLER lowercases (audit_dataset passes
        # resume_text.lower()); matching itself is plain substring search.
        result, _ = classify_field("microsoft excel expert".lower(), alias_index, taxonomy, min_hits=1)
        assert result == "tools"

    def test_no_hits_returns_none(self, taxonomy, alias_index):
        result, hits = classify_field("chef with 10 years experience", alias_index, taxonomy, min_hits=2)
        assert result is None
        assert hits == {}

    def test_three_char_minimum_skips_short_aliases(self, taxonomy, alias_index):
        # A 2-char alias like "C" would false-positive everywhere; the
        # indexer only feeds aliases >= 3 chars into matching.
        result, hits = classify_field("c language", alias_index, taxonomy, min_hits=1)
        assert result is None
        assert hits == {}
