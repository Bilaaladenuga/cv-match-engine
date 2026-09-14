"""
Tests for Phase 13 ranking metrics (ml/evaluation/ranking_metrics.py).

Values are hand-computed. The module lives outside the backend package, so
the repo root is added to sys.path the same way the evaluation scripts do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.ranking_metrics import (  # noqa: E402
    GAIN_MAP,
    aggregates_to_dict,
    dcg,
    evaluate_grouped_ranking,
    ndcg,
    precision_at_k,
    recall_at_k,
)

LOG2_3 = 1.5849625007211562  # log2(3)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


class TestDcg:
    def test_perfect_grading(self):
        # 2 + 1/log2(3) + 0/log2(4)
        assert dcg([2, 1, 0]) == pytest.approx(2 + 1 / LOG2_3)

    def test_discount_matches_definition(self):
        assert dcg([1, 1, 1]) == pytest.approx(1 + 1 / LOG2_3 + 0.5)

    def test_empty(self):
        assert dcg([]) == 0.0


class TestNdcg:
    def test_perfect_ordering_is_one(self):
        assert ndcg([2, 2, 1, 0]) == 1.0

    def test_imperfect_ordering_below_one(self):
        assert 0.0 < ndcg([1, 2, 0]) < 1.0

    def test_all_zero_gains_returns_zero(self):
        # idcg == 0 -> avoid division by zero
        assert ndcg([0, 0, 0]) == 0.0


class TestPrecisionRecall:
    def test_precision_counts_relevant_in_top_k(self):
        # top-2 = [2, 0] -> 1 of 2 relevant
        assert precision_at_k([2, 0, 1], 2) == pytest.approx(0.5)

    def test_precision_all_relevant(self):
        assert precision_at_k([2, 1, 0], 2) == 1.0

    def test_recall_fraction_of_all_relevant(self):
        # 4 relevant total, 3 in top-3
        assert recall_at_k([2, 2, 1, 1, 0, 0], 3) == pytest.approx(0.75)

    def test_recall_zero_when_no_relevant_items(self):
        assert recall_at_k([0, 0, 0], 2) == 0.0


# ---------------------------------------------------------------------------
# Grouped evaluation
# ---------------------------------------------------------------------------


def _slate(gains: list[int], scores: list[float], group: str = "jd1") -> pd.DataFrame:
    return pd.DataFrame({"jd_group": [group] * len(gains), "gain": gains, "score": scores})


class TestEvaluateGroupedRanking:
    def test_perfect_ordering_scores_one(self):
        df = _slate([2, 2, 1, 1, 0, 0], [9, 8, 7, 6, 1, 0])
        agg = evaluate_grouped_ranking(df, "score", "jd_group", k_values=(3, 5))
        assert agg.precision_at_k[3] == 1.0
        assert agg.recall_at_k[3] == pytest.approx(0.75)
        assert agg.ndcg_at_k[3] == 1.0
        # random baseline cannot beat a perfect ordering
        assert agg.random_baseline_ndcg[3] <= 1.0

    def test_small_groups_are_skipped(self):
        df = _slate([2, 1, 0], [3, 2, 1])  # only 3 rows < min_group_size 4
        agg = evaluate_grouped_ranking(df, "score", "jd_group", k_values=(3,))
        assert agg.n_groups_used.get(3, 0) == 0
        assert agg.skipped_groups == 1
        assert agg.precision_at_k[3] == 0.0

    def test_k_larger_than_group_is_skipped_per_k(self):
        # 5-row group: usable at K=3 but not K=10
        df = _slate([2, 1, 1, 0, 0], [5, 4, 3, 2, 1])
        agg = evaluate_grouped_ranking(df, "score", "jd_group", k_values=(3, 10))
        assert agg.n_groups_used[3] == 1
        assert agg.n_groups_used.get(10, 0) == 0

    def test_multiple_groups_averaged(self):
        perfect = _slate([2, 2, 1, 0], [9, 8, 7, 1], group="a")
        half = _slate([2, 2, 1, 0], [9, 1, 8, 7], group="b")  # one relevant buried
        agg = evaluate_grouped_ranking(
            pd.concat([perfect, half], ignore_index=True), "score", "jd_group", k_values=(3,)
        )
        assert agg.n_groups_used[3] == 2
        # group a top-3 gains [2,2,1] -> P@3 = 1.0
        # group b by score desc: 9->2, 8->1, 7->0 -> top-3 [2,1,0] -> P@3 = 2/3
        assert agg.precision_at_k[3] == pytest.approx((1.0 + 2 / 3) / 2)

    def test_inverted_ordering_scores_worse_than_random(self):
        gains = [2, 2, 1, 1, 0, 0]
        df = _slate(gains, [1, 2, 5, 6, 9, 8])  # worst first
        agg = evaluate_grouped_ranking(df, "score", "jd_group", k_values=(3,))
        assert agg.ndcg_at_k[3] < agg.random_baseline_ndcg[3]


class TestGainMapAndSerialization:
    def test_gain_map_is_ordinal(self):
        assert GAIN_MAP == {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

    def test_aggregates_to_dict_json_safe(self):
        df = _slate([2, 2, 1, 0], [9, 8, 7, 1])
        agg = evaluate_grouped_ranking(df, "score", "jd_group", k_values=(3,))
        d = aggregates_to_dict(agg)
        assert d["k_values"] == [3]
        assert isinstance(d["ndcg_at_k"]["3"], float)
        assert d["n_groups_used"]["3"] == 1


# ---------------------------------------------------------------------------
# Serving-consistent score convention (mirrors run_ranking_eval)
# ---------------------------------------------------------------------------


class TestFitScoreConvention:
    def test_fit_score_formula(self):
        from ml.evaluation.run_ranking_eval import fit_score_from_proba

        proba = np.array([0.1, 0.5, 0.4])
        classes = np.array([0, 1, 2])
        # P(Good) + 0.5*P(Potential)
        assert fit_score_from_proba(proba, classes) == pytest.approx(0.4 + 0.25)

    def test_fit_score_robust_to_class_permutation(self):
        from ml.evaluation.run_ranking_eval import fit_score_from_proba

        proba = np.array([0.4, 0.1, 0.5])
        classes = np.array([2, 0, 1])  # same probs, different class order
        assert fit_score_from_proba(proba, classes) == pytest.approx(0.4 + 0.25)
