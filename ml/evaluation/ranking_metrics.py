"""
Ranking metrics for candidate-ranking quality (Phase 13).

The ranking use case: for one job, order candidates by predicted fit and
show the top K to a recruiter. Pointwise classification metrics (accuracy,
F1) do not measure ordering quality; ranked-retrieval metrics do.

Label -> gain mapping
---------------------
Fit labels are ordinal, so graded relevance is the right relevance model:

    No Fit = 0, Potential Fit = 1, Good Fit = 2

NDCG uses standard exponential gain (2**rel - 1). For the binary metrics
(Precision@K, Recall@K) an item counts as relevant when its gain >= 1
(Potential or Good Fit), mirroring what a recruiter would shortlist.

Grouping / leakage rules
------------------------
- JD groups (recruiter slate): all candidate rows sharing one job
  description. This is the production ranking scenario (Phase 15).
- CV groups (candidate view): all job rows sharing one CV. The public
  dataset repeats CV bodies across the upstream train/test boundary, so
  per-CV aggregates are exploratory only (docs/ml-methodology.md 6.1).
- Groups smaller than ``min_group_size`` are skipped; coverage is reported
  so the numbers cannot be silently inflated by tiny groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GAIN_MAP: dict[str, int] = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
DEFAULT_MIN_GROUP = 4
DEFAULT_KS = (3, 5, 10)


# ---------------------------------------------------------------------------
# Core per-list metrics (gains ordered best-first by the model's ranking)
# ---------------------------------------------------------------------------


def dcg(gains: list[float] | np.ndarray) -> float:
    """Discounted cumulative gain with standard log2(i+1) discount."""
    g = np.asarray(gains, dtype=float)
    positions = np.arange(1, len(g) + 1)
    return float(np.sum(g / np.log2(positions + 1)))


def ndcg(gains: list[float] | np.ndarray) -> float:
    """NDCG@len(gains): DCG of the model ordering vs the ideal ordering."""
    ideal = sorted(gains, reverse=True)
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at_k(gains: list[float] | np.ndarray, k: int, relevant_min_gain: int = 1) -> float:
    """Fraction of the top-K that is relevant. Requires len(gains) >= k."""
    top = list(gains)[:k]
    if not top:
        return 0.0
    return float(np.mean([1.0 if g >= relevant_min_gain else 0.0 for g in top]))


def recall_at_k(gains: list[float] | np.ndarray, k: int, relevant_min_gain: int = 1) -> float:
    """Fraction of all relevant items in the group that appear in the top K."""
    top = list(gains)[:k]
    n_relevant_total = sum(1 for g in gains if g >= relevant_min_gain)
    if n_relevant_total == 0:
        return 0.0
    n_relevant_top = sum(1 for g in top if g >= relevant_min_gain)
    return n_relevant_top / n_relevant_total


# ---------------------------------------------------------------------------
# Group-level aggregation
# ---------------------------------------------------------------------------


@dataclass
class GroupAggregates:
    """Aggregated ranking metrics over all usable groups."""

    k_values: list[int]
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    n_groups_used: dict[int, int] = field(default_factory=dict)
    n_rows_used: dict[int, int] = field(default_factory=dict)
    random_baseline_ndcg: dict[int, float] = field(default_factory=dict)
    random_baseline_precision_at_k: dict[int, float] = field(default_factory=dict)
    random_baseline_recall_at_k: dict[int, float] = field(default_factory=dict)
    skipped_groups: int = 0


def evaluate_grouped_ranking(
    df: pd.DataFrame,
    score_col: str,
    group_col: str,
    gain_col: str = "gain",
    k_values: tuple[int, ...] = DEFAULT_KS,
    min_group_size: int = DEFAULT_MIN_GROUP,
    random_perms: int = 50,
    seed: int = 13,
) -> GroupAggregates:
    """Evaluate ranking quality within ``group_col`` groups.

    For each group, rows are ordered by descending ``score_col``; the
    resulting gain sequence is scored with P@K / R@K / NDCG@K. Groups with
    fewer than ``min_group_size`` rows are excluded from every metric for
    which they are too small (their count lands in ``skipped_groups`` per
    first exclusion). A seeded within-group shuffle provides the random
    ordering baseline so "better than chance" is measured, not assumed.
    """
    agg = GroupAggregates(k_values=list(k_values))
    p_sums = {k: 0.0 for k in k_values}
    r_sums = {k: 0.0 for k in k_values}
    n_sums = {k: 0.0 for k in k_values}
    rand_p_sums = {k: 0.0 for k in k_values}
    rand_r_sums = {k: 0.0 for k in k_values}
    rand_sums = {k: 0.0 for k in k_values}
    rng = np.random.default_rng(seed)

    for _, grp in df.groupby(group_col):
        ordered = grp.sort_values(score_col, ascending=False)
        gains = ordered[gain_col].to_list()
        for k in k_values:
            if len(gains) < max(k, min_group_size):
                continue
            p_sums[k] += precision_at_k(gains, k)
            r_sums[k] += recall_at_k(gains, k)
            n_sums[k] += ndcg(gains[:k])
            perm_scores = rng.permutation(ordered[score_col].to_numpy())
            perm_gains = (
                ordered.assign(_perm=perm_scores).sort_values("_perm", ascending=False)[gain_col].to_list()
            )
            rand_sums[k] += ndcg(perm_gains[:k])
            rand_p_sums[k] += precision_at_k(perm_gains, k)
            rand_r_sums[k] += recall_at_k(perm_gains, k)
            agg.n_groups_used[k] = agg.n_groups_used.get(k, 0) + 1
            agg.n_rows_used[k] = agg.n_rows_used.get(k, 0) + len(gains)

    n_total_groups = df[group_col].nunique()
    for k in k_values:
        n_used = agg.n_groups_used.get(k, 0)
        agg.precision_at_k[k] = p_sums[k] / n_used if n_used else 0.0
        agg.recall_at_k[k] = r_sums[k] / n_used if n_used else 0.0
        agg.ndcg_at_k[k] = n_sums[k] / n_used if n_used else 0.0
        agg.random_baseline_ndcg[k] = rand_sums[k] / n_used if n_used else 0.0
        agg.random_baseline_precision_at_k[k] = rand_p_sums[k] / n_used if n_used else 0.0
        agg.random_baseline_recall_at_k[k] = rand_r_sums[k] / n_used if n_used else 0.0
    agg.skipped_groups = n_total_groups - max(agg.n_groups_used.values(), default=0)
    return agg


def aggregates_to_dict(agg: GroupAggregates) -> dict:
    """Serialize aggregates with integer-string keys for JSON output."""
    return {
        "k_values": agg.k_values,
        "precision_at_k": {str(k): round(v, 4) for k, v in agg.precision_at_k.items()},
        "recall_at_k": {str(k): round(v, 4) for k, v in agg.recall_at_k.items()},
        "ndcg_at_k": {str(k): round(v, 4) for k, v in agg.ndcg_at_k.items()},
        "random_baseline_ndcg": {str(k): round(v, 4) for k, v in agg.random_baseline_ndcg.items()},
        "random_baseline_precision_at_k": {
            str(k): round(v, 4) for k, v in agg.random_baseline_precision_at_k.items()
        },
        "random_baseline_recall_at_k": {
            str(k): round(v, 4) for k, v in agg.random_baseline_recall_at_k.items()
        },
        "n_groups_used": {str(k): v for k, v in agg.n_groups_used.items()},
        "n_rows_used": {str(k): v for k, v in agg.n_rows_used.items()},
        "skipped_groups": agg.skipped_groups,
    }
