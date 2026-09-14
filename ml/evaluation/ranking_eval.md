# Ranking Evaluation (Phase 13)

Held-out test sample: 1759 rows (857 No / 444 Potential / 458 Good). Groups with <4 rows are excluded.

Score convention (identical to serving): `P(Good Fit) + 0.5 * P(Potential Fit)`.
Relevance gains: No Fit = 0, Potential Fit = 1, Good Fit = 2 (graded).
Random baseline = seeded within-group score shuffle (50 permutations).

## JD groups - recruiter slates (production ranking scenario)

| Model | P@3 | rand P@3 | R@3 | N@3 | rand N@3 | P@5 | rand P@5 | R@5 | N@5 | rand N@5 | groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.5411 | 0.4638 | 0.2564 | 0.6809 | 0.574 | 0.5072 | 0.458 | 0.3644 | 0.7055 | 0.6581 | 69 |
| random_forest | 0.5121 | 0.4251 | 0.2553 | 0.7099 | 0.5816 | 0.487 | 0.4522 | 0.3702 | 0.7176 | 0.6588 | 69 |
| gradient_boosting | 0.5411 | 0.43 | 0.289 | 0.7317 | 0.5802 | 0.5072 | 0.4377 | 0.3796 | 0.7323 | 0.6372 | 69 |

### Interpretation (JD slates, full test set)

- The full 1,759-row set is imbalanced (857 No / 444 Potential / 458
  Good), so relevant candidates are the MINORITY in a slate. This is
  the honest production distribution - the earlier 300-row balanced
  sample was an easier, non-representative regime (NDCG@5 ~0.94 there
  vs ~0.71-0.73 here).
- NDCG@5 lift over random: +0.05 to +0.10 depending on model - the
  models DO put the strongest candidates (Good Fit, gain 2) near the
  top of a slate.
- P@5 shows NO lift over random (~0.50 both): the models surface
  Good fits but cannot separate Potential Fit from No Fit when the
  relevant class is the minority. Binary shortlist precision is the
  weak spot - consistent with Potential Fit being the weakest class
  in classification too.
- Practical read: the ranking is useful for surfacing excellent
  candidates, not for binary go/no-go shortlisting.

## CV groups - one candidate, many jobs (exploratory)

The public dataset reuses CV bodies across the upstream train/test
boundary, so per-CV aggregates may be optimistic. See
docs/ml-methodology.md section 6.1.

| Model | P@3 | rand P@3 | R@3 | N@3 | rand N@3 | P@5 | rand P@5 | R@5 | N@5 | rand N@5 | groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.5612 | 0.5424 | 0.4896 | 0.7101 | 0.6998 | 0.5431 | 0.5277 | 0.6806 | 0.706 | 0.7132 | 130 |
| random_forest | 0.5593 | 0.5386 | 0.4912 | 0.6949 | 0.6968 | 0.5369 | 0.5338 | 0.6756 | 0.6991 | 0.7122 | 130 |
| gradient_boosting | 0.5537 | 0.5348 | 0.4877 | 0.6952 | 0.6958 | 0.5338 | 0.5446 | 0.673 | 0.7072 | 0.7268 | 130 |

### Interpretation (CV groups)

- 177 CV groups with >=4 applications (1,224 rows) - statistically
  meaningful, unlike the earlier 300-row sample.
- There is NO lift over random on any metric for any model: ordering
  a candidate's job fits is currently at chance level. The features
  aggregate the whole CV, so they capture what the candidate IS but
  barely vary across jobs - a listwise comparison problem our
  pointwise model cannot see. Likely fixes: per-pair interaction
  features or a pairwise/listwise objective.

