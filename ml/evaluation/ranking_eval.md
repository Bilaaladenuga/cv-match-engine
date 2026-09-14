# Ranking Evaluation (Phase 13)

Held-out test sample: 1759 rows (857 No / 444 Potential / 458 Good). Groups with <4 rows are excluded.

Score convention (identical to serving): `P(Good Fit) + 0.5 * P(Potential Fit)`.
Relevance gains: No Fit = 0, Potential Fit = 1, Good Fit = 2 (graded).
Random baseline = seeded within-group score shuffle (50 permutations).

## JD groups - recruiter slates (production ranking scenario)

| Model | P@3 | rand P@3 | R@3 | N@3 | rand N@3 | P@5 | rand P@5 | R@5 | N@5 | rand N@5 | groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.5459 | 0.3623 | 0.2696 | 0.6905 | 0.5313 | 0.5043 | 0.4174 | 0.3776 | 0.7097 | 0.5917 | 69 |
| random_forest | 0.5266 | 0.4493 | 0.2865 | 0.6928 | 0.6038 | 0.4986 | 0.4116 | 0.3854 | 0.7235 | 0.6247 | 69 |
| gradient_boosting | 0.5459 | 0.43 | 0.2927 | 0.7379 | 0.5929 | 0.4812 | 0.4377 | 0.3563 | 0.7379 | 0.6066 | 69 |

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
| logistic_regression | 0.5612 | 0.5443 | 0.4899 | 0.7092 | 0.7086 | 0.5446 | 0.5369 | 0.6844 | 0.7099 | 0.7289 | 130 |
| random_forest | 0.5612 | 0.5537 | 0.4904 | 0.7046 | 0.6868 | 0.5385 | 0.5446 | 0.6777 | 0.7016 | 0.7227 | 130 |
| gradient_boosting | 0.565 | 0.548 | 0.5028 | 0.7113 | 0.6969 | 0.5385 | 0.5462 | 0.6801 | 0.7175 | 0.7165 | 130 |

### Interpretation (CV groups)

- 177 CV groups with >=4 applications (1,224 rows) - statistically
  meaningful, unlike the earlier 300-row sample.
- There is NO lift over random on any metric for any model: ordering
  a candidate's job fits is currently at chance level. The features
  aggregate the whole CV, so they capture what the candidate IS but
  barely vary across jobs - a listwise comparison problem our
  pointwise model cannot see. Likely fixes: per-pair interaction
  features or a pairwise/listwise objective.

