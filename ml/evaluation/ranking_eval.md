# Ranking Evaluation (Phase 13)

Held-out test sample: 300 rows (100 No / 100 Potential / 100 Good). Groups with <4 rows are excluded.

Score convention (identical to serving): `P(Good Fit) + 0.5 * P(Potential Fit)`.
Relevance gains: No Fit = 0, Potential Fit = 1, Good Fit = 2 (graded).
Random baseline = seeded within-group score shuffle (50 permutations).

## JD groups - recruiter slates (production ranking scenario)

| Model | P@3 | R@3 | N@3 | P@5 | R@5 | N@5 | P@10 | R@10 | N@10 | random N@5 | groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.775 | 0.5644 | 0.8979 | 0.8 | 0.8169 | 0.941 | 0.8 | 0.8718 | 0.9639 | 0.873 | 28 |
| random_forest | 0.8167 | 0.5975 | 0.9123 | 0.8143 | 0.8255 | 0.9595 | 0.7667 | 0.8301 | 0.9824 | 0.8824 | 28 |
| gradient_boosting | 0.7917 | 0.5769 | 0.885 | 0.8 | 0.8124 | 0.9367 | 0.8333 | 0.8974 | 0.9751 | 0.8862 | 28 |

### Interpretation (JD slates)

- P@5 ~ 0.80 means 4 of the top-5 shortlisted candidates are relevant
  (Potential or Good Fit) - directly usable as a recruiter shortlist.
- The random baseline is high (~0.88 NDCG@5) because slates are large
  (up to 15 candidates) and mostly relevant, so almost any ordering
  puts relevant items near the top. The honest signal is the LIFT over
  random (~+0.05-0.08 NDCG@5, ~+0.06-0.08 NDCG@3), not the raw value.
- Random Forest edges out the other models on NDCG@3/5 despite losing
  on pointwise accuracy to Logistic Regression - ordering quality and
  thresholding quality are different skills.

## CV groups - one candidate, many jobs (exploratory)

The public dataset reuses CV bodies across the upstream train/test
boundary, so per-CV aggregates may be optimistic. See
docs/ml-methodology.md section 6.1.

NOTE: only 1-3 CV groups in the 300-row sample reach the minimum
group size - these numbers are NOT statistically meaningful. A
reliable per-candidate evaluation needs the full test set extracted
(1,759 rows) and is deferred until the feature pipeline is fast enough.

| Model | P@3 | R@3 | N@3 | P@5 | R@5 | N@5 | P@10 | R@10 | N@10 | random N@5 | groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.8889 | 0.7556 | 0.9023 | 1.0 | 1.0 | 0.9606 | n/a | n/a | n/a | 0.9606 | 1 |
| random_forest | 1.0 | 0.8667 | 0.8935 | 1.0 | 1.0 | 0.9864 | n/a | n/a | n/a | 1.0 | 1 |
| gradient_boosting | 1.0 | 0.8667 | 0.8935 | 1.0 | 1.0 | 1.0 | n/a | n/a | n/a | 0.8879 | 1 |
