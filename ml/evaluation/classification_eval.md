# Classification Evaluation (Phase 13)

Held-out test set: 1759 rows (857 No / 444 Potential / 458 Good). Chance accuracy = 0.487 (majority class).

## logistic_regression

As trained: accuracy **0.4042** · macro-F1 **0.3906** · ROC-AUC (OvR macro) **0.5761** · Good-vs-rest AUC **0.5918**

Prior corrected: accuracy **0.4315** · macro-F1 **0.3792** · Good-vs-rest AUC **0.585** — same model, probabilities reweighted to the natural class prior (training table is stratified 1/3 per class, the source distribution is ~50/25/25).

| true\pred | No | Potential | Good |
|---|---|---|---|
| No | **364** | 214 | 279 |
| Potential | 140 | **153** | 151 |
| Good | 161 | 103 | **194** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| No Fit | 0.5474 | 0.4247 | 0.4783 | 857 |
| Potential Fit | 0.3255 | 0.3446 | 0.3348 | 444 |
| Good Fit | 0.3109 | 0.4236 | 0.3586 | 458 |

**Errors:** 1048 total — 644 overrated (FP) vs 404 underrated (FN); 608 adjacent, 440 distant (skipped a class).

Top confusion cells: No Fit -> Good Fit (279), No Fit -> Potential Fit (214), Good Fit -> No Fit (161), Potential Fit -> Good Fit (151).

**Feature deltas — potential to good** (151 error vs 153 correct rows):

- `n_candidate_skills`: 10.497 (errors) vs 6.837 (correct), Δ=3.66
- `n_required_skills`: 4.497 (errors) vs 2.157 (correct), Δ=2.34
- `n_preferred_skills`: 1.914 (errors) vs 0.987 (correct), Δ=0.927
- `cov_database_n`: 0.821 (errors) vs 0.32 (correct), Δ=0.501
- `cov_machine_learning_n`: 0.437 (errors) vs 0.026 (correct), Δ=0.411
- `cov_cloud_n`: 0.45 (errors) vs 0.046 (correct), Δ=0.405

**Feature deltas — good to potential** (103 error vs 194 correct rows):

- `n_candidate_skills`: 8.99 (errors) vs 12.186 (correct), Δ=3.195
- `n_preferred_skills`: 0.592 (errors) vs 3.263 (correct), Δ=2.671
- `n_required_skills`: 3.058 (errors) vs 4.454 (correct), Δ=1.395
- `cov_cloud_n`: 0.107 (errors) vs 0.959 (correct), Δ=0.852
- `cov_database_n`: 0.204 (errors) vs 0.881 (correct), Δ=0.678
- `cov_machine_learning_n`: 0.049 (errors) vs 0.691 (correct), Δ=0.642

**Calibration (Good-vs-rest, as trained):**

- [0.0,0.2): n=384, predicted 0.128, empirical 0.169
- [0.2,0.4): n=743, predicted 0.295, empirical 0.26
- [0.4,0.6): n=357, predicted 0.491, empirical 0.266
- [0.6,0.8): n=211, predicted 0.694, empirical 0.389
- [0.8,1.0]: n=64, predicted 0.878, empirical 0.359

**Calibration (Good-vs-rest, prior corrected):**

- [0.0,0.2): n=660, predicted 0.116, empirical 0.217
- [0.2,0.4): n=619, predicted 0.282, empirical 0.25
- [0.4,0.6): n=279, predicted 0.484, empirical 0.315
- [0.6,0.8): n=161, predicted 0.689, empirical 0.335
- [0.8,1.0]: n=40, predicted 0.895, empirical 0.45

## random_forest

As trained: accuracy **0.4025** · macro-F1 **0.3897** · ROC-AUC (OvR macro) **0.5652** · Good-vs-rest AUC **0.5755**

Prior corrected: accuracy **0.4503** · macro-F1 **0.377** · Good-vs-rest AUC **0.5703** — same model, probabilities reweighted to the natural class prior (training table is stratified 1/3 per class, the source distribution is ~50/25/25).

| true\pred | No | Potential | Good |
|---|---|---|---|
| No | **361** | 258 | 238 |
| Potential | 155 | **158** | 131 |
| Good | 156 | 113 | **189** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| No Fit | 0.5372 | 0.4212 | 0.4722 | 857 |
| Potential Fit | 0.2987 | 0.3559 | 0.3248 | 444 |
| Good Fit | 0.3387 | 0.4127 | 0.372 | 458 |

**Errors:** 1051 total — 627 overrated (FP) vs 424 underrated (FN); 657 adjacent, 394 distant (skipped a class).

Top confusion cells: No Fit -> Potential Fit (258), No Fit -> Good Fit (238), Good Fit -> No Fit (156), Potential Fit -> No Fit (155).

**Feature deltas — potential to good** (131 error vs 158 correct rows):

- `n_candidate_skills`: 11.023 (errors) vs 6.278 (correct), Δ=4.744
- `n_required_skills`: 4.573 (errors) vs 2.468 (correct), Δ=2.104
- `n_preferred_skills`: 1.756 (errors) vs 1.139 (correct), Δ=0.616
- `cov_database_n`: 0.748 (errors) vs 0.373 (correct), Δ=0.375
- `cov_programming_n`: 0.87 (errors) vs 0.544 (correct), Δ=0.326
- `cov_devops_n`: 0.603 (errors) vs 0.316 (correct), Δ=0.287

**Feature deltas — good to potential** (113 error vs 189 correct rows):

- `n_candidate_skills`: 8.912 (errors) vs 12.259 (correct), Δ=3.348
- `n_preferred_skills`: 0.664 (errors) vs 2.825 (correct), Δ=2.162
- `n_required_skills`: 3.088 (errors) vs 4.651 (correct), Δ=1.562
- `experience_gap_years`: 1.459 (errors) vs 0.694 (correct), Δ=0.765
- `cov_devops_n`: 0.354 (errors) vs 0.974 (correct), Δ=0.62
- `cov_cloud_n`: 0.327 (errors) vs 0.868 (correct), Δ=0.54

**Calibration (Good-vs-rest, as trained):**

- [0.0,0.2): n=507, predicted 0.138, empirical 0.213
- [0.2,0.4): n=766, predicted 0.3, empirical 0.249
- [0.4,0.6): n=422, predicted 0.483, empirical 0.353
- [0.6,0.8): n=64, predicted 0.644, empirical 0.156

**Calibration (Good-vs-rest, prior corrected):**

- [0.0,0.2): n=784, predicted 0.117, empirical 0.228
- [0.2,0.4): n=692, predicted 0.291, empirical 0.266
- [0.4,0.6): n=265, predicted 0.471, empirical 0.343
- [0.6,0.8): n=18, predicted 0.633, empirical 0.222

## gradient_boosting

As trained: accuracy **0.4036** · macro-F1 **0.388** · ROC-AUC (OvR macro) **0.5594** · Good-vs-rest AUC **0.5611**

Prior corrected: accuracy **0.4252** · macro-F1 **0.3695** · Good-vs-rest AUC **0.5553** — same model, probabilities reweighted to the natural class prior (training table is stratified 1/3 per class, the source distribution is ~50/25/25).

| true\pred | No | Potential | Good |
|---|---|---|---|
| No | **375** | 234 | 248 |
| Potential | 142 | **162** | 140 |
| Good | 154 | 131 | **173** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| No Fit | 0.5589 | 0.4376 | 0.4908 | 857 |
| Potential Fit | 0.3074 | 0.3649 | 0.3337 | 444 |
| Good Fit | 0.3084 | 0.3777 | 0.3395 | 458 |

**Errors:** 1049 total — 622 overrated (FP) vs 427 underrated (FN); 647 adjacent, 402 distant (skipped a class).

Top confusion cells: No Fit -> Good Fit (248), No Fit -> Potential Fit (234), Good Fit -> No Fit (154), Potential Fit -> No Fit (142).

**Feature deltas — potential to good** (140 error vs 162 correct rows):

- `n_candidate_skills`: 10.507 (errors) vs 6.765 (correct), Δ=3.742
- `n_required_skills`: 4.721 (errors) vs 1.975 (correct), Δ=2.746
- `n_preferred_skills`: 1.0 (errors) vs 1.654 (correct), Δ=0.654
- `cov_devops_n`: 0.6 (errors) vs 0.265 (correct), Δ=0.335
- `cov_machine_learning_n`: 0.371 (errors) vs 0.086 (correct), Δ=0.285
- `cov_programming_required`: 0.331 (errors) vs 0.088 (correct), Δ=0.242

**Feature deltas — good to potential** (131 error vs 173 correct rows):

- `n_candidate_skills`: 8.962 (errors) vs 12.434 (correct), Δ=3.472
- `n_required_skills`: 2.679 (errors) vs 4.78 (correct), Δ=2.101
- `experience_gap_years`: 1.405 (errors) vs 0.517 (correct), Δ=0.888
- `n_preferred_skills`: 1.71 (errors) vs 2.179 (correct), Δ=0.469
- `cov_programming_n`: 0.733 (errors) vs 1.11 (correct), Δ=0.377
- `cov_cloud_n`: 0.412 (errors) vs 0.786 (correct), Δ=0.374

**Calibration (Good-vs-rest, as trained):**

- [0.0,0.2): n=619, predicted 0.114, empirical 0.208
- [0.2,0.4): n=576, predicted 0.293, empirical 0.267
- [0.4,0.6): n=374, predicted 0.487, empirical 0.321
- [0.6,0.8): n=175, predicted 0.696, empirical 0.28
- [0.8,1.0]: n=15, predicted 0.818, empirical 0.4

**Calibration (Good-vs-rest, prior corrected):**

- [0.0,0.2): n=829, predicted 0.1, empirical 0.229
- [0.2,0.4): n=536, predicted 0.29, empirical 0.271
- [0.4,0.6): n=263, predicted 0.484, empirical 0.338
- [0.6,0.8): n=130, predicted 0.688, empirical 0.262
- [0.8,1.0]: n=1, predicted 0.844, empirical 0.0

## Summary — the honest read

1. **No model beats the majority-class baseline on accuracy.** Always predicting
   'No Fit' scores 0.487; the best model variant scores
   0.450. On raw classification accuracy these features are not yet
   competitive — the models' value lies in ranking and grade separation, not in
   thresholded labels.
2. **Discrimination is weak but real:** best Good-vs-rest AUC 0.5918
   (0.5 = chance). This matches the ranking eval: useful for surfacing Good
   candidates within a slate, useless for binary shortlisting.
3. **Models are overconfident on Good Fit:** in the top calibration bin the
   model predicts ~0.88 probability of Good while the empirical rate is ~0.36.
   Root cause: training on a stratified table taught a uniform prior while the
   source distribution is ~50/25/25. Prior correction recovers 3-5 accuracy
   points by reweighting thresholds but does not change ranking (AUC ~equal).
   Raw probabilities must never be shown to users as 'confidence'.
4. **Volume-proxy shortcut (feature deltas):** candidates overrated into Good
   list ~10.5 skills on average vs ~6.8 for correctly-potential ones; real Good
   fits list ~12.4. The model conflates 'long CV' with 'good fit'. This is a
   spurious correlation the next feature generation should correct for
   (e.g. normalize by CV length, add per-pair depth features).
5. **Ordinal structure is underused:** 38% of errors skip a class entirely
   (e.g. No predicted Good). Ordinal-aware training (ordinal targets, class
   margins) or a two-stage No-vs-rest / grade classifier is the natural fix.

