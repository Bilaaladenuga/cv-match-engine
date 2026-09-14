# Classification Evaluation (Phase 13)

Held-out test set: 1759 rows (857 No / 444 Potential / 458 Good). Chance accuracy = 0.487 (majority class).

## logistic_regression

As trained: accuracy **0.4042** · macro-F1 **0.3903** · ROC-AUC (OvR macro) **0.5778** · Good-vs-rest AUC **0.5902**

Prior corrected: accuracy **0.4298** · macro-F1 **0.3771** · Good-vs-rest AUC **0.5836** — same model, probabilities reweighted to the natural class prior (training table is stratified 1/3 per class, the source distribution is ~50/25/25).

| true\pred | No | Potential | Good |
|---|---|---|---|
| No | **365** | 214 | 278 |
| Potential | 136 | **153** | 155 |
| Good | 160 | 105 | **193** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| No Fit | 0.5522 | 0.4259 | 0.4809 | 857 |
| Potential Fit | 0.3242 | 0.3446 | 0.3341 | 444 |
| Good Fit | 0.3083 | 0.4214 | 0.3561 | 458 |

**Errors:** 1048 total — 647 overrated (FP) vs 401 underrated (FN); 610 adjacent, 438 distant (skipped a class).

Top confusion cells: No Fit -> Good Fit (278), No Fit -> Potential Fit (214), Good Fit -> No Fit (160), Potential Fit -> Good Fit (155).

**Feature deltas — potential to good** (155 error vs 153 correct rows):

- `cv_word_count`: 708.348 (errors) vs 647.85 (correct), Δ=60.499
- `n_candidate_skills`: 10.271 (errors) vs 6.98 (correct), Δ=3.291
- `n_required_skills`: 4.432 (errors) vs 2.183 (correct), Δ=2.249
- `n_preferred_skills`: 1.871 (errors) vs 0.987 (correct), Δ=0.884
- `cov_database_n`: 0.794 (errors) vs 0.307 (correct), Δ=0.486
- `cov_machine_learning_n`: 0.426 (errors) vs 0.026 (correct), Δ=0.4

**Feature deltas — good to potential** (105 error vs 193 correct rows):

- `cv_word_count`: 633.457 (errors) vs 675.72 (correct), Δ=42.263
- `n_candidate_skills`: 8.59 (errors) vs 12.021 (correct), Δ=3.43
- `n_preferred_skills`: 0.6 (errors) vs 3.254 (correct), Δ=2.654
- `n_required_skills`: 3.029 (errors) vs 4.415 (correct), Δ=1.386
- `cov_cloud_n`: 0.105 (errors) vs 0.927 (correct), Δ=0.823
- `cov_database_n`: 0.19 (errors) vs 0.886 (correct), Δ=0.696

**Calibration (Good-vs-rest, as trained):**

- [0.0,0.2): n=392, predicted 0.128, empirical 0.171
- [0.2,0.4): n=745, predicted 0.296, empirical 0.263
- [0.4,0.6): n=341, predicted 0.49, empirical 0.255
- [0.6,0.8): n=219, predicted 0.697, empirical 0.388
- [0.8,1.0]: n=62, predicted 0.884, empirical 0.371

**Calibration (Good-vs-rest, prior corrected):**

- [0.0,0.2): n=663, predicted 0.115, empirical 0.219
- [0.2,0.4): n=622, predicted 0.283, empirical 0.244
- [0.4,0.6): n=272, predicted 0.488, empirical 0.331
- [0.6,0.8): n=161, predicted 0.692, empirical 0.335
- [0.8,1.0]: n=41, predicted 0.893, empirical 0.415

## random_forest

As trained: accuracy **0.4076** · macro-F1 **0.3946** · ROC-AUC (OvR macro) **0.5646** · Good-vs-rest AUC **0.5743**

Prior corrected: accuracy **0.4406** · macro-F1 **0.3612** · Good-vs-rest AUC **0.5692** — same model, probabilities reweighted to the natural class prior (training table is stratified 1/3 per class, the source distribution is ~50/25/25).

| true\pred | No | Potential | Good |
|---|---|---|---|
| No | **362** | 261 | 234 |
| Potential | 152 | **150** | 142 |
| Good | 150 | 103 | **205** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| No Fit | 0.5452 | 0.4224 | 0.476 | 857 |
| Potential Fit | 0.2918 | 0.3378 | 0.3132 | 444 |
| Good Fit | 0.3528 | 0.4476 | 0.3946 | 458 |

**Errors:** 1042 total — 637 overrated (FP) vs 405 underrated (FN); 658 adjacent, 384 distant (skipped a class).

Top confusion cells: No Fit -> Potential Fit (261), No Fit -> Good Fit (234), Potential Fit -> No Fit (152), Good Fit -> No Fit (150).

**Feature deltas — potential to good** (142 error vs 150 correct rows):

- `cv_word_count`: 735.845 (errors) vs 648.0 (correct), Δ=87.845
- `n_candidate_skills`: 11.437 (errors) vs 5.653 (correct), Δ=5.783
- `n_required_skills`: 4.338 (errors) vs 2.42 (correct), Δ=1.918
- `skills_per_100_words`: 1.818 (errors) vs 0.999 (correct), Δ=0.819
- `n_preferred_skills`: 1.718 (errors) vs 0.993 (correct), Δ=0.725
- `cov_devops_n`: 0.599 (errors) vs 0.253 (correct), Δ=0.345

**Feature deltas — good to potential** (103 error vs 205 correct rows):

- `cv_word_count`: 613.67 (errors) vs 721.161 (correct), Δ=107.491
- `n_candidate_skills`: 7.447 (errors) vs 12.512 (correct), Δ=5.066
- `n_required_skills`: 2.583 (errors) vs 4.751 (correct), Δ=2.169
- `n_preferred_skills`: 0.631 (errors) vs 2.702 (correct), Δ=2.071
- `cov_devops_n`: 0.252 (errors) vs 0.985 (correct), Δ=0.733
- `skills_per_100_words`: 1.377 (errors) vs 2.028 (correct), Δ=0.65

**Calibration (Good-vs-rest, as trained):**

- [0.0,0.2): n=465, predicted 0.134, empirical 0.215
- [0.2,0.4): n=758, predicted 0.295, empirical 0.239
- [0.4,0.6): n=458, predicted 0.484, empirical 0.349
- [0.6,0.8): n=78, predicted 0.654, empirical 0.218

**Calibration (Good-vs-rest, prior corrected):**

- [0.0,0.2): n=760, predicted 0.119, empirical 0.228
- [0.2,0.4): n=699, predicted 0.29, empirical 0.27
- [0.4,0.6): n=262, predicted 0.476, empirical 0.347
- [0.6,0.8): n=38, predicted 0.637, empirical 0.132

## gradient_boosting

As trained: accuracy **0.4036** · macro-F1 **0.3876** · ROC-AUC (OvR macro) **0.5669** · Good-vs-rest AUC **0.5741**

Prior corrected: accuracy **0.4269** · macro-F1 **0.3702** · Good-vs-rest AUC **0.5694** — same model, probabilities reweighted to the natural class prior (training table is stratified 1/3 per class, the source distribution is ~50/25/25).

| true\pred | No | Potential | Good |
|---|---|---|---|
| No | **374** | 237 | 246 |
| Potential | 147 | **149** | 148 |
| Good | 145 | 126 | **187** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| No Fit | 0.5616 | 0.4364 | 0.4911 | 857 |
| Potential Fit | 0.291 | 0.3356 | 0.3117 | 444 |
| Good Fit | 0.3219 | 0.4083 | 0.36 | 458 |

**Errors:** 1049 total — 631 overrated (FP) vs 418 underrated (FN); 658 adjacent, 391 distant (skipped a class).

Top confusion cells: No Fit -> Good Fit (246), No Fit -> Potential Fit (237), Potential Fit -> Good Fit (148), Potential Fit -> No Fit (147).

**Feature deltas — potential to good** (148 error vs 149 correct rows):

- `cv_word_count`: 723.047 (errors) vs 656.805 (correct), Δ=66.242
- `n_candidate_skills`: 10.709 (errors) vs 5.973 (correct), Δ=4.736
- `n_required_skills`: 4.189 (errors) vs 2.248 (correct), Δ=1.941
- `skills_per_100_words`: 1.73 (errors) vs 1.047 (correct), Δ=0.683
- `n_preferred_skills`: 1.628 (errors) vs 1.101 (correct), Δ=0.528
- `experience_gap_years`: 0.872 (errors) vs 0.371 (correct), Δ=0.501

**Feature deltas — good to potential** (126 error vs 187 correct rows):

- `cv_word_count`: 689.833 (errors) vs 671.321 (correct), Δ=18.512
- `n_candidate_skills`: 8.397 (errors) vs 12.652 (correct), Δ=4.256
- `n_required_skills`: 2.325 (errors) vs 4.909 (correct), Δ=2.584
- `n_preferred_skills`: 1.381 (errors) vs 2.401 (correct), Δ=1.02
- `skills_per_100_words`: 1.351 (errors) vs 2.184 (correct), Δ=0.833
- `experience_gap_years`: 1.369 (errors) vs 0.561 (correct), Δ=0.808

**Calibration (Good-vs-rest, as trained):**

- [0.0,0.2): n=640, predicted 0.114, empirical 0.217
- [0.2,0.4): n=551, predicted 0.289, empirical 0.247
- [0.4,0.6): n=347, predicted 0.497, empirical 0.32
- [0.6,0.8): n=192, predicted 0.692, empirical 0.307
- [0.8,1.0]: n=29, predicted 0.824, empirical 0.448

**Calibration (Good-vs-rest, prior corrected):**

- [0.0,0.2): n=862, predicted 0.101, empirical 0.217
- [0.2,0.4): n=477, predicted 0.288, empirical 0.306
- [0.4,0.6): n=271, predicted 0.492, empirical 0.299
- [0.6,0.8): n=143, predicted 0.693, empirical 0.28
- [0.8,1.0]: n=6, predicted 0.812, empirical 0.667

## Summary — the honest read

1. **No model beats the majority-class baseline on accuracy.** Always predicting
   'No Fit' scores 0.487; the best model variant scores
   0.441. On raw classification accuracy these features are not yet
   competitive — the models' value lies in ranking and grade separation, not in
   thresholded labels.
2. **Discrimination is weak but real:** best Good-vs-rest AUC 0.5902
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

