# Model Card — CV–Job Match Classifier

**Version:** `match-model-v0.4.0-baseline` (36-feature schema, prior-calibrated, explainable) · hybrid engine: `match-model-v0.1`
**Status:** baseline (Phase 12–13) — decision-support only, **not** an automated hiring decision tool.
**Date:** 2026-09-14

---

## 1. Model description

Two cooperating scoring layers:

1. **Hybrid deterministic engine** (`match-model-v0.1`): hand-weighted
   combination of skills / semantic / experience / education / certification
   scores from the deterministic pipeline. Fully explainable; weights in
   `backend/app/scoring/weights.py`.
2. **Trained classifier** (`v0.3.0-baseline`, contributes when the artifact
   is present): 3-class fit classifier (No / Potential / Good Fit) over the
   33-feature Phase 12 schema. `fit_score = P(Good) + 0.5·P(Potential)`.
   Combined version stamp: `match-model-v0.1+v0.3.0-baseline`.

Three scikit-learn baselines were trained; Gradient Boosting serves the API
by default (`backend/app/ml/model_scorer.py`), Random Forest is the best
*ranker* (see §4).

## 2. Training data

- Source: HuggingFace `cnamuangtoun/resume-job-description-fit`
  (CV text, JD text, 3-class fit label). 6,241 train / 1,759 test upstream.
- Working train table: **2,100 rows (700 per class, stratified)** extracted
  with the production feature pipeline (`ml/features/extract_features.py`).
- Test table: **all 1,759 upstream test rows**, 0 extraction failures.
- Labels: `No Fit=0, Potential Fit=1, Good Fit=2` (ordinal).
- **Prior note:** the train table is stratified (uniform 1/3 prior); the
  natural source distribution is ~50/25/25. Consequences in §5.

## 3. Features (36, v0.4.0)

| Group | Features |
|---|---|
| Skill overlap (5) | `skill_overlap_ratio`, `required_skill_coverage`, `required_plus_partial`, `preferred_skill_coverage`, `certification_match_ratio` |
| Semantic (1) | `semantic_similarity` (int8-quantized MiniLM, shared with serving) |
| Experience (4) | `experience_gap_years`, `experience_score`, `experience_data_available`, `seniority_match` |
| Education (2) | `education_level_score`, `education_field_score` |
| Title (1) | `job_title_similarity` |
| Volume counts (3) | `n_candidate_skills`, `n_required_skills`, `n_preferred_skills` |
| Per-category coverage (17) | `cov_<cat>_required` ×9 + `cov_<cat>_n` ×8 demand counts |
| CV-length normalization (3, v0.4.0) | `cv_word_count`, `skills_per_100_words`, `cv_length_bucket` |

The v0.4.0 CV-length block exists to kill the volume-proxy shortcut (§5.3):
it gives the model explicit CV length and skill density so the raw skill
count no longer acts as a hidden length proxy.

All features are computed by the same code path at training and serving
time (single source of truth: `backend/app/ml/feature_extraction.py`).

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-d), int8
quantized, served through ONNX Runtime (`fastembed`) — no PyTorch in the
serving path. Per-text cosine against the previous torch vectors is
0.92–0.96 with pairwise ranking behaviour unchanged; peak process memory
falls from ~620 MB to ~261 MB. Changing the embedding model itself is a
major version bump; changing only the serving runtime is not.

## 4. Evaluation (full 1,759-row held-out set)

Reports: `ml/evaluation/classification_eval.md`, `ml/evaluation/ranking_eval.md`,
`ml/evaluation/perm_importance.json`.

### Classification (v0.4.0)

| model | acc (as trained) | acc (prior-corrected) | macro-F1 | OvR AUC | Good-vs-rest AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.404 | 0.430 | 0.390 | **0.578** | **0.590** |
| random_forest | 0.408 | **0.441** | 0.395 | 0.565 | 0.574 |
| gradient_boosting | 0.404 | 0.427 | 0.388 | 0.567 | 0.574 |

Majority-class baseline: **0.487** — still unbeaten on raw accuracy, and
this was EXPECTED for the shortcut fix: the shortcut helped accuracy while
corrupting the ranking signal. The success metric was importance
reallocation (below), not accuracy.

### The volume-proxy shortcut, before and after (the v0.4.0 point)

Permutation importance of `n_candidate_skills` on the fit score:

| schema | importance | rank |
|---|---|---|
| v0.3.2 (33 features) | 0.0231 | **#1 (dominant, 2.2× runner-up)** |
| v0.4.0 (36 features) | 0.0049 | #5 (behind demand counts + overlap) |

A 4.7× importance collapse: the model no longer leans on "long CV".
Reassuringly, `skills_per_100_words` (the density feature) is itself
informative (0.0056) — length normalization transferred signal rather
than deleting it. Ranking quality held up (GB NDCG@5 0.738 vs 0.732
random-baseline-lift similar), error balance shifted slightly toward
fewer distant errors, and the calibrated top-bin reliability is
unchanged (overconfidence is a prior/separation issue, not a shortcut
issue — §5.2 stands).

### Ranking (recruiter slates, 69 JD groups)

| model | P@5 (rand 0.45) | NDCG@5 (rand 0.61–0.66) |
|---|---|---|
| gradient_boosting | 0.481 | **0.738** |
| random_forest | 0.493 | 0.717 |
| logistic_regression | 0.507 | 0.715 |

Ordering a candidate's jobs (CV groups, 177 groups): **no lift over random**
on any metric — the pointwise feature set does not support listwise
comparison. This remains the key open problem.

## 5. Known failure modes (from error analysis)

1. **Below-majority accuracy** — the model's value is grade separation and
   within-slate ordering, not thresholded labels. Never present the raw
   class decision as "the verdict".
2. **Overconfidence on Good Fit** — top calibration bin: predicted ~0.88 vs
   empirical ~0.36. Root cause: uniform training prior vs 50/25/25 world.
   **Mitigation (v0.3.1):** Saerens-style prior correction is applied at
   serving time (`backend/app/ml/calibration.py`); the artifact carries its
   own calibration metadata (`calibration_`), the API exposes both
   calibrated and raw probabilities plus the method name, and the audited
   natural prior is regression-guarded by a unit test against
   `data/raw/train.csv`. Prior correction recovers 3–5 accuracy points but
   does not fix ranking or AUC; calibrated values are compatibility
   estimates, not probabilities of being hired.
3. **Volume-proxy shortcut** — candidates overrated into Good Fit list
   ~10.5 skills on average vs ~6.8 for correct Potential rows (real Good
   rows: ~12.4). "Long CV" is being conflated with "good fit". Next feature
   generation should normalize by CV length.
4. **Ordinal structure underused** — 38% of errors skip a class (e.g. No →
   Good). Ordinal-aware objectives or a two-stage classifier is the natural
   fix.
5. **Potential Fit is the weakest class everywhere** (F1 ≈ 0.33, and binary
   shortlist precision shows no lift over random) — the middle grade is
   where label noise concentrates.

## 6. Limitations & bias

- **Data**: single public dataset; English tech-sector CVs; labels are
  dataset-author judgments, not hiring outcomes. 476 CV bodies repeat
  across the upstream train/test boundary (no exact pair leakage — audited).
- **Protected characteristics** (gender, age, ethnicity, disability),
  university prestige, and employment gaps are **not** features and must
  never become features. No compensation is made for dataset bias against
  these groups; the mitigations are transparency (this card, per-match
  explanations) and the decision-support framing below.
- **Coverage**: prose-style CVs were a historical failure mode (0/40 parsed
  for experience); the full-text fallback now catches ~30%. The
  `experience_data_available` flag lets the model learn missingness instead
  of penalizing it.
- **Scope**: the system reports *model-estimated compatibility*, never
  "chance of getting hired". It is decision-support and must not be the
  sole basis for hiring decisions.

## 7. Serving calibration (v0.3.1) & explainability (v0.3.2)

- Artifact metadata: `calibration_ = {method, natural_prior,
  training_prior}` and `reference_stats_ = {medians, n_train}` written by
  `ml/training/train_baseline.py` at train time — model, calibration, and
  reference statistics are inseparable.
- Serving: `model_scorer.score_features` corrects raw posteriors to the
  natural prior; on missing/invalid metadata it degrades to raw
  probabilities and logs a warning (never fails a match request).
- API: `ml_details.probabilities` = calibrated, `ml_details
  .raw_probabilities` = pre-correction, `ml_details.calibration_method` =
  applied method (`saerens_prior_correction` or `none`).
- Combined version stamp: `match-model-v0.1+v0.3.2-baseline`.

### Explanations (Phase 14, v0.3.2)

- Per match, `ml_details.explanation` carries reference-substitution
  factor contributions computed against the CALIBRATED fit score: each
  factor is the score change from replacing one feature with the training
  median ("a typical applicant"). For linear models the decomposition is
  exact (unit-tested); tree models are approximate.
- Plain-language layer is grounded in the Phase 13 findings: a
  volume-proxy signature (skill count far above typical while required
  coverage is weak) triggers an explicit caution; close grade
  probabilities are reported as borderline instead of a verdict; every
  explanation carries the decision-support disclaimer.
- Global view: `ml/evaluation/perm_importance.py` (permutation importance
  on the fit score). Its output confirms the Phase 13 volume finding at
  the global level — `n_candidate_skills` is the model's most-relied-on
  feature (0.023, ~2.2x the runner-up), while semantic similarity ranks
  far lower (0.0015). The volume caution therefore guards the model's
  PRIMARY signal, not an edge case.
- Local explanations can surface counterintuitive model behavior (e.g.
  penalizing high title similarity, which is rare in training data).
  That is the explainer working: it reports what the model actually
  learned, including its quirks.

## 8. Reproducibility

- Training: `ml/training/train_baseline.py` (artifacts are gitignored;
  `ml/models/training_report.json` is committed evidence).
- Features: `ml/features/extract_features.py` (deterministic, checkpointed).
- Evaluation: `ml/evaluation/run_classification_eval.py`,
  `ml/evaluation/run_ranking_eval.py`.
- Every Match row and API response carries the combined `model_version`.
