# Model Card — CV–Job Match Classifier

**Version:** `match-model-v0.3.1-baseline` (trained classifier, prior-calibrated) · hybrid engine: `match-model-v0.1`
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

## 3. Features (33)

| Group | Features |
|---|---|
| Skill overlap (5) | `skill_overlap_ratio`, `required_skill_coverage`, `required_plus_partial`, `preferred_skill_coverage`, `certification_match_ratio` |
| Semantic (1) | `semantic_similarity` (int8-quantized MiniLM, shared with serving) |
| Experience (4) | `experience_gap_years`, `experience_score`, `experience_data_available`, `seniority_match` |
| Education (2) | `education_level_score`, `education_field_score` |
| Title (1) | `job_title_similarity` |
| Volume counts (3) | `n_candidate_skills`, `n_required_skills`, `n_preferred_skills` |
| Per-category coverage (17) | `cov_<cat>_required` ×9 (programming, frontend, backend, database, cloud, devops, data_science, machine_learning, other) + `cov_<cat>_n` ×8 demand counts |

All features are computed by the same code path at training and serving
time (single source of truth: `backend/app/ml/feature_extraction.py`).

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-d),
dynamic int8 quantization + 4 torch threads (~2.4× CPU speedup, cosine
retention ≥ 0.95 vs fp32). Changing the embedding model is a major version
bump.

## 4. Evaluation (full 1,759-row held-out set)

Reports: `ml/evaluation/classification_eval.md`, `ml/evaluation/ranking_eval.md`.

### Classification

| model | acc (as trained) | acc (prior-corrected) | macro-F1 | OvR AUC | Good-vs-rest AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.404 | **0.432** | 0.391 | **0.576** | **0.592** |
| random_forest | 0.403 | 0.450 | 0.390 | 0.565 | 0.576 |
| gradient_boosting | 0.404 | 0.425 | 0.388 | 0.559 | 0.561 |

Majority-class baseline: **0.487**. **No model beats it on accuracy.**

### Ranking (recruiter slates, 69 JD groups)

| model | P@5 (rand 0.45) | NDCG@5 (rand 0.64–0.66) |
|---|---|---|
| gradient_boosting | **0.507** | **0.732** |
| random_forest | 0.487 | 0.718 |
| logistic_regression | 0.507 | 0.706 |

Ordering a candidate's jobs (CV groups, 177 groups): **no lift over random**
on any metric — the pointwise feature set does not support listwise
comparison.

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

## 7. Serving calibration (v0.3.1)

- Artifact metadata: `calibration_ = {method, natural_prior,
  training_prior}` written by `ml/training/train_baseline.py` at train
  time — model and calibration are inseparable.
- Serving: `model_scorer.score_features` corrects raw posteriors to the
  natural prior; on missing/invalid metadata it degrades to raw
  probabilities and logs a warning (never fails a match request).
- API: `ml_details.probabilities` = calibrated, `ml_details
  .raw_probabilities` = pre-correction, `ml_details.calibration_method` =
  applied method (`saerens_prior_correction` or `none`).
- Combined version stamp: `match-model-v0.1+v0.3.1-baseline`.

## 8. Reproducibility

- Training: `ml/training/train_baseline.py` (artifacts are gitignored;
  `ml/models/training_report.json` is committed evidence).
- Features: `ml/features/extract_features.py` (deterministic, checkpointed).
- Evaluation: `ml/evaluation/run_classification_eval.py`,
  `ml/evaluation/run_ranking_eval.py`.
- Every Match row and API response carries the combined `model_version`.
