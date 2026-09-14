# PROJECT STATUS

## Current Phase
**Phase 12 — Dataset & Feature Engineering** ✅ COMPLETE (model training next)

## Completed Phases

### Phase 1 — Project Initialization ✅
### Phase 2 — Database Architecture ✅
### Phase 3 — CV Document Parser ✅
### Phase 4 — CV Information Extraction ✅
### Phase 5 — Skill Taxonomy ✅
### Phase 6 — Job Description Parser ✅
### Phase 7 — Embedding Engine ✅
### Phase 8 — Skill Matching Engine ✅
### Phase 9 — Experience Matching ✅
- Improved: now infers per-skill experience from CV skills list when not found in role titles
- Skills listed on CV get total career duration with confidence 0.50

### Phase 10 — Semantic Matching ✅
- Overall semantic compatibility scoring (0–100 scale)
- Three strategies: full document (35%), section-level (30%), skill-level (35%)
- Configurable weights with automatic re-normalization
- Human-readable labels: Strong/Good/Moderate/Fair/Weak
- One-paragraph explanation with strongest/weakest signal and disclaimer
- Direct skill-to-skill similarity comparison

### Phase 11 — Matching Model ✅
- `app/scoring/weights.py` — validated, configurable MATCHING_WEIGHTS
  (skills 0.40 / semantic 0.25 / experience 0.20 / education 0.10 / certs 0.05)
  with documented rationale; normalization + JSON persistence helpers
- `app/scoring/education_matcher.py` — deterministic degree-level hierarchy
  (HS < Associate < Bachelor < Master < PhD; higher satisfies lower) + field
  relevance (exact 1.0 / related 0.8 / tech-family 0.6 / unrelated 0.3),
  blended 60/40; neutral 0.85 when no requirement
- `app/scoring/certification_matcher.py` — exact/alias (local alias map),
  semantic fallback at >= 0.80 threshold, no partial credit for credentials;
  neutral 0.85 when no certs required
- `app/scoring/matching_model.py` — hybrid score with clamped components,
  weighted-impact-ranked positive/negative factors, grounded recommendations,
  ethics disclaimer, model_version stamping (match-model-v0.1)
- End-to-end verified: sample CV vs sample JD = 87/100 (Excellent match)

## Key Dependencies
```
torch==2.4.1+cpu
sentence-transformers==3.1.1
transformers==4.44.2
scikit-learn==1.9.0
numpy==1.26.4
```

### Post-Phase 11 Additions ✅
- `docs/ml-methodology.md` — full methodology: pipeline architecture,
  per-component scoring design, weight rationale, feature engineering plan,
  dataset plan, evaluation metrics, versioning, ethics
- `POST /api/matches` endpoint (Phase 19 slice):
  - `app/schemas/match.py` — MatchRequest/MatchResponse Pydantic models
  - `app/services/matching_service.py` — pipeline orchestration + Match/
    MatchExplanation persistence (entity mode: stored resume+job; text mode:
    raw texts persisted under a demo user until Phase 20 auth)
  - `app/api/matches.py` — route with 400/404/422 error mapping
  - 9 API tests on SQLite in-memory (no live PostgreSQL needed)
  - custom weights accepted and validated via request body

### Phase 12 — Dataset & Feature Engineering ✅
- Dataset: `cnamuangtoun/resume-job-description-fit` (HF) — 6,241 train /
  1,759 test CV-JD pairs, labels No/Potential/Good Fit; stored under
  `data/raw/` (gitignored), cleaned + profiled in `data/processed/`
  (30 dupes removed; ~50/25/25 label split)
- `ml/preprocessing/build_dataset.py` — download/clean/profile
- Canonical feature builder: `backend/app/ml/feature_extraction.py`
  (train/serve consistency — same engines as inference), 16 features:
  skill overlap/coverage, semantic similarity, experience gap +
  **experience_data_available missingness gate**, seniority, education,
  certifications, title similarity, skill-count metadata
- `ml/features/extract_features.py` — batch extractor with checkpoint/resume
  (600 train + 300 test rows extracted, 0 failures, ~0.7 rows/s CPU)
- **Bugs found & fixed via dataset validation:**
  - `_trim_education_field` infinite loop on single-word trailer captures
    ("...in in...") — would hang the live API; regression tests added
  - Experience extraction caught 0/40 dataset resumes (over-fit to
    sectioned CV formats): added full-text fallback scan
    (`extract_experience_from_text`, education lines excluded) + prose
    years-claim parser (`extract_total_years_claim`, digits + words,
    "thirteen years of experience")
  - Unknown candidate years no longer encoded as a negative gap (missing ≠ bad)
- Separation verified: top discriminators are skill counts, experience gap,
  required coverage, semantic similarity (Good vs No Fit)

### Phase 12 (continued) — Baseline Models Trained ✅
- `ml/training/train_baseline.py` — Logistic Regression (scaled, class-weighted),
  Random Forest (balanced subsample), Gradient Boosting (sample-weighted);
  fixed seeds; artifacts joblib-dumped to `ml/models/` (gitignored —
  reproducible from script), metrics in `ml/models/training_report.json`
- Test set (300 rows, 100/class): **all three ~0.42-0.44 accuracy vs 0.33
  chance**; macro-F1 ≈ 0.41-0.43; errors concentrate in adjacent classes
  (Potential↔Good), consistent with the ordinal label structure
- End-to-end inference verified: sample CV×JD -> Good Fit p=0.59,
  Potential p=0.38, No Fit p=0.03 (consistent with hybrid 87/100)
- Honest limitations: 16 features are mostly coarse aggregate ratios;
  Potential Fit is the hardest class (its definition is closest to
  annotator judgment); more rows + richer features are the levers,
  documented in docs/ml-methodology.md

### Trained Model Wired Into the Matching Engine ✅
- `app/ml/model_scorer.py` — lazy, cached artifact loading (gitignored;
  returns None on fresh clones so the engine degrades to pure hybrid);
  expected-value mapping P(Good) + 0.5·P(Potential) -> 0..1;
  feature-schema contract enforced against the artifact's feature_names_in_
- `app/ml/feature_extraction.py` refactored: `build_feature_vector`
  assembles features from engine outputs the service already computed —
  the API and the batch trainer land on the same code path (train/serve
  consistency by construction)
- `matching_model.py`: optional sixth `ml_model` component; engine weights
  scaled by 0.75 (ratios preserved), ML takes 0.25 (ML_WEIGHT_SHARE);
  version stamp combines as `match-model-v0.1+v0.2-baseline`; ml_details
  (per-class probabilities) exposed in API responses and persisted in
  Match.feature_values
- Live check: sample CV×JD = 85/100 Excellent (hybrid alone was 87; the
  model's 0.857 fit score agrees)
- API tests made artifact-agnostic (pass with or without the model file)
- Bug caught by tests: integer-encoded classes_ (0/1/2) now mapped onto
  canonical label names before probability lookup

## Dataset Expansion + Retrain (Phase 12 conclusion)
- Training set expanded 600 -> 2,100 rows (700/class); test set 300 (100/class)
- Provenance root cause fixed: extract_features.py now preserves the source
  CSV index in split_row (reset_index had silently destroyed it, which
  corrupted an earlier merge); all ambiguous part files deleted, tables
  re-extracted end-to-end from scratch
- Leakage audit: 0 exact (CV, JD) pair overlaps across the HF train/test
  boundary; 476 CV bodies repeat across splits (upstream dataset property,
  narrow channel with 16 aggregate features) - documented in
  docs/ml-methodology.md as a limitation
- Performance: embedding engine now int8-quantized (EMBEDDINGS_INT8=1
  default) with 4 torch threads -> ~2.4x faster encode measured on this box;
  shared by batch extraction AND API inference, so no train/serve skew;
  cosine retention vs fp32 >= 0.95
- Full re-extraction: 2,400 rows, 0 failures
- Retrain results (test = 300 rows, chance = 33%):
  - LogisticRegression acc 0.413 / macro-F1 0.404 (saturated - feature-bound)
  - RandomForest      acc 0.430 / macro-F1 0.419
  - GradientBoosting  acc 0.463 / macro-F1 0.451  <- best, up from ~0.44
- Caveat: test features were re-extracted under int8 embeddings, so the
  600->2,100 comparison is not a perfectly controlled ablation
- Artifacts re-stamped match-model-v0.2.1-baseline; training_report.json
  regenerated; model-scorer wiring re-verified against the new artifact

## v0.3.0 — Per-Category Coverage Features (breaking the feature ceiling)
- Motivation: the 600->2,100 expansion (+2 acc pts) showed the model is
  feature-starved, not data-starved; aggregate ratios say HOW MUCH matched
  but not WHICH KINDS of skills a candidate covers vs what the job demands
- 17 new cov_* features: 9 per-category required-coverage ratios
  (programming, frontend, backend, database, cloud, devops, data_science,
  machine_learning, other) + 8 per-category demand counts; categories with
  no demanded skills score 0.5 (neutral, not penalized)
- Shared implementation: compute_category_coverage() in
  app/ml/feature_extraction.py used by both batch extraction and the API
  feature path — no train/serve drift
- Full re-extraction with the 33-feature schema: 2,100 train (700/class)
  + 300 test (100/class), 0 failures, 0 NaNs, disjoint split_row ranges
- Retrain results (test = 300 rows, chance = 33%):
  - LogisticRegression acc 0.457 / macro-F1 0.452 (up from 0.413)
  - RandomForest      acc 0.453 / macro-F1 0.446 (up from 0.430)
  - GradientBoosting  acc 0.443 / macro-F1 0.434 (down from 0.463 — within
    noise on a 300-row test set)
- Read: the linear model clearly benefits (category coverage is a directly
  linearly-separable signal); tree ensembles are statistically flat, so
  the remaining ceiling likely needs richer features rather than more rows
- Artifacts re-stamped match-model-v0.3.0-baseline (feature-schema change);
  training_report.json regenerated; model scorer re-verified end-to-end

## Phase 13 (part 1) — Ranking Evaluation (Precision@K / Recall@K / NDCG@K)
- New ml/evaluation/ranking_metrics.py: graded-gain (0/1/2) metrics with
  group aggregation, min-group-size exclusion, coverage reporting, and a
  seeded within-group random-ordering baseline
- New ml/evaluation/run_ranking_eval.py: scores the 300-row test sample
  with all three baselines using the SERVING score convention
  P(Good) + 0.5*P(Potential); groups by JD (recruiter slates, production
  scenario) and by CV (one candidate x many jobs, exploratory)
- JD-slate results (40 groups at K=3, 28 at K=5/10, 244 rows):
  - RandomForest NDCG@3 0.912 / P@3 0.817 (random baseline 0.834)
  - RandomForest NDCG@5 0.960 / P@5 0.814 (random baseline 0.882)
  - LogisticRegression NDCG@5 0.941 / P@5 0.800; GradientBoosting 0.937 / 0.800
  - P@5 ~ 0.80 = 4 of the top-5 shortlisted candidates are relevant
  - Honest framing: random baseline is high (~0.88 NDCG@5) because slates
    are large and mostly relevant; the LIFT (+0.05-0.08 NDCG) is the signal
  - RandomForest wins on ordering despite losing on pointwise accuracy
    to LogReg — ordering and thresholding are different skills
- CV-group numbers: only 1-3 usable groups in the sample — reported as
  NOT statistically meaningful in the report; full-test-set extraction
  (1,759 rows) deferred until the feature pipeline is faster
- 19 new ranking-metric tests (hand-computed values), incl. the class-
  permutation robustness of fit_score_from_proba
- Findings written into ml/evaluation/ranking_eval.md (committable)

## Phase 13 (part 2) — Full-Set Classification Evaluation & Error Analysis
- Extracted the FULL 1,759-row test set (0 failures) — ranking eval re-run
  on the honest imbalanced distribution (857/444/458):
  - JD slates (69 groups): NDCG@5 lift over random +0.05-0.10 (GB best
    0.732 vs 0.637 random); P@5 ~0.507 vs ~0.45 random — models surface
    Good fits but CANNOT binary-shortlist
  - CV groups (177 groups, 1,224 rows): NO lift over random on any metric
    for any model — pointwise features cannot do listwise comparison;
    documented as the key open problem (needs per-pair interaction
    features or pairwise/listwise objectives)
- New ml/evaluation/run_classification_eval.py: per-class P/R/F1,
  confusion matrices, ROC-AUC (OvR macro + Good-vs-rest + adjacent-pair
  AUCs), calibration bins, ordinal FP/FN analysis (overrated/underrated,
  adjacent/distant), confusion-cell decomposition, feature-delta profiles,
  top-confident-errors, and a Saerens-style prior-corrected variant
- Honest headline: NO model beats the 0.487 majority baseline on accuracy
  (best prior-corrected 0.450); best Good-vs-rest AUC 0.592 (LogReg)
- Error analysis findings (all documented in classification_eval.md):
  1. Overconfidence on Good (predicted ~0.88 vs empirical ~0.36 in the top
     bin) — root cause: stratified training table taught a uniform prior
     vs the ~50/25/25 source distribution; prior correction recovers 3-5
     acc pts but does not change AUC
  2. Volume-proxy shortcut: overrated candidates list ~10.5 skills vs ~6.8
     for correct Potential rows — 'long CV' conflated with 'good fit'
  3. 38% of errors skip a class (ordinal structure underused)
  4. Potential Fit is the weakest class in classification AND ranking
- 9 new helper tests (prior correction, calibration bins, error
  direction/distance, feature-delta profiles); suite at 400 passing
- docs/model-card.md CREATED (Phase 23 deliverable pulled forward): full
  v0.3.0 documentation with honest metrics, failure modes, limitations

## Test Summary
```
Total: 400 tests passing
- 9 classification-eval helper tests (NEW)
- 19 ranking metric tests
- 17 feature extraction tests (+6 category-coverage tests)
- 11 model scorer / hybrid-ML integration tests
- 11 model / 39 parser / 27 extraction / 27 taxonomy / 5 seed skills
- 38 job parser tests (+4 infinite-loop regressions)
- 36 embedding / 35 skill matcher / 26 experience matcher / 22 semantic
- 11 weights / 22 education+cert / 23 matching model / 12 API (SQLite)
- 10 experience-extractor fallback tests
- 3 misc
```

## Next Up
- Phase 13 COMPLETE — next: Phase 14 (explainability: translate ML
  features + error-analysis findings into human-readable factors;
  feature-delta deltas are the template)
- Feature generation v0.4 candidates (in priority order, from the error
  analysis): CV-length normalization to kill the volume-proxy shortcut,
  per-pair interaction features for CV-group ranking, ordinal-aware
  training objective, prior-aware training (or calibrate at serving)
- Phase 15 — candidate ranking endpoint (use the ranking eval findings:
  optimize for Good-Fit surfacing, do not promise binary shortlisting)
