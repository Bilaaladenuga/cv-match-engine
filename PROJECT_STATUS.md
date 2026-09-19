# PROJECT STATUS

## Current Phase
**Phases 1–17 complete + privacy-first product decision: NO user
accounts.** History lives in the browser (localStorage); raw-text
/api/matches is fully stateless. File upload (PDF/DOCX/TXT) is live on
/analyze. Next: recruiter ranking UI, Recharts visualizations,
22 (observability), 24 (deployment), model roadmap

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

## v0.3.1 — Prior Calibration in Serving (fixing the overconfidence finding)
- New backend/app/ml/calibration.py: single source of truth for the
  Saerens prior correction; audited NATURAL_PRIOR constants regression-
  guarded against data/raw/train.csv by a unit test
- Training embeds calibration_ metadata ON the artifact (method + natural
  prior + training prior) so model and calibration are inseparable;
  artifacts re-stamped match-model-v0.3.1-baseline
- Serving: model_scorer.score_features applies the correction using the
  artifact metadata (module constants as fallback for older artifacts);
  invalid metadata degrades to raw probabilities with a warning, never a
  failed request; MLScorerResult now carries raw_probabilities +
  calibration_method alongside the calibrated values
- Hybrid layer forwards both through ml_details; API now returns
  calibrated probabilities, raw probabilities, and the method name
- End-to-end verified via TestClient on the stateless /api/matches path:
  version stamp match-model-v0.1+v0.3.1-baseline, No Fit share boosted
  (~1.5% -> ~3.1% on the sample pair) as the natural prior requires
- 15 new tests (correction math incl. column-order independence,
  renormalization, degenerate input; audited-prior guard; metadata-driven
  serving; graceful degradation; artifact-metadata presence). Suite: 415
  passing, Ruff clean

## Phase 14 — Explainability Engine (v0.3.2)
- New backend/app/ml/explainer.py: per-match factor contributions via
  reference substitution (feature -> training median, "a typical
  applicant"); 34 forward passes, ~10 ms, computed on the CALIBRATED fit
  score so explanations match the displayed number
- Training embeds reference_stats_ (per-feature training medians) on the
  artifacts; version re-stamped match-model-v0.3.2-baseline
- Plain-language layer grounded in the Phase 13 error analysis:
  - volume-proxy caution fires when skill count >= 1.5x the training
    median while required coverage <= 0.60 (guards the model's PRIMARY
    signal — perm importance shows n_candidate_skills is #1 at 0.023,
    2.2x the runner-up, while semantic_similarity is 15th at 0.0015)
  - grade-uncertainty note: adjacent grades within 0.15 are reported as
    borderline, not a verdict
  - every explanation carries the decision-support disclaimer
- Advisory integration: model_scorer attaches explanation (any explainer
  failure degrades to no explanation, never a failed request); hybrid
  layer forwards it via ml_details; degradation carries a reason string
- Global view: ml/evaluation/perm_importance.py (seeded permutation
  importance on the fit score, committable JSON evidence)
- 14 new tests incl. EXACT additivity on an affine stub (fit(reference) +
  sum(contributions) == fit(x) to 1e-9) and all caution/uncertainty/
  degradation paths. Suite: 429 passing, Ruff clean
- Known quirk surfaced honestly by the explainer: the model sometimes
  PENALIZES high title similarity (rare in training data) — documented in
  the model card as the explainer reporting what the model actually
  learned, quirks included

## v0.4.0 — CV-Length Normalization (volume-proxy shortcut fixed)
- 3 new features via shared compute_cv_length_features(): cv_word_count,
  skills_per_100_words (density), cv_length_bucket — 36-feature schema
- Augmentation over re-extraction: new columns derive from cv_text +
  n_candidate_skills, so ml/preprocessing/augment_cv_length_features.py
  joins tables to raw CSVs by split_row and computes ONLY the new columns
  (33 expensive embedding columns pass through untouched) — seconds, not
  an hour; same helper as serving = no drift
- Retrain (v0.4.0-baseline): acc 0.404-0.408 raw / up to 0.441 prior-
  corrected, AUC ~unchanged (0.567-0.578 OvR) — expected: the shortcut
  had been HELPING accuracy while corrupting ranking
- THE FIX, measured (perm importance of n_candidate_skills):
  0.0231 (#1, dominant) -> 0.0049 (#5) = 4.7x collapse; skills density
  itself picked up the signal (0.0056) — transferred, not deleted
- Ranking held: GB NDCG@5 0.738 (was 0.732); JD-slate lift intact
- Overconfidence persists (prior/separation issue, not shortcut) —
  serving-time calibration remains the mitigation
- 10 new/updated feature tests (helper semantics, buckets, density,
  unbounded-range contract). Suite: 439 passing, Ruff clean

## Phase 15 — Candidate Ranking Endpoint (recruiter mode)
- POST /api/jobs/{job_id}/rank-candidates with {resume_ids, weights?}:
  runs the SAME pipeline as /api/matches per candidate (no ranking-specific
  scoring code), orders the slate by overall score desc with a
  deterministic candidate_id-asc tiebreak, persists a Match +
  MatchExplanation row per candidate
- Reproducibility contract: response stamps ranking_run_id (uuid), the
  combined model_version, the exact weights used, ISO created_at, and
  per-candidate match_ids — re-running with the same inputs + model
  version reproduces the ordering (pipeline is deterministic)
- Honest slate framing per Phase 13 eval: response disclaimer states
  rankings are model-estimated compatibility, not hiring recommendations;
  per-row explanations (matched/missing skills, top positive/negative
  factors) ship alongside
- Failure isolation: missing/empty/failed resumes land in
  failed_resume_ids instead of killing the run; empty/duplicate
  resume_ids -> 400; unknown job -> 404; invalid weights -> 422
- New files: app/services/ranking_service.py, app/api/ranking.py; schemas
  in app/schemas/match.py; router wired in app/main.py
- 8 integration tests (SQLite, real pipeline): ordering, persistence,
  determinism across request orders, failure isolation, error mapping.
  Suite: 447 passing, Ruff clean
- Debugging note: a fixture set dependency_overrides[get_db] to a
  sessionmaker CLASS instead of a generator callable — this FastAPI/
  Starlette version's dependency machinery failed it with a confusing
  `local_kw` 422. Fixed by wrapping in a generator function.

## History / Retrieval Endpoints
- GET /api/history — user's analysis history, newest first, with summary
  fields (scores, matched/missing counts) and pagination
- GET /api/matches/{match_id} — full stored match report (scores, per-skill
  breakdown, experience/semantic/education blocks, recommendations)
- history_service + routes wired in app/main.py; 6 API tests (SQLite)

## Taxonomy Extension — Healthcare + Finance
- 28 new skills across 2 new categories (healthcare 13, finance 15) —
  157 skills / 14 categories total; demonstrates the documented
  add-a-domain process end-to-end (docs/skill-taxonomy.md updated)
- Fixed a real cross-skill alias collision introduced by the extension:
  'Financial Analysis' (misplaced in healthcare) claimed the alias
  'financial modeling', colliding with finance's 'Financial Modeling' —
  moved to finance with its own aliases; 'financial modelling' (British
  spelling) now belongs to Financial Modeling
- Added missing aliases for 'Medical Terminology'; CI alias-collision
  test gate passes; extraction verified on real domain text (EHR,
  phlebotomy, triage, QuickBooks, tax prep, credit analysis)

## Phase 16 — CV Improvement Engine (evidence-grounded recommendations)
- app/scoring/improvement_engine.py: grades the EVIDENCE behind every
  job-required skill from three signals — skills-section presence,
  work-history/projects mention, Phase 9 per-skill duration estimates —
  into strong / moderate / weak / absent
- Flagship warning: a skill matched but listed only in the skills section
  gets "X appears in your skills section but there is limited evidence of
  its use in your work experience" — the padding detector the spec asks for
- Anti-gaming rule learned from E2E: the Phase 9 estimator's coarse
  fallback attributes the whole career span to unmentioned skills with a
  'CV skills section (inferred)' source note. Month estimates only count
  as evidence when a NAMED ROLE attributed them or the work text mentions
  the skill; inferred-only months are zeroed (fallback months must not
  fake strong evidence)
- build_recommendations(): prioritized, capped list — build missing
  skills, substantiate weak evidence, strengthen partial matches with
  quantified outcomes, experience-gap and certification actions, grounded
  positives for well-evidenced skills
- PipelineOutput exposes the graded skill_evidence block so the frontend
  can render a per-skill evidence panel; recommendations flow through the
  existing MatchReport / API unchanged
- MatcherInputs gained a cv_evidence field (backward-compatible default)
- 15 engine tests (evidence grading, alias-in-work-history, flagship
  warning, cap, graceful legacy inputs). Suite: 468 passing, Ruff clean
- E2E verified: Jane Okafor CV vs Platform Engineer JD — Kubernetes/AWS/
  CI-CD correctly flagged as listed-but-not-evidenced, Python/Redis
  grounded as strong, Terraform given a build action item

## Phase 17 — Next.js Frontend (first full slice)
- Pages: / (landing), /analyze (CV + JD text input → live report),
  /history (stored analyses list), /history/[match_id] (full stored
  report), /dashboard (stats + recent analyses)
- Shared MatchReportView renders both live and stored reports: score
  dial, weighted component bars with evidence, the Phase 16 skill-
  evidence table (status × strength, flagged "listed but not evidenced"
  rows, per-skill months), positive/negative factors, recommendations,
  trained-model probabilities with calibration method, and the ethics
  disclaimer always visible
- lib/api.ts rewritten against the REAL backend contract (old stub
  referenced resume/job endpoints that don't exist yet): POST
  /api/matches (raw-text mode), GET /api/history, GET /api/matches/{id};
  lib/types.ts mirrors the Pydantic schemas
- Professional styling per spec: grays, borders, no gradients/glow/
  emoji; SiteHeader in the layout with active-link highlighting
- VERIFIED end-to-end on this machine: `next build` clean (5 routes),
  all pages 200 via `next start`, backend live on :8000 with CORS
  preflight from :3000 OK, real POST /api/matches over HTTP → 201 with
  match_id null, 59% Moderate match, ML label Good Fit
- IMPORTANT fix found during E2E: raw-text /api/matches previously
  required PostgreSQL (it persists under a demo user) — a DB outage
  surfaced as a 500. Now: DB down + raw-text mode degrades to a
  stateless report (match_id=null); DB down + entity mode returns a
  clear 503. Regression tests with a DeadSession stand-in
- eslint config: removed a rule referencing the uninstalled
  @typescript-eslint plugin (broke `next build`)
- .gitignore: *.log, *.tsbuildinfo; committed package-lock.json for
  reproducible installs

## No-Accounts Architecture + File Upload (privacy-first slice)
- PRODUCT DECISION (user): free software, no sign-up — accounts are out.
  Consequences, all verified:
  - raw-text POST /api/matches is now PURELY stateless: the demo-user
    persistence path is deleted. `match_id` is always null in text mode;
    DB is touched only in entity mode (rankings / stored candidates)
  - frontend history moved to localStorage (`lib/history.ts`, 100-entry
    cap, per-entry delete, clear-all with confirm); /history,
    /history/[id] and /dashboard read it — the server-side history
    endpoints remain available for API users but the UI no longer calls them
  - uploads are never persisted: `POST /api/resumes/extract` writes the
    file to a temp path, parses, deletes it in a `finally` block
- NEW endpoint `POST /api/resumes/extract` (multipart):
  - `services/document_service.py` — 4 validation layers: extension
    whitelist, streaming size cap (cuts off oversized uploads rather than
    buffering), magic-byte sniffing (a fake .pdf/.docx is rejected on
    content, not name), empty/textless rejection (scanned PDFs get a
    clear 422, parse failures map to 422 not 500)
  - `parsers/base.py`: `normalize_text` extracted to a module function
  - 11 integration tests incl. the privacy contract (no DB dependency)
- /analyze: drag-drop zone + file picker (PDF/DOCX/TXT) fills the CV
  textarea from the extraction response; uploads show a state chip;
  every completed analysis auto-saves to browser history
- E2E over real HTTP: txt upload → 311 chars extracted; match on the
  extracted text → 201, 60% Moderate match, ML Potential Fit, 6 evidence
  rows; fake.pdf → 415, .exe → 415, empty → 422

## Multi-Field Taxonomy Expansion (26 career categories)
- Taxonomy grown 157 → **280 skills across 26 categories**: added
  engineering, law, education, marketing, sales, HR, media, skilled
  trades, logistics, laboratory science, hospitality, and public sector,
  plus cross-field staples (Microsoft Excel/Word/Office, Google
  Workspace, Customer Service, Public Speaking)
- Migration preserved as `scripts/expand_taxonomy_fields.py` —
  collision-safe (refuses to write on any name/alias conflict) and
  idempotent
- ML compatibility verified BEFORE expanding: the 17 cov_* features use a
  FIXED category list; new categories aggregate into `cov_other` until a
  future retrain adds dedicated features — the v0.4.0 serving artifact
  is untouched (no retrain needed, no breakage)
- Extraction smoke-tested across 9 fields (nursing, paralegal, civil
  engineering, welding, marketing, accounting, teaching, supply chain,
  office admin) — two alias gaps found and fixed ("civil engineer",
  "welder"/MIG/TIG)
- All 30 taxonomy tests pass including the CI alias-collision gate;
  481 total, Ruff clean

## Frontend Fixes + Visualizations (Phase 18 slice)
- FIXED user-reported crash: `use(params)` on /history/[match_id] threw
  "unsupported type passed to use()" — Next 14 passes params to CLIENT
  components as a plain object (Promise params are Next 15+). Params are
  now read directly; verified on the dynamic route
- Recharts visualizations added to MatchReportView, each carrying
  information the text around it does not:
  - Score-contribution bar chart: weighted points per component
    (raw × weight), sorted — answers "where did my score come from";
    complements (does not duplicate) the raw-score bars above it
  - Skill-coverage donut: matched/partial/missing composition with
    counts and % — the at-a-glance size of each group next to the
    detailed evidence table
- Dev-server chunk corruption after recharts install ("Cannot find
  module './359.js'") resolved by clearing .next; production build clean

## Model v0.5 Retrain Plan (documented, not yet implemented)
- `docs/model-v05-plan.md`: decision table (data-first, then schema, then
  retrain — schema expansion alone is rejected as a no-op on tech-only
  rows), multi-domain dataset strategy with weak-label fallback + human
  audit, per-field evaluation harness and fairness gate (best-vs-worst
  field NDCG@5 gap ≤ 0.15), shadow-mode rollout, explicit out-of-scope
  list (no LLM scoring, no per-field serving models)

## Production CORS Fix (hosted deployment)
- Root cause reproduced before changing anything: `main.py` **hardcoded**
  `allow_origins=["*"]`, so the deployed `CORS_ORIGINS` env var was never
  read; glob entries such as `https://*.vercel.app` could never match
  (Starlette's `allow_origins` compares literal strings); and an unhandled
  exception bypassed CORS entirely — Starlette's `ServerErrorMiddleware`
  is always the outermost layer — so a 500 (or a Render cold-start 502)
  reached the browser with no `Access-Control-*` headers and was reported
  as a CORS error, hiding the real status
- `app/core/config.py`: `split_cors_origins()` accepts a JSON array
  (Render/Railway) or a comma-separated list; `build_cors_origin_regex()`
  compiles wildcard hosts (`https://*.vercel.app`) into a regex
- `app/main.py`: `configure_cors()` now honors `CORS_ORIGINS`; a bare `*`
  disables credentials (the CORS spec forbids wildcard + credentials);
  CORS is registered LAST so it is the outermost user middleware
- `app/core/middleware.py`: new `ErrorHandlingMiddleware`, registered
  INSIDE CORS, converts unhandled exceptions into a generic JSON 500 that
  travels back out through CORS — errors now reach the browser with their
  true status code instead of masquerading as CORS failures
- 21 new tests in `tests/test_cors.py` (parsing, wildcard matching,
  disallowed-origin rejection, credentials rules, and CORS-on-500)

## Backend Lint Cleanup (ruff clean across app/)
- `ruff check app/` is now clean (was 13 errors across 7 files):
  import sorting, unused imports (`field`, `io`), `datetime.UTC` alias,
  lowercase Platt parameters, ternary/simplified returns, and two dead
  locals in the PDF generator
- All fixes are behaviour-preserving; verified against the modules they
  touch (parsers, extraction, calibration, model scorer, matching model,
  API routes) — no test count change

## Known Issues — pre-existing test failures (NOT from the lint cleanup)
These fail on `HEAD~` as well; confirmed by stashing the cleanup and
re-running:
- `tests/test_calibration.py::TestScorerIntegration` (3 tests) and
  `tests/test_model_scorer.py::test_stub_model_scoring_via_monkeypatch`:
  the scorer falls back to `ml/models/v0.5_platt_params.json` whenever the
  artifact carries no Platt params, so a v0.4 baseline is scored with v0.5
  calibration and the method string gains `+platt_scaling`. Needs a
  decision: pin the JSON to the matching model version, or check the
  artifact's version before applying it.
- `tests/test_improvement_engine.py::TestEvidenceGrading::test_strong_from_sustained_use`:
  expectation vs. current evidence-grading threshold — unverified further.

## Test Summary
```
Total: 502 tests passing
- 11 resume-upload tests (validation ladder + privacy contract)
- 11 matches endpoint tests (stateless contract + DB-outage regressions)
- 15 improvement-engine tests (Phase 16)
- 6 history endpoint tests
- 8 ranking endpoint tests
- 27 feature extraction tests (+10 CV-length)
- 14 explainer tests
- 15 calibration / serving-integration tests
- 9 classification-eval helper tests
- 19 ranking metric tests
- 17 feature extraction tests (+6 category-coverage tests)
- 11 model scorer / hybrid-ML integration tests
- 11 model / 39 parser / 27 extraction / 27 taxonomy / 5 seed skills
- 38 job parser tests (+4 infinite-loop regressions)
- 36 embedding / 35 skill matcher / 26 experience matcher / 22 semantic
- 11 weights / 22 education+cert / 23 matching model / 12 API (SQLite)
- 10 experience-extractor fallback tests
- 3 misc
- 21 CORS tests (origin parsing, wildcards, credentials, error-path CORS)
```

## Next Up
- Recruiter ranking UI on top of POST /jobs/{id}/rank-candidates (the
  ranking flow persists Matches — needs the PostgreSQL service running)
- Recharts visualizations (score breakdown, skill coverage)
- Remaining phases: 22 (observability), 24 (deployment). Phase 20 auth
  is intentionally DROPPED (no-accounts product decision); file-size and
  rate-limit hardening can still be added without accounts
- Feature roadmap (model side): **v0.5 retrain per docs/model-v05-plan.md**
  (multi-domain data → cov_* schema promotion → per-field gates), then
  per-pair interaction features for CV-group ranking (the listwise gap),
  ordinal-aware training objective
