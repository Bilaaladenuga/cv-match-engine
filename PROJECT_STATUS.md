# PROJECT STATUS

## Current Phase
**Phase 11 — Matching Model** ✅ COMPLETE

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

## Test Summary
```
Total: 330 tests passing (9 API tests NEW)
- 11 model tests
- 39 parser tests
- 27 extraction tests
- 27 taxonomy tests
- 5 seed skills tests
- 34 job parser tests
- 36 embedding tests
- 35 skill matcher tests
- 26 experience matcher tests
- 22 semantic matcher tests
- 11 weights tests (NEW)
- 22 education/cert matcher tests (NEW)
- 23 matching model tests (NEW)
- 3 misc
```

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

## Next Up
**Phase 12 — Train a Real ML Model** (dataset acquisition, feature engineering,
baseline models: Logistic Regression / Random Forest / Gradient Boosting)
