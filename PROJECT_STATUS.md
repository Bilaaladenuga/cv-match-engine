# PROJECT STATUS

## Completed Phases

- [x] Phase 1 — Project Initialization
- [x] Phase 2 — Database Architecture (models + Alembic + PostgreSQL running)
- [x] Phase 3 — CV Document Parser
- [x] Phase 4 — CV Information Extraction
- [x] Phase 5 — Skill Taxonomy
- [ ] Phase 6 — Job Description Parser
- [ ] Phase 7 — Embedding Engine
- [ ] Phase 8 — Skill Matching Engine
- [ ] Phase 9 — Experience Matching
- [ ] Phase 10 — Semantic Matching
- [ ] Phase 11 — Matching Model
- [ ] Phase 12 — Train a Real ML Model
- [ ] Phase 13 — Model Evaluation
- [ ] Phase 14 — Explainable AI
- [ ] Phase 15 — Candidate Ranking
- [ ] Phase 16 — CV Improvement Engine
- [ ] Phase 17 — Frontend
- [ ] Phase 18 — Visualization
- [ ] Phase 19 — API Design
- [ ] Phase 20 — Security
- [ ] Phase 21 — Testing
- [ ] Phase 22 — Observability
- [ ] Phase 23 — Model Versioning
- [ ] Phase 24 — Deployment
- [ ] Phase 25 — Performance
- [ ] Phase 26 — Dataset & Research Documentation
- [ ] Phase 27 — Ethical Considerations
- [ ] Phase 28 — Final Product

## Current Phase

**Phase 6 — Job Description Parser**

## Phase 5 — Completed Tasks

1. ✅ Taxonomy extracted from code into a data file: `backend/app/nlp/skill_taxonomy.json` (118 skills, schema_version field)
2. ✅ 12 categories with labels/descriptions: programming, frontend, backend, database, cloud, devops, data_science, machine_learning, gis, design, tools, soft_skills
3. ✅ `SkillTaxonomy` loader (`app/nlp/taxonomy.py`) — canonical alias resolution, `canonicalize()`, category lookups, `lookup_items()`
4. ✅ Data integrity enforced at insert time: duplicate canonicals, unknown categories, and ambiguous aliases raise `SkillTaxonomyError`
5. ✅ Runtime extension API: `register_skill()` (merges into existing skills, auto-creates categories, rejects stolen aliases)
6. ✅ Extensible loading: `load_taxonomy(path)`, env override `SKILL_TAXONOMY_PATH`, cached default instance
7. ✅ Taxonomy cleanup — fixed conflicting mappings: SQL → database (MySQL stays distinct), "ruby on rails" → Ruby on Rails (not Ruby), removed ".net" from C# aliases (now its own skill), dropped opencv from Computer Vision
8. ✅ Spec examples verified: `JS`/`Javascript`/`Java Script` → JavaScript, `Postgres`/`PostgreSQL database` → PostgreSQL
9. ✅ New-category coverage: machine learning (TensorFlow, PyTorch, sklearn, Keras, XGBoost, LLM, NLP…), GIS (QGIS, ArcGIS, GeoPandas, PostGIS, Leaflet, Mapbox…), design (Figma, Photoshop, Canva, UI/UX), soft skills (Communication, Leadership, Agile…)
10. ✅ Word-boundary safety: single-letter alias "r" requires whitespace edges (R&D is not matched); short aliases ("ml", "go") never match inside larger words
11. ✅ Extractor rewritten on top of the taxonomy — public API unchanged, no test breakage
12. ✅ 27 taxonomy tests (104 total passing), Ruff clean

## Test Summary

- 11 model tests — passing
- 39 parser tests — passing
- 27 extraction tests — passing
- 27 taxonomy tests — passing
- 5 seed-script planner tests — passing
- **Total: 109 tests passing**

## Interim Tasks (between phases)

- ✅ Seed script `backend/scripts/seed_skills.py` — idempotent upsert of the taxonomy JSON into the `skills` table
  - INSERT new skills, UPDATE stale categories, leaves orphaned DB rows alone (may be referenced by FKs)
  - `--dry-run` preview mode; planner is a pure function with unit tests
  - Verified live against PostgreSQL: 118 skills seeded, re-run is a no-op, category drift corrected

## Known Issues

- Full ML dependencies (torch, sentence-transformers) not yet installed — will be added in Phase 7
- PyMuPDF `fitz` deprecation warning — cosmetic, API still works

## Architectural Decisions

- **Parser pattern**: Abstract base class with per-format implementations
- **Section detection**: Regex + heuristic approach (deterministic, no ML needed)
- **Encoding handling**: Try multiple encodings for TXT files
- **SQLite vs PostgreSQL**: Using `sqlalchemy.types.JSON` for cross-dialect compatibility
- **Extraction approach**: Deterministic NLP (regex + curated dictionaries) first; ML/NLP models layered in where rules prove unreliable
- **Date-range convention**: Month-precision end dates treated as inclusive (person worked through that month); year-only ranges exclusive
- **Skill taxonomy as data, not code**: Taxonomy lives in JSON so it can be extended/versioned without touching Python; validation at insert time keeps alias lookup unambiguous
- **Canonical names are resolvable aliases**: A canonical skill name always resolves to itself, so list items like "MySQL" match exactly
- **Extraction layers**: Exact/canonical match first (confidence 0.95), then whole-text alias scan (0.9); taxonomy and matching share one source of truth

## Infrastructure

- PostgreSQL running via Docker (career-match-db container)
- Alembic migrations applied (9 tables created)
- Connection: `postgresql://career_match:career_match@localhost:5432/career_match`
- No Docker in project itself — Docker is only used locally to host the PostgreSQL instance
