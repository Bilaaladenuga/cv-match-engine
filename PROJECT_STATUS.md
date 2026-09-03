# PROJECT STATUS

## Completed Phases

- [x] Phase 1 — Project Initialization
- [x] Phase 2 — Database Architecture (models + Alembic + PostgreSQL running)
- [x] Phase 3 — CV Document Parser
- [x] Phase 4 — CV Information Extraction
- [x] Phase 5 — Skill Taxonomy
- [x] Phase 6 — Job Description Parser
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

**Phase 7 — Embedding Engine**

## Phase 6 — Completed Tasks

1. ✅ `app/nlp/job_parser.py` — parses a job description into a structured `JobProfile` with `to_dict()`
2. ✅ Job title extraction — labeled lines ("Job Title:"), standalone Title Case first lines, and "looking for / hiring / seeking" phrases; titles truncated at the role word ("Principal Engineer at Rocket" → "Principal Engineer")
3. ✅ Company / location extraction from labeled lines; remote/hybrid detection
4. ✅ Required vs. preferred skills — heading-aware blocks (Required Qualifications / Preferred Skills / Responsibilities / Education) plus sentence-level signals ("preferred", "nice to have", "bonus", ...); skills default to required like real job posts; explicit required wins over preferred mentions; a "is a plus" line inside a Required block demotes that line's skills
5. ✅ Skill classification reuses the Phase 5 taxonomy (canonical names + categories, whole-line + list scanning so hyphenated aliases like scikit-learn survive)
6. ✅ Minimum experience — "3+ years", "at least 5 years", "minimum of 2", "5–7 years" (floor), preferred-context numbers ignored, seniority level (junior/mid/senior/lead/...) detected
7. ✅ Education requirement — degree level (Bachelor's/Master's/Doctorate...), field, and required vs. preferred
8. ✅ Certifications — cert-shaped entries only (issuer + cert markers), "Certifications:" labels stripped
9. ✅ Responsibilities captured as cleaned bullet lists (and excluded from skill extraction)
10. ✅ `data/sample/sample_job.txt` — realistic JD covering every feature; end-to-end profile verified
11. ✅ 34 parser tests (146 total passing), Ruff clean
12. ✅ Housekeeping: dev deps (pytest, ruff) added to `requirements-core.txt`; stray `asyncio_mode` pytest option removed
13. ✅ Cross-cutting fix: alias word boundaries now treat a preceding `.` as inside-a-word, so `js` no longer matches inside `next.js` / `node.js` (fixed in the shared skill extractor)

## Test Summary

- 11 model tests — passing
- 39 parser tests — passing
- 27 extraction tests — passing
- 27 taxonomy tests — passing
- 34 job parser tests — passing
- 5 seed-script planner tests — passing
- **Total: 146 tests passing**

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
- **Skill taxonomy as data, not code**: JSON vocabulary validated at insert time; single source of truth for extractors, matching, and the DB seed
- **Required vs. preferred**: Heading-aware blocks + per-sentence signals; default is required; explicit required wins over preferred mentions
- **Sentence boundaries include newlines**: JD bullets are per-line units, so one bullet can't poison the classification of others
- **Extraction layers**: Exact/canonical match first (confidence 0.95), then whole-text alias scan (0.9); taxonomy and matching share one source of truth

## Interim Tasks (between phases)

- ✅ Seed script `backend/scripts/seed_skills.py` — idempotent upsert of the taxonomy JSON into the `skills` table (INSERT new, UPDATE stale categories, orphans reported not deleted; `--dry-run`)
- ✅ Taxonomy docs `docs/skill-taxonomy.md` — file format, category ids, safe add/remove/rename checklist, verification steps, runtime/custom-file extension
- ✅ Safety hardening: unknown category ids in the taxonomy JSON fail fast at load; runtime `register_skill` may still create categories
- ✅ Worked example: 11 niche skills added (Celery, RabbitMQ, REST API, WebSocket, pytest, Jest, Playwright, Cypress, AWS certs ×2, Google Cloud Certified) → taxonomy at 129 skills, seeded to PostgreSQL
- ✅ CI alias-collision gate test — raw JSON scan reporting every ambiguous alias in one message

## Infrastructure

- PostgreSQL running via Docker (career-match-db container)
- Alembic migrations applied (9 tables created)
- `skills` table seeded: 129 rows from the taxonomy
- Connection: `postgresql://career_match:career_match@localhost:5432/career_match`
- No Docker in project itself — Docker is only used locally to host the PostgreSQL instance
