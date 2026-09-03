# PROJECT STATUS

## Completed Phases

- [x] Phase 1 — Project Initialization
- [x] Phase 2 — Database Architecture (models + Alembic + PostgreSQL running)
- [x] Phase 3 — CV Document Parser
- [x] Phase 4 — CV Information Extraction
- [ ] Phase 5 — Skill Taxonomy
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

**Phase 5 — Skill Taxonomy**

## Phase 4 — Completed Tasks

1. ✅ Skill extraction — dictionary-driven alias matching with word-boundary safety (Python, React, PostgreSQL, Docker, AWS + aliases like `js` → JavaScript)
2. ✅ Work experience extraction — company, role, start/end dates, duration in months from patterns like `Role | Company | Jan 2020 - Dec 2022`
3. ✅ Date handling — month-precision end dates are inclusive (Jan 2020 – Dec 2022 = 36 mo), year-only ranges exclusive (2019–2021 = 24 mo); `Present` supported
4. ✅ Education extraction — degree type, field of study, institution, year
5. ✅ Certification extraction — certification name, issuer, year (AWS, PMI/PMP, Kubernetes, etc.)
6. ✅ Job title extraction — title + seniority from experience text
7. ✅ Total years of experience calculation from parsed entries
8. ✅ `CandidateProfileBuilder` — assembles name/email/phone/summary/skills/experience/education/certs/titles into one structured profile with `to_dict()`
9. ✅ End-to-end pipeline verified on sample CV: name, email, 8.7 yrs, 19 skills, 3 job entries, education
10. ✅ 27 extraction tests (77 total passing), Ruff clean

## Test Summary

- 11 model tests — passing
- 39 parser tests — passing
- 27 extraction tests — passing
- **Total: 77 tests passing**

## Known Issues

- Full ML dependencies (torch, sentence-transformers) not yet installed — will be added in Phase 7
- PyMuPDF `fitz` deprecation warning — cosmetic, API still works

## Architectural Decisions

- **Parser pattern**: Abstract base class with per-format implementations
- **Section detection**: Regex + heuristic approach (deterministic, no ML needed)
- **Encoding handling**: Try multiple encodings for TXT files
- **SQLite vs PostgreSQL**: Using `sqlalchemy.types.JSON` for cross-dialect compatibility
- **Extraction approach**: Deterministic NLP (regex + curated dictionaries) first; ML/NLP models to be layered in where rules prove unreliable (per Phase 4 spec)
- **Date-range convention**: Month-precision end dates treated as inclusive (person worked through that month); year-only ranges exclusive

## Infrastructure

- PostgreSQL running via Docker (career-match-db container)
- Alembic migrations applied (9 tables created)
- Connection: `postgresql://career_match:career_match@localhost:5432/career_match`
- No Docker in project itself — Docker is only used locally to host the PostgreSQL instance
