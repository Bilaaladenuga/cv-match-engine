# PROJECT STATUS

## Completed Phases

- [x] Phase 1 — Project Initialization
- [x] Phase 2 — Database Architecture (models + Alembic setup)
- [ ] Phase 3 — CV Document Parser
- [ ] Phase 4 — CV Information Extraction
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

**Phase 3 — CV Document Parser**

## Phase 1 — Completed Tasks

1. ✅ Created monorepo directory structure
2. ✅ Set up FastAPI backend skeleton (main.py, config, database)
3. ✅ Set up Next.js frontend skeleton (package.json, tsconfig, app layout)
4. ✅ Created ML directory structure
5. ✅ Created environment variable templates (.env.example)
6. ✅ Created README.md
7. ✅ Created .gitignore files
8. ✅ Installed Python dependencies and verified backend starts
9. ✅ Installed Node dependencies and verified frontend starts (HTTP 200)
10. ✅ Added linting/formatting config (ESLint, Ruff, pytest)
11. ✅ Updated PROJECT_STATUS.md

## Phase 2 — Completed Tasks

1. ✅ Created SQLAlchemy ORM models (9 tables):
   - `users` — authentication and ownership
   - `resumes` — uploaded CV documents
   - `candidate_profiles` — structured data from CVs
   - `skills` — normalized skill taxonomy
   - `candidate_skills` — candidate-skill associations
   - `jobs` — job descriptions
   - `job_requirements` — job-skill requirements
   - `matches` — compatibility scores
   - `match_explanations` — human-readable explanations
2. ✅ Set up Alembic for database migrations
3. ✅ Configured env.py to import all models for autogenerate
4. ✅ Used `JSON` type (not `JSONB`) for SQLite-compatible testing
5. ✅ All 11 model tests pass
6. ✅ No Docker dependency

## Known Issues

- Full ML dependencies (torch, sentence-transformers) not yet installed — will be added in Phase 7
- Port 3000 occupied on this machine — use port 3001 for frontend dev
- PostgreSQL not installed locally — Alembic migrations will work once PG is available

## Architectural Decisions

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (67M params, fast inference, good quality)
- **Backend framework**: FastAPI (auto-generated OpenAPI docs, async support)
- **Frontend framework**: Next.js 14 with App Router
- **Database ORM**: SQLAlchemy 2.0 with DeclarativeBase
- **Python linting**: Ruff
- **Frontend linting**: ESLint (next/core-web-vitals)
- **No Docker**: All services run locally or via managed services
- **JSON vs JSONB**: Using `sqlalchemy.types.JSON` for cross-dialect compatibility (SQLite for tests, PostgreSQL for production)

## Next Steps

1. Begin Phase 3 (CV document parser — PDF, DOCX, TXT extraction)
2. Build section detection for name, email, phone, skills, experience, education
3. Create tests for multiple CV layouts
