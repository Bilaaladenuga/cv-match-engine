# PROJECT STATUS

## Current Phase
**Phase 7 — Embedding Engine** ✅ COMPLETE

## Completed Phases

### Phase 1 — Project Initialization ✅
- FastAPI backend with health endpoint
- Next.js 14 frontend with landing page
- Alembic migrations configured
- PostgreSQL setup
- Basic testing infrastructure
- Ruff linting

### Phase 2 — Database Architecture ✅
- 9 SQLAlchemy ORM models (User, Resume, CandidateProfile, Skill, CandidateSkill, Job, JobRequirement, Match, MatchExplanation)
- Alembic autogenerate configured
- JSON/JSONB support for SQLite + PostgreSQL
- 11 model tests passing

### Phase 3 — CV Document Parser ✅
- PDF parser (PyMuPDF)
- DOCX parser (python-docx)
- TXT parser (multi-encoding fallback)
- Section detector (summary, skills, experience, education, certs, projects, languages)
- Contact extractor (email, phone, LinkedIn, GitHub)
- Name extractor (heuristic from CV header)
- Text normalization (Unicode NFKD, tab→space, collapse whitespace)
- 39 parser tests passing

### Phase 4 — CV Information Extraction ✅
- Skill extraction via taxonomy alias dictionary
- Work experience extraction (company, role, dates, duration)
- Education extraction (degree, field, institution)
- Certification extraction (name, issuer, year)
- Job title extraction with seniority detection
- CandidateProfileBuilder assembles all components
- 27 extraction tests passing

### Phase 5 — Skill Taxonomy ✅
- 129 skills across 12 categories in JSON data file
- SkillTaxonomy loader with validation
- Runtime extension API (register_skill, load_taxonomy)
- CI alias-collision gate test
- 27 taxonomy tests passing

### Phase 6 — Job Description Parser ✅
- JobProfile extraction (title, company, location, remote status)
- Required vs preferred skills (heading-aware blocks)
- Minimum experience extraction (seniority level)
- Education requirements
- Certification requirements
- Responsibilities extraction
- 34 job parser tests passing

### Phase 7 — Embedding Engine ✅
- MiniLM model (384-dim) via sentence-transformers
- Embedding generation (single + batch)
- Cosine similarity computation
- Similarity matrix (pairwise)
- Strategy A: Full document embedding
- Strategy B: Section-level embedding
- Strategy C: Skill-level embedding
- Strategy D: Weighted hybrid
- Global model singleton (lazy-loaded)
- 36 embedding tests passing

## Test Summary
```
Total: 182 tests passing
- 11 model tests
- 39 parser tests
- 27 extraction tests
- 27 taxonomy tests
- 5 seed skills tests
- 34 job parser tests
- 36 embedding tests
```

## Key Dependencies
```
torch==2.4.1+cpu
sentence-transformers==3.1.1
transformers==4.44.2
scikit-learn==1.9.0
numpy==1.26.4
```

## Next Up
**Phase 8 — Skill Matching Engine** (exact, alias, semantic, related matching)

---

## Recent Activity

### 2026-09-04: Phase 7 Complete
- Installed torch 2.4.1+cpu, sentence-transformers 3.1.1, transformers 4.44.2
- Fixed venv activation scripts (old path → new path)
- Copied working venv from career-match/ to career-match-engine/
- Created `backend/app/ml/embeddings.py` with 4 embedding strategies
- Created `backend/tests/test_embeddings.py` with 36 comprehensive tests
- Verified: all 182 tests passing, Ruff clean
