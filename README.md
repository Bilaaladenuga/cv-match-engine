# Career Match — CV–Job Matching Engine

An explainable machine learning system for CV–Job compatibility analysis.

## Overview

Career Match analyzes a candidate's CV against a job description and produces a detailed compatibility report including:

- **Overall match score** with weighted components
- **Skill matching** (exact, alias, semantic, partial)
- **Experience comparison** (required vs. candidate)
- **Education analysis**
- **Semantic similarity** using sentence embeddings
- **Explainability** — understand why a score was given
- **Recommendations** — actionable steps to improve compatibility
- **Candidate ranking** — compare multiple candidates for one job

## Architecture

```
Next.js Frontend → FastAPI Backend → ML/NLP Pipeline → PostgreSQL
```

### Separation of Concerns

- **Deterministic processing**: PDF/DOCX parsing, section detection, skill extraction
- **ML/NLP processing**: embeddings, semantic similarity, matching model
- **Application layer**: auth, uploads, dashboards, API

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js, TypeScript, Tailwind CSS, Recharts |
| Backend | Python, FastAPI, Pydantic |
| ML/NLP | ONNX Runtime (fastembed), scikit-learn, pandas, NumPy |
| Database | PostgreSQL, SQLAlchemy |
| Document Processing | PyMuPDF, python-docx |

## Project Structure

```
cv-match-engine/
├── frontend/          # Next.js application
├── backend/           # FastAPI application
├── ml/                # ML training, evaluation, datasets
├── data/              # Raw, processed, and sample data
├── docs/              # Architecture, API, and ML documentation
└── scripts/           # Utility scripts
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your database credentials
uvicorn app.main:app --reload
```

The API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

#### Seeding the skill taxonomy

The canonical skill vocabulary lives in `backend/app/nlp/skill_taxonomy.json`.
Load it into the `skills` table (idempotent — safe to rerun):

```bash
cd backend
python -m scripts.seed_skills          # apply
python -m scripts.seed_skills --dry-run  # preview without writing
```

To add or change skills safely, read [`docs/skill-taxonomy.md`](docs/skill-taxonomy.md) first.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The app runs at [http://localhost:3000](http://localhost:3000).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/resumes/upload` | Upload a CV document |
| POST | `/api/resumes/{id}/parse` | Parse an uploaded CV |
| POST | `/api/jobs` | Create a job description |
| POST | `/api/jobs/{id}/parse` | Parse a job description |
| POST | `/api/matches` | Run CV–Job matching |
| GET | `/api/matches/{id}` | Get match results |
| POST | `/api/jobs/{id}/rank-candidates` | Rank candidates for a job |
| GET | `/api/candidates/{id}` | Get candidate profile |
| GET | `/api/history` | Get match history |

## Development

This project is built in phases. See `PROJECT_STATUS.md` for current progress.

### Testing

```bash
# Backend tests
cd backend && pytest

# Frontend type checking
cd frontend && npm run type-check
```

## Ethical Considerations

This system is a **decision-support tool** and should not be used as the sole basis for hiring decisions. Scores represent model-estimated compatibility, not a guarantee of hiring outcomes.

The system does not use protected characteristics (gender, age, ethnicity, disability) as scoring features.

## License

MIT
