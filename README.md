# ResearchPath

ResearchPath is a local-first research paper recommender that turns a topic or research goal into a structured reading path.

Instead of returning a flat list of papers, it organizes recommendations into:

1. Background
2. Foundational
3. Core methods
4. Recent frontier

The project is built as a full-stack ML/search system: data ingestion, retrieval baselines, learned ranking experiments, evaluation, backend APIs, and a React product UI.

## Why It Exists

Beginner researchers often know what they want to learn, but not which papers to read first. Search engines can surface relevant papers, but they rarely explain whether a paper is introductory, foundational, too advanced, or a good next step.

ResearchPath ranks papers using relevance, difficulty, citations, recency, semantic similarity, and user feedback so the output is closer to a guided research path than a search-results page.

## Features

- Search papers from a local research corpus.
- Build beginner-aware reading paths by section.
- Compare multiple ranking strategies.
- Edit a reader profile with background, goal, preferred topics, avoided topics, and paper taste.
- Save papers to a local library with tags.
- Give feedback such as Save, More like this, Not useful, Too easy, and Too hard.
- Open recommended papers through paper links, PDF links, and DOI links when available.
- View explanations for why papers were recommended.

## Product UI

The frontend is a Vite + React app with:

- branded `ResearchPath` landing/workspace layout,
- editable reader profile,
- saved-paper library,
- search and reading-path modes,
- readable method labels instead of internal model version IDs,
- paper action buttons for source/PDF/DOI access.

## Retrieval and Ranking

The backend supports multiple retrieval and ranking methods:

- keyword search,
- TF-IDF,
- citation and recency scoring,
- frozen transformer embeddings,
- FAISS vector search,
- hybrid ranking,
- learned ranking and guarded blend experiments,
- opt-in safe-fusion ranking for offline/runtime parity demos.

The default backend method remains `bm25` for reliability. More advanced learned methods are exposed as explicit opt-in options.

## Tech Stack

- **Frontend:** React, TypeScript, Vite, CSS
- **Backend:** FastAPI, SQLAlchemy, Pydantic
- **Database:** PostgreSQL
- **Retrieval:** BM25, TF-IDF, sentence-transformer embeddings, FAISS
- **ML/Reranking:** scikit-learn rankers, calibrated blends, neural-reranker diagnostics
- **Data sources:** OpenAlex and arXiv pipelines
- **Testing:** pytest, frontend production build checks

## Current Dataset State

The local development corpus contains about **50k ML/AI papers** with metadata, citations, and provenance from arXiv/OpenAlex pipelines.

Large raw datasets, processed indexes, model artifacts, and full-text PDFs are kept out of git.

## Run Locally

Start PostgreSQL:

```powershell
docker compose up -d db
```

Start the backend:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
cd frontend
npm install
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

The frontend expects the backend at:

```text
http://localhost:8000
```

## Useful API Examples

Health check:

```powershell
curl.exe http://localhost:8000/health
```

Search recommendations:

```powershell
curl.exe "http://localhost:8000/recommend/query?query=transformer&k=5&method=bm25"
```

Reading path:

```powershell
curl.exe "http://localhost:8000/path/query?query=AI%20agents%20for%20scientific%20discovery&method=bm25&background_level=basic_ml&k=4"
```

## Validation

Focused checks used during the latest product pass:

```powershell
cd frontend
npm.cmd run build
```

```powershell
cd ..
.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_feedback_profile_api.py backend\tests\test_product_recommendations.py backend\tests\test_formatting.py backend\tests\test_reading_path_api.py
```

## What This Demonstrates

- End-to-end ML product engineering, not just model training.
- Search and recommender-system fundamentals.
- Offline evaluation before model promotion.
- Conservative deployment of learned rankers behind explicit opt-in methods.
- Full-stack product work with a real frontend, backend, database, and feedback loop.
- Practical handling of data provenance, deduplication, ranking diagnostics, and runtime constraints.

## Documentation

- `PROJECT_CONTEXT.md` contains the detailed progress log and technical history.
- `docs/v6_7_portfolio_demo_packaging.md` summarizes the accepted ML/ranking pipeline.
- `docs/product_frontend_profile_pass_2026_07_03.md` documents the latest frontend/profile/product pass.

## Current Limitations

- Multi-user support is not implemented yet; the app uses one local default profile.
- Advanced learned/fusion rankers are opt-in and can be slower than keyword or hybrid retrieval.
- Large datasets and model artifacts are local-only and not committed to the repository.
