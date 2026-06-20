# ResearchPath Project Context and Progress Log

Last updated: 2026-06-20

## Project Identity
 
ResearchPath is a learning-to-rank research paper recommendation system for beginner researchers.

The project should not be framed as only "semantic search for papers." The stronger framing is:

> ResearchPath learns to rank research papers and build beginner-friendly reading paths from a user's research goal, background level, retrieval signals, citation structure, and model-based relevance scores.

Target user story:

> "I want to learn AI agents for scientific discovery, and I know basic ML. What should I read first, next, and later?"

Final product output:

1. Background papers
2. Foundational papers
3. Core method papers
4. Recent frontier papers

The product should help beginners avoid two common failure modes:

- Reading recent frontier papers before understanding prerequisites.
- Reading only similar papers instead of following a structured learning path.

## Resume-Level Goal

ResearchPath should become a serious MLE portfolio project showing:

- Full-stack ML product engineering
- Data ingestion and dataset construction
- Classical information retrieval baselines
- Offline evaluation with ranking metrics
- Transformer embedding retrieval
- Vector indexing with FAISS
- Contrastive learning for a fine-tuned bi-encoder
- Hard negative mining
- Cross-encoder reranking
- Difficulty prediction
- Citation graph features
- Reading path planning
- Clear experiment tracking and model comparison

The strongest resume framing should eventually be:

> Built a learning-to-rank research paper recommendation system that combines BM25, FAISS vector retrieval, contrastively fine-tuned transformer bi-encoders, cross-encoder reranking, weak-supervision evaluation, and beginner-aware reading path planning.

## Current State

ResearchPath now has the foundation, real OpenAlex data pipeline, weak-label evaluation, frozen transformer retrieval, FAISS indexing, and first bi-encoder fine-tuning runs implemented.

Implemented backend:

- FastAPI app
- SQLAlchemy database layer
- PostgreSQL-compatible `Paper` schema
- Environment-variable database configuration
- JSONL/CSV ingestion script
- BM25 baseline retriever over title plus abstract
- Query-to-paper recommendation endpoint
- Seed-paper recommendation endpoint
- Recommendation response format with score, method, snippet, authors, and explanation
- Evaluation metric utilities for Recall@K, Precision@K, NDCG@K, MRR, and latency timing
- Unit tests for parsing, BM25, metrics, API health, and response formatting

Implemented frontend:

- Vite React app
- Search input
- Result cards
- Method label
- Paper title, year, authors, abstract snippet, and explanation
- Simple maintainable layout

Implemented infrastructure:

- Docker Compose PostgreSQL service
- Backend Dockerfile
- Sample JSONL dataset
- README with setup and roadmap
- `.gitignore`

Operational notes:

- Docker Desktop has been installed and PostgreSQL has been started successfully.
- The Postgres container `researchpath-db` has been healthy on port `5432`.
- Sample data has been ingested into Postgres.
- Backend tests have passed with `46 passed`.
- Frontend build has passed.
- A blank frontend page bug was fixed by importing `React` in JSX-using components.

Known local environment quirks:

- PowerShell may block `.venv\Scripts\Activate.ps1`. Use direct venv Python commands instead.
- If `docker` is not recognized, restart PowerShell/VS Code or add this to PATH:

```text
C:\Program Files\Docker\Docker\resources\bin
```

- If `npm.ps1` is blocked, use `npm.cmd`.

## Current Run Commands

Start PostgreSQL:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
docker compose up -d db
```

Start backend without activating the virtual environment:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Start frontend:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\frontend
npm.cmd run dev
```

Run backend tests:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\backend
.\.venv\Scripts\python.exe -m pytest
```

Ingest sample data:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\sample_papers.jsonl
```

## Immediate Assessment

The project has crossed into real MLE portfolio territory: it now has a measurable retrieval benchmark, several baselines, a semantic retrieval baseline, FAISS, and a trained bi-encoder experiment. The current strongest result is MNRL bi-encoder fine-tuning, which slightly beats the frozen embedding baseline on the 1,000-query weak-label development slice.

The main weakness is still data and supervision quality. Current labels are citation-derived weak labels, so metrics are useful for iteration but not a final claim about human reading-path quality. The next high-value work is a reranking stage and better evaluation splits, followed by beginner-aware difficulty and path planning.

## Core Technical Principle

Evaluation must come before advanced deep learning.

Every future model should be compared against:

1. TF-IDF
2. BM25
3. Citation/recency heuristic baseline
4. Frozen pretrained embeddings
5. FAISS-backed frozen embeddings
6. Fine-tuned bi-encoder
7. Bi-encoder plus hard negatives
8. Cross-encoder reranker
9. Reading-path planner using relevance, difficulty, citation, and recency signals

The project should avoid saying "the deep learning model is better" unless there is a saved evaluation report proving it.

## Near-Term Todo List

### Data Layer

- Add a `citations` table or edge file format.
- Add external paper IDs such as arXiv ID, Semantic Scholar ID, OpenAlex ID, or DOI.
- Add ingestion support for a real dataset source.
- Add deduplication by source ID, DOI, or normalized title.
- Add indexes for title, year, categories, and source IDs.
- Add a dataset manifest file describing source, filters, counts, and date collected.
- Add scripts to create train/dev/test splits.

Suggested next schema additions:

- `external_id`
- `source`
- `doi`
- `url`
- `references_count`
- `influential_citation_count`
- `abstract_word_count`
- `updated_at`

### Evaluation Layer

- Create `backend/app/ml/evaluation_runner.py` or `scripts/evaluate_retrieval.py`.
- Define an evaluation query format:

```json
{
  "query_id": "agents_scientific_discovery_intro",
  "query": "AI agents for scientific discovery",
  "relevant_paper_ids": [1, 2, 3],
  "notes": "Weak labels from citations or curated seed list"
}
```

- Save evaluation reports to `data/processed/evaluations/`.
- Track Recall@5, Recall@10, NDCG@10, MRR, and latency.
- Add a baseline comparison table.
- Add tests for metric edge cases.

### Retrieval Baselines

- Add TF-IDF retriever using scikit-learn.
- Keep BM25 as the main sparse baseline.
- Add a simple citation/recency score:

```text
score = relevance_score + alpha * log1p(citation_count) + beta * recency_score
```

- Add a shared retriever interface so models can be swapped cleanly.

Suggested interface:

```python
class Retriever:
    method_name: str

    def fit(self, papers: list[Paper]) -> None:
        ...

    def search(self, query: str, k: int) -> list[ScoredDocument]:
        ...
```

## Deep Learning Roadmap

### Phase 1: Frozen Transformer Embeddings

Goal:

Build semantic retrieval without training yet.

Models to try:

- `sentence-transformers/all-MiniLM-L6-v2` as a fast baseline
- `allenai/specter2_base` or SPECTER-style scientific paper embeddings later
- SciBERT-based sentence embedding models if available

Implementation tasks:

- Add an embedding service under `backend/app/ml/embeddings/`.
- Encode `title + abstract`.
- Store embeddings in `data/processed/embeddings/`.
- Store metadata mapping:

```text
paper_id -> embedding row index
model_name -> embedding file
created_at -> timestamp
```

- Add cosine similarity search.
- Add endpoint option like `method=embedding`.
- Compare against BM25 using the evaluation runner.

Key design choice:

Do not put sentence-transformer logic directly in API routes. Keep it behind a retrieval interface so FAISS and fine-tuned models can replace the index later.

Expected resume value:

Shows semantic retrieval and embedding-based inference, but not yet training.

### Phase 2: FAISS Vector Search

Goal:

Make embedding retrieval scalable and production-like.

Implementation tasks:

- Add `faiss-cpu`.
- Build a FAISS index from paper embeddings.
- Store FAISS index files under `data/processed/faiss/`.
- Add index metadata:

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "paper_count": 10000,
  "index_type": "IndexFlatIP",
  "normalized": true
}
```

- Add script:

```text
scripts/build_faiss_index.py
```

- Compare latency against brute-force cosine similarity.

Expected resume value:

Strong MLE retrieval systems keyword: vector search, approximate nearest neighbors, indexing, latency measurement.

### Phase 3: Weak Supervision From Citations

Goal:

Create training and evaluation labels without hand-labeling thousands of examples.

Positive pairs:

- Paper cites another paper.
- Paper is cited by another paper.
- Papers are co-cited.
- Papers are bibliographically coupled.
- Papers share categories and citation neighborhoods.

Negative pairs:

- Random papers from unrelated categories.
- Same broad category but no citation relationship.
- Papers retrieved by BM25 but not citation-related.
- Papers retrieved by frozen embeddings but not citation-related.

Important:

Weak labels are noisy. The README and reports should be honest and call them weak supervision, not perfect relevance labels.

### Phase 4: Fine-Tuned Bi-Encoder

Goal:

Train a transformer so related papers and query-paper pairs are close in embedding space.

Architecture:

```text
paper/query text -> transformer encoder -> dense vector
```

Training objective:

- Contrastive loss
- Multiple negatives ranking loss
- InfoNCE-style loss

Training examples:

```text
anchor: paper title + abstract
positive: cited paper title + abstract
negative: BM25/FAISS hard negative paper
```

Implementation tasks:

- Add `backend/app/ml/training/`.
- Add dataset builder:

```text
scripts/build_contrastive_pairs.py
```

- Add hard negative mining:

```text
scripts/mine_hard_negatives.py
```

- Add training script:

```text
scripts/train_biencoder.py
```

- Save checkpoints under:

```text
data/processed/models/biencoder/
```

- Save training config and metrics:

```text
data/processed/runs/<run_id>/config.json
data/processed/runs/<run_id>/metrics.json
```

Metrics to compare:

- Recall@10
- Recall@50
- NDCG@10
- MRR
- embedding latency
- index build time

Expected resume value:

This is the first major deep learning part. It shows representation learning, contrastive training, hard negatives, and measurable improvement over baselines.

### Phase 5: Hard Negative Mining

Goal:

Make fine-tuning meaningfully harder and more realistic.

Hard negative sources:

- BM25 top results that are not cited or co-cited.
- FAISS nearest neighbors from frozen embeddings that are not positives.
- Same category papers from different citation communities.
- Papers with similar titles but different technical focus.

Why this matters:

Random negatives are often too easy. Hard negatives force the model to distinguish superficially similar papers, which is essential for ranking quality.

Expected resume value:

Hard negative mining is a strong signal that the project understands modern retrieval training, not just model fine-tuning.

### Phase 6: Cross-Encoder Reranker

Goal:

Improve final ranking quality by jointly scoring query-paper pairs.

Architecture:

```text
BM25 / FAISS retrieves top 100
cross-encoder scores query + paper text pairs
return reranked top 10
```

Input format:

```text
[query or goal] [SEP] [paper title + abstract]
```

Training labels:

- Positive: cited/co-cited/relevant papers
- Negative: hard negatives from retrieval candidates

Implementation tasks:

- Add `scripts/train_reranker.py`.
- Add a reranker service interface.
- Add reranking latency measurement.
- Add config for candidate pool size.
- Compare:

```text
BM25
FAISS frozen embeddings
fine-tuned bi-encoder
BM25 + cross-encoder
FAISS + cross-encoder
bi-encoder + cross-encoder
```

Expected tradeoff:

Cross-encoder should improve NDCG/MRR but increase latency. This is an important MLE discussion point.

Expected resume value:

This shows a modern two-stage search/recommender architecture.

### Phase 7: Difficulty Prediction

Goal:

Predict whether a paper is beginner, intermediate, or advanced.

Initial weak labels:

- Beginner: surveys, tutorials, older foundational papers, lower technical density
- Intermediate: method papers with moderate prerequisites
- Advanced: dense frontier papers, high equation/term density, very recent specialized papers

Features:

- Title and abstract text
- Citation count
- Year
- Venue
- Categories
- Technical term density
- Abstract length
- Number of references if available

Model progression:

1. Heuristic rules
2. Logistic regression or gradient boosted trees
3. Transformer classifier

Evaluation:

- Accuracy if labels exist
- Macro F1
- Confusion matrix
- Manual spot checks

Expected resume value:

This differentiates ResearchPath from ordinary paper search. It connects ML output to beginner usefulness.

### Phase 8: Reading Path Planner

Goal:

Turn ranked papers into a structured reading path.

Inputs:

- Retrieval relevance
- Difficulty prediction
- Citation count
- Publication year
- Citation graph position
- Paper category/topic
- User background level

Output groups:

1. Background
2. Foundational
3. Core methods
4. Recent frontier

Potential scoring:

```text
path_score = relevance
           + foundation_weight * citation_signal
           + recency_weight * recency_signal
           - difficulty_penalty_for_beginner
           + graph_centrality_weight * graph_score
```

Important:

The planner should not only return the most similar papers. It should organize papers by learning usefulness.

Expected resume value:

This is the product differentiator. It turns retrieval into an intelligent learning assistant without relying on LLMs.

### Phase 9: Citation Graph Features

Goal:

Add graph-aware ranking signals.

Features:

- In-degree citation count
- PageRank
- Citation neighborhood overlap
- Graph distance from seed paper
- Co-citation count
- Bibliographic coupling count
- Community detection topic cluster

Later optional deep learning:

- GraphSAGE
- GAT
- Node2Vec-style graph embeddings

Recommendation:

Do graph neural networks only after the retrieval/reranking system works. A GNN is impressive, but it should solve a measured ranking problem, not exist as decoration.

### Phase 10: LLM Explanations

Goal:

Use LLMs only after the recommender works.

LLM should explain:

- Why a paper is recommended
- What prerequisites are needed
- How it fits into the reading path
- Beginner-friendly summary

LLM should not be the main recommender.

Reason:

The portfolio should demonstrate MLE/retrieval/deep-learning skill. LLM explanations are polish, not the core system.

## Suggested Next Milestone

The next milestone should be:

> Milestone 8: Trained Difficulty Classifier and Larger Reranker

Deliverables:

- Create a weak-label dataset for paper difficulty.
- Train a baseline classifier for beginner/intermediate/advanced difficulty.
- Compare heuristic difficulty vs trained classifier with manual spot checks.
- Train a larger cross-encoder reranker on thousands of examples.
- Add validation metrics for cross-encoder training.
- Feed trained difficulty predictions into the path planner.

Why this is next:

- The retrieval and reranking stack now exists.
- The first heuristic reading-path planner now exists.
- The next gap is replacing heuristic difficulty with a measured model.
- A larger reranker and trained difficulty classifier make the system stronger both as an ML project and as a beginner-focused product.

## Milestone Checklist

### Milestone 1: Foundation

- [x] Repo structure
- [x] FastAPI backend
- [x] PostgreSQL schema
- [x] JSONL/CSV ingestion
- [x] BM25 retrieval
- [x] Recommendation API
- [x] Evaluation metric utilities
- [x] Simple React frontend
- [x] Unit tests
- [x] Docker Compose Postgres
- [x] README roadmap

### Milestone 2: Dataset and Evaluation

- [ ] Choose real data source
- [x] Add external IDs
- [x] Add citation edge representation
- [x] Add deduplication
- [x] Create dataset manifest
- [x] Create weak relevance label format
- [x] Implement TF-IDF baseline
- [x] Implement citation/recency baseline
- [x] Implement evaluation runner
- [x] Save baseline reports

Milestone 2 implementation note:

- Paper schema now supports external IDs, source, DOI, URL, references count, influential citation count, abstract word count, and updated timestamp.
- Citation edges are represented in a database table.
- Local ingestion supports richer JSONL/CSV records and deduplicates by external ID, DOI, then normalized title.
- Ingestion writes manifests to `data/processed/manifests/`.
- Evaluation examples use JSONL under `data/raw/evaluation_examples.jsonl`.
- Retrieval evaluation writes JSON reports to `data/processed/evaluations/`.
- Deep learning remains intentionally out of scope until the real dataset and baseline evaluation are stable.

### Milestone 3: Frozen Embedding Retrieval

- [x] Add sentence-transformers dependency
- [x] Add embedding generation script
- [x] Store embeddings and metadata
- [x] Implement cosine search
- [x] Add embedding recommendation method
- [x] Compare against BM25

### Milestone 3A: Real OpenAlex Dataset

- [x] Choose OpenAlex as the first real metadata/citation source
- [x] Add OpenAlex works fetcher
- [x] Convert OpenAlex works to ResearchPath JSONL
- [x] Preserve OpenAlex IDs as `external_id`
- [x] Convert referenced works to local reference identifiers
- [x] Add parser tests for OpenAlex records
- [x] Fetch first real starter dataset
- [x] Ingest starter dataset into Postgres
- [x] Run BM25 / TF-IDF / citation-recency evaluation on starter data
- [x] Inspect citation edge coverage
- [ ] Decide larger data filter for 10k-50k CS/ML papers

OpenAlex implementation note:

- Script: `scripts/fetch_openalex_papers.py`
- Default output: `data/raw/openalex_papers.jsonl`
- Raw generated dumps matching `data/raw/openalex_*.jsonl` are git-ignored.
- The script supports `OPENALEX_API_KEY` and `OPENALEX_EMAIL`.
- It uses the OpenAlex `/works` API with `search`, `filter`, `sort`, cursor pagination, `per_page`, and `select`.
- It intentionally does not add deep learning yet.

Recommended starter fetch:

```powershell
.\backend\.venv\Scripts\python.exe scripts\fetch_openalex_papers.py --query "machine learning artificial intelligence" --max-results 500 --output data\raw\openalex_ml_ai.jsonl
```

Then ingest:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\openalex_ml_ai.jsonl --dataset-name openalex_ml_ai --source openalex
```

Current local OpenAlex starter status:

- Fetched `data/raw/openalex_agents_scidisc.jsonl`.
- Ingested 50 OpenAlex records into Postgres.
- Expanded the starter set with top cited/referenced OpenAlex records.
- Current local database after sample + OpenAlex starter + reference expansion: 85 papers, 39 citation edges.
- Low edge coverage is expected because most referenced works are outside the first seed fetch.

### Milestone 3B: Citation Expansion and Weak Labels

- [x] Add citation expansion script
- [x] Rank missing OpenAlex references by frequency
- [x] Fetch selected missing references as new JSONL records
- [x] Add weak-label generation script from citation edges
- [x] Add tests for expansion and weak-label generation
- [x] Expand current OpenAlex seed with top missing references
- [x] Ingest expanded references
- [x] Generate weak-label evaluation examples
- [x] Run baseline evaluation on weak labels
- [x] Inspect whether labels are too sparse or noisy

Commands:

```powershell
.\backend\.venv\Scripts\python.exe scripts\expand_openalex_references.py --input data\raw\openalex_agents_scidisc.jsonl --output data\raw\openalex_agents_scidisc_expanded.jsonl --max-references 100
```

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\openalex_agents_scidisc_expanded.jsonl --dataset-name openalex_agents_scidisc_expanded --source openalex_reference_expansion
```

```powershell
.\backend\.venv\Scripts\python.exe scripts\generate_weak_labels.py --output data\processed\evaluation_examples\weak_labels.jsonl --query-mode title --min-relevant 1
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\processed\evaluation_examples\weak_labels.jsonl
```

Why this matters for deep learning:

- Citation expansion increases positive-pair coverage.
- Weak labels create the evaluation set and later contrastive training pairs.
- This gives future frozen embeddings, FAISS, bi-encoder training, and cross-encoder reranking a measurable target.

Latest local evaluation run:

- Seed fetch: 50 OpenAlex records for `AI agents scientific discovery`.
- Reference expansion: 50 candidate references selected, 29 usable papers fetched.
- Current DB size: 85 papers, 39 citation edges.
- Weak-label examples: 31.
- Report: `data/processed/evaluations/retrieval_evaluation_20260619_173542.json`.
- Baseline metrics:
  - BM25: Recall@5 0.258, Recall@10 0.334, NDCG@10 0.214, MRR 0.205, latency 0.28 ms.
  - TF-IDF: Recall@5 0.267, Recall@10 0.366, NDCG@10 0.218, MRR 0.193, latency 0.64 ms.
  - Citation/recency: Recall@5 0.258, Recall@10 0.334, NDCG@10 0.214, MRR 0.206, latency 0.36 ms.

Interpretation:

- The dataset is still tiny, so these numbers are not resume-ready yet.
- The pipeline is now doing the right thing: fetch real papers, expand citations, generate weak relevance labels, and compare baselines.
- The next data goal is to scale from 85 papers to at least a few thousand papers before adding transformer embeddings.

### Milestone 3C: Larger OpenAlex Dataset Pipeline

- [x] Add `scripts/build_openalex_dataset.py`
- [x] Fetch a 1k-scale OpenAlex CS/ML seed dataset
- [x] Expand references for the larger seed set
- [x] Ingest seed and reference papers through the existing ingestion service
- [x] Generate weak-label examples from the expanded citation graph
- [x] Run BM25 / TF-IDF / citation-recency evaluation on the larger dataset
- [x] Scale from 1k to 5k papers
- [ ] Scale from 5k to 10k papers if needed
- [ ] Improve reference expansion strategy beyond top missing references
- [ ] Add frozen transformer embedding retrieval after the 5k-10k baseline is stable

Pipeline command:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\build_openalex_dataset.py --dataset-name openalex_ai_ml_5k --query "machine learning artificial intelligence" --max-results 5000 --max-references 750 --from-year 2020 --min-citations 5 --query-mode title --min-relevant 1
```

Latest 1k-scale run:

- Dataset: `openalex_ai_ml_1k`
- Seed records fetched: 1000
- Reference records fetched: 126
- Seed/reference ingestion: 956 / 110 inserted
- Citation edges inserted during run: 1703
- Current local DB size: 1151 papers, 1742 citation edges
- Weak-label queries: 831
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260619_174254.json`
- Pipeline summary: `data/processed/manifests/openalex_ai_ml_1k_pipeline_summary.json`
- Baseline metrics:
  - BM25: Recall@5 0.189, Recall@10 0.268, NDCG@10 0.166, MRR 0.169, latency 5.32 ms.
  - TF-IDF: Recall@5 0.175, Recall@10 0.261, NDCG@10 0.159, MRR 0.164, latency 1.60 ms.
  - Citation/recency: Recall@5 0.190, Recall@10 0.271, NDCG@10 0.168, MRR 0.171, latency 6.60 ms.

Interpretation:

- The system now has enough data to start seeing baseline differences.
- Citation/recency slightly beats pure text baselines on this weak-label setup, which is plausible because labels come from citation links.
- The next meaningful ML step is not training yet; it is either scaling to 5k-10k papers or adding frozen transformer embeddings as the first semantic retrieval comparison.

Latest 5k-scale run:

- Dataset: `openalex_ai_ml_5k`
- Seed records fetched: 5000
- Reference records fetched: 490
- Seed/reference ingestion: 4002 / 362 inserted
- Citation edges inserted during run: 11589
- Current local DB size: 5515 papers, 13331 citation edges
- Weak-label queries: 4511
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260619_203600.json`
- Pipeline summary: `data/processed/manifests/openalex_ai_ml_5k_pipeline_summary.json`
- Baseline metrics:
  - BM25: Recall@5 0.123, Recall@10 0.179, NDCG@10 0.119, MRR 0.144, latency 29.99 ms.
  - TF-IDF: Recall@5 0.108, Recall@10 0.164, NDCG@10 0.108, MRR 0.134, latency 6.21 ms.
  - Citation/recency: Recall@5 0.124, Recall@10 0.181, NDCG@10 0.121, MRR 0.145, latency 38.04 ms.

Interpretation:

- The 5k dataset is now large enough to justify adding pretrained transformer embeddings.
- The weak labels are still citation-derived, so citation/recency has an expected advantage.
- The next result to target is frozen transformer retrieval vs BM25/TF-IDF on the same weak-label file.

### Milestone 3D: Frozen Transformer Retrieval

- [x] Add optional `sentence-transformers` dependency group
- [x] Add `EmbeddingRetriever` using cosine similarity over local NumPy embeddings
- [x] Add `scripts/build_embeddings.py`
- [x] Add optional embedding-index support to `scripts/evaluate_retrieval.py`
- [x] Build smoke-test embedding index
- [x] Build full 5k embedding index
- [x] Evaluate frozen embeddings against BM25 / TF-IDF / citation-recency
- [ ] Optimize embedding evaluation by batching query encoding
- [x] Add embedding retrieval endpoint or API method selector
- [ ] Compare a science-paper-specific model such as SPECTER-style embeddings

Commands:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\backend
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
```

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\build_embeddings.py --output data\processed\embeddings\all_minilm_l6_v2_5k.npz --batch-size 32
```

```powershell
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\processed\evaluation_examples\openalex_ai_ml_5k_weak_labels.jsonl --embedding-index data\processed\embeddings\all_minilm_l6_v2_5k.npz --embedding-model sentence-transformers/all-MiniLM-L6-v2 --max-examples 1000
```

Latest frozen embedding run:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding shape: 5515 papers x 384 dimensions
- Embedding index: `data/processed/embeddings/all_minilm_l6_v2_5k.npz`
- Metadata: `data/processed/embeddings/all_minilm_l6_v2_5k.npz.meta.json`
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260619_205744.json`
- Query cap: 1000 weak-label examples
- Metrics:
  - BM25: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.181, latency 50.18 ms.
  - TF-IDF: Recall@5 0.116, Recall@10 0.175, NDCG@10 0.128, MRR 0.171, latency 9.74 ms.
  - Citation/recency: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.182, latency 62.91 ms.
  - Frozen embedding: Recall@5 0.152, Recall@10 0.225, NDCG@10 0.169, MRR 0.216, latency 38.32 ms.

Interpretation:

- Frozen embeddings beat the lexical and citation/recency baselines on the 1k-query dev comparison.
- This validates adding semantic retrieval before training a custom bi-encoder.
- The next engineering step is FAISS because brute-force cosine similarity will not scale cleanly past tens of thousands of papers.

### Milestone 4: FAISS Retrieval

- [x] Add optional FAISS dependency group
- [x] Add `FaissRetriever`
- [x] Add `scripts/build_faiss_index.py`
- [x] Build FAISS index from the 5k frozen embedding file
- [x] Add optional FAISS support to `scripts/evaluate_retrieval.py`
- [x] Measure latency against brute-force embedding retrieval
- [x] Compare quality and speed
- [x] Add API method selector for `embedding` vs `faiss_embedding`
- [x] Add frontend retrieval method selector
- [ ] Add approximate FAISS index option such as IVF or HNSW after scaling beyond 50k papers

Commands:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\backend
.\.venv\Scripts\python.exe -m pip install -e ".[faiss]"
```

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
.\backend\.venv\Scripts\python.exe scripts\build_faiss_index.py --embeddings data\processed\embeddings\all_minilm_l6_v2_5k.npz --output data\processed\faiss\all_minilm_l6_v2_5k.faiss
```

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\processed\evaluation_examples\openalex_ai_ml_5k_weak_labels.jsonl --embedding-index data\processed\embeddings\all_minilm_l6_v2_5k.npz --faiss-index data\processed\faiss\all_minilm_l6_v2_5k.faiss --faiss-id-map data\processed\faiss\all_minilm_l6_v2_5k.ids.npz --embedding-model sentence-transformers/all-MiniLM-L6-v2 --max-examples 1000
```

Latest FAISS run:

- FAISS index: `data/processed/faiss/all_minilm_l6_v2_5k.faiss`
- ID map: `data/processed/faiss/all_minilm_l6_v2_5k.ids.npz`
- Metadata: `data/processed/faiss/all_minilm_l6_v2_5k.faiss.meta.json`
- Vectors indexed: 5515
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260619_222312.json`
- Query cap: 1000 weak-label examples
- Metrics:
  - BM25: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.181, latency 41.92 ms.
  - TF-IDF: Recall@5 0.116, Recall@10 0.175, NDCG@10 0.128, MRR 0.171, latency 7.94 ms.
  - Citation/recency: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.182, latency 51.50 ms.
  - Brute-force embedding: Recall@5 0.152, Recall@10 0.225, NDCG@10 0.169, MRR 0.216, latency 18.31 ms.
  - FAISS embedding: Recall@5 0.152, Recall@10 0.225, NDCG@10 0.169, MRR 0.216, latency 14.98 ms.

Interpretation:

- FAISS exact search returns the same rankings as brute-force cosine similarity, as expected for normalized vectors with `IndexFlatIP`.
- The measured latency improvement is modest at 5k papers because query encoding dominates runtime.
- The value of FAISS becomes more important when scaling to tens or hundreds of thousands of papers and when approximate indexes are introduced.

API/product surface update:

- Added `method` query parameter to `/papers/search`, `/recommend/query`, and `/recommend/paper/{paper_id}`.
- Added `/recommend/methods`.
- Supported methods: `bm25`, `tfidf`, `citation_recency`, `embedding`, `faiss_embedding`.
- Added frontend method selector.
- Real-service smoke test:
  - `bm25` returned 2 recommendations for `AI agents for scientific discovery`.
  - `faiss_embedding` returned 2 recommendations for the same query using the local FAISS index.

### Milestone 5: Fine-Tuned Bi-Encoder

- [x] Build contrastive training pairs
- [x] Mine hard negatives
- [x] Save train/val/test dataset JSONL files
- [x] Save dataset stats report
- [x] Add `scripts/train_biencoder.py`
- [x] Train bi-encoder smoke model
- [x] Save checkpoints
- [x] Build embeddings from trained smoke model
- [x] Evaluate against frozen embeddings
- [x] Document result and failure case
- [x] Train larger bi-encoder run with more triplets
- [x] Add MultipleNegativesRankingLoss training option
- [x] Rebuild FAISS index for trained model
- [x] Compare trained model against frozen embeddings and FAISS
- [ ] Train full-data MNRL run on GPU
- [ ] Add validation evaluator during training

Training data command:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\build_biencoder_dataset.py --use-faiss --negatives-per-positive 3 --bm25-candidates-k 50 --bm25-query-field title
```

Latest bi-encoder dataset build:

- Script: `scripts/build_biencoder_dataset.py`
- Report: `data/processed/training/bi_encoder_dataset_report.json`
- Train: `data/processed/training/bi_encoder_train.jsonl`
- Val: `data/processed/training/bi_encoder_val.jsonl`
- Test: `data/processed/training/bi_encoder_test.jsonl`
- Papers: 5515
- Citation edges: 13331
- Query papers: 4511
- Positive pairs/examples: 26624
- Negatives: 79872
- Average positives per query: 5.90
- Average negatives per example: 3.0
- Train/val/test examples: 21049 / 2644 / 2931
- Label source: citation graph
- Negative source: BM25 + FAISS + random fallback

Implementation notes:

- Split assignment is stable by query paper id to avoid query leakage across train/val/test.
- BM25 hard-negative mining uses paper titles as queries for runtime control; exported training text still uses title + abstract.
- FAISS hard-negative mining uses precomputed paper embeddings directly instead of re-encoding query papers.
- Initial full-title+abstract BM25 mining timed out; this is recorded as a useful engineering constraint.

Smoke training command:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
.\backend\.venv\Scripts\python.exe scripts\train_biencoder.py --max-train-triplets 256 --max-val-triplets 64 --epochs 1 --batch-size 8 --no-evaluator --output-dir data\processed\models\biencoder_all_minilm_smoke
```

Smoke training result:

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Loss: `TripletLoss`
- Train triplets used: 256
- Val triplets sampled: 64
- Runtime: 145.6 seconds on CPU
- Training loss: 5.059
- Model output: `data/processed/models/biencoder_all_minilm_smoke`
- Metadata: `data/processed/models/biencoder_all_minilm_smoke/training_metadata.json`

Trained smoke model evaluation:

- Embedding index: `data/processed/embeddings/biencoder_smoke_5k.npz`
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260620_005152.json`
- Query cap: 1000 weak-label examples
- Metrics:
  - BM25: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.181, latency 42.16 ms.
  - TF-IDF: Recall@5 0.116, Recall@10 0.175, NDCG@10 0.128, MRR 0.171, latency 8.01 ms.
  - Citation/recency: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.182, latency 52.79 ms.
  - Trained smoke embedding: Recall@5 0.131, Recall@10 0.205, NDCG@10 0.151, MRR 0.201, latency 18.13 ms.

Interpretation:

- This is the first real deep-learning training run in the project.
- The smoke model underperforms the frozen embedding baseline, which had Recall@10 0.225 on the comparable 1k-query run.
- That is expected because the smoke run used only 256 triplets and no evaluator; the purpose was to validate the training and evaluation path.
- A serious run should use substantially more pairs, validation evaluation, and preferably GPU.

Larger TripletLoss run:

```powershell
.\backend\.venv\Scripts\python.exe scripts\train_biencoder.py --max-train-triplets 1024 --max-val-triplets 256 --epochs 1 --batch-size 8 --output-dir data\processed\models\biencoder_all_minilm_1k
```

- Model output: `data/processed/models/biencoder_all_minilm_1k`
- Embedding index: `data/processed/embeddings/biencoder_1k_5k.npz`
- FAISS index: `data/processed/faiss/biencoder_1k_5k.faiss`
- Runtime: 536.4 seconds on CPU
- Training loss: 5.049
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260620_013938.json`
- Query cap: 1000 weak-label examples
- Metrics:
  - BM25: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.181, latency 42.87 ms.
  - TF-IDF: Recall@5 0.116, Recall@10 0.175, NDCG@10 0.128, MRR 0.171.
  - Citation/recency: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.182.
  - Triplet embedding: Recall@5 0.056, Recall@10 0.081, NDCG@10 0.069, MRR 0.119, latency 17.97 ms.
  - Triplet FAISS: Recall@5 0.056, Recall@10 0.081, NDCG@10 0.069, MRR 0.119, latency 12.45 ms.

Interpretation:

- TripletLoss degraded retrieval badly on the current weak-label setup.
- Likely causes: default margin is not tuned, citation positives are noisy, and the triplet construction may be too brittle for a small CPU run.
- Do not use TripletLoss as the default training objective until it is tuned.

MNRL training command:

```powershell
.\backend\.venv\Scripts\python.exe scripts\train_biencoder.py --loss mnrl --max-train-triplets 1024 --max-val-triplets 256 --epochs 1 --batch-size 8 --output-dir data\processed\models\biencoder_all_minilm_mnrl_1k
```

MNRL result:

- Loss: `MultipleNegativesRankingLoss`
- Model output: `data/processed/models/biencoder_all_minilm_mnrl_1k`
- Embedding index: `data/processed/embeddings/biencoder_mnrl_1k_5k.npz`
- FAISS index: `data/processed/faiss/biencoder_mnrl_1k_5k.faiss`
- Train pairs used: 1024
- Runtime: 377.3 seconds on CPU
- Training loss: 1.251
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260620_015210.json`
- Query cap: 1000 weak-label examples
- Metrics:
  - BM25: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.181, latency 42.95 ms.
  - TF-IDF: Recall@5 0.116, Recall@10 0.175, NDCG@10 0.128, MRR 0.171, latency 8.15 ms.
  - Citation/recency: Recall@5 0.130, Recall@10 0.189, NDCG@10 0.140, MRR 0.182, latency 52.62 ms.
  - MNRL embedding: Recall@5 0.155, Recall@10 0.227, NDCG@10 0.170, MRR 0.217, latency 18.48 ms.
  - MNRL FAISS: Recall@5 0.155, Recall@10 0.227, NDCG@10 0.170, MRR 0.217, latency 12.40 ms.

Interpretation:

- MNRL is the current best trained bi-encoder objective.
- It slightly beats the frozen embedding baseline on the comparable 1,000-query weak-label slice: Recall@10 0.227 vs 0.225.
- The improvement is small, so it should be framed honestly as a first measured neural training improvement, not a final model.
- The next serious bi-encoder experiment should train on the full dataset with GPU and add validation-time retrieval metrics.

### Milestone 6: Cross-Encoder Reranker

- [x] Build reranking dataset
- [x] Train cross-encoder smoke model
- [x] Add reranker evaluation script
- [x] Evaluate NDCG/MRR and Recall@10 tradeoff
- [x] Measure latency tradeoff
- [ ] Train larger cross-encoder run
- [ ] Add API reranker service
- [ ] Compare candidate pools of 50-100

Implementation notes:

- Dataset builder: `scripts/build_reranker_dataset.py`
- Training script: `scripts/train_cross_encoder.py`
- Evaluation script: `scripts/evaluate_reranker.py`
- Tests:
  - `backend/tests/test_reranker_dataset.py`
  - `backend/tests/test_train_cross_encoder.py`
  - `backend/tests/test_evaluate_reranker.py`

Cross-encoder dataset:

- Report: `data/processed/training/cross_encoder_dataset_report.json`
- Examples: 40157
- Query papers: 4511
- Positives: 26624
- Negatives: 13533
- Train/val/test examples: 31981 / 3943 / 4233
- Positive source: citation graph weak labels
- Negative source: BM25/FAISS mined hard negatives from the bi-encoder dataset

Smoke training command:

```powershell
.\backend\.venv\Scripts\python.exe scripts\train_cross_encoder.py --max-train-examples 64 --max-val-examples 32 --epochs 1 --batch-size 4 --no-evaluator --output-dir data\processed\models\cross_encoder_minilm_smoke
```

Smoke training result:

- Base model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Model output: `data/processed/models/cross_encoder_minilm_smoke`
- Metadata: `data/processed/models/cross_encoder_minilm_smoke/training_metadata.json`
- Train examples used: 64
- Val examples sampled: 32
- Label balance: 32 positive / 32 negative
- Runtime: about 33 seconds of training on CPU
- Training loss: 1.451

Smoke evaluation command:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\evaluate_reranker.py --cross-encoder-model data\processed\models\cross_encoder_minilm_smoke --candidate-method faiss_embedding --candidate-k 25 --rerank-k 10 --max-examples 100 --batch-size 16
```

Smoke evaluation result:

- Candidate retriever: MNRL FAISS
- Candidate pool: 25
- Query cap: 100 weak-label examples
- Evaluation report: `data/processed/evaluations/reranker_evaluation_20260620_021829.json`
- Cross-encoder reranker: Recall@5 0.121, Recall@10 0.200, NDCG@10 0.169, MRR 0.225, latency 586.97 ms.
- Same-slice MNRL FAISS baseline: Recall@5 0.127, Recall@10 0.185, NDCG@10 0.167, MRR 0.249, latency 13.99 ms.

Interpretation:

- The full two-stage neural ranking path now works.
- The smoke reranker slightly improves Recall@10 and NDCG@10 over MNRL FAISS on the same 100-query slice.
- It worsens MRR and is far slower, so it is not a final quality win yet.
- A serious cross-encoder run should use thousands of examples, validation metrics, and candidate pools of 50-100.

### Milestone 7: Beginner-Aware Recommendations

- [x] Add difficulty heuristics
- [ ] Train difficulty classifier
- [x] Add reading path planner
- [x] Add path endpoint
- [x] Update frontend to show path sections
- [x] Add tests for difficulty, planner behavior, and API response shape

Implementation notes:

- Difficulty scorer: `backend/app/services/difficulty.py`
- Reading path planner: `backend/app/services/reading_path.py`
- API endpoint: `GET /path/query`
- Frontend component: `frontend/src/components/PathSection.tsx`
- Frontend page mode: `Reading path` vs `Search`

Path endpoint:

```powershell
curl "http://localhost:8000/path/query?query=AI%20agents%20for%20scientific%20discovery&method=bm25&background_level=basic_ml&k=4"
```

Path sections:

1. `background`
2. `foundational`
3. `core_methods`
4. `frontier`

Difficulty heuristic signals:

- survey/tutorial/introduction keywords reduce difficulty
- long abstracts increase difficulty
- technical term density increases difficulty
- large reference lists increase difficulty
- very recent low-citation papers get a frontier difficulty boost
- very high citation counts can reduce difficulty because the paper may be foundational

Latest verification:

- Backend tests: 59 passed
- Frontend build: passed
- Local Postgres service smoke check returned all four sections for `AI agents for scientific discovery`

Interpretation:

- ResearchPath now does more than semantic search: it can structure retrieved papers into a beginner-aware path.
- The current difficulty model is heuristic and should be described as such.
- The next ML step is to train and evaluate a difficulty classifier, then compare it against this heuristic baseline.

## Notes for Future Updates

Every time a milestone changes, update this file with:

- What changed
- Commands that were run
- Metrics before and after
- Known bugs
- Next experiment
- Resume bullet draft if the milestone is significant

Keep the project honest: save numbers, note failures, and compare against baselines.
