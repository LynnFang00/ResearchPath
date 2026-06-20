# ResearchPath

ResearchPath is a learning-to-rank research paper recommendation system for beginner researchers.

The long-term goal is not just semantic paper search. Given a goal like "I want to learn AI agents for scientific discovery and I know basic ML," ResearchPath should recommend an ordered reading path:

1. Background papers
2. Foundational papers
3. Core method papers
4. Recent frontier papers

The current milestone has moved past the foundation into measured retrieval experiments: BM25/TF-IDF, frozen transformer embeddings, FAISS, and first bi-encoder fine-tuning runs are all evaluated against the same weak-label benchmark.

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the living progress log, current status, todos, and detailed deep learning roadmap.

## Why This Helps Beginner Researchers

Beginner researchers often do not know which paper should come first. Search engines can return relevant papers, but they rarely explain whether a paper is foundational, too advanced, outdated, or a good next step. ResearchPath is designed to become a system that ranks papers by relevance, learning value, difficulty, recency, and citation context.

## Current Milestone

Milestone 1 focused on a clean, extensible foundation:

- FastAPI backend
- PostgreSQL paper schema with SQLAlchemy
- Local JSONL/CSV ingestion
- BM25 baseline retrieval over title and abstract
- Recommendation response format with simple explanations
- Evaluation metric skeleton for Recall@K, Precision@K, NDCG@K, MRR, and latency
- Simple React frontend
- Unit tests

Deep learning is now implemented in the bi-encoder training path, but the project still treats evaluation as the source of truth. A model is only considered useful when it improves saved retrieval metrics against BM25, TF-IDF, frozen embeddings, and FAISS.

## Milestone 2: Dataset and Evaluation Foundation

Milestone 2 adds the dataset and evaluation layer needed before deep learning:

- Richer paper metadata fields: external IDs, DOI, source, URL, reference counts, influential citation counts, abstract word counts, and update timestamps
- Citation edge support through a `citation_edges` table
- Deduplication by external ID, DOI, then normalized title
- Dataset manifests saved under `data/processed/manifests/`
- TF-IDF retrieval baseline
- Citation/recency heuristic baseline
- Shared retriever interface for BM25, TF-IDF, future embeddings, FAISS, and trained models
- Evaluation runner for Recall@5, Recall@10, NDCG@10, MRR, and latency
- JSON evaluation reports saved under `data/processed/evaluations/`

Evaluation comes before deep learning because later transformer models need a scoreboard. Without BM25, TF-IDF, and citation/recency baselines, it is impossible to know whether embeddings, FAISS, a fine-tuned bi-encoder, or a cross-encoder reranker actually improve ranking quality.

Weak supervision means using imperfect but useful labels instead of manually labeling every query-paper pair. For ResearchPath, citation links can act as noisy relevance labels:

- If one paper cites another, the target paper is probably related.
- If two papers are co-cited, they may belong to the same topic.
- If two papers cite many of the same works, they may be methodologically related.

These labels are not perfect. A citation can be background, negative, historical, or incidental. But weak labels are good enough to build initial evaluation sets and training pairs, as long as reports describe them honestly.

## Roadmap

1. Data ingestion for CS/ML paper metadata
2. FastAPI backend and PostgreSQL database
3. BM25 baseline retrieval
4. Evaluation pipeline with Recall@K, NDCG@K, MRR, and latency
5. Pretrained transformer embedding retrieval
6. FAISS vector indexing
7. Simple React frontend
8. Fine-tuned transformer bi-encoder with contrastive learning
9. Hard negative mining from BM25/FAISS
10. Cross-encoder reranker
11. Difficulty prediction for beginner/intermediate/advanced papers
12. Reading path planner
13. Citation graph features
14. User feedback personalization
15. LLM explanations after the recommender works
16. Deployment and README polish

Evaluation comes before advanced deep learning so every improvement can be measured against BM25, TF-IDF, frozen embeddings, fine-tuned bi-encoders, and cross-encoder reranking.

## Milestone 3: OpenAlex Dataset Fetching

ResearchPath now includes an OpenAlex fetcher that converts works from the OpenAlex API into the local JSONL format used by the ingestion pipeline.

OpenAlex is the preferred first real data source because it provides large-scale paper metadata, citation/reference links, topics, authors, venues, publication years, DOI fields, and citation counts. This gives ResearchPath the raw material needed for weak relevance labels before deep learning.

Fetch a small starter dataset:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
.\backend\.venv\Scripts\python.exe scripts\fetch_openalex_papers.py --query "machine learning artificial intelligence" --max-results 500 --output data\raw\openalex_ml_ai.jsonl
```

Optionally use a free OpenAlex API key:

```powershell
$env:OPENALEX_API_KEY="your_key_here"
$env:OPENALEX_EMAIL="you@example.com"
```

Preview the API URL without fetching:

```powershell
.\backend\.venv\Scripts\python.exe scripts\fetch_openalex_papers.py --query "AI agents scientific discovery" --max-results 100 --dry-run
```

Ingest fetched papers:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\openalex_ml_ai.jsonl --dataset-name openalex_ml_ai --source openalex
```

Generated OpenAlex raw dumps matching `data/raw/openalex_*.jsonl` are ignored by git so large local datasets are not accidentally committed.

### Citation Expansion and Weak Labels

A small OpenAlex search fetch often has low internal citation coverage because papers cite many works outside the fetched subset. Expand the dataset by fetching the most common missing referenced works:

```powershell
.\backend\.venv\Scripts\python.exe scripts\expand_openalex_references.py --input data\raw\openalex_agents_scidisc.jsonl --output data\raw\openalex_agents_scidisc_expanded.jsonl --max-references 100
```

Ingest the expanded references:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\openalex_agents_scidisc_expanded.jsonl --dataset-name openalex_agents_scidisc_expanded --source openalex_reference_expansion
```

Generate weak-label evaluation examples from citation edges:

```powershell
.\backend\.venv\Scripts\python.exe scripts\generate_weak_labels.py --output data\processed\evaluation_examples\weak_labels.jsonl --query-mode title --min-relevant 1
```

Run evaluation on those weak labels:

```powershell
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\processed\evaluation_examples\weak_labels.jsonl
```

### Larger Dataset Pipeline

Use the pipeline script when you want to fetch a larger OpenAlex seed set, expand references, ingest both files, generate weak labels, and run the baseline evaluation in one command:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\build_openalex_dataset.py --dataset-name openalex_ai_ml_5k --query "machine learning artificial intelligence" --max-results 5000 --max-references 750 --from-year 2020 --min-citations 5 --query-mode title --min-relevant 1
```

This writes:

- seed papers to `data/raw/<dataset_name>.jsonl`
- expanded reference papers to `data/raw/<dataset_name>_references.jsonl`
- weak-label examples to `data/processed/evaluation_examples/<dataset_name>_weak_labels.jsonl`
- evaluation reports to `data/processed/evaluations/`
- pipeline summaries to `data/processed/manifests/`

Latest local scaled run:

- Dataset: `openalex_ai_ml_5k`
- Seed records fetched: 5000
- Reference records fetched: 490
- Current local database after ingestion: 5515 papers, 13331 citation edges
- Weak-label queries: 4511
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260619_203600.json`
- BM25: Recall@10 0.179, NDCG@10 0.119, MRR 0.144
- TF-IDF: Recall@10 0.164, NDCG@10 0.108, MRR 0.134
- Citation/recency: Recall@10 0.181, NDCG@10 0.121, MRR 0.145

For faster development checks on a large local database, add `--max-eval-examples 1000`. Full reports should omit that cap.

## Milestone 4: Frozen Transformer Retrieval

ResearchPath now supports a frozen semantic retrieval baseline using `sentence-transformers`. This is not fine-tuning yet. It encodes paper title/abstract text into local NumPy embeddings, then ranks papers by cosine similarity.

Install ML extras:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\backend
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
```

Build the local embedding index:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\build_embeddings.py --output data\processed\embeddings\all_minilm_l6_v2_5k.npz --batch-size 32
```

Evaluate BM25, TF-IDF, citation/recency, and frozen embeddings together:

```powershell
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\processed\evaluation_examples\openalex_ai_ml_5k_weak_labels.jsonl --embedding-index data\processed\embeddings\all_minilm_l6_v2_5k.npz --embedding-model sentence-transformers/all-MiniLM-L6-v2 --max-examples 1000
```

Latest frozen embedding dev run:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings: `5515 x 384`
- Embedding index: `data/processed/embeddings/all_minilm_l6_v2_5k.npz`
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260619_205744.json`
- Query cap: 1000 weak-label examples
- BM25: Recall@10 0.189, NDCG@10 0.140, MRR 0.181
- TF-IDF: Recall@10 0.175, NDCG@10 0.128, MRR 0.171
- Citation/recency: Recall@10 0.189, NDCG@10 0.140, MRR 0.182
- Frozen embedding: Recall@10 0.225, NDCG@10 0.169, MRR 0.216

This is the first semantic retrieval result. It gives future FAISS indexing, hard negative mining, and bi-encoder fine-tuning a concrete baseline to beat.

## Milestone 5: FAISS Vector Search

ResearchPath includes a FAISS exact vector-search baseline using `IndexFlatIP` over normalized frozen-transformer embeddings. This keeps ranking equivalent to cosine similarity while moving vector search into a scalable retrieval library.

Install FAISS extras:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\backend
.\.venv\Scripts\python.exe -m pip install -e ".[faiss]"
```

Build the FAISS index from the embedding file:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
.\backend\.venv\Scripts\python.exe scripts\build_faiss_index.py --embeddings data\processed\embeddings\all_minilm_l6_v2_5k.npz --output data\processed\faiss\all_minilm_l6_v2_5k.faiss
```

Evaluate lexical baselines, brute-force embeddings, and FAISS embeddings together:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\processed\evaluation_examples\openalex_ai_ml_5k_weak_labels.jsonl --embedding-index data\processed\embeddings\all_minilm_l6_v2_5k.npz --faiss-index data\processed\faiss\all_minilm_l6_v2_5k.faiss --faiss-id-map data\processed\faiss\all_minilm_l6_v2_5k.ids.npz --embedding-model sentence-transformers/all-MiniLM-L6-v2 --max-examples 1000
```

Latest FAISS dev run:

- FAISS index: `data/processed/faiss/all_minilm_l6_v2_5k.faiss`
- ID map: `data/processed/faiss/all_minilm_l6_v2_5k.ids.npz`
- Vectors indexed: 5515
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260619_222312.json`
- Query cap: 1000 weak-label examples
- Brute-force embedding: Recall@10 0.225, NDCG@10 0.169, MRR 0.216, latency 18.31 ms
- FAISS embedding: Recall@10 0.225, NDCG@10 0.169, MRR 0.216, latency 14.98 ms

## Milestone 6: Bi-Encoder Training Data

ResearchPath now builds weakly supervised contrastive training data for a fine-tuned bi-encoder. Citation graph edges create positives, while BM25 and FAISS retrieve hard negatives.

Build the dataset:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\build_biencoder_dataset.py --use-faiss --negatives-per-positive 3 --bm25-candidates-k 50 --bm25-query-field title
```

Outputs:

- `data/processed/training/bi_encoder_train.jsonl`
- `data/processed/training/bi_encoder_val.jsonl`
- `data/processed/training/bi_encoder_test.jsonl`
- `data/processed/training/bi_encoder_dataset_report.json`

Latest dataset build:

- Papers: 5515
- Citation edges: 13331
- Query papers: 4511
- Positive pairs/examples: 26624
- Negatives: 79872
- Average positives per query: 5.90
- Average negatives per example: 3.0
- Train/val/test examples: 21049 / 2644 / 2931
- Negative sources: BM25 + FAISS + random fallback

Example row shape:

```json
{
  "query_paper_id": 7,
  "positive_paper_id": 273,
  "negative_paper_ids": [3050, 1022, 5268],
  "split": "train",
  "label_source": "citation_graph",
  "negative_source": "bm25_faiss_random",
  "query_text": "...",
  "positive_text": "...",
  "negative_texts": ["...", "...", "..."]
}
```

## Milestone 7: Bi-Encoder Fine-Tuning

ResearchPath now has a working deep-learning training path. It supports `TripletLoss` and `MultipleNegativesRankingLoss` (MNRL) over weakly supervised citation pairs and hard negatives.

Install ML dependencies:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath\backend
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
```

Run smoke fine-tuning:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
.\backend\.venv\Scripts\python.exe scripts\train_biencoder.py --max-train-triplets 256 --max-val-triplets 64 --epochs 1 --batch-size 8 --no-evaluator --output-dir data\processed\models\biencoder_all_minilm_smoke
```

Run a stronger MNRL development experiment:

```powershell
.\backend\.venv\Scripts\python.exe scripts\train_biencoder.py --loss mnrl --max-train-triplets 1024 --max-val-triplets 256 --epochs 1 --batch-size 8 --output-dir data\processed\models\biencoder_all_minilm_mnrl_1k
```

Build embeddings for the trained smoke model:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\build_embeddings.py --model-name data\processed\models\biencoder_all_minilm_smoke --output data\processed\embeddings\biencoder_smoke_5k.npz --batch-size 32
```

Evaluate the trained smoke model:

```powershell
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\processed\evaluation_examples\openalex_ai_ml_5k_weak_labels.jsonl --embedding-index data\processed\embeddings\biencoder_smoke_5k.npz --embedding-model data\processed\models\biencoder_all_minilm_smoke --max-examples 1000
```

Smoke TripletLoss result:

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Loss: `TripletLoss`
- Train triplets used: 256
- Val triplets sampled: 64
- Runtime: 145.6 seconds on CPU
- Training loss: 5.059
- Model output: `data/processed/models/biencoder_all_minilm_smoke`
- Embedding index: `data/processed/embeddings/biencoder_smoke_5k.npz`
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260620_005152.json`
- Smoke model Recall@10: 0.205
- Frozen baseline Recall@10 on the comparable 1k-query run: 0.225

Larger TripletLoss run:

- Model output: `data/processed/models/biencoder_all_minilm_1k`
- Train triplets used: 1024
- Runtime: 536.4 seconds on CPU
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260620_013938.json`
- Triplet embedding Recall@10: 0.081
- Triplet FAISS Recall@10: 0.081

MNRL development run:

- Model output: `data/processed/models/biencoder_all_minilm_mnrl_1k`
- Train pairs used: 1024
- Runtime: 377.3 seconds on CPU
- Training loss: 1.251
- Embedding index: `data/processed/embeddings/biencoder_mnrl_1k_5k.npz`
- FAISS index: `data/processed/faiss/biencoder_mnrl_1k_5k.faiss`
- Evaluation report: `data/processed/evaluations/retrieval_evaluation_20260620_015210.json`
- MNRL embedding Recall@10: 0.227
- MNRL FAISS Recall@10: 0.227
- Comparable BM25 Recall@10: 0.189
- Comparable frozen embedding Recall@10: 0.225

Interpretation: the 1k TripletLoss experiment degraded retrieval, so TripletLoss is not the right default without margin tuning or better triplet construction. MNRL is the stronger current training objective: it slightly beats the frozen embedding baseline on the same 1,000-query weak-label evaluation slice while keeping FAISS-compatible retrieval.

## Milestone 8: Cross-Encoder Reranker

ResearchPath now has the second-stage neural ranking path. The cross-encoder scores a query and candidate paper together after a first-stage retriever such as BM25, FAISS, or the MNRL bi-encoder has selected candidates.

Build cross-encoder training data from the existing weakly supervised bi-encoder data:

```powershell
cd c:\Users\Lynn\UofT\projects\researchpath
.\backend\.venv\Scripts\python.exe scripts\build_reranker_dataset.py
```

Train a small smoke reranker:

```powershell
.\backend\.venv\Scripts\python.exe scripts\train_cross_encoder.py --max-train-examples 64 --max-val-examples 32 --epochs 1 --batch-size 4 --no-evaluator --output-dir data\processed\models\cross_encoder_minilm_smoke
```

Evaluate the reranker on top of MNRL FAISS candidates:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\evaluate_reranker.py --cross-encoder-model data\processed\models\cross_encoder_minilm_smoke --candidate-method faiss_embedding --candidate-k 25 --rerank-k 10 --max-examples 100 --batch-size 16
```

Latest cross-encoder dataset:

- Dataset report: `data/processed/training/cross_encoder_dataset_report.json`
- Examples: 40157
- Query papers: 4511
- Positives: 26624
- Negatives: 13533
- Train/val/test examples: 31981 / 3943 / 4233

Latest smoke training result:

- Base model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Model output: `data/processed/models/cross_encoder_minilm_smoke`
- Train examples used: 64
- Val examples sampled: 32
- Label balance: 32 positive / 32 negative
- Runtime: about 33 seconds of training on CPU
- Training loss: 1.451

Latest smoke reranking result:

- Candidate retriever: MNRL FAISS
- Candidate pool: 25
- Query cap: 100 weak-label examples
- Evaluation report: `data/processed/evaluations/reranker_evaluation_20260620_021829.json`
- Cross-encoder reranker: Recall@10 0.200, NDCG@10 0.169, MRR 0.225, latency 586.97 ms
- Same-slice MNRL FAISS baseline: Recall@10 0.185, NDCG@10 0.167, MRR 0.249, latency 13.99 ms

Interpretation: the smoke reranker proves the full two-stage neural ranking path works. It slightly improves Recall@10 and NDCG@10 on the 100-query slice, but worsens MRR and is far slower. This is not a final reranker yet; the next serious run should train on thousands of examples, use validation metrics, and compare candidate pools of 50-100.

## Milestone 9: Beginner-Aware Reading Paths

ResearchPath now has the first version of its product differentiator: it can organize retrieved papers into a beginner-aware reading path instead of returning only a flat ranked list.

The path planner groups candidate papers into:

1. `background`
2. `foundational`
3. `core_methods`
4. `frontier`

This milestone is heuristic, not a trained difficulty classifier yet. That is intentional: the project now has a testable product layer that future trained difficulty models can replace.

New backend pieces:

- Difficulty scorer: `backend/app/services/difficulty.py`
- Reading path planner: `backend/app/services/reading_path.py`
- API endpoint: `GET /path/query`
- Response schemas: `PathPaper` and `ReadingPathResponse`
- Tests for difficulty, planner behavior, and API response shape

Example API call:

```powershell
curl "http://localhost:8000/path/query?query=AI%20agents%20for%20scientific%20discovery&method=bm25&background_level=basic_ml&k=4"
```

The endpoint returns grouped sections. Each paper includes:

- title, year, authors, snippet
- retrieval score and method
- difficulty label: `beginner`, `intermediate`, or `advanced`
- difficulty explanation
- reading-path section
- path reason

The React frontend now defaults to a reading-path mode and can switch back to flat search. It shows grouped sections with difficulty labels and path-role explanations.

Latest verification:

- Backend tests: 59 passed
- Frontend build: passed
- Local Postgres service smoke check returned all four path sections for `AI agents for scientific discovery`

Next improvement: train a real difficulty classifier using weak labels and manual checks, then use the classifier output in the planner.

## Milestone 10: Fresh Reading Paths

ResearchPath now has a freshness layer so it feels less like a static search demo and more like a dataset-backed reading path system.

Freshness matters because beginner recommendations degrade when the corpus is stale. A reading path needs older foundational papers, but it also needs recent frontier papers from an up-to-date index. The backend exposes dataset status at:

```powershell
curl "http://localhost:8000/dataset/status"
```

The status payload includes dataset name, source, paper count, citation edge count, last updated timestamp, model/index version, embedding model name, and FAISS index path. The frontend shows the total paper count and last updated time.

Incremental updates are handled by:

```powershell
.\backend\.venv\Scripts\python.exe scripts\update_papers.py --file data\raw\new_papers.jsonl --dataset-name openalex_incremental --source local_incremental
```

For this milestone, the fetch layer is pluggable but the implemented provider is local JSONL. The script reads the latest manifest, keeps only records newer than the last update timestamp, deduplicates against existing papers, ingests only new papers, adds citation edges when referenced papers are available, writes a new manifest, and prints an update summary. OpenAlex and arXiv live fetchers can be added behind the same fetcher interface later.

Deduplication now checks external ID, DOI, normalized title, and near-duplicate titles. This improves recommendation quality by preventing repeated survey papers or metadata variants from occupying multiple reading-path slots. When duplicates are found, ResearchPath keeps the more complete metadata record and merges useful fields such as DOI, venue, authors, categories, citation counts, abstracts, and references.

Reading path scoring now uses clearer section-specific rules:

- Background prefers surveys, tutorials, books, broad overviews, and beginner-friendly papers.
- Foundational prefers older central papers with high citation or influence signals.
- Core methods prefers method, architecture, and technical contribution papers with medium difficulty.
- Recent frontier prefers recent, relevant, non-survey papers and penalizes weakly relevant recent papers.

Path responses include debug scoring fields for development: relevance, citation, recency, difficulty, section score, duplicate penalty, and final path score. The frontend does not need to display these fields by default.

Manual reading-path evaluation starts with 20 seed queries in `data/eval/manual_queries.jsonl`. Generate review outputs with:

```powershell
.\backend\.venv\Scripts\python.exe scripts\evaluate_reading_paths.py --queries data\eval\manual_queries.jsonl
```

The stable review file is written to `data/processed/evaluations/reading_path_manual_review.json` and includes empty manual score fields for later human review.

Current limitation: difficulty is still a heuristic model based on metadata, abstract length, technical terms, survey/tutorial signals, citations, and recency. It is useful for coarse routing, but it can misclassify dense surveys, short technical papers, or accessible papers with specialized vocabulary. A trained difficulty model should replace this once there are enough manual labels.

## Repository Layout

```text
researchpath/
  backend/
    app/
      api/
      core/
      db/
      ml/
      models/
      schemas/
      services/
      main.py
    data/
    tests/
    Dockerfile
    pyproject.toml
  frontend/
    src/
      api/
      components/
      pages/
    package.json
  data/
    raw/
    processed/
  notebooks/
  scripts/
  docker-compose.yml
```

## Backend Setup

From the repo root:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Create a `.env` file in `backend/`:

```env
DATABASE_URL=postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath
```

Start PostgreSQL from the repo root:

```bash
docker compose up db
```

Run the API:

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Ingest Sample Data

From the repo root, with the backend virtual environment activated:

```bash
python scripts/ingest_papers.py --file data/raw/sample_papers.jsonl
```

With PowerShell activation disabled, use:

```powershell
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\sample_papers.jsonl --dataset-name sample_papers --source sample
```

Each ingestion run writes a dataset manifest to:

```text
data/processed/manifests/
```

Or use the API:

```bash
curl -X POST "http://localhost:8000/papers/ingest" \
  -H "Content-Type: application/json" \
  -d "{\"file_path\":\"../data/raw/sample_papers.jsonl\"}"
```

## Example API Calls

Health:

```bash
curl http://localhost:8000/health
```

BM25 search:

```bash
curl "http://localhost:8000/papers/search?query=AI%20agents%20scientific%20discovery&k=5"
```

Query recommendations:

```bash
curl "http://localhost:8000/recommend/query?query=beginner%20machine%20learning%20agents&k=5"
```

Available retrieval methods:

```bash
curl "http://localhost:8000/recommend/methods"
```

Method-specific recommendations:

```bash
curl "http://localhost:8000/recommend/query?query=AI%20agents%20scientific%20discovery&k=5&method=faiss_embedding"
```

Supported methods are `bm25`, `tfidf`, `citation_recency`, `embedding`, and `faiss_embedding`.

Seed-paper recommendations:

```bash
curl "http://localhost:8000/recommend/paper/1?k=5&method=embedding"
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

By default, the frontend expects the backend at `http://localhost:8000`. Override it with:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Tests

```bash
cd backend
pytest
```

## Evaluation

Run retrieval evaluation from the repo root:

```powershell
$env:DATABASE_URL='postgresql+psycopg://researchpath:researchpath@localhost:5432/researchpath'
.\backend\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --examples data\raw\evaluation_examples.jsonl
```

The evaluation example format is JSONL:

```json
{"query_id":"agents_scientific_discovery_intro","query":"AI agents for scientific discovery","relevant_paper_ids":[5,6],"notes":"Weak labels from citations or curated seed list"}
```

Reports are saved to:

```text
data/processed/evaluations/
```

The current runner compares:

- BM25
- TF-IDF
- Citation/recency heuristic baseline

## Planned Evaluation Metrics

ResearchPath will compare retrieval and ranking systems using:

- Recall@K: fraction of relevant papers retrieved in the top K
- Precision@K: fraction of top K results that are relevant
- NDCG@K: ranking quality with stronger credit for relevant papers near the top
- MRR: how high the first relevant paper appears
- Latency: query-time cost for retrieval and reranking

Future weak ground truth can come from citation links, co-citation, shared topics, and curated reading lists.
