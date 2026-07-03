# ResearchPath Project Context and Progress Log

Last updated: 2026-07-03

## Current Stage: V6.7 Documentation and Demo Packaging

ResearchPath has completed the V3.9 through V6.6 offline and opt-in runtime pipeline. V6.7 is a packaging/documentation pass: summarize the ML progression, document the accepted safe-fusion ranker, and prepare frontend demo support for comparing explicit opt-in methods. No new labels, retraining, corpus expansion, default changes, or commits are part of this stage.

Current runtime and corpus state:

- Local corpus size: `50,424` papers.
- Backend default runtime method: `bm25`.
- Frontend defaults remain unchanged from the current app state.
- Opt-in backend learned methods: `method=v3_3_ltr`, `method=v4_1_blend`, `method=v4_9_guarded_text_blend`, and `method=v6_4_safe_fusion`.
- `method=v6_4_safe_fusion` is the runtime method name for the accepted V6 safe-fusion path; V6.6 adds a serialized ridge scorer so the safe-fusion implementation can reproduce the offline fusion score when the required inference-safe component scores are present.

Judgment state:

- Total judged labels: `2,400`.
- V3.9 semantic expansion labels: `1,569 / 1,569`.
- V3.9 batches complete: `01-16`.
- V4.8 targeted contrastive override labels are included in the accepted evaluation view, without editing protected V2/V3/V4 label files.

Completed ML pipeline:

1. V3.3 established the strongest compact random-forest LTR model without V2.7 leakage features and was integrated as `method=v3_3_ltr` behind an explicit opt-in.
2. V4.1 added weighted retraining and a calibrated blend, integrated as `method=v4_1_blend` without changing defaults.
3. V4.4/V4.5/V4.6 diagnosed text-score regressions and accepted a guarded text blend only after leakage and robustness audits.
4. V5.0 integrated the accepted guarded text blend as `method=v4_9_guarded_text_blend`, still opt-in only.
5. V6.0 through V6.2 built neural reranker datasets and offline neural baselines. The neural models were useful diagnostic signals but were not promoted directly.
6. V6.3/V6.4 learned a safe fusion of V4.9 and neural signals, then constrained it with top-10 swap limits to avoid severe topic regressions.
7. V6.5 integrated `method=v6_4_safe_fusion` as opt-in only.
8. V6.6 serialized the ridge fusion scorer and completed runtime parity so the safe-fusion path is reproducible rather than summary-dependent.

V6.6 accepted safe-fusion evidence:

- Scorer config: `data/processed/models/v6_6_safe_fusion_ridge_scorer.json`.
- Candidate config: `data/processed/models/v6_6_safe_fusion_candidate.json`.
- Runtime method: `method=v6_4_safe_fusion`.
- V6.6 test Reading NDCG@10: `0.7443` vs V4.9 `0.7242`.
- V6.6 test Topic NDCG@10: `0.8373` vs V4.9 `0.8274`.
- V6.6 test Hard-neg@10: `0.1000`, not worse than V4.9.
- Severe regressions: `0`.
- Non-weak severe regressions: `0`.
- Runtime formula parity max delta: `0.0`.
- Offline/live candidate-set Jaccard in parity smoke: `1.0`.
- Forbidden runtime features required: `[]`.

V6.6 runtime caveat:

- The safe-fusion route can compute V6.6 ridge scores only when the required inference-safe component scores are available: BM25, V3.3 LTR, V4.1 blend, V4.9 guarded text, V6.1 neural, V6.2 multitask neural, and V4.9 confidence features.
- If neural component scores are absent in a live environment, the implementation falls back to preserving the V4.9 guarded-text ordering rather than inventing unavailable neural features.

V6.7 packaging outputs:

- Project context refreshed to the current V6.6/V6.7 state.
- Frontend demo support exposes learned methods as explicit selections and adds an unchecked method-comparison toggle for BM25, V4.9 guarded text, and V6.6 safe fusion.
- Portfolio summary: `docs/v6_7_portfolio_demo_packaging.md`.
- Packaging report: `data/eval/results/v6_7_documentation_packaging_report.json` and `.md`.

Protected artifacts remain unchanged for V2.1 labels, V2.5 labels, V3.2 labels, V3.5 labels, selected 240, and V3.9 labels.

V6.7 constraints:

- Do not train new models.
- Do not add or edit labels.
- Do not expand the corpus.
- Do not change backend runtime defaults or frontend defaults.
- Do not overwrite existing model artifacts.
- Do not commit or push without explicit permission.

## 2026-07-03 Product Frontend and Profile Personalization Pass

After freezing model training work, ResearchPath shifted toward frontend/product polish for portfolio and demo readiness. This pass did not add labels, retrain models, expand the corpus, change backend default ranking, or overwrite model artifacts.

Focused documentation:

- `docs/product_frontend_profile_pass_2026_07_03.md`

Frontend/product changes:

- Reworked the frontend from a single dense demo page into a clearer product layout:
  - top brand/header area,
  - left workflow panel for search/profile/library controls,
  - right workspace for search results, reading paths, and method comparisons.
- Made `ResearchPath` the large first-viewport brand title instead of a generic "Research reading paths" heading.
- Changed typography:
  - UI font: `Plus Jakarta Sans`.
  - Brand/display title font: `Fraunces`.
- Added an editable Reader Profile panel:
  - background level,
  - current status,
  - research goal,
  - paper taste,
  - preferred topics,
  - avoided topics.
- Simplified paper feedback controls:
  - visible buttons: Save, More like this, Not useful,
  - difficulty dropdown: Too easy, Too hard, Already read.
- Moved saved-paper library into the left workflow panel to keep recommendations visually dominant.
- Added readable frontend labels for backend retrieval methods while keeping backend method values unchanged:
  - `bm25` -> "Keyword match"
  - `tfidf` -> "Term-weighted match"
  - `citation_recency` -> "Influential and recent"
  - `embedding` -> "Semantic similarity"
  - `faiss_embedding` -> "Fast semantic search"
  - `hybrid` -> "Personalized blend"
  - `learned_hybrid` -> "Personalized learned blend"
  - `v3_3_ltr` -> "Classic learned ranker"
  - `v4_1_blend` -> "Guardrailed learned blend"
  - `v4_9_guarded_text_blend` -> "Text-aware guarded ranker"
  - `v6_4_safe_fusion` -> "Best offline fusion"
- Added method help text under the selector so users understand the ranking strategy without knowing version IDs.
- Added source actions to result cards:
  - Open paper,
  - PDF,
  - DOI.

Backend/API changes:

- Extended `RecommendationResponse` with paper access fields:
  - `paper_url`,
  - `pdf_url`,
  - `doi_url`,
  - `source_url`,
  - `doi`.
- Added URL derivation in `backend/app/services/formatting.py`:
  - DOI values produce `https://doi.org/...`.
  - arXiv IDs/URLs/DOIs produce `https://arxiv.org/pdf/...`.
  - direct PDF URLs are reused when available.
- Extended the local profile model/API with:
  - `avoid_topics`,
  - `current_status`,
  - `research_goal`,
  - `paper_taste`.
- Added runtime schema patching for those profile columns in `backend/app/db/schema.py`, consistent with the current dev-time schema bridge.
- Updated lightweight personalization for hybrid-style ranking:
  - preferred topics can boost similar papers,
  - avoided topics can penalize similar papers,
  - paper taste and research goal can lightly boost surveys, foundational papers, recent papers, or method-oriented papers.

Important behavior notes:

- Multi-user support is still not implemented in the product. The database has `user_key`, but the profile service still uses the single default profile key, `default`.
- Profile personalization is most meaningful for `hybrid` / `learned_hybrid`. Other opt-in rankers are still mainly evaluation/demo methods and may not adapt strongly to profile edits.
- `method=v6_4_safe_fusion` remains opt-in and can be slow for cold interactive requests because it depends on heavier component scoring.
- Backend method identifiers remain unchanged for compatibility; only frontend labels were made user-facing.
- Backend default remains `bm25`.

Verification run during this pass:

```powershell
npm.cmd run build
.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_feedback_profile_api.py backend\tests\test_product_recommendations.py
.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_formatting.py backend\tests\test_reading_path_api.py
```

Runtime smoke after backend restart:

```powershell
curl.exe -sS http://localhost:8000/health
curl.exe -sS "http://localhost:8000/recommend/query?query=transformer&k=1&method=bm25"
```

The sample recommendation response included link fields such as `paper_url`, `doi_url`, `source_url`, and `doi`.

## Historical Stage: Starting V2 Evaluation and Corpus Expansion

ResearchPath is moving from the completed v1 full-text manual benchmark into v2. The v1 benchmark was useful for comparing the current local retrieval methods, but it is small and pooled from existing retrieval outputs. The next stage is to reduce pooling bias, expand the local corpus, and produce a stronger benchmark before doing more learned-model work.

Current runtime data source:

- Runtime search uses the local PostgreSQL `papers` table.
- Current local DB size checked on 2026-06-24 after V2 50k expansion: `50,424` papers.
- Source breakdown: `44,909` arXiv, `5,509` OpenAlex, `6` seed.
- Runtime search does not currently call OpenAlex live.
- External collection sources currently implemented:
  - OpenAlex `/works` API through `scripts/fetch_openalex_papers.py`.
  - arXiv OAI-PMH through `scripts/fetch_arxiv_papers.py`.

V1 benchmark status:

- Gold labels: `data/eval/manual_labels_fulltext_v1.jsonl`
- Candidate pool: `data/eval/manual_label_pool_v1.jsonl`
- Total labels: `80`
- Topics: `8`
- Labels per topic: `10`
- Last validation: `Valid=True`
- Duplicate count: `1`
- Intentional duplicate: GNN survey duplicate row for `v1_graph_neural_networks`
- Important corrected issue: `paper_id=2700` for `v1_diffusion_image_generation` had mismatched extracted full text and was corrected to `abstract_only`.

V1 evaluation results:

- Baseline report: `data/eval/results/manual_benchmark_method_comparison.json`
- Markdown report: `data/eval/results/manual_benchmark_method_comparison.md`
- Best current baseline: `hybrid`
- Hybrid mean NDCG@10: `0.757`
- Hybrid mean Recall@10: `0.822`
- Hybrid mean Precision@10: `0.312`
- Hybrid mean would_recommend_count@10: `3.12`

Leakage-safe learned reranker result:

- CV report: `data/eval/results/learned_reranker_cv_comparison.json`
- Markdown report: `data/eval/results/learned_reranker_cv_comparison.md`
- Protocol: leave-one-topic-out cross-validation over the 8 v1 topics.
- Best learned model: `learned_relevance_gbr`
- Best learned mean NDCG@10: `0.692`
- Delta vs hybrid NDCG@10: `-0.066`
- Delta vs hybrid Precision@10: `-0.050`
- Delta vs hybrid would_recommend_count@10: `-0.500`
- Conclusion: keep `hybrid` as the default. The learned reranker is data-starved and should not be promoted.

V2 direction:

- Keep the app local-first.
- Expand the local OpenAlex/arXiv corpus before relying on learned rerankers.
- Add optional live external expansion later, but do not make live APIs the only runtime search source.
- Build a mixed-source v2 benchmark with internal candidates, external API candidates, canonical seeds, hard negatives, and random weak-match negatives.
- Rename benchmark metrics honestly as judged metrics because the pool is not exhaustive.

Near-term V2 milestones:

1. Document current system state.
   - DB size/source breakdown.
   - Runtime API flow.
   - Current benchmark caveats.
   - V1 baseline and learned-reranker results.

2. Expand corpus offline.
   - Use OpenAlex as the main metadata/citation source.
   - Add arXiv as an ML/AI and full-text-oriented source.
   - Target `30k-50k` ML/AI papers first.
   - Preserve source provenance.
   - Deduplicate by DOI, OpenAlex ID, arXiv ID, and normalized title.

3. Rebuild retrieval indexes.
   - Rebuild embedding index for the expanded corpus.
   - Rebuild FAISS index for the expanded corpus.
   - Keep BM25/TF-IDF/hybrid comparable on the same corpus.

4. Export raw retrieval scores.
   - Add raw `bm25_score`, `tfidf_score`, `embedding_similarity`, `faiss_similarity`, and `hybrid_score`.
   - Keep old label files unchanged.
   - Use raw scores for later tuned hybrid and reranker features.

5. Build v2 benchmark.
   - Target: `16` topics x `15` labels = `240` labels.
   - Candidate sources per topic:
     - top internal BM25
     - top internal embedding/FAISS
     - top internal hybrid
     - deeper-ranked internal candidates
     - OpenAlex API candidates
     - arXiv API candidates
     - manually seeded canonical/foundational papers
     - hard negatives
     - random weak-match negatives
   - Add `source_provenance` to labeling packets/labels.

6. Rerun baselines on v2.
   - Compare BM25, TF-IDF, embedding, FAISS, hybrid.
   - Use judged NDCG@10, judged Precision@10, judged Recall@10, rec@10, judged@10, duplicate@10, and canonical Coverage@20.

7. Tune hybrid before training larger models.
   - Use leave-one-topic-out CV to tune a simple weighted formula.
   - This is more data-efficient than training a full supervised model on the small manual benchmark.

V2 implementation status on 2026-06-24:

- Added source-aware provenance models:
  - `backend/app/models/paper_identifier.py`
  - `backend/app/models/paper_source.py`
- Added both models to `backend/app/models/__init__.py` so dev startup/schema creation can create the tables.
- Updated ingestion to preserve source identity:
  - DOI, arXiv ID, and OpenAlex ID identifiers are recorded when present.
  - Raw source metadata is stored in `paper_sources`.
  - Existing `Paper.external_id` remains backward-compatible.
  - Exact identifier matches are used for dedupe; uncertain fuzzy matches are not auto-merged.
- Added arXiv harvester:
  - Script: `scripts/fetch_arxiv_papers.py`
  - Source: `https://export.arxiv.org/oai2`
  - Default ML/AI categories: `cs.AI`, `cs.CL`, `cs.CV`, `cs.IR`, `cs.LG`, `cs.NE`, `cs.RO`, `stat.ML`, `eess.IV`
  - Output: normalized JSONL plus raw arXiv metadata.
- Added OpenAlex enrichment for arXiv records:
  - Script: `scripts/enrich_arxiv_with_openalex.py`
  - Match priority: DOI first, then conservative title/year similarity.
  - Output: OpenAlex-normalized JSONL with OpenAlex IDs, citations, topics, references, and raw metadata.
- Fixed OpenAlex timestamp normalization so both date-only and full timestamp `updated_date` values produce valid `updated_at` values.
- Added raw retrieval score/provenance export to `scripts/export_labeling_candidates.py`:
  - `candidate_source`
  - `source_provenance`
  - `retrieval_scores_by_method`
  - `raw_bm25_score`
  - `raw_tfidf_score`
  - `embedding_similarity`
  - `faiss_similarity`
  - `hybrid_score`
  - `appears_in_n_methods`
  - `best_rank`
  - `mean_rank`
  - `has_arxiv_full_text`
  - `has_openalex_metadata`

V2 smoke checks run:

```powershell
.\backend\.venv\Scripts\python.exe -m py_compile scripts\fetch_openalex_papers.py scripts\enrich_arxiv_with_openalex.py scripts\fetch_arxiv_papers.py backend\app\services\ingestion.py scripts\export_labeling_candidates.py
.\backend\.venv\Scripts\python.exe scripts\fetch_arxiv_papers.py --dry-run --max-records 10
.\backend\.venv\Scripts\python.exe scripts\fetch_arxiv_papers.py --output data\raw\arxiv_smoke.jsonl --max-records 2 --sets cs --categories cs.LG,cs.AI,cs.CL --from-date 2024-01-01 --until-date 2024-01-10 --sleep-seconds 0.2
.\backend\.venv\Scripts\python.exe scripts\enrich_arxiv_with_openalex.py --input data\raw\arxiv_smoke.jsonl --output data\raw\arxiv_openalex_smoke.jsonl --max-records 1 --sleep-seconds 0.2
.\backend\.venv\Scripts\python.exe scripts\export_labeling_candidates.py --queries data\eval\query_set_v1.json --out data\eval\v2_candidate_export_smoke.jsonl --top-k 2 --methods bm25
```

Smoke results:

- arXiv smoke fetch wrote `2` normalized rows to `data/raw/arxiv_smoke.jsonl`.
- OpenAlex enrichment matched `1` of those rows and preserved both `arxiv` and `openalex` identifiers.
- Candidate export smoke wrote `16` rows and includes raw BM25 scores, source provenance, and OpenAlex/arXiv flags.
- The smoke records were not intentionally ingested into the main corpus; keep the real corpus expansion staged.

Staged V2 corpus expansion plan:

1. Fetch the first 10k arXiv ML/AI records.
2. Ingest those 10k and inspect dedupe/source provenance.
3. Enrich the same 10k against OpenAlex.
4. Ingest the OpenAlex enrichment and inspect merge quality.
5. Rebuild BM25/TF-IDF, embedding, and FAISS indexes.
6. Run latency and retrieval smoke tests.
7. Repeat at 50k, then 100k.

Do not jump directly to 100k. The expensive part is not only downloading records; it is verifying that arXiv and OpenAlex identity resolution is not creating bad merges.

Suggested first 10k commands:

```powershell
.\backend\.venv\Scripts\python.exe scripts\fetch_arxiv_papers.py --output data\raw\arxiv_ml_ai_10k.jsonl --max-records 10000 --sets cs,stat,eess --categories cs.AI,cs.CL,cs.CV,cs.IR,cs.LG,cs.NE,cs.RO,stat.ML,eess.IV --sleep-seconds 3
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\arxiv_ml_ai_10k.jsonl --dataset-name arxiv_ml_ai_10k --source arxiv
.\backend\.venv\Scripts\python.exe scripts\enrich_arxiv_with_openalex.py --input data\raw\arxiv_ml_ai_10k.jsonl --output data\raw\arxiv_openalex_enrichment_10k.jsonl --max-records 10000 --sleep-seconds 0.15
.\backend\.venv\Scripts\python.exe scripts\ingest_papers.py --file data\raw\arxiv_openalex_enrichment_10k.jsonl --dataset-name arxiv_openalex_enrichment_10k --source openalex
```

V2 10k ingest result on 2026-06-24:

- Fetched `10,000` arXiv ML/AI records:
  - `data/raw/arxiv_ml_ai_10k.jsonl`
  - `data/raw/arxiv_ml_ai_10k.jsonl.meta.json`
- First arXiv ingest:
  - Inserted: `9,977`
  - Skipped/merged: `23`
  - Manifest: `data/processed/manifests/arxiv_ml_ai_10k_2026-06-24T022956_845678_0000.json`
- Added `--exact-dedupe-only` to `scripts/ingest_papers.py` for large source imports.
  - Exact merges still use DOI, external ID, source identifiers, and exact normalized title.
  - Near-title scans are disabled for large imports because they are slow and risky for source identity.
- Widened `papers.venue` from `VARCHAR(256)` to `TEXT` for arXiv comments/proceedings strings.
- Backfilled missing arXiv identifier rows after fixing identifier upserts:
  - Inserted: `0`
  - Skipped: `10,000`
  - Manifest: `data/processed/manifests/arxiv_ml_ai_10k_backfill_identifiers_2026-06-24T023125_896966_0000.json`
- OpenAlex enrichment:
  - Input arXiv records: `10,000`
  - Matched OpenAlex records: `1,204`
  - Output: `data/raw/arxiv_openalex_enrichment_10k.jsonl`
  - Miss sample: `data/raw/arxiv_openalex_enrichment_10k.jsonl.misses.json`
- OpenAlex enrichment ingest:
  - Inserted: `3`
  - Skipped/merged: `1,201`
  - Citation edges inserted: `700`
  - Manifest: `data/processed/manifests/arxiv_openalex_enrichment_10k_2026-06-24T030926_391774_0000.json`
- Current DB after 10k V2 ingest:
  - Papers: `15,495`
  - Source counts: `9,977` arXiv, `5,512` OpenAlex, `6` null source.
  - Citation edges: `14,031`
  - Paper identifiers: `12,398`
  - Identifier counts: `9,987` arXiv, `1,211` DOI, `1,200` OpenAlex.
  - Paper source records: `11,178`
  - Source record counts: `9,978` arXiv, `1,200` OpenAlex.

V2 15k index rebuild:

```powershell
.\backend\.venv\Scripts\python.exe scripts\build_embeddings.py --output data\processed\embeddings\all_minilm_l6_v2_v2_15k.npz --batch-size 32
.\backend\.venv\Scripts\python.exe scripts\build_faiss_index.py --embeddings data\processed\embeddings\all_minilm_l6_v2_v2_15k.npz --output data\processed\faiss\all_minilm_l6_v2_v2_15k.faiss
```

Index outputs:

- Embeddings: `data/processed/embeddings/all_minilm_l6_v2_v2_15k.npz`
- FAISS: `data/processed/faiss/all_minilm_l6_v2_v2_15k.faiss`
- FAISS IDs: `data/processed/faiss/all_minilm_l6_v2_v2_15k.ids.npz`
- Vectors indexed after pre-50k cleanup: `15,492`
- Runtime config defaults now point to these V2 15k indexes.

Retrieval latency smoke after process-local retriever caching:

- BM25 warm average over 2 queries: `39.4 ms`
- TF-IDF warm average: `19.4 ms`
- Embedding warm average: `22.5 ms`
- FAISS warm average: `32.2 ms`
- Hybrid warm average: `172.7 ms`
- Cold first calls remain slower because they build/load retrievers:
  - BM25 first call: about `2.0 s`
  - TF-IDF first call: about `4.6 s`
  - Embedding first call: about `6.6 s`
  - FAISS first call: about `1.1 s`
  - Hybrid first call: about `7.8 s`

V2 10k caveats:

- OpenAlex matched only `1,204 / 10,000` arXiv records in the first pass. This is acceptable for old arXiv-heavy harvests but should improve with DOI-focused categories/newer slices.
- The OpenAlex enrichment ingest inserted `3` records instead of merging all records. These appear to be old/ambiguous arXiv identifier edge cases and should be reviewed before scaling to 50k.
- Existing pre-V2 OpenAlex rows still mostly rely on legacy `Paper.external_id` provenance; a later cleanup can backfill `paper_identifiers` and `paper_sources` for those old rows.
- The service now uses process-local caches. This is good enough for local dev and demos, but production should use explicit startup index loading and invalidation.

V2 provenance correctness pass on 2026-06-24:

- Inspected the `3` OpenAlex-only enrichment inserts:
  - `paper_id=15800` should have merged with arXiv `paper_id=5880`, arXiv ID `cmp-lg/9806001`.
  - `paper_id=15801` should have merged with arXiv `paper_id=5901`, arXiv ID `cmp-lg/9807002`.
  - `paper_id=15802` should have merged/enriched arXiv `paper_id=15577`, arXiv ID `1211.5740`; OpenAlex title is a container/proceedings title but the abstract/source row points to the arXiv paper.
- Root cause:
  - The OpenAlex enrichment JSONL carried arXiv IDs, but ingestion duplicate lookup only used `external_id`, DOI, and exact normalized title.
  - It did not use incoming `identifiers.arxiv` / `identifiers.openalex` rows during duplicate lookup.
  - Old-style arXiv DOI extraction did not handle `10.48550/arxiv.cmp-lg/...`.
- Fixes:
  - Ingestion duplicate lookup now includes normalized source identifiers from incoming records.
  - Old-style arXiv IDs such as `cmp-lg/9806001` are extracted from `10.48550/arxiv.cmp-lg/9806001`.
  - Added regression tests for OpenAlex enrichment merging by arXiv identifier and old-style arXiv DOI extraction.
- Backfilled legacy OpenAlex provenance:
  - Script: `scripts/backfill_openalex_provenance.py`
  - Candidate papers: `5,512`
  - OpenAlex identifiers added: `5,508`
  - OpenAlex source rows added: `5,508`
  - Existing rows: `4`
- Added corpus provenance validation:
  - Script: `scripts/validate_corpus_provenance.py`
  - JSON report: `data/processed/reports/corpus_provenance_validation_v2_15k.json`
  - Markdown report: `data/processed/reports/corpus_provenance_validation_v2_15k.md`
- Validation after backfill, before explicit duplicate cleanup:
  - Papers: `15,495`
  - Paper count by source: `9,977` arXiv, `5,512` OpenAlex, `6` null.
  - Identifier count by type: `9,987` arXiv, `1,211` DOI, `6,708` OpenAlex.
  - Papers with both arXiv and OpenAlex identifiers: `1,197`
  - Papers missing identifiers: `6`
  - Papers missing source provenance: `6`
  - Duplicate DOI/arXiv/OpenAlex identifiers: `0`
  - Duplicate normalized titles shown in report: `30`
  - OpenAlex-only enrichment inserts still present in current DB: `3`
- The next pass removed these 3 duplicate OpenAlex enrichment rows explicitly, after transferring useful metadata, identifiers, and source provenance.

Pre-50k final cleanup on 2026-06-24:

- Inspected the `6` papers missing identifiers and source provenance.
  - They are the original seed/demo records, not the OpenAlex enrichment duplicates.
  - Seed papers: IDs `1` through `6`.
  - Titles include `Attention Is All You Need`, `ReAct`, `Toolformer`, `SPECTER`, the LLM agents survey, and the automated scientific discovery survey.
- Backfilled those seed rows as intentional internal records:
  - Script: `scripts/backfill_seed_provenance.py`
  - Added `seed` identifiers like `seed:1`.
  - Added `seed` source provenance rows.
  - Updated blank `Paper.source` values to `seed`.
- Added explicit duplicate cleanup script:
  - Script: `scripts/cleanup_known_openalex_enrichment_duplicates.py`
  - Cleanup report: `data/processed/reports/cleanup_known_openalex_enrichment_duplicates_v2_15k.json`
  - Merged `15800 -> 5880`.
  - Merged `15801 -> 5901`.
  - Merged `15802 -> 15577`.
  - Transferred OpenAlex identifiers, DOI identifiers where useful, source rows, and stronger metadata before deleting duplicate paper rows.
  - Citation edge transfer was checked; these duplicate rows had no citation edges to move.
- Rebuilt indexes after deleting the duplicate rows:
  - Embeddings: `data/processed/embeddings/all_minilm_l6_v2_v2_15k.npz`
  - FAISS: `data/processed/faiss/all_minilm_l6_v2_v2_15k.faiss`
  - Vectors indexed: `15,492`
- Final validation:
  - Report: `data/processed/reports/corpus_provenance_validation_v2_15k.json`
  - Markdown: `data/processed/reports/corpus_provenance_validation_v2_15k.md`
  - Papers: `15,492`
  - Paper count by source: `9,977` arXiv, `5,509` OpenAlex, `6` seed.
  - Identifier count by type: `9,987` arXiv, `1,211` DOI, `6,708` OpenAlex, `6` seed.
  - Papers with both arXiv and OpenAlex identifiers: `1,200`
  - Papers missing identifiers: `0`
  - Papers missing source provenance: `0`
  - Duplicate DOI/arXiv/OpenAlex identifiers: `0`
  - OpenAlex-only enrichment inserts from the known bug: `0`
  - Duplicate normalized titles still shown in report: `30`; these are legacy OpenAlex/title duplicates, not duplicate source identifiers.
- Verification:
  - Required targeted suite: `19 passed`
  - Full backend suite: `99 passed`, with one existing Starlette/httpx deprecation warning.

Pre-50k status:

- The known source-identifier merge bug is fixed for future enrichment imports.
- The three already-inserted duplicate OpenAlex rows have been explicitly merged away.
- The seed/demo rows are now documented as `seed` provenance, not malformed ingestion rows.
- It is reasonable to proceed to the 50k arXiv/OpenAlex expansion after reviewing whether the remaining duplicate normalized-title report needs a separate legacy OpenAlex cleanup.

V2 50k expansion on 2026-06-24:

- Pre-expansion baseline validation:
  - Report: `data/processed/reports/corpus_provenance_validation_v2_15k_pre_50k_baseline.json`
  - Papers: `15,492`
  - Missing identifiers: `0`
  - Missing source provenance: `0`
  - Duplicate DOI/arXiv/OpenAlex identifiers: `0`
  - OpenAlex-only enrichment inserts: `0`
  - Duplicate normalized titles: `30`
- arXiv expansion fetch:
  - Script: `scripts/fetch_arxiv_papers.py`
  - Output: `data/raw/arxiv_ml_ai_50k_incremental.jsonl`
  - Records fetched: `35,000`
  - Category filter: `cs.AI`, `cs.CL`, `cs.CV`, `cs.IR`, `cs.LG`, `cs.NE`, `cs.RO`, `stat.ML`, `eess.IV`
  - Date bias: `from-date=2018-01-01`
  - Batch years were mostly `2018` and `2019`.
- arXiv expansion ingest:
  - Inserted: `34,932`
  - Skipped/merged: `68`
  - Dedupe mode: exact/source-identifier only; no near-title fuzzy merge.
  - Manifest: `data/processed/manifests/arxiv_ml_ai_50k_incremental_2026-06-24T062500_600315_0000.json`
- OpenAlex enrichment:
  - Existing sequential enrichment timed out on a 4,471 DOI subset because it wrote only after completion.
  - Added `--doi-only` and conservative parallel `--workers` support to `scripts/enrich_arxiv_with_openalex.py`.
  - DOI-bearing arXiv subset: `data/raw/arxiv_ml_ai_50k_incremental_doi_subset.jsonl`
  - DOI subset size: `4,471`
  - DOI-only OpenAlex matches: `4,405`
  - Enrichment output: `data/raw/arxiv_openalex_enrichment_50k_doi_subset.jsonl`
  - Miss sample: `data/raw/arxiv_openalex_enrichment_50k_doi_subset.jsonl.misses.json`
- OpenAlex enrichment ingest:
  - Inserted: `0`
  - Merged/skipped: `4,405`
  - Citation edges inserted: `9,973`
  - Manifest: `data/processed/manifests/arxiv_openalex_enrichment_50k_doi_subset_2026-06-24T064416_641000_0000.json`
  - This was the desired behavior: OpenAlex rows enriched existing arXiv papers rather than creating new paper rows.
- 50k corpus validation:
  - JSON report: `data/processed/reports/corpus_provenance_validation_v2_50k.json`
  - Markdown report: `data/processed/reports/corpus_provenance_validation_v2_50k.md`
  - Papers: `50,424`
  - Paper count by source: `44,909` arXiv, `5,509` OpenAlex, `6` seed.
  - Citation edges: `24,004`
  - Paper identifiers: `61,757`
  - Paper source provenance records: `56,065`
  - arXiv identifiers: `44,976`
  - DOI identifiers: `5,676`
  - OpenAlex identifiers: `11,099`
  - Papers with both arXiv and OpenAlex identifiers: `5,640`
  - Papers missing identifiers: `0`
  - Papers missing source provenance: `0`
  - Duplicate DOI/arXiv/OpenAlex identifiers: `0`
  - OpenAlex-only enrichment inserts from known bug pattern: `0`
  - Duplicate normalized titles: `30`; these did not increase from the clean 15k baseline.
- 50k index rebuild:
  - Embeddings: `data/processed/embeddings/all_minilm_l6_v2_50k.npz`
  - Embedding metadata: `data/processed/embeddings/all_minilm_l6_v2_50k.npz.meta.json`
  - FAISS: `data/processed/faiss/all_minilm_l6_v2_50k.faiss`
  - FAISS IDs: `data/processed/faiss/all_minilm_l6_v2_50k.ids.npz`
  - Embeddings indexed: `50,424`
  - FAISS IDs indexed: `50,424`
  - Runtime config now points to the 50k index files.
- Retrieval latency report:
  - JSON: `data/processed/reports/retrieval_latency_v2_50k.json`
  - Markdown: `data/processed/reports/retrieval_latency_v2_50k.md`
  - Warm average latency over 8 representative topics:
    - BM25: `106.566 ms`
    - TF-IDF: `75.859 ms`
    - Embedding: `40.409 ms`
    - FAISS: `92.104 ms`
    - Hybrid: `288.973 ms`
  - Cold first query:
    - BM25: `6,250.321 ms`
    - TF-IDF: `16,949.477 ms`
    - Embedding: `8,593.913 ms`
    - FAISS: `2,661.195 ms`
    - Hybrid: `24,506.464 ms`
  - Memory usage was not captured in this pass.
- Retrieval sanity report:
  - JSON: `data/processed/reports/retrieval_sanity_v2_50k.json`
  - Markdown: `data/processed/reports/retrieval_sanity_v2_50k.md`
  - Topics: 8 representative ML/AI topics.
  - Methods: BM25, TF-IDF, embedding, FAISS, hybrid.
  - Duplicate titles in top 10: `0` across all method/topic combinations.
  - Weak query-term overlap zero count: `0` across all method/topic combinations.
- V2 benchmark candidate scaffolding:
  - Query seed file: `data/eval/query_set_v2_seed.json`
  - Export script: `scripts/export_v2_labeling_candidates.py`
  - Candidate pool: `data/eval/v2_labeling_candidate_pool.jsonl`
  - Metadata: `data/eval/v2_labeling_candidate_pool.jsonl.meta.json`
  - Topics: `16`
  - Rows: `1,274`
  - Rows per topic range: `56` to `99`
  - Candidate source tags include internal method top/deeper results, arXiv/OpenAlex source candidates, canonical seed hooks, hard-negative candidates, and random weak negatives.
  - This is a pre-labeling pool, not final labels.
- Verification after 50k expansion:
  - Targeted suite: `19 passed`
  - Full backend suite: `99 passed`, with one existing Starlette/httpx deprecation warning.

V2 labeling readiness checkpoint on 2026-06-26:

- Candidate pool review:
  - Candidate pool: `data/eval/v2_labeling_candidate_pool.jsonl`
  - Metadata: `data/eval/v2_labeling_candidate_pool.jsonl.meta.json`
  - Rows: `1,274`
  - Topics: `16`
  - Pool construction intentionally mixes:
    - top internal BM25 candidates
    - top internal embedding/FAISS candidates
    - top internal hybrid candidates
    - deeper internal candidates
    - arXiv source candidates
    - OpenAlex source candidates
    - canonical/foundational seed hooks
    - hard negatives
    - random weak-match negatives
  - This reduces pooling bias compared with v1, where labels mostly came from current retrieval outputs.
- Selected manual-labeling subset:
  - Selection script: `scripts/select_v2_labeling_subset.py`
  - Selected file: `data/eval/v2_labeling_selected_240.jsonl`
  - Selection report: `data/eval/v2_labeling_selection_report.md`
  - Selected rows: `240`
  - Topics: `16`
  - Rows per topic: exactly `15`
  - No relevance labels have been created yet.
  - Selection policy: include canonical/foundational rows where available, then balance top BM25, top embedding/FAISS, top hybrid, deeper-rank candidates, hard negatives, random weak negatives, and year/source diversity.
- Canonical coverage audit before manual labeling:
  - Audit script: `scripts/audit_v2_canonical_coverage.py`
  - Audit report: `data/eval/v2_canonical_coverage_audit.md`
  - Goal: preserve exactly `240` rows while checking that each topic has enough likely positive/core papers before human labeling.
  - Likely coverage tags added for triage only:
    - `likely core/foundational positive`
    - `likely relevant survey/background`
    - `likely recent frontier/application`
    - `likely hard negative`
    - `likely random/irrelevant negative`
  - These tags are not gold labels and must not be used as evaluation/training targets.
  - Validation after audit:
    - Rows: `240`
    - Topics: `16`
    - Bad per-topic counts: `0`
    - Coverage counts: `25` likely core/foundational, `51` likely survey/background, `114` likely recent frontier/application, `18` likely hard negative, `32` likely random/irrelevant negative.
- Canonical coverage replacements made:
  - `v2_retrieval_augmented_generation`: replaced `9920` Storing and Indexing Plan Derivations through Explanation-based Analysis of Retrieval Failures with `46348` Retrieval-Enhanced Adversarial Training for Neural Response Generation.
  - `v2_diffusion_image_generation`: replaced `11847` Benchmarking the Quality of Diffusion-Weighted Images with `313` T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models.
  - `v2_diffusion_image_generation`: replaced `30257` Vector Learning for Cross Domain Representations with `3876` A taxonomy of prompt modifiers for text-to-image generation.
  - `v2_efficient_transformers`: replaced `49227` Transfer NAS: Knowledge Transfer between Search Spaces with Transformer Agents with `49828` Sharing Attention Weights for Fast Transformer.
  - `v2_robot_learning`: replaced `9017` Avoider robot design to dim the fire with dt basic mini system with `33621` An Algorithmic Perspective on Imitation Learning.
  - `v2_ai_for_scientific_discovery`: replaced `1850` Managing artificial intelligence with `3953` Self-driving laboratories to autonomously navigate the protein fitness landscape.
  - `v2_ai_for_scientific_discovery`: replaced `35425` Building Ethically Bounded AI with `36` Autonomous chemical research with large language models.
  - All replacements came from the existing candidate pool, not from newly fetched external data.
- Focus-topic coverage after audit:
  - `v2_retrieval_augmented_generation`: `13` likely positive/background/frontier and `2` likely negatives.
  - `v2_diffusion_image_generation`: `13` likely positive/background/frontier and `2` likely negatives.
  - `v2_efficient_transformers`: `12` likely positive/background/frontier and `3` likely negatives.
  - `v2_robot_learning`: `13` likely positive/background/frontier and `2` likely negatives.
  - `v2_ai_for_scientific_discovery`: `10` likely positive/background/frontier and `5` likely negatives.
- Verification:
  - Compile check passed:
    - `scripts/audit_v2_canonical_coverage.py`
    - `scripts/select_v2_labeling_subset.py`
  - No labels were created.
  - No model training was run.

V2.1 benchmark upgrade on 2026-06-26:

- The V2 selected candidate file remains frozen:
  - `data/eval/v2_labeling_selected_240.jsonl`
  - SHA256 after V2.1 packet generation: `6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F`
- Added V2.1 labeling docs:
  - `data/eval/v2_1_labeling_guide.md`
  - `data/eval/v2_1_labeling_schema.md`
  - `data/eval/v2_1_anchor_calibration.md`
- Added V2.1 packet builder:
  - Script: `scripts/build_v2_1_labeling_packets.py`
  - Output packet: `data/eval/v2_1_labeling_packets.jsonl`
  - Metadata: `data/eval/v2_1_labeling_packets.meta.json`
  - Rows: `240`
  - Topics: `16`
  - Rows per topic: exactly `15`
  - The packet preserves candidate membership and order while adding schema metadata and topic-local anchors.
  - Anchors are calibration hints only, not labels or training targets.
- Added V2.1 validation:
  - Script: `scripts/validate_v2_1_labels.py`
  - Checks all required continuous score fields are present and in `[0, 1]`.
  - Checks required `intent_scores` fields.
  - Checks enum values for roles, duplicate status, evidence level, and label confidence.
  - Emits soft consistency warnings.
  - Prints per-topic distribution summaries.
- Added V2.1 evaluation scaffolding:
  - Script: `scripts/evaluate_v2_1_benchmark.py`
  - Reports topic-match NDCG, reading-value NDCG, audience-specific NDCG, intent-specific NDCG, role coverage, duplicate penalty, and path-level coverage.
  - It consumes future V2.1 labels and packet ranks only; it does not train models.
- Verification:
  - Compile check passed for the three V2.1 scripts.
  - Focused V2.1 tests: `3 passed`.
- No labels were created.
- No model training was run.
- The selected 240 candidates were not modified.

V2.1 packet generation rule update on 2026-06-26:

- Updated `scripts/build_v2_1_labeling_packets.py` so generated V2.1 packets are clean labeling payloads, not raw copies of every selected-row field.
- No-truncation policy:
  - Full abstracts are included without truncation.
  - `abstract_snippet` and other snippet fields are omitted from generated V2.1 packets.
  - Generated packet text must not use character limits, ellipsized abstracts, or partial paragraphs.
  - If packet text becomes too large, the builder splits packets instead of shortening text.
- Added one-topic packet outputs:
  - Directory: `data/eval/v2_1_labeling_packets_by_topic/`
  - Files: `16`
  - Each current topic packet contains exactly `15` selected candidates.
  - Current packet sizes are below the split threshold, so no topic needed a split part in this run.
- Candidate packet fields now include:
  - `query_id`, `query`, `paper_id`, `title`, `year`, `venue`
  - full `abstract`
  - `sources_provenance`
  - normalized `identifiers` with `arxiv_id`, `doi`, and `openalex_id`
  - `source_url` and `pdf_url` when available or derivable
  - `selection_reasons`
  - `likely_coverage` as a heuristic-only object
  - retrieval ranks and scores by method
  - `citation_count`
  - duplicate-title cluster info
  - optional authors, arXiv/OpenAlex category/topic fields, source-specific metadata, and anchor notes
- Full-text rule:
  - Full text is not included for every paper by default.
  - If the abstract is missing, very short, or ambiguous, the builder uses local full-text manifests when available.
  - Included full-text evidence uses complete detected section headings, introduction, and/or conclusion text; no included section is truncated.
  - Current generated evidence levels: `234` `title_abstract`, `6` `title_abstract_intro_conclusion`.
- Verification:
  - Frozen selected file hash unchanged: `6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F`
  - Generated aggregate packet rows: `240`
  - Generated topic packet files: `16`
  - Snippet-field audit found no `abstract_snippet` or `"snippet"` fields in generated packets.
  - Compile check passed for V2.1 scripts.
  - Focused V2.1 tests: `3 passed`.
- No labels were created.
- No model training was run.
- The selected 240 candidates were not modified.

V2.1 ChatGPT upload packet generation on 2026-06-26:

- Added upload-file generator:
  - Script: `scripts/build_v2_1_chatgpt_upload_files.py`
  - Input: `data/eval/v2_1_labeling_packets_by_topic/`
  - Output directory: `data/eval/v2_1_chatgpt_upload/`
- Generated one ChatGPT-upload-ready Markdown file per topic:
  - `topic_01_v2_transformer_architecture.md`
  - `topic_02_v2_retrieval_augmented_generation.md`
  - `topic_03_v2_graph_neural_networks.md`
  - `topic_04_v2_contrastive_learning.md`
  - `topic_05_v2_bayesian_optimization.md`
  - `topic_06_v2_large_language_model_agents.md`
  - `topic_07_v2_recommendation_systems.md`
  - `topic_08_v2_diffusion_image_generation.md`
  - `topic_09_v2_ai_for_scientific_discovery.md`
  - `topic_10_v2_multimodal_learning.md`
  - `topic_11_v2_graph_recommendation.md`
  - `topic_12_v2_efficient_transformers.md`
  - `topic_13_v2_llm_evaluation.md`
  - `topic_14_v2_self_supervised_vision.md`
  - `topic_15_v2_causal_representation_learning.md`
  - `topic_16_v2_robot_learning.md`
- Each upload file includes:
  - V2.1 label schema
  - allowed enum values
  - score anchors
  - topic query and `query_id`
  - topic-specific anchor calibration
  - exactly `15` selected candidate papers
  - full abstracts with no truncation
  - source/provenance, identifiers, URLs, citation counts, selection reasons, heuristic-only likely coverage, retrieval ranks/scores, duplicate-title info, and evidence availability
- No upload file needed to be split in the current run.
- Added workflow README:
  - `data/eval/v2_1_chatgpt_upload/README.md`
  - It instructs to upload one topic file, ask ChatGPT for JSONL labels only, save the returned JSONL, append labels, validate, and repeat for all topics.
- Added append helper so the README points to an executable command:
  - Script: `scripts/append_v2_1_labels.py`
  - Validates incoming rows before appending.
  - Rejects duplicate `(query_id, paper_id)` labels unless `--replace` is used.
- Verification:
  - Frozen selected file hash unchanged: `6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F`
  - Upload Markdown files: `16` topic files plus `README.md`
  - Candidate heading count: `15` in every topic file
  - Snippet-field audit found no `abstract_snippet` or `"snippet"` fields in upload files.
  - `scripts/append_v2_1_labels.py --help` runs successfully.
  - Compile check passed for upload/append/validation scripts.
  - Focused V2.1 tests: `3 passed`.
- No labels were created.
- No model training was run.
- No evaluation was run.

V2.1 labeling completed:

- Final topic labels appended:
  - Topic 16: `v2_robot_learning`
  - Labels appended: `15`
  - Total labels: `240 / 240`
  - Labels remaining: `0`
- Completed label file:
  - `data/eval/manual_labels_v2_1.jsonl`
- Final validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Labels: `240`
  - Topics complete: `16 / 16`
  - Labels per topic: exactly `15`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `0`
- Final topic distribution for `v2_robot_learning`:
  - Roles: `application=3`, `core_methods=9`, `foundational=1`, `negative=2`
  - Topic match buckets: `high=12`, `medium=1`, `low=2`
  - Reading value buckets: `high=6`, `medium=7`, `low=2`
- No evaluation was run.
- Next recommended step:
  - Run V2.1 benchmark evaluation comparing BM25, TF-IDF, embedding, FAISS, and hybrid on the completed labels.

V2.1 benchmark evaluation completed:

- Validation before evaluation:
  - Labels: `240`
  - Valid: `True`
  - Warnings: `0`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
- Evaluation command:
  - `.\backend\.venv\Scripts\python.exe scripts\evaluate_v2_1_benchmark.py --labels data\eval\manual_labels_v2_1.jsonl --packet data\eval\v2_1_labeling_packets.jsonl --json-out data\eval\results\v2_1_benchmark_method_comparison.json --md-out data\eval\results\v2_1_benchmark_method_comparison.md`
- Output files:
  - `data/eval/results/v2_1_benchmark_method_comparison.json`
  - `data/eval/results/v2_1_benchmark_method_comparison.md`
- Best method by topic-match NDCG@10:
  - `bm25`: `0.876`
- Best method by reading-value NDCG@10:
  - `hybrid`: `0.832`
- Method averages:
  - `hybrid`: topic NDCG `0.866`, reading NDCG `0.832`, beginner `0.839`, intermediate `0.845`, advanced `0.835`, expert `0.802`, role coverage `0.958`, path coverage `0.859`, duplicate penalty `0.006`, judged@10 `9.12`
  - `bm25`: topic NDCG `0.876`, reading NDCG `0.821`, beginner `0.754`, intermediate `0.827`, advanced `0.853`, expert `0.832`, role coverage `0.946`, path coverage `0.828`, duplicate penalty `0.006`, judged@10 `9.38`
  - `embedding`: topic NDCG `0.790`, reading NDCG `0.764`, beginner `0.732`, intermediate `0.770`, advanced `0.764`, expert `0.741`, role coverage `0.944`, path coverage `0.875`, duplicate penalty `0.006`, judged@10 `7.69`
  - `faiss_embedding`: topic NDCG `0.790`, reading NDCG `0.764`, beginner `0.732`, intermediate `0.770`, advanced `0.764`, expert `0.741`, role coverage `0.944`, path coverage `0.875`, duplicate penalty `0.006`, judged@10 `7.69`
  - `tfidf`: topic NDCG `0.773`, reading NDCG `0.721`, beginner `0.693`, intermediate `0.730`, advanced `0.744`, expert `0.721`, role coverage `0.943`, path coverage `0.812`, duplicate penalty `0.006`, judged@10 `8.25`
- Skipped methods:
  - `learned_hybrid`: no materialized ranks in packet.
- No evaluation errors.
- No model training was run.
- The selected 240 candidates were not modified.

V2.1 labeling progress update:

- Next two-topic batch labels appended:
  - Topic 14: `v2_self_supervised_vision`
  - Topic 15: `v2_causal_representation_learning`
  - Labels appended: `30`
  - Total labels: `225 / 240`
  - Labels remaining: `15`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `0`
- Per-topic distribution:
  - `v2_self_supervised_vision`
    - Roles: `application=4`, `background=5`, `core_methods=2`, `evaluation_benchmark=2`, `negative=2`
    - Topic match buckets: `high=10`, `medium=1`, `low=4`
    - Reading value buckets: `high=7`, `medium=3`, `low=5`
  - `v2_causal_representation_learning`
    - Roles: `application=2`, `background=3`, `core_methods=6`, `foundational=2`, `negative=2`
    - Topic match buckets: `high=5`, `medium=8`, `low=2`
    - Reading value buckets: `high=2`, `medium=9`, `low=4`
- No evaluation was run.
- One topic remains before final V2.1 benchmark evaluation:
  - Topic 16: `v2_robot_learning`

V2.1 validation warning policy update:

- Kept `paper_id=708`, `Holistic Evaluation of Language Models`, as labeled.
- Updated `scripts/validate_v2_1_labels.py` so `high_beginner_and_expert_fit` is not warned when a row has a broad reference role:
  - `background`
  - `foundational`
  - `evaluation_benchmark`
- Reason:
  - Some benchmark/framework/survey-style references can be beginner-accessible overviews and expert-relevant references at the same time.
  - HELM is a valid example: broad readable LLM evaluation coverage plus serious benchmark/framework value.
- Validation after update:
  - Labels: `195`
  - Valid: `True`
  - Warnings: `0`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
- No evaluation was run.

V2.1 labeling progress update:

- Next two-topic batch labels appended:
  - Topic 12: `v2_efficient_transformers`
  - Topic 13: `v2_llm_evaluation`
  - Labels appended: `30`
  - Total labels: `195 / 240`
  - Labels remaining: `45`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `1`
- Soft warning:
  - Line `195`, `high_beginner_and_expert_fit`: `paper_id=708`, `Holistic Evaluation of Language Models`.
  - This is non-blocking; HELM can plausibly be useful both as a beginner-accessible evaluation benchmark overview and as an expert-relevant reference.
- Per-topic distribution:
  - `v2_efficient_transformers`
    - Roles: `application=3`, `background=2`, `core_methods=6`, `foundational=2`, `negative=2`
    - Topic match buckets: `high=7`, `medium=6`, `low=2`
    - Reading value buckets: `high=4`, `medium=6`, `low=5`
  - `v2_llm_evaluation`
    - Roles: `application=3`, `background=1`, `evaluation_benchmark=9`, `negative=2`
    - Topic match buckets: `high=6`, `medium=5`, `low=4`
    - Reading value buckets: `high=5`, `medium=4`, `low=6`
- No evaluation was run.

V2.1 validation warning policy update:

- Updated `scripts/validate_v2_1_labels.py` to suppress `high_topic_match_low_reading_value` warnings when a row is explicitly marked as a duplicate:
  - `duplicate_status in {"near_duplicate", "exact_duplicate"}` or `primary_role="duplicate"`.
- Reason:
  - Duplicate rows can be topically excellent while having low reading value because they are redundant with another selected paper.
- Validation after update:
  - Labels: `165`
  - Valid: `True`
  - Warnings: `0`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
- No evaluation was run.

V2.1 labeling progress update:

- Next two-topic batch labels appended:
  - Topic 10: `v2_multimodal_learning`
  - Topic 11: `v2_graph_recommendation`
  - Labels appended: `30`
  - Total labels: `165 / 240`
  - Labels remaining: `75`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `1`
- Soft warning:
  - Existing line `79`, `high_topic_match_low_reading_value`: `paper_id=26` for `v2_large_language_model_agents`.
  - Still explainable because it is marked `duplicate_status=exact_duplicate` and `duplicate_of_paper_id=5`.
  - No new warning was introduced by this batch.
- Per-topic distribution:
  - `v2_multimodal_learning`
    - Roles: `application=2`, `background=4`, `core_methods=6`, `evaluation_benchmark=1`, `negative=2`
    - Topic match buckets: `high=11`, `medium=2`, `low=2`
    - Reading value buckets: `high=7`, `medium=6`, `low=2`
  - `v2_graph_recommendation`
    - Roles: `application=1`, `background=5`, `core_methods=4`, `evaluation_benchmark=2`, `foundational=1`, `negative=2`
    - Topic match buckets: `high=7`, `medium=6`, `low=2`
    - Reading value buckets: `high=5`, `medium=5`, `low=5`
- No evaluation was run.

V2.1 labeling progress update:

- Topics 8 and 9 labels appended:
  - Topic 8: `v2_diffusion_image_generation`
  - Topic 9: `v2_ai_for_scientific_discovery`
  - Labels appended: `30`
  - Total labels: `135 / 240`
  - Labels remaining: `105`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `1`
- Soft warning:
  - Existing line `79`, `high_topic_match_low_reading_value`: `paper_id=26` for `v2_large_language_model_agents`.
  - Still explainable because it is marked `duplicate_status=exact_duplicate` and `duplicate_of_paper_id=5`.
  - No new warning was introduced by topics 8 and 9.
- Per-topic distribution:
  - `v2_diffusion_image_generation`
    - Roles: `application=7`, `background=1`, `foundational=1`, `negative=5`, `recent_frontier=1`
    - Topic match buckets: `high=3`, `medium=4`, `low=8`
    - Reading value buckets: `high=2`, `medium=2`, `low=11`
  - `v2_ai_for_scientific_discovery`
    - Roles: `application=7`, `background=2`, `core_methods=2`, `evaluation_benchmark=1`, `negative=3`
    - Topic match buckets: `high=7`, `medium=5`, `low=3`
    - Reading value buckets: `high=6`, `medium=3`, `low=6`
- No evaluation was run.

V2.1 labeling progress update:

- Topics 6 and 7 labels appended:
  - Topic 6: `v2_large_language_model_agents`
  - Topic 7: `v2_recommendation_systems`
  - Labels appended: `30`
  - Total labels: `105 / 240`
  - Labels remaining: `135`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `1`
- Soft warning:
  - Line `79`, `high_topic_match_low_reading_value`: `paper_id=26` for `v2_large_language_model_agents`.
  - This is explainable because it is marked `duplicate_status=exact_duplicate` and `duplicate_of_paper_id=5`, so topical match is high but reading value is low.
- Per-topic distribution:
  - `v2_large_language_model_agents`
    - Roles: `application=3`, `background=5`, `core_methods=1`, `duplicate=1`, `foundational=2`, `negative=3`
    - Topic match buckets: `high=6`, `medium=3`, `low=6`
    - Reading value buckets: `high=3`, `medium=3`, `low=9`
  - `v2_recommendation_systems`
    - Roles: `application=6`, `background=3`, `core_methods=3`, `evaluation_benchmark=1`, `negative=2`
    - Topic match buckets: `high=8`, `medium=5`, `low=2`
    - Reading value buckets: `high=3`, `medium=8`, `low=4`
- No evaluation was run.

V2.1 labeling progress update:

- Topics 4 and 5 labels appended:
  - Topic 4: `v2_contrastive_learning`
  - Topic 5: `v2_bayesian_optimization`
  - Labels appended: `30`
  - Total labels: `75 / 240`
  - Labels remaining: `165`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `0`
- Per-topic distribution:
  - `v2_contrastive_learning`
    - Roles: `application=5`, `background=2`, `core_methods=3`, `foundational=2`, `negative=2`, `recent_frontier=1`
    - Topic match buckets: `high=10`, `medium=2`, `low=3`
    - Reading value buckets: `high=4`, `medium=8`, `low=3`
  - `v2_bayesian_optimization`
    - Roles: `application=2`, `background=2`, `core_methods=8`, `negative=2`, `recent_frontier=1`
    - Topic match buckets: `high=12`, `medium=1`, `low=2`
    - Reading value buckets: `high=5`, `medium=7`, `low=3`
- No evaluation was run.

V2.1 labeling progress update:

- Topics 2 and 3 labels appended:
  - Topic 2: `v2_retrieval_augmented_generation`
  - Topic 3: `v2_graph_neural_networks`
  - Labels appended: `30`
  - Total labels: `45 / 240`
  - Labels remaining: `195`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `0`
- Per-topic distribution:
  - `v2_retrieval_augmented_generation`
    - Roles: `application=4`, `core_methods=4`, `evaluation_benchmark=4`, `negative=2`, `recent_frontier=1`
    - Topic match buckets: `high=7`, `medium=3`, `low=5`
    - Reading value buckets: `high=3`, `medium=7`, `low=5`
  - `v2_graph_neural_networks`
    - Roles: `application=1`, `background=3`, `core_methods=7`, `foundational=1`, `negative=2`, `recent_frontier=1`
    - Topic match buckets: `high=12`, `medium=1`, `low=2`
    - Reading value buckets: `high=4`, `medium=9`, `low=2`
- No evaluation was run.
- The selected 240 candidates were not modified.

Immediate next step:

Manually label `data/eval/v2_1_labeling_packets.jsonl` into a new file such as `data/eval/manual_labels_v2_1.jsonl`. Use anchors only for calibration while assigning real V2.1 continuous scores. After labeling, validate the completed label file, then evaluate BM25, TF-IDF, embedding, FAISS, and hybrid on the V2.1 benchmark before tuning or training any new reranker.

V2.1 labeling progress:

- Topic 1 labels appended:
  - Topic: `v2_transformer_architecture`
  - Label file: `data/eval/manual_labels_v2_1.jsonl`
  - Labels appended: `15`
  - Total labels: `15 / 240`
  - Labels remaining: `225`
- Validation report:
  - JSON: `data/eval/results/v2_1_label_validation.json`
  - Valid: `True`
  - Missing required fields: `0`
  - Invalid values: `0`
  - Duplicate query-paper rows: `0`
  - Labels not in packet: `0`
  - Warnings: `0`
- Follow-up correction:
  - Updated `paper_id=1`, `query_id=v2_transformer_architecture` from `duplicate_status=near_duplicate` to `duplicate_status=none`.
  - Kept `duplicate_of_paper_id=null`.
  - This removed the prior `duplicate_missing_target` soft warning.
- Per-topic distribution for `v2_transformer_architecture`:
  - Roles: `application=3`, `background=2`, `core_methods=5`, `foundational=2`, `negative=2`, `recent_frontier=1`
  - Topic match buckets: `high=8`, `medium=5`, `low=2`
  - Reading value buckets: `high=3`, `medium=9`, `low=3`
- No evaluation was run.


V2 50k caveats:

- OpenAlex enrichment for the 50k pass was DOI-only for reliability and speed. It matched `4,405 / 4,471` DOI-bearing arXiv records. Non-DOI arXiv papers were not title-enriched in this pass.
- The ingestion warning text currently labels some identifier-key duplicate merges imprecisely in the console. The underlying merge behavior is correct, but manifest wording can be improved later.
- Cold retrieval latency is too high for a polished demo unless retrievers are explicitly warmed at startup.

Recommended next step:

Build and review the V2 benchmark candidate pool, then manually label about `240` final labels across the 16 topics. After labels exist, evaluate BM25/TF-IDF/dense/FAISS/hybrid on V2 and only then tune hybrid or train rerankers.

Project framing update:

ResearchPath should not be framed as "I trained a model." The stronger framing is:

> I built a real scientific-paper recommendation system, expanded the corpus, designed a fairer benchmark, compared multiple retrieval/ranking methods, and used evaluation to guide ranking improvements.

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

## V2.1 Learned-Ranker Milestone Prep

Generated the next-step analysis artifacts for a lightweight learned reranker without changing labels, selected candidates, or training any model.

Command run:

```powershell
.\backend\.venv\Scripts\python.exe scripts\analyze_v2_1_learned_ranker_milestone.py --eval data\eval\results\v2_1_benchmark_method_comparison.json --packet data\eval\v2_1_labeling_packets.jsonl --labels data\eval\manual_labels_v2_1.jsonl --json-out data\eval\results\v2_1_learned_ranker_milestone.json --md-out data\eval\results\v2_1_learned_ranker_milestone.md
```

Outputs:

- `data/eval/results/v2_1_learned_ranker_milestone.md`
- `data/eval/results/v2_1_learned_ranker_milestone.json`
- `scripts/analyze_v2_1_learned_ranker_milestone.py`

Main findings:

- Hybrid wins the overall reading-value benchmark, but BM25 remains strongest on overall topic-match NDCG.
- Hybrid's largest reading-value losses against BM25 are `v2_graph_recommendation`, `v2_recommendation_systems`, `v2_efficient_transformers`, and `v2_ai_for_scientific_discovery`.
- Hybrid's weakest absolute reading-value topics are `v2_efficient_transformers`, `v2_large_language_model_agents`, and `v2_graph_recommendation`.
- Intent winners split by role: hybrid wins background, foundational, recent-frontier, and application; BM25 wins core-methods and evaluation-benchmark.
- Proposed learned target: `v2_1_beginner_path_gain`, a weighted gain over reading value, topic match, beginner/intermediate fit, path/application intent scores, and a small duplicate penalty.
- Recommended validation: grouped cross-validation by `query_id`, preferably leave-one-topic-out or 4-fold grouped CV over the 16 topics.
- Selected 240 SHA256 after this milestone: `6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F`.

## V2.2 Lightweight learned_hybrid Reranker

Implemented and evaluated a leakage-safe lightweight `learned_hybrid` reranker using V2.1 labels and packet retrieval outputs. This trained only a Ridge regression model with `StandardScaler`; no neural embedding, bi-encoder, or cross-encoder model was trained.

Command run:

```powershell
.\backend\.venv\Scripts\python.exe scripts\train_v2_2_lightweight_learned_hybrid.py --labels data\eval\manual_labels_v2_1.jsonl --packet data\eval\v2_1_labeling_packets.jsonl --baseline data\eval\results\v2_1_benchmark_method_comparison.json --milestone data\eval\results\v2_1_learned_ranker_milestone.json --json-out data\eval\results\v2_2_learned_hybrid_cv_report.json --md-out data\eval\results\v2_2_learned_hybrid_cv_report.md --model-out data\processed\models\v2_2_lightweight_learned_hybrid.json --alpha 5.0 --folds 4 --k 10
```

Outputs:

- `scripts/train_v2_2_lightweight_learned_hybrid.py`
- `data/eval/results/v2_2_learned_hybrid_cv_report.json`
- `data/eval/results/v2_2_learned_hybrid_cv_report.md`
- `data/processed/models/v2_2_lightweight_learned_hybrid.json`

Model:

- Type: Ridge regression with `StandardScaler`
- Hyperparameters: `alpha=5.0`, `fit_intercept=True`, `random_state=17`
- Features: 78 runtime-reproducible rank, score, text-match, metadata, source, evidence, and title-dedup heuristic features
- Explicitly excluded leakage fields: label scores, audience scores, intent scores, roles, duplicate label status, label confidence, notes, `selection_reasons`, and `likely_coverage`

Grouped 4-fold CV results:

- `learned_hybrid`: topic NDCG@10 `0.931`, reading NDCG@10 `0.895`, beginner `0.868`, intermediate `0.902`, advanced `0.896`, expert `0.871`, role coverage `1.000`, path coverage `0.891`, duplicate penalty `0.006`, judged@10 `10.00`
- Current `hybrid`: topic NDCG@10 `0.866`, reading NDCG@10 `0.832`
- `bm25`: topic NDCG@10 `0.876`, reading NDCG@10 `0.821`

Success criteria:

- Beat current hybrid on reading-value NDCG@10: `True` (`+0.063`)
- Avoid losing more than `0.01` to BM25 on topic-match NDCG@10: `True` (`+0.055`)
- Overall pass: `True`

Failure cases:

- Worse than current hybrid on reading-value NDCG@10: `v2_retrieval_augmented_generation` (`-0.025`), `v2_diffusion_image_generation` (`-0.010`), `v2_graph_neural_networks` (`-0.005`)
- Worse than BM25 on topic-match NDCG@10: `v2_self_supervised_vision` (`-0.028`), `v2_bayesian_optimization` (`-0.020`), `v2_recommendation_systems` (`-0.014`), `v2_graph_neural_networks` (`-0.002`), `v2_retrieval_augmented_generation` (`-0.000`)

Selected 240 SHA256 after this milestone: `6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F`.

## V2.2b Fair learned_hybrid Evaluation

Corrected the V2.2 evaluation after audit. The original V2.2 result is preserved and explicitly classified as `exploratory_upper_bound_not_apples_to_apples` because it ranked all 15 packet candidates for learned_hybrid while baselines used pre-materialized method ranks with lower judged@10.

Command run:

```powershell
.\backend\.venv\Scripts\python.exe scripts\evaluate_v2_2b_fair_learned_hybrid.py --labels data\eval\manual_labels_v2_1.jsonl --packet data\eval\v2_1_labeling_packets.jsonl --old-v22 data\eval\results\v2_2_learned_hybrid_cv_report.json --json-out data\eval\results\v2_2b_fair_learned_hybrid_report.json --md-out data\eval\results\v2_2b_fair_learned_hybrid_report.md --model-out data\processed\models\v2_2b_lightweight_learned_hybrid.json --alpha 5.0 --folds 4 --k 10
```

Files created/modified:

- `backend/app/services/v2_2_learned_ranker.py`
- `backend/app/services/learned_ranker.py`
- `scripts/evaluate_v2_2b_fair_learned_hybrid.py`
- `data/eval/results/v2_2b_fair_learned_hybrid_report.json`
- `data/eval/results/v2_2b_fair_learned_hybrid_report.md`
- `data/processed/models/v2_2b_lightweight_learned_hybrid.json`

Same-packet reranking results, where every method ranks the same 15 labeled packet candidates:

- `learned_hybrid`: topic NDCG@10 `0.931`, reading NDCG@10 `0.895`, beginner `0.868`, intermediate `0.902`, advanced `0.896`, expert `0.871`, path coverage `0.891`, judged@10 `10.00`
- `hybrid`: topic NDCG@10 `0.907`, reading NDCG@10 `0.867`
- `bm25`: topic NDCG@10 `0.904`, reading NDCG@10 `0.849`
- Same-packet deltas: learned vs hybrid reading `+0.028`; learned vs BM25 topic `+0.026`

Restricted current-hybrid candidate results, where every method ranks only candidates with a materialized current-hybrid rank:

- `learned_hybrid`: topic NDCG@10 `0.875`, reading NDCG@10 `0.838`, beginner `0.814`, intermediate `0.845`, advanced `0.846`, expert `0.822`, path coverage `0.859`, judged@10 `9.12`
- `hybrid`: topic NDCG@10 `0.866`, reading NDCG@10 `0.832`
- `bm25`: topic NDCG@10 `0.861`, reading NDCG@10 `0.814`
- Restricted deltas: learned vs hybrid reading `+0.005`; learned vs BM25 topic `+0.014`

Reproducibility and leakage checks:

- Grouped CV by `query_id`: `True`
- Label fields used as features: `[]`
- `selection_reasons` used: `False`
- `likely_coverage` used: `False`
- Model artifact schema: `v2.2b_ridge_packet_feature_model`
- Feature/scaler/weight lengths: `78`
- Saved artifact regenerates all-240 predictions: `True`, max absolute difference `1.11e-16`
- Labels SHA256: `0ECC4A0F638E77E76D44913281EC6431ADDF1873474C3B80A48531B9529359F7`
- Selected 240 SHA256: `6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F`

Conclusion:

- Corrected V2.2b is safe to present as a validated lightweight learned-ranker improvement, with the caveat that the production backend still needs a full packet-row/candidate-context integration path before replacing the current runtime `learned_hybrid` behavior.
- The improvement is modest under the stricter restricted-candidate setup, so present both same-packet and restricted-candidate results.

## V2.3 Production-Style Candidate-Pool Evaluation

Implemented a V2.3 shared candidate-pool evaluation for the V2.2b learned ranker. This is an explicit fallback experiment, not true full 50K production retrieval, because the local Postgres database was not reachable and the available embedding/FAISS artifacts do not provide a standalone paper-id-to-metadata mapping.

Command run:

```powershell
.\backend\.venv\Scripts\python.exe scripts\evaluate_v2_3_production_candidate_pool.py --labels data\eval\manual_labels_v2_1.jsonl --packet data\eval\v2_1_labeling_packets.jsonl --v22b-report data\eval\results\v2_2b_fair_learned_hybrid_report.json --model data\processed\models\v2_2b_lightweight_learned_hybrid.json --raw-corpus data\raw\arxiv_ml_ai_50k_incremental.jsonl --json-out data\eval\results\v2_3_production_candidate_pool_report.json --md-out data\eval\results\v2_3_production_candidate_pool_report.md --top-k 50 --eval-k 10
```

Files created:

- `scripts/evaluate_v2_3_production_candidate_pool.py`
- `data/eval/results/v2_3_production_candidate_pool_report.json`
- `data/eval/results/v2_3_production_candidate_pool_report.md`

Candidate pool actually used:

- Local raw corpus: `data/raw/arxiv_ml_ai_50k_incremental.jsonl`
- Injected labeled V2.1 packet papers with their true label `paper_id`s
- Corpus size after packet injection: `35,224`
- Shared pool per topic: union of top-50 BM25, top-50 TF-IDF, and top-50 current hybrid
- Dense embedding and FAISS were marked unavailable in this run because metadata mapping could not be recovered without the DB
- Average candidate pool size: `81.88`
- Average judged packet candidates in pool: `12.06 / 15`
- Average packet coverage: `0.804`

V2.3 method averages:

- `hybrid`: topic NDCG@10 `0.661`, reading NDCG@10 `0.648`, beginner `0.711`, intermediate `0.661`, advanced `0.627`, expert `0.594`, judged@10 `6.50`, unjudged@10 `3.50`
- `learned_hybrid`: topic NDCG@10 `0.579`, reading NDCG@10 `0.582`, beginner `0.578`, intermediate `0.580`, advanced `0.565`, expert `0.546`, judged@10 `5.12`, unjudged@10 `4.88`
- `bm25`: topic NDCG@10 `0.533`, reading NDCG@10 `0.513`, judged@10 `5.12`
- `tfidf`: topic NDCG@10 `0.540`, reading NDCG@10 `0.510`, judged@10 `5.56`

Interpretation:

- learned_hybrid vs hybrid reading-value NDCG@10: `-0.066`
- learned_hybrid vs BM25 topic-match NDCG@10: `+0.047`
- learned_hybrid does not beat current hybrid on reading-value in the V2.3 fallback candidate-pool setting.
- The result is not safe to present as production-style learned reranking improvement. Present it as a production-style fallback stress test showing that candidate-pool coverage and unjudged top-10 items are the next bottleneck.
- Runtime backend integration remains scaffold-only; full candidate-context feature construction is not wired into live retrieval.

V2.3 reproducibility checks:

- No V2.3 training/CV was run; grouped CV is not applicable.
- Label fields used as features: `[]`
- `selection_reasons` used: `False`
- `likely_coverage` used: `False`
- Saved V2.2b model packet prediction max diff: `0.0`
- Labels SHA256: `0ECC4A0F638E77E76D44913281EC6431ADDF1873474C3B80A48531B9529359F7`
- Selected 240 SHA256: `6C85E0C340A2547DFA79482ED52213E7BF759774D45F70095978E14178DA237F`

## V2.4 True Production Candidate-Pool Evaluation

V2.4 corrected the V2.3 fallback limitation by evaluating against the actual backend PostgreSQL paper corpus and 50k MiniLM/FAISS production assets.

Core facts:

- DB reachable: `true`
- Backend papers table size: `50,424`
- FAISS id-to-paper metadata mapping works.
- Dense exact matrix retrieval and FAISS retrieval are both included.
- Candidate pool: union of top-50 BM25, top-50 TF-IDF, top-50 exact embedding, top-50 FAISS embedding, and top-50 current hybrid, deduplicated by `paper_id`.
- Report: `data/eval/results/v2_4_true_production_candidate_pool_report.json`
- Markdown: `data/eval/results/v2_4_true_production_candidate_pool_report.md`

Important result:

- Old learned_hybrid did not beat production baselines in the true production pool.
- learned_hybrid reading NDCG@10: about `0.623`
- hybrid reading NDCG@10: about `0.627`
- learned_hybrid vs hybrid reading delta: about `-0.004`
- learned_hybrid vs BM25 topic delta: about `-0.026`
- Oracle reading NDCG@10: about `0.984`

Interpretation:

- Candidate pool quality is high; the pool contains strong papers.
- Main bottleneck is ranking robustness: learned_hybrid promoted too many unjudged or weak production candidates.
- This motivated V2.5 hard-negative labeling.

## V2.5 Hard-Negative Labeling Packet

V2.5 exported a small high-value hard-negative/hidden-positive labeling packet from V2.4 production-pool failures.

Files:

- Export script: `scripts/export_v2_5_hard_negative_labeling_packet.py`
- Candidate packet: `data/eval/v2_5_hard_negative_labeling_packet.jsonl`
- Summary JSON: `data/eval/results/v2_5_hard_negative_packet_summary.json`
- Summary Markdown: `data/eval/results/v2_5_hard_negative_packet_summary.md`
- Batch files:
  - `data/eval/v2_5_labeling_batches/batch_01.jsonl`
  - `data/eval/v2_5_labeling_batches/batch_02.jsonl`
  - `data/eval/v2_5_labeling_batches/batch_03.jsonl`
  - `data/eval/v2_5_labeling_batches/batch_04.jsonl`
- Label target: `data/eval/manual_labels_v2_5_hard_negatives.jsonl`
- Validator: `scripts/validate_manual_labels_v2_5.py`
- Labeling guide: `data/eval/v2_5_hard_negative_labeling_guide.md`

V2.5 packet:

- Total candidates: `96`
- Topics: `16`
- About `6` candidates per topic.
- Selection buckets:
  - `learned_promoted_unjudged`
  - `dense_promoted_unjudged`
  - `ranker_disagreement`

Validation intent:

- No duplicate `(query_id, paper_id)` rows.
- No accidental duplicate with V2.1 manual labels.
- Same V2.1 label schema.
- No labels auto-created by export.
- V2.1 labels and selected 240 protected.

Completed label file:

- `data/eval/manual_labels_v2_5_hard_negatives.jsonl`
- Label count: `96`
- SHA256: `F3CEFD7ED5C89D79796AD487C255879A8246DE87692289C9A3A6C67157F7453C`

## V2.6 Production-Aware learned_hybrid Retraining

V2.6 trained a production-aware lightweight Ridge learned_hybrid using V2.1 plus V2.5 labels. This was a non-neural model and did not create labels.

Inputs:

- V2.1 labels: `240`
- V2.5 hard-negative labels: `96`
- Combined training rows: `336`
- Old model: `data/processed/models/v2_2b_lightweight_learned_hybrid.json`

Files:

- Script: `scripts/train_v2_6_production_aware_learned_hybrid.py`
- Model: `data/processed/models/v2_6_production_aware_learned_hybrid.json`
- Report JSON: `data/eval/results/v2_6_production_aware_learned_hybrid_report.json`
- Report Markdown: `data/eval/results/v2_6_production_aware_learned_hybrid_report.md`
- Audit summary: `data/eval/results/v2_6_audit_summary.md`

Key expanded production-pool results:

- V2.6 reading NDCG@10: `0.728`
- Current hybrid reading NDCG@10: `0.635`
- Delta vs current hybrid: `+0.093`
- V2.6 topic NDCG@10: `0.743`
- BM25 topic NDCG@10: `0.670`
- Delta vs BM25 topic: `+0.073`
- V2.5 negative rows old mean predicted score: `0.595`
- V2.5 negative rows V2.6 mean predicted score: `0.318`

Important caveat:

- Old V2.2b still beat V2.6 overall in expanded production-pool evaluation.
- Old V2.2b reading NDCG@10: `0.788`
- V2.6 reading NDCG@10: `0.728`
- Delta: `-0.061`

Interpretation:

- V2.6 successfully demoted many known hard negatives.
- It also lowered scores for many hidden positives.
- It is useful as a hard-negative/calibration signal, not as the replacement production ranker.
- Safe wording: V2.6 is a current-hybrid improvement and hard-negative-fix experiment, not the best learned-ranker upgrade.

Metadata note:

- The V2.6 artifact metadata was audited and polished to state that it is `v2_6_production_aware`, trained on V2.1 + V2.5 labels, using mixed feature contexts:
  - V2.1 packet universe
  - V2.5 / production candidate-pool universe
- Runtime integration remains scaffold-only for V2.6.

## V2.7 Score-Level Blending and Calibration

V2.7 evaluated lightweight score-level blends intended to preserve old V2.2b ranking strength while using V2.6 as a hard-negative signal.

Files:

- Script: `scripts/evaluate_v2_7_score_blends.py`
- Model/artifact: `data/processed/models/v2_7_score_blend.json`
- Report JSON: `data/eval/results/v2_7_score_blend_report.json`
- Report Markdown: `data/eval/results/v2_7_score_blend_report.md`
- Audit summary: `data/eval/results/v2_7_audit_summary.md`

Selected method:

- `blend_0_6_old_0_2_v26_0_1_hybrid_0_1_embedding`

Formula:

```text
V2.7 score = 0.60 * old_v2_2b_score
           + 0.20 * v2_6_score
           + 0.10 * hybrid_score
           + 0.10 * embedding_score
```

Selected method results:

- V2.1 packet:
  - reading NDCG@10: `0.930`
  - topic NDCG@10: `0.951`
  - judged@10: `10.00`
  - unjudged@10: `0.00`
- V2.5 hard-negative:
  - reading NDCG@10: `0.955`
  - topic NDCG@10: `0.956`
  - judged@10: `6.00`
  - unjudged@10: `0.00`
- Expanded production pool:
  - reading NDCG@10: `0.837`
  - topic NDCG@10: `0.853`
  - judged@10: `9.12`
  - unjudged@10: `0.88`

Compared with old V2.2b on expanded production:

- Reading delta: `+0.048`
- Topic delta: `+0.057`
- Unjudged@10 delta: `-0.50`
- Fixed V2.5 negative promotions vs old: `9`
- Hidden positives harmed vs old: `16`

Interpretation:

- V2.7 is the strongest current offline production-style learned reranking candidate under the current evaluation setup.
- It blends old V2.2b, V2.6, current hybrid, and dense embedding scores.
- It is not a neural model.
- It should not be overclaimed as live/default behavior until runtime parity is verified.

Recommended wording:

> V2.7 is the strongest current offline production-style learned reranking candidate. It blends the original learned ranker, the production-aware V2.6 ranker, hybrid retrieval, and dense embedding scores. On the expanded production-pool evaluation, it improved reading NDCG@10 from 0.788 to 0.837 over the previous learned ranker, improved topic NDCG@10 from 0.796 to 0.853, and reduced unjudged@10 from 1.38 to 0.88.

## V2.8 Guarded Runtime Integration

V2.8 added the V2.7 selected blend as an opt-in backend runtime method only. It did not change frontend defaults or backend defaults.

Method name:

- `learned_blend_v2_7`

Files:

- Runtime retriever: `backend/app/services/retrievers/learned_blend_v2_7.py`
- Routing/config updates:
  - `backend/app/services/recommendation_service.py`
  - `backend/app/core/config.py`
- Tests:
  - `backend/tests/test_v2_7_learned_blend.py`
  - `backend/tests/test_recommendation_methods.py`

Runtime behavior:

- Opt-in only.
- Existing default recommendation behavior unchanged.
- Frontend behavior unchanged.
- No labels modified.
- No selected 240 modifications.
- No retraining.
- No commits or pushes performed as part of the milestone.

Runtime candidate pool mirrors offline V2.7/V2.4 expanded production setup:

- top-50 BM25
- top-50 TF-IDF
- top-50 exact embedding
- top-50 FAISS embedding
- top-50 current hybrid
- deduplicated by `paper_id`

Runtime scoring details:

- Old V2.2b and V2.6 scores are computed over the full shared candidate pool.
- Hybrid and embedding scores are min-max normalized over the same shared pool.
- Missing scores default to zero.
- Deterministic tie-breaking: score descending, then `paper_id` ascending.
- V2.7 artifact validation fails loudly if the selected method or weights do not match the expected V2.7 fixed blend.

Targeted tests run:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests\test_v2_7_learned_blend.py tests\test_recommendation_methods.py tests\test_hybrid_ranking.py
```

Result:

- `11 passed`

Remaining V2.8 caveats:

- Runtime logic mirrors the offline evaluator rather than sharing one extracted helper, so future drift is possible.
- Current parity test is targeted/in-memory; add a DB/FAISS-backed parity smoke before making V2.7 default.
- Runtime rows may be less enriched than offline DB rows for identifier/source-derived features, which can cause small real-query score differences.
- V2.8 is ready for a guarded PR, but not ready to become the frontend/default recommendation path.

## V2.9 Supervised LTR Dataset Foundation

V2.9 created the training-ready supervised learning-to-rank table and score-column baseline evaluator for future V3.0 training. It did not train a model.

Files:

- Dataset builder: `scripts/build_v2_9_ltr_dataset.py`
- Score-column evaluator: `scripts/evaluate_v2_9_score_columns.py`
- Tests: `backend/tests/test_v2_9_ltr_dataset.py`
- Dataset: `data/eval/training/v2_9_ltr_dataset.jsonl`
- Topic splits: `data/eval/training/v2_9_splits.json`
- Baseline report JSON: `data/eval/results/v2_9_score_column_baselines.json`
- Baseline report Markdown: `data/eval/results/v2_9_score_column_baselines.md`

Dataset contents:

- Total rows: `336`
- V2.1 rows: `240`
- V2.5 hard-negative rows: `96`
- Duplicate `(query_id, paper_id)` rows: `0`
- Topics: `16`
- Judged-only training table.

Each row includes:

- Query/topic metadata.
- Paper metadata.
- Label fields.
- Difficulty/audience fields.
- Intent/role/duplicate labels.
- BM25/TF-IDF/embedding/FAISS/hybrid scores and ranks.
- Old V2.2b score.
- V2.6 score.
- V2.7 score.
- `hard_negative` flag.
- `hidden_positive` flag.
- `positive` flag.
- Judged source: `v2_1` or `v2_5`.

Split metadata:

- Train topics: `10`
- Dev topics: `3`
- Test topics: `3`
- Leave-topic-out folds: `16`
- No topic leakage.

V2.9 score-column baseline results:

- V2.7 score reproduced existing judged-scope V2.7 metrics within tolerance.
- V2.1 V2.7 score:
  - reading NDCG@10: `0.929638`
  - topic NDCG@10: `0.950513`
- V2.5 V2.7 score:
  - reading NDCG@10: `0.954722`
  - topic NDCG@10: `0.955553`
- Combined judged table V2.7 score:
  - reading NDCG@10: `0.887328`
  - topic NDCG@10: `0.877326`

Commands run:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests\test_v2_9_ltr_dataset.py

cd ..
.\backend\.venv\Scripts\python.exe scripts\build_v2_9_ltr_dataset.py
.\backend\.venv\Scripts\python.exe scripts\evaluate_v2_9_score_columns.py
```

Results:

- V2.9 tests: `3 passed`
- Dataset build: `336` rows
- V2.7 reproduction check: `true`
- Protected hashes unchanged.

V2.9 caveat:

- `unjudged@10` is zero in the V2.9 score-column report because the V2.9 training table is judged-only. Expanded production-pool unjudged metrics still require the production candidate-pool evaluator.

## V3.0 Starting Point

V3.0 should train the first supervised non-neural LTR model using the V2.9 dataset.

Recommended deliverables:

- `scripts/train_v3_0_ltr_ranker.py`
- `data/processed/models/v3_0_ltr_ranker.json`
- `data/eval/results/v3_0_ltr_training_report.json`
- `data/eval/results/v3_0_ltr_training_report.md`

Recommended model types:

- Ridge
- ElasticNet
- Optionally tree-based non-neural ranker/baseline if available
- No neural models yet

Recommended evaluation:

- Leave-one-topic-out CV.
- Train/dev/test by topic.
- No query/topic leakage.
- Compare against:
  - `v2_7_score`
  - `old_v2_2b_score`
  - `v2_6_score`
  - `hybrid_score`
  - `embedding_score`
  - `faiss_embedding_score`
- Report:
  - reading NDCG@10
  - topic NDCG@10
  - MRR@10
  - positive Recall@10
  - hard-negative promotion rate
  - per-topic metrics
  - hidden-positive harm diagnostics

Success bar:

- V3.0 should beat or match V2.7 under grouped/topic-held-out evaluation.
- If it only wins full-data apparent metrics but loses grouped CV, do not present it as an upgrade.

V3.0 constraints:

- Do not modify labels.
- Do not modify selected 240.
- Do not train neural models.
- Do not change runtime defaults.
- Do not change frontend defaults.
- Do not commit or push without explicit permission.

## Notes for Future Updates

Every time a milestone changes, update this file with:

- What changed
- Commands that were run
- Metrics before and after
- Known bugs
- Next experiment
- Resume bullet draft if the milestone is significant

Keep the project honest: save numbers, note failures, and compare against baselines.
