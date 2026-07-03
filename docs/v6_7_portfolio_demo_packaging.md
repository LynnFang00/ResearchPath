# V6.7 Portfolio and Demo Packaging

V6.7 packages the completed ResearchPath ML pipeline for review and demo use. It does not train models, add labels, expand the corpus, change defaults, or overwrite model artifacts.

## One-Sentence Project Story

ResearchPath is a local-first research paper recommender that turns a topic into beginner-aware reading paths, then validates every ranker improvement against versioned judged labels before exposing it as an opt-in backend method.

## Current Runtime Position

- Backend default method: `bm25`.
- Frontend default behavior: unchanged.
- Corpus size: `50,424` papers.
- Judged labels: `2,400`.
- Opt-in learned methods: `v3_3_ltr`, `v4_1_blend`, `v4_9_guarded_text_blend`, `v6_4_safe_fusion`.
- Current accepted safe-fusion runtime method: `method=v6_4_safe_fusion`.

## Completed ML Progression

| Stage | Outcome |
|---|---|
| V3.3 | Random-forest LTR without V2.7 leakage features became the first safe offline production-pool upgrade. |
| V4.1 | Weighted retraining and calibrated blending improved robustness while staying opt-in. |
| V4.5-V4.6 | Text-assisted ranking was audited for regressions and leakage before candidate promotion. |
| V4.9-V5.0 | Guarded text blend passed validation and was integrated as `method=v4_9_guarded_text_blend`. |
| V6.0-V6.2 | Neural reranker datasets and neural baselines were built offline; neural scores were retained as diagnostic signals. |
| V6.3-V6.4 | Learned fusion combined V4.9 and neural signals, then constrained swaps to avoid topic regressions. |
| V6.5 | Safe fusion was integrated as explicit `method=v6_4_safe_fusion`. |
| V6.6 | Safe-fusion scorer serialization made the runtime path reproducible from stored coefficients and scaler parameters. |

## V6.6 Safe-Fusion Evidence

- Scorer config: `data/processed/models/v6_6_safe_fusion_ridge_scorer.json`.
- Candidate config: `data/processed/models/v6_6_safe_fusion_candidate.json`.
- Test Reading NDCG@10: `0.7443` vs V4.9 `0.7242`.
- Test Topic NDCG@10: `0.8373` vs V4.9 `0.8274`.
- Test Hard-neg@10: `0.1000`, not worse than V4.9.
- Severe regressions: `0`.
- Non-weak severe regressions: `0`.
- Runtime formula parity max delta: `0.0`.
- Offline/live candidate Jaccard in parity smoke: `1.0`.
- Forbidden runtime features required: `[]`.

## Demo Flow

1. Start the backend and frontend normally.
2. Run a query in search mode with the existing default UI state.
3. Use the retrieval-method selector to choose an explicit opt-in method for a single-method demo.
4. Enable the unchecked method-comparison toggle to compare BM25, V4.9 guarded text, and V6.6 safe fusion side by side.

The comparison panel calls the same recommendation endpoints as normal search. It does not make learned rankers default and does not create a separate runtime integration path.

## Caveats To Say Out Loud

- The backend default remains `bm25` because the strongest learned methods are still opt-in demo methods.
- V6.6 safe fusion uses only inference-safe component scores, but live full-strength scoring depends on those component scores being available.
- If neural component scores are unavailable, the safe-fusion runtime path preserves the V4.9 ordering rather than fabricating missing neural signals.
- The neural models are useful portfolio evidence and diagnostic signals, not standalone promoted rankers.

## What This Demonstrates

- Versioned label management and protected-hash discipline.
- Offline-to-runtime parity for learned rankers.
- Leakage audits before model promotion.
- Regression diagnosis at topic and rank levels.
- Conservative opt-in deployment of learned ranking methods.
- Frontend demo support that keeps defaults stable.
