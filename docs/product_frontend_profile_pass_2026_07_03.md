# Product Frontend and Profile Personalization Pass

Date: 2026-07-03

This pass moved ResearchPath from an internal ML demo toward a more portfolio-ready product interface. It did not train models, add labels, expand the corpus, change backend defaults, or overwrite model artifacts.

## Goals

- Make the app feel more like a real research product instead of a model-evaluation demo.
- Make `ResearchPath` the visible brand.
- Let users edit profile preferences that can affect recommendations.
- Hide backend version labels from normal users.
- Let users open recommended papers, PDFs, and DOI pages directly.

## Frontend Changes

- Reworked the layout into:
  - top brand/header area,
  - left workflow panel,
  - right results workspace.
- Enlarged the `ResearchPath` title and made it the first-viewport brand signal.
- Switched typography:
  - UI: `Plus Jakarta Sans`.
  - brand/display title: `Fraunces`.
- Added an editable Reader Profile panel:
  - background level,
  - current status,
  - research goal,
  - paper taste,
  - preferred topics,
  - avoided topics.
- Moved the saved library into the left workflow panel.
- Simplified per-paper feedback:
  - Save,
  - More like this,
  - Not useful,
  - Difficulty dropdown with Too easy, Too hard, Already read.
- Added paper action links on result cards:
  - Open paper,
  - PDF,
  - DOI.

## User-Facing Method Labels

Backend method IDs are unchanged. The frontend now displays readable labels:

| Backend method | Frontend label |
|---|---|
| `bm25` | Keyword match |
| `tfidf` | Term-weighted match |
| `citation_recency` | Influential and recent |
| `embedding` | Semantic similarity |
| `faiss_embedding` | Fast semantic search |
| `hybrid` | Personalized blend |
| `learned_hybrid` | Personalized learned blend |
| `v3_3_ltr` | Classic learned ranker |
| `v4_1_blend` | Guardrailed learned blend |
| `v4_9_guarded_text_blend` | Text-aware guarded ranker |
| `v6_4_safe_fusion` | Best offline fusion |

The method selector also includes short help text explaining each ranking strategy.

## Backend Changes

- Extended recommendation responses with:
  - `paper_url`,
  - `pdf_url`,
  - `doi_url`,
  - `source_url`,
  - `doi`.
- Added link derivation in `backend/app/services/formatting.py`:
  - DOI values become DOI links.
  - arXiv URLs, arXiv IDs, and arXiv DOI values produce PDF links.
  - Direct PDF source URLs are reused.
- Extended the user profile model/API with:
  - `avoid_topics`,
  - `current_status`,
  - `research_goal`,
  - `paper_taste`.
- Added dev-time runtime schema patching for the new profile columns.
- Updated hybrid personalization to use:
  - preferred topics as boosts,
  - avoided topics as penalties,
  - paper taste and research goal as small ranking signals.

## Current Behavior

- Profile settings are persisted through `/profile`.
- Profile personalization most directly affects `hybrid` and `learned_hybrid`.
- Opt-in learned/evaluation methods still mostly demonstrate benchmark behavior.
- `v6_4_safe_fusion` remains opt-in and can be slow for cold interactive requests.
- Backend default ranking remains `bm25`.
- Frontend labels changed, but backend method values did not.

## Multi-User Status

Multi-user support is not implemented yet.

The database has a `user_key` field, but the profile service still uses the single local key:

```text
default
```

To support multiple users later, the app needs a `user_key` or auth-aware path through:

- profile endpoints,
- feedback endpoints,
- library endpoints,
- recommendation/personalization calls.

## Verification

Commands run during this pass:

```powershell
npm.cmd run build
.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_feedback_profile_api.py backend\tests\test_product_recommendations.py
.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_formatting.py backend\tests\test_reading_path_api.py
```

Backend smoke after restart:

```powershell
curl.exe -sS http://localhost:8000/health
curl.exe -sS "http://localhost:8000/recommend/query?query=transformer&k=1&method=bm25"
```

The sample recommendation returned paper access fields, including DOI/source links.

## Runtime Method Comparison

A local smoke comparison was run across representative queries:

- `transformer`
- `AI agents for scientific discovery`
- `graph neural networks`

Compared modes:

- Search: flat top-k recommendation list.
- Reading Path: sectioned recommendations for Background, Foundational, Core Methods, and Recent Frontier.

Compared methods:

- Keyword match (`bm25`)
- Personalized blend (`hybrid`)
- Semantic similarity (`embedding`)
- Text-aware guarded ranker (`v4_9_guarded_text_blend`)
- Best offline fusion (`v6_4_safe_fusion`) as a single-query smoke

Observed behavior:

- Search and Reading Path can return different papers because Reading Path re-ranks candidates for section fit.
- Keyword match is literal and usually fast.
- Semantic similarity is fast after warmup and useful for conceptual queries.
- Personalized blend is the main product-facing profile-aware method.
- Text-aware guarded ranker often returns more canonical/evaluation-aware papers but is slower.
- Best offline fusion is useful for portfolio/evaluation evidence but is not a good default interactive method.

Example observations:

- `transformer` with Keyword match surfaced `Multimodal Learning With Transformers: A Survey`.
- `transformer` with Text-aware guarded ranker surfaced `Attention Is All You Need`.
- `AI agents for scientific discovery` with Semantic similarity produced stronger conceptual path candidates, including LLM-agent and autonomous-chemistry style papers.
- `graph neural networks` with Semantic similarity surfaced `A Comprehensive Survey on Graph Neural Networks` and `The Graph Neural Network Model` in the reading path.

## Personalization Evidence

Profile tailoring was tested with `method=hybrid`, because that is the method where profile and feedback signals are applied.

### Profile Preference Test

Query:

```text
neural networks
```

Baseline empty profile produced broad neural-network results such as:

- `A survey of uncertainty in deep neural networks`
- `Survey on categorical data for neural networks`
- `The Future of Neural Networks`

Then the profile was temporarily changed to:

- preferred topics: `graph neural networks`, `graph representation`
- avoided topics: `medical imaging`
- research goal: `literature_review`
- paper taste: `surveys_first`

After the profile change, scores/ranking changed and a graph-related paper appeared in the top results:

- `Graph neural networks in particle physics`

Reading Path diagnostics exposed reasons such as:

- `matches preferred topics`
- `profile prefers surveys first`
- `matches avoided topics`

### Feedback Test

Query:

```text
transformer
```

Baseline Personalized blend ranking placed this first:

- `Multimodal Learning With Transformers: A Survey`

Then real feedback events were posted:

- `more_like_this` on `Transformer-XL: Attentive Language Models beyond a Fixed-Length Context`
- `not_relevant` on `Multimodal Learning With Transformers: A Survey`
- `too_hard` on `Multimodal Learning With Transformers: A Survey`

After feedback, `Transformer-XL: Attentive Language Models beyond a Fixed-Length Context` moved to rank 1 for the same query/method.

Interpretation:

- Feedback and profile edits do tailor recommendations for `hybrid` / `learned_hybrid`.
- This is transparent heuristic personalization, not online reinforcement learning or model retraining.
- Other methods remain less profile-aware and are better understood as retrieval/evaluation methods.
- The user's profile was restored after the experiment.

## Local Dev Server Note

On this Windows setup, Vite may bind to IPv6 localhost by default. If the browser cannot connect to the page, restart the frontend with an explicit IPv4 host:

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

Then open:

```text
http://127.0.0.1:5173
```

## Next Product Work

- Add true multi-user profile selection or auth.
- Make profile effects more visible in explanations.
- Add loading skeletons and optimistic feedback updates.
- Add a polished empty state with suggested research goals.
- Consider hiding advanced rankers behind an "Advanced ranking" disclosure.
