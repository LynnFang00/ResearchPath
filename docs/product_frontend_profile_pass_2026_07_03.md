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

## Next Product Work

- Add true multi-user profile selection or auth.
- Make profile effects more visible in explanations.
- Add loading skeletons and optimistic feedback updates.
- Add a polished empty state with suggested research goals.
- Consider hiding advanced rankers behind an "Advanced ranking" disclosure.
