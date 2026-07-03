import React, { FormEvent, useEffect, useState } from 'react';
import { BarChart3, Database, Search, UserRound } from 'lucide-react';
import {
  compareRecommendationMethods,
  deleteLibraryItem,
  fetchRecommendationMethods,
  fetchProfile,
  fetchDatasetStatus,
  fetchLibrary,
  fetchReadingPath,
  patchProfile,
  postFeedback,
  saveLibraryItem,
  searchRecommendations,
  type DatasetStatus,
  type FeedbackAction,
  type LibraryResponse,
  type MethodComparisonResult,
  type PathPaper,
  type ReadingPath,
  type Recommendation,
  type UserProfile,
} from '../api/client';
import { PathSection } from '../components/PathSection';
import { ResultCard } from '../components/ResultCard';

const retrievalMethods = [
  { value: 'hybrid', label: 'Personalized blend' },
  { value: 'learned_hybrid', label: 'Personalized learned blend' },
  { value: 'bm25', label: 'Keyword match' },
  { value: 'tfidf', label: 'Term-weighted match' },
  { value: 'citation_recency', label: 'Influential and recent' },
  { value: 'embedding', label: 'Semantic similarity' },
  { value: 'faiss_embedding', label: 'Fast semantic search' },
  { value: 'v3_3_ltr', label: 'Classic learned ranker' },
  { value: 'v4_1_blend', label: 'Guardrailed learned blend' },
  { value: 'v4_9_guarded_text_blend', label: 'Text-aware guarded ranker' },
  { value: 'v6_4_safe_fusion', label: 'Best offline fusion' },
];

const comparisonMethods = ['bm25', 'v4_9_guarded_text_blend', 'v6_4_safe_fusion'];
const methodLabels = Object.fromEntries(retrievalMethods.map((retrievalMethod) => [retrievalMethod.value, retrievalMethod.label]));
const methodHelpText: Record<string, string> = {
  hybrid: 'Best for normal use: combines text match, semantic match, citations, recency, and your profile.',
  learned_hybrid: 'Personalized blend with a lightweight learned adjustment from judged examples.',
  bm25: 'Fast keyword search for exact topic terms and paper titles.',
  tfidf: 'Classic term-weighted search that emphasizes distinctive words.',
  citation_recency: 'Prioritizes papers that are cited, influential, and recent.',
  embedding: 'Finds semantically similar papers even when wording differs.',
  faiss_embedding: 'Fast vector search over semantic embeddings.',
  v3_3_ltr: 'Older learned ranker kept for comparison and evaluation history.',
  v4_1_blend: 'Learned ranker with guardrails against obvious hard negatives.',
  v4_9_guarded_text_blend: 'Text-aware learned ranker with stricter relevance guardrails.',
  v6_4_safe_fusion: 'Strongest offline fusion result; useful for demos, but slower interactively.',
};
const pathSectionOrder = ['background', 'foundational', 'core_methods', 'recent_frontier'];

type ProfileDraft = {
  background_level: string;
  current_status: string;
  research_goal: string;
  paper_taste: string;
  preferred_topics: string;
  avoid_topics: string;
};

const defaultProfileDraft: ProfileDraft = {
  background_level: 'basic_ml',
  current_status: 'exploring',
  research_goal: 'learn_topic',
  paper_taste: 'balanced',
  preferred_topics: '',
  avoid_topics: '',
};

const profileOptions = {
  currentStatus: [
    { value: 'exploring', label: 'Exploring' },
    { value: 'coursework', label: 'Coursework' },
    { value: 'thesis', label: 'Thesis / research' },
    { value: 'industry', label: 'Industry project' },
    { value: 'paper_reproduction', label: 'Paper reproduction' },
  ],
  researchGoal: [
    { value: 'learn_topic', label: 'Learn a topic' },
    { value: 'literature_review', label: 'Literature review' },
    { value: 'start_project', label: 'Start a project' },
    { value: 'find_baseline', label: 'Find baselines' },
    { value: 'implement_method', label: 'Implement a method' },
  ],
  paperTaste: [
    { value: 'balanced', label: 'Balanced' },
    { value: 'surveys_first', label: 'Surveys first' },
    { value: 'foundational_first', label: 'Foundational first' },
    { value: 'recent_first', label: 'Recent work first' },
    { value: 'implementation_first', label: 'Implementation focused' },
  ],
};

export function SearchPage() {
  const [query, setQuery] = useState('transformer');
  const [method, setMethod] = useState('embedding');
  const [mode, setMode] = useState<'search' | 'path'>('search');
  const [backgroundLevel, setBackgroundLevel] = useState('basic_ml');
  const [availableMethods, setAvailableMethods] = useState<string[]>(retrievalMethods.map((retrievalMethod) => retrievalMethod.value));
  const [compareMethods, setCompareMethods] = useState(false);
  const [results, setResults] = useState<Recommendation[]>([]);
  const [comparisonResults, setComparisonResults] = useState<MethodComparisonResult[]>([]);
  const [readingPath, setReadingPath] = useState<ReadingPath | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [profileDraft, setProfileDraft] = useState<ProfileDraft>(defaultProfileDraft);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [library, setLibrary] = useState<LibraryResponse | null>(null);
  const [activeLibraryTag, setActiveLibraryTag] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
  const [libraryTagInputs, setLibraryTagInputs] = useState<Record<number, string>>({});
  const [pendingLibraryPaperId, setPendingLibraryPaperId] = useState<number | null>(null);
  const savedPaperIds = new Set(library?.items.map((item) => item.paper_id) ?? []);
  const savedTagsByPaperId = new Map(library?.items.map((item) => [item.paper_id, item.tags] as const) ?? []);
  const resultCount = readingPath
    ? pathSectionOrder.reduce((count, section) => count + (readingPath.sections[section]?.length ?? 0), 0)
    : results.length;
  const outputTitle = mode === 'path' ? 'Reading path' : 'Search results';
  const outputDetail =
    resultCount > 0 ? `${resultCount} papers for "${query}"` : isLoading ? 'Searching the local corpus' : 'No results yet';

  useEffect(() => {
    fetchRecommendationMethods()
      .then((loadedMethods) => {
        if (loadedMethods.length > 0) setAvailableMethods(loadedMethods);
      })
      .catch(() => null);
    fetchDatasetStatus()
      .then(setDatasetStatus)
      .catch(() => setDatasetStatus(null));
    fetchProfile()
      .then((loadedProfile) => {
        setProfile(loadedProfile);
        setBackgroundLevel(loadedProfile.background_level);
        setProfileDraft(profileToDraft(loadedProfile));
      })
      .catch(() => setProfile(null));
    fetchLibrary()
      .then(setLibrary)
      .catch(() => setLibrary(null));
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResults([]);
    setComparisonResults([]);
    setReadingPath(null);

    try {
      if (mode === 'path') {
        const path = await fetchReadingPath(query.trim(), 4, method, backgroundLevel);
        setReadingPath(path);
      } else {
        const recommendations = await searchRecommendations(query.trim(), 10, method);
        setResults(recommendations);
        if (compareMethods) {
          const methods = comparisonMethods.filter((comparisonMethod) => availableMethods.includes(comparisonMethod));
          setComparisonResults(await compareRecommendationMethods(query.trim(), 5, methods));
        }
      }
    } catch (searchError) {
      setError(searchError instanceof Error ? searchError.message : 'Search failed.');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleBackgroundChange(value: string) {
    setBackgroundLevel(value);
    setProfileDraft((current) => ({ ...current, background_level: value }));
    try {
      const updated = await patchProfile({ background_level: value });
      setProfile(updated);
      setProfileDraft(profileToDraft(updated));
    } catch {
      // Keep UI responsive even if local profile persistence is temporarily unavailable.
    }
  }

  async function handleProfileSave() {
    setIsSavingProfile(true);
    setFeedbackMessage(null);
    try {
      const updated = await patchProfile({
        background_level: profileDraft.background_level,
        current_status: profileDraft.current_status,
        research_goal: profileDraft.research_goal,
        paper_taste: profileDraft.paper_taste,
        preferred_topics: parseTags(profileDraft.preferred_topics),
        avoid_topics: parseTags(profileDraft.avoid_topics),
      });
      setProfile(updated);
      setBackgroundLevel(updated.background_level);
      setProfileDraft(profileToDraft(updated));
      setFeedbackMessage('Profile updated');
    } catch (profileError) {
      setFeedbackMessage(profileError instanceof Error ? profileError.message : 'Profile update failed.');
    } finally {
      setIsSavingProfile(false);
    }
  }

  async function refreshLibrary(tag: string | null = activeLibraryTag) {
    const updated = await fetchLibrary(tag);
    setLibrary(updated);
  }

  async function handleLibraryTagChange(tag: string | null) {
    setActiveLibraryTag(tag);
    await refreshLibrary(tag);
  }

  async function handleFeedback(paper: Recommendation | PathPaper, action: FeedbackAction, tags: string[] = []) {
    try {
      await postFeedback({
        paper_id: paper.paper_id,
        query,
        section: 'path_section' in paper ? paper.path_section : null,
        action,
        method,
        background_level: backgroundLevel,
        tags,
      });
      setFeedbackMessage(`Feedback saved: ${action.replaceAll('_', ' ')}`);
      const updated = await fetchProfile();
      setProfile(updated);
      if (action === 'save') {
        await refreshLibrary();
      }
    } catch (feedbackError) {
      setFeedbackMessage(feedbackError instanceof Error ? feedbackError.message : 'Feedback failed.');
      throw feedbackError;
    }
  }

  async function handleUnsave(paper: Recommendation | PathPaper) {
    try {
      await deleteLibraryItem(paper.paper_id);
      setFeedbackMessage('Removed from library');
      const updatedProfile = await fetchProfile();
      setProfile(updatedProfile);
      await refreshLibrary();
    } catch (unsaveError) {
      setFeedbackMessage(unsaveError instanceof Error ? unsaveError.message : 'Unsave failed.');
      throw unsaveError;
    }
  }

  async function handleAddLibraryTags(paperId: number) {
    const tags = parseTags(libraryTagInputs[paperId] ?? '');
    if (tags.length === 0) return;

    setPendingLibraryPaperId(paperId);
    try {
      await saveLibraryItem(paperId, tags);
      setFeedbackMessage(`Added tags: ${tags.join(', ')}`);
      setLibraryTagInputs((current) => ({ ...current, [paperId]: '' }));
      await refreshLibrary();
    } catch (tagError) {
      setFeedbackMessage(tagError instanceof Error ? tagError.message : 'Tag update failed.');
    } finally {
      setPendingLibraryPaperId(null);
    }
  }

  const selectableMethods = retrievalMethods.filter((retrievalMethod) => availableMethods.includes(retrievalMethod.value));

  return (
    <main className="page">
      <header className="app-header">
        <div>
          <p className="eyebrow">Local-first ML research recommender</p>
          <h1>ResearchPath</h1>
          <p className="subtle">Build a reading path through ML research with ranked papers, staged sections, and profile-aware feedback.</p>
        </div>
        <div className="status-grid">
          <div className="status-card">
            <Database size={18} />
            <span>Dataset</span>
            <strong>{datasetStatus ? datasetStatus.paper_count.toLocaleString() : 'Loading'}</strong>
            <p>Updated {formatDatasetTime(datasetStatus?.last_updated_timestamp ?? null)}</p>
          </div>
          <div className="status-card">
            <UserRound size={18} />
            <span>Profile</span>
            <strong>{profile?.current_status.replaceAll('_', ' ') ?? 'Exploring'}</strong>
            <p>{profile?.saved_paper_ids.length ?? 0} saved / {profile?.preferred_topics.length ?? 0} topics</p>
          </div>
        </div>
      </header>

      <div className="workspace-shell">
        <aside className="workflow-panel" aria-label="Search controls">
        <form className="search-form" onSubmit={handleSubmit}>
          <div className="mode-toggle" aria-label="Result mode">
            <button type="button" className={mode === 'path' ? 'active' : ''} onClick={() => setMode('path')}>
              Reading path
            </button>
            <button type="button" className={mode === 'search' ? 'active' : ''} onClick={() => setMode('search')}>
              Search
            </button>
          </div>
          <label htmlFor="query">Research goal or topic</label>
          <div className="search-row">
            <input
              id="query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="I want to learn AI agents and know basic ML"
            />
            <button type="submit" disabled={isLoading} aria-label="Search papers">
              <Search size={18} />
              <span>{isLoading ? 'Searching' : 'Search'}</span>
            </button>
          </div>
          <div className="control-grid">
            <div>
              <label htmlFor="method">Retrieval method</label>
              <select id="method" value={method} onChange={(event) => setMethod(event.target.value)}>
                {selectableMethods.map((retrievalMethod) => (
                  <option key={retrievalMethod.value} value={retrievalMethod.value}>
                    {retrievalMethod.label}
                  </option>
                ))}
              </select>
              <p className="field-hint">{methodHelpText[method] ?? 'Choose how ResearchPath ranks candidate papers.'}</p>
            </div>
            <div className="demo-compare-control">
              <label htmlFor="compare-methods">Method comparison</label>
              <label className="checkbox-row">
                <input
                  id="compare-methods"
                  type="checkbox"
                  checked={compareMethods}
                  onChange={(event) => setCompareMethods(event.target.checked)}
                  disabled={mode !== 'search'}
                />
              <span>Keyword / guarded / fusion</span>
              </label>
            </div>
          </div>
        </form>

        <section className="profile-panel" aria-label="Editable profile">
          <div className="profile-panel__header">
            <div>
              <p className="eyebrow">Reader profile</p>
              <h2>Tune recommendations</h2>
            </div>
            <span>{profile?.paper_taste.replaceAll('_', ' ') ?? 'balanced'}</span>
          </div>

          <div className="profile-grid">
            <div>
              <label htmlFor="profile-background">Background</label>
              <select
                id="profile-background"
                value={profileDraft.background_level}
                onChange={(event) => handleBackgroundChange(event.target.value)}
              >
                <option value="basic_ml">Basic ML</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>
            <div>
              <label htmlFor="current-status">Current status</label>
              <select
                id="current-status"
                value={profileDraft.current_status}
                onChange={(event) => setProfileDraft((current) => ({ ...current, current_status: event.target.value }))}
              >
                {profileOptions.currentStatus.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="research-goal">Goal</label>
              <select
                id="research-goal"
                value={profileDraft.research_goal}
                onChange={(event) => setProfileDraft((current) => ({ ...current, research_goal: event.target.value }))}
              >
                {profileOptions.researchGoal.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="paper-taste">Paper taste</label>
              <select
                id="paper-taste"
                value={profileDraft.paper_taste}
                onChange={(event) => setProfileDraft((current) => ({ ...current, paper_taste: event.target.value }))}
              >
                {profileOptions.paperTaste.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="profile-text-fields">
            <label htmlFor="preferred-topics">Prefer topics</label>
            <input
              id="preferred-topics"
              value={profileDraft.preferred_topics}
              onChange={(event) => setProfileDraft((current) => ({ ...current, preferred_topics: event.target.value }))}
              placeholder="agents, transformers, retrieval"
            />
            <label htmlFor="avoid-topics">Avoid topics</label>
            <input
              id="avoid-topics"
              value={profileDraft.avoid_topics}
              onChange={(event) => setProfileDraft((current) => ({ ...current, avoid_topics: event.target.value }))}
              placeholder="too theoretical, medical imaging"
            />
          </div>

          <div className="profile-actions">
            <button type="button" onClick={handleProfileSave} disabled={isSavingProfile}>
              {isSavingProfile ? 'Saving profile' : 'Save profile'}
            </button>
            <p>Hybrid ranking uses these preferences; other opt-in rankers still mainly show benchmark behavior.</p>
          </div>
        </section>

        {library && (
          <section className="library-panel" aria-label="Saved library">
            <div className="library-header">
              <div>
                <p className="eyebrow">Library</p>
                <h2>Saved papers</h2>
              </div>
              <span>{library.items.length}</span>
            </div>
            <div className="library-tags" aria-label="Library tags">
              <button type="button" className={activeLibraryTag === null ? 'active' : ''} onClick={() => handleLibraryTagChange(null)}>
                All
              </button>
              {library.tags.map((tag) => (
                <button
                  key={tag}
                  type="button"
                  className={activeLibraryTag === tag ? 'active' : ''}
                  onClick={() => handleLibraryTagChange(tag)}
                >
                  {tag}
                </button>
              ))}
            </div>
            <div className="library-items">
              {library.items.map((item) => (
                <article className="library-item" key={item.id}>
                  <div>
                    <h3>{item.paper.title}</h3>
                    <p>{[item.paper.year ?? 'Year unknown', item.paper.venue].filter(Boolean).join(' / ')}</p>
                    <div className="library-item__tags">
                      {item.tags.length ? item.tags.map((tag) => <span key={tag}>{tag}</span>) : <span>untagged</span>}
                    </div>
                  </div>
                  <div className="library-item__actions">
                    <div className="library-tag-editor">
                      <input
                        aria-label={`Tags for ${item.paper.title}`}
                        value={libraryTagInputs[item.paper_id] ?? ''}
                        onChange={(event) =>
                          setLibraryTagInputs((current) => ({ ...current, [item.paper_id]: event.target.value }))
                        }
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') {
                            event.preventDefault();
                            handleAddLibraryTags(item.paper_id);
                          }
                        }}
                        placeholder="transformer, resnet"
                      />
                      <button
                        type="button"
                        disabled={pendingLibraryPaperId === item.paper_id}
                        onClick={() => handleAddLibraryTags(item.paper_id)}
                      >
                        {pendingLibraryPaperId === item.paper_id ? 'Adding' : 'Add'}
                      </button>
                    </div>
                    <button type="button" onClick={() => handleUnsave(item.paper)}>
                      Unsave
                    </button>
                  </div>
                </article>
              ))}
              {library.items.length === 0 && (
                <div className="empty-section">Saved papers will appear here after you click Save.</div>
              )}
            </div>
          </section>
        )}
        </aside>

        <section className="results-workspace" aria-label={outputTitle}>
          <div className="workspace-header">
            <div>
              <p className="eyebrow">Workspace</p>
              <h2>{outputTitle}</h2>
              <p>{outputDetail}</p>
            </div>
            {compareMethods && mode === 'search' && (
              <span className="comparison-badge">
                <BarChart3 size={15} />
                Comparison
              </span>
            )}
          </div>

          {error && <p className="error">{error}</p>}
          {feedbackMessage && <p className="feedback-message">{feedbackMessage}</p>}

          {comparisonResults.length > 0 && (
            <section className="comparison-panel" aria-label="Method comparison">
              <div className="comparison-header">
                <BarChart3 size={18} />
                <h2>Method comparison</h2>
              </div>
              <div className="comparison-grid">
                {comparisonResults.map((comparison) => (
                  <article className="comparison-column" key={comparison.method}>
                    <h3>{methodLabels[comparison.method] ?? comparison.method}</h3>
                    {comparison.error ? (
                      <p className="error">{comparison.error}</p>
                    ) : (
                      <ol>
                        {comparison.recommendations.map((recommendation) => (
                          <li key={recommendation.paper_id}>
                            <span>{recommendation.title}</span>
                            <strong>{recommendation.score.toFixed(3)}</strong>
                          </li>
                        ))}
                      </ol>
                    )}
                  </article>
                ))}
              </div>
            </section>
          )}

          {readingPath && (
            <section className="path-results" aria-label="Reading path">
              {pathSectionOrder.map((section) => (
                <PathSection
                  key={section}
                  title={section}
                  papers={readingPath.sections[section] ?? []}
                  status={readingPath.section_status[section]}
                  query={readingPath.query}
                  backgroundLevel={readingPath.background_level}
                  savedPaperIds={savedPaperIds}
                  savedTagsByPaperId={savedTagsByPaperId}
                  onFeedback={handleFeedback}
                  onUnsave={handleUnsave}
                />
              ))}
            </section>
          )}

          {!readingPath && (
            <section className="results" aria-label="Recommended papers">
              {results.map((result) => (
                <ResultCard
                  key={result.paper_id}
                  result={result}
                  isSaved={savedPaperIds.has(result.paper_id)}
                  savedTags={savedTagsByPaperId.get(result.paper_id) ?? []}
                  onFeedback={handleFeedback}
                  onUnsave={handleUnsave}
                />
              ))}
            </section>
          )}
        </section>
      </div>
    </main>
  );
}

function parseTags(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(',')
        .map((tag) => tag.trim().toLowerCase())
        .filter(Boolean),
    ),
  );
}

function profileToDraft(profile: UserProfile): ProfileDraft {
  return {
    background_level: profile.background_level,
    current_status: profile.current_status,
    research_goal: profile.research_goal,
    paper_taste: profile.paper_taste,
    preferred_topics: profile.preferred_topics.join(', '),
    avoid_topics: profile.avoid_topics.join(', '),
  };
}

function formatDatasetTime(timestamp: string | null): string {
  if (!timestamp) return 'unknown';
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return 'unknown';
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}
