import React, { FormEvent, useEffect, useState } from 'react';
import { Search } from 'lucide-react';
import {
  fetchDatasetStatus,
  fetchReadingPath,
  searchRecommendations,
  type DatasetStatus,
  type ReadingPath,
  type Recommendation,
} from '../api/client';
import { PathSection } from '../components/PathSection';
import { ResultCard } from '../components/ResultCard';

const retrievalMethods = [
  { value: 'hybrid', label: 'Hybrid' },
  { value: 'bm25', label: 'BM25' },
  { value: 'tfidf', label: 'TF-IDF' },
  { value: 'citation_recency', label: 'Citation + recency' },
  { value: 'embedding', label: 'Embedding' },
  { value: 'faiss_embedding', label: 'FAISS' },
];

const pathSectionOrder = ['background', 'foundational', 'core_methods', 'recent_frontier'];

export function SearchPage() {
  const [query, setQuery] = useState('AI agents for scientific discovery');
  const [method, setMethod] = useState('hybrid');
  const [mode, setMode] = useState<'search' | 'path'>('path');
  const [backgroundLevel, setBackgroundLevel] = useState('basic_ml');
  const [results, setResults] = useState<Recommendation[]>([]);
  const [readingPath, setReadingPath] = useState<ReadingPath | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDatasetStatus()
      .then(setDatasetStatus)
      .catch(() => setDatasetStatus(null));
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResults([]);
    setReadingPath(null);

    try {
      if (mode === 'path') {
        const path = await fetchReadingPath(query.trim(), 4, method, backgroundLevel);
        setReadingPath(path);
      } else {
        const recommendations = await searchRecommendations(query.trim(), 10, method);
        setResults(recommendations);
      }
    } catch (searchError) {
      setError(searchError instanceof Error ? searchError.message : 'Search failed.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="search-panel">
        <div>
          <p className="eyebrow">ResearchPath</p>
          <h1>Find the next papers worth reading.</h1>
          <p className="subtle">
            Search papers directly or organize recommendations into a beginner-aware reading path.
          </p>
          {datasetStatus && (
            <div className="dataset-status" aria-label="Dataset status">
              <span>{datasetStatus.paper_count.toLocaleString()} papers</span>
              <span>Updated {formatDatasetTime(datasetStatus.last_updated_timestamp)}</span>
            </div>
          )}
        </div>

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
          <label htmlFor="method">Retrieval method</label>
          <select id="method" value={method} onChange={(event) => setMethod(event.target.value)}>
            {retrievalMethods.map((retrievalMethod) => (
              <option key={retrievalMethod.value} value={retrievalMethod.value}>
                {retrievalMethod.label}
              </option>
            ))}
          </select>
          {mode === 'path' && (
            <>
              <label htmlFor="background">Background level</label>
              <select
                id="background"
                value={backgroundLevel}
                onChange={(event) => setBackgroundLevel(event.target.value)}
              >
                <option value="basic_ml">Basic ML</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </>
          )}
        </form>
      </section>

      {error && <p className="error">{error}</p>}

      {readingPath && (
        <section className="path-results" aria-label="Reading path">
          {pathSectionOrder.map((section) => (
            <PathSection key={section} title={section} papers={readingPath.sections[section] ?? []} />
          ))}
        </section>
      )}

      {!readingPath && (
        <section className="results" aria-label="Recommended papers">
          {results.map((result) => (
            <ResultCard key={result.paper_id} result={result} />
          ))}
        </section>
      )}
    </main>
  );
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
