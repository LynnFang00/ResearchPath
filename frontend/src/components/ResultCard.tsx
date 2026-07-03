import React, { useEffect, useState } from 'react';
import { Bookmark, Check, ExternalLink, FileText, ThumbsUp, XCircle } from 'lucide-react';
import type { FeedbackAction, PathPaper, Recommendation } from '../api/client';

type ResultCardProps = {
  result: Recommendation | PathPaper;
  query?: string;
  backgroundLevel?: string;
  onFeedback?: (paper: Recommendation | PathPaper, action: FeedbackAction, tags?: string[]) => Promise<void> | void;
  isSaved?: boolean;
  savedTags?: string[];
  onUnsave?: (paper: Recommendation | PathPaper) => Promise<void> | void;
};

const primaryFeedbackButtons: Array<{ action: FeedbackAction; label: string; icon: React.ReactNode }> = [
  { action: 'save', label: 'Save', icon: <Bookmark size={15} /> },
  { action: 'more_like_this', label: 'More like this', icon: <ThumbsUp size={15} /> },
  { action: 'not_relevant', label: 'Not useful', icon: <XCircle size={15} /> },
];

const methodDisplayNames: Record<string, string> = {
  hybrid: 'Personalized blend',
  learned_hybrid: 'Personalized learned blend',
  bm25: 'Keyword match',
  tfidf: 'Term-weighted match',
  citation_recency: 'Influential and recent',
  embedding: 'Semantic similarity',
  faiss_embedding: 'Fast semantic search',
  v3_3_ltr: 'Classic learned ranker',
  v4_1_blend: 'Guardrailed learned blend',
  v4_9_guarded_text_blend: 'Text-aware guarded ranker',
  v6_4_safe_fusion: 'Best offline fusion',
};

export function ResultCard({ result, onFeedback, isSaved = false, savedTags: initialSavedTags = [], onUnsave }: ResultCardProps) {
  const pathPaper = result as PathPaper;
  const tags = pathPaper.paper_type_tags ?? [];
  const signals = pathPaper.explanation_signals ?? [];
  const [selectedAction, setSelectedAction] = useState<FeedbackAction | null>(null);
  const [pendingAction, setPendingAction] = useState<FeedbackAction | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [showTagEditor, setShowTagEditor] = useState(false);
  const [tagInput, setTagInput] = useState('');
  const [localSaved, setLocalSaved] = useState(false);
  const [savedTags, setSavedTags] = useState<string[]>(initialSavedTags);

  const cardIsSaved = isSaved || localSaved;
  const initialSavedTagKey = initialSavedTags.join('\u0000');

  useEffect(() => {
    setSavedTags(initialSavedTags);
    if (isSaved) {
      setLocalSaved(false);
    }
  }, [initialSavedTagKey, isSaved]);

  async function handleFeedback(action: FeedbackAction, tags: string[] = []) {
    if (!onFeedback || pendingAction) return;
    setPendingAction(action);
    setFeedbackStatus(null);
    try {
      await onFeedback(result, action, tags);
      setSelectedAction(action);
      if (action === 'save') {
        setLocalSaved(true);
        setSavedTags(tags);
        setShowTagEditor(false);
      }
      setFeedbackStatus(`Saved: ${action.replaceAll('_', ' ')}`);
    } catch (error) {
      setFeedbackStatus(error instanceof Error ? error.message : 'Feedback failed.');
    } finally {
      setPendingAction(null);
    }
  }

  function handleSaveClick() {
    setShowTagEditor((value) => !value);
  }

  function submitSaveWithTags() {
    const tags = parseTags(tagInput);
    handleFeedback('save', tags);
  }

  async function handleUnsave() {
    if (!onUnsave || pendingAction) return;
    setPendingAction('save');
    setFeedbackStatus(null);
    try {
      await onUnsave(result);
      setLocalSaved(false);
      setSavedTags([]);
      setSelectedAction(null);
      setShowTagEditor(false);
      setFeedbackStatus('Removed from library');
    } catch (error) {
      setFeedbackStatus(error instanceof Error ? error.message : 'Unsave failed.');
    } finally {
      setPendingAction(null);
    }
  }

  return (
    <article className="result-card">
      <div className="result-card__meta">
        <span>{[result.year ?? 'Year unknown', result.venue].filter(Boolean).join(' / ')}</span>
        <span className="method-label">{methodDisplayNames[result.method] ?? result.method}</span>
      </div>
      <h2>{result.title}</h2>
      <p className="authors">{result.authors.length ? result.authors.join(', ') : 'Authors unknown'}</p>
        <div className="paper-tags">
        {pathPaper.difficulty_label && <span>{pathPaper.difficulty_label}</span>}
        {pathPaper.confidence_label && <span>{pathPaper.confidence_label} confidence</span>}
        {tags.map((tag) => (
          <span key={tag}>{tag}</span>
        ))}
        {savedTags.map((tag) => (
          <span className="user-tag" key={`saved-${tag}`}>{tag}</span>
        ))}
      </div>
      <p className="snippet">{result.abstract_snippet}</p>
      {(result.paper_url || result.pdf_url || result.doi_url) && (
        <div className="paper-link-row" aria-label="Paper links">
          {result.paper_url && (
            <a href={result.paper_url} target="_blank" rel="noreferrer">
              <ExternalLink size={15} />
              <span>Open paper</span>
            </a>
          )}
          {result.pdf_url && (
            <a href={result.pdf_url} target="_blank" rel="noreferrer">
              <FileText size={15} />
              <span>PDF</span>
            </a>
          )}
          {result.doi_url && (
            <a href={result.doi_url} target="_blank" rel="noreferrer">
              <ExternalLink size={15} />
              <span>DOI</span>
            </a>
          )}
        </div>
      )}
      <div className="reason">
        <span>Why recommended</span>
        <p>{pathPaper.why_recommended ?? result.explanation}</p>
        {pathPaper.why_this_section && <p>{pathPaper.why_this_section}</p>}
      </div>
      {(pathPaper.read_before?.length || pathPaper.read_after?.length) && (
        <div className="read-order">
          <div>
            <span>Read before</span>
            <p>{pathPaper.read_before?.length ? pathPaper.read_before.join(', ') : 'Start here'}</p>
          </div>
          <div>
            <span>Read after</span>
            <p>{pathPaper.read_after?.length ? pathPaper.read_after.join(', ') : 'End of path'}</p>
          </div>
        </div>
      )}
      {signals.length > 0 && (
        <div className="signal-list">
          {signals.map((signal) => (
            <span key={signal}>{signal}</span>
          ))}
        </div>
      )}
      {onFeedback && (
        <div className="feedback-row" aria-label="Paper feedback">
          {primaryFeedbackButtons.map((button) => (
            <button
              key={button.action}
              type="button"
              className={selectedAction === button.action || (button.action === 'save' && cardIsSaved) ? 'selected' : ''}
              disabled={pendingAction !== null}
              onClick={() => {
                if (button.action === 'save' && cardIsSaved) {
                  handleUnsave();
                  return;
                }
                if (button.action === 'save') {
                  handleSaveClick();
                  return;
                }
                handleFeedback(button.action);
              }}
            >
              {button.icon}
              <span>
                {pendingAction === button.action
                  ? cardIsSaved && button.action === 'save'
                    ? 'Removing'
                    : 'Saving'
                  : cardIsSaved && button.action === 'save'
                    ? 'Saved'
                    : button.label}
              </span>
            </button>
          ))}
          <label className="difficulty-feedback">
            <Check size={15} />
            <select
              aria-label="Difficulty feedback"
              value=""
              disabled={pendingAction !== null}
              onChange={(event) => {
                const action = event.target.value as FeedbackAction;
                if (action) handleFeedback(action);
              }}
            >
              <option value="">Difficulty</option>
              <option value="too_easy">Too easy</option>
              <option value="too_hard">Too hard</option>
              <option value="already_read">Already read</option>
            </select>
          </label>
          {feedbackStatus && <span className="feedback-status">{feedbackStatus}</span>}
        </div>
      )}
      {showTagEditor && (
        <div className="tag-editor">
          <label htmlFor={`tags-${result.paper_id}`}>Save with tags</label>
          <div>
            <input
              id={`tags-${result.paper_id}`}
              value={tagInput}
              onChange={(event) => setTagInput(event.target.value)}
              placeholder="transformer, resnet, agents"
            />
            <button type="button" disabled={pendingAction !== null} onClick={submitSaveWithTags}>
              Save to library
            </button>
          </div>
        </div>
      )}
      {'final_path_score' in pathPaper && (
        <details className="signal-details">
          <summary>Ranking signals</summary>
          <dl>
            {signalValue('BM25', pathPaper.bm25_score)}
            {signalValue('TF-IDF', pathPaper.tfidf_score)}
            {signalValue('FAISS', pathPaper.faiss_score)}
            {signalValue('Citation', pathPaper.citation_score)}
            {signalValue('Recency', pathPaper.recency_score)}
            {signalValue('Difficulty fit', pathPaper.difficulty_fit_score)}
            {signalValue('Section', pathPaper.section_score)}
            {signalValue('Duplicate penalty', pathPaper.duplicate_penalty)}
            {signalValue('Personalization', pathPaper.personalization_score)}
            {signalValue('Topic match', pathPaper.topic_similarity)}
            {signalValue('Learned ranker', pathPaper.learned_ranker_score)}
            {signalValue('Learned adjustment', pathPaper.learned_ranker_adjustment)}
          </dl>
          {pathPaper.learned_ranker_version && <p>Ranker version: {pathPaper.learned_ranker_version}</p>}
          {pathPaper.personalization_reason && <p>{pathPaper.personalization_reason}</p>}
        </details>
      )}
    </article>
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

function signalValue(label: string, value?: number | null) {
  return (
    <React.Fragment key={label}>
      <dt>{label}</dt>
      <dd>{typeof value === 'number' ? value.toFixed(2) : '0.00'}</dd>
    </React.Fragment>
  );
}
