import React from 'react';
import type { FeedbackAction, PathPaper } from '../api/client';
import { ResultCard } from './ResultCard';

type PathSectionProps = {
  title: string;
  papers: PathPaper[];
  status?: { section_complete: boolean; fill_reason: string | null };
  query: string;
  backgroundLevel: string;
  savedPaperIds?: Set<number>;
  savedTagsByPaperId?: Map<number, string[]>;
  onFeedback: (paper: PathPaper, action: FeedbackAction, tags?: string[]) => void;
  onUnsave?: (paper: PathPaper) => void;
};

const sectionLabels: Record<string, string> = {
  background: 'Background',
  foundational: 'Foundational',
  core_methods: 'Core methods',
  recent_frontier: 'Recent frontier',
};

export function PathSection({
  title,
  papers,
  status,
  query,
  backgroundLevel,
  savedPaperIds = new Set<number>(),
  savedTagsByPaperId = new Map<number, string[]>(),
  onFeedback,
  onUnsave,
}: PathSectionProps) {
  return (
    <section className="path-section" aria-label={sectionLabels[title] ?? title}>
      <div className="path-section__header">
        <div>
          <h2>{sectionLabels[title] ?? title}</h2>
          {status && !status.section_complete && (
            <p>Fewer strong candidates found for this stage.</p>
          )}
        </div>
        <span className={status?.section_complete ? 'complete' : 'incomplete'}>
          {status?.section_complete ? 'Complete' : 'Sparse'} / {papers.length}
        </span>
      </div>
      <div className="path-section__papers">
        {papers.map((paper) => (
          <div className="path-paper" key={`${title}-${paper.paper_id}`}>
            <ResultCard
              result={paper}
              query={query}
              backgroundLevel={backgroundLevel}
              isSaved={savedPaperIds.has(paper.paper_id)}
              savedTags={savedTagsByPaperId.get(paper.paper_id) ?? []}
              onFeedback={(result, action, tags) => onFeedback(result as PathPaper, action, tags)}
              onUnsave={onUnsave ? (result) => onUnsave(result as PathPaper) : undefined}
            />
          </div>
        ))}
        {papers.length === 0 && (
          <div className="empty-section">
            No high-confidence papers for this stage yet.
          </div>
        )}
      </div>
    </section>
  );
}
