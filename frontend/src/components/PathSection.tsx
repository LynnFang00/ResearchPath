import React from 'react';
import type { PathPaper } from '../api/client';
import { ResultCard } from './ResultCard';

type PathSectionProps = {
  title: string;
  papers: PathPaper[];
};

const sectionLabels: Record<string, string> = {
  background: 'Background',
  foundational: 'Foundational',
  core_methods: 'Core methods',
  recent_frontier: 'Recent frontier',
};

export function PathSection({ title, papers }: PathSectionProps) {
  return (
    <section className="path-section" aria-label={sectionLabels[title] ?? title}>
      <div className="path-section__header">
        <h2>{sectionLabels[title] ?? title}</h2>
        <span>{papers.length}</span>
      </div>
      <div className="path-section__papers">
        {papers.map((paper) => (
          <div className="path-paper" key={`${title}-${paper.paper_id}`}>
            <div className="path-paper__labels">
              <span>{paper.difficulty_label}</span>
              <span>{paper.difficulty_score.toFixed(2)}</span>
            </div>
            <ResultCard result={paper} />
            <div className="path-reason">
              <span>Path role</span>
              <p>{paper.path_reason}</p>
              <p>{paper.difficulty_explanation}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
