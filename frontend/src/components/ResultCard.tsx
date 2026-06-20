import React from 'react';
import type { Recommendation } from '../api/client';

type ResultCardProps = {
  result: Recommendation;
};

export function ResultCard({ result }: ResultCardProps) {
  return (
    <article className="result-card">
      <div className="result-card__meta">
        <span>{result.year ?? 'Year unknown'}</span>
        <span className="method-label">{result.method}</span>
      </div>
      <h2>{result.title}</h2>
      <p className="authors">{result.authors.length ? result.authors.join(', ') : 'Authors unknown'}</p>
      <p className="snippet">{result.abstract_snippet}</p>
      <div className="reason">
        <span>Reason</span>
        <p>{result.explanation}</p>
      </div>
    </article>
  );
}
