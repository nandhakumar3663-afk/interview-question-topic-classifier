import React from 'react';
import { Sparkles } from 'lucide-react';

const CATEGORY_COLOR_MAP = {
  'Technical Knowledge': 'var(--cat-tech)',
  'Communication': 'var(--cat-comm)',
  'Problem Solving': 'var(--cat-prob)',
  'Leadership': 'var(--cat-lead)',
  'Role-Specific Skills': 'var(--cat-role)',
};

export default function SimilarQuestions({ questions = [] }) {
  if (!questions || questions.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.86rem' }}>
        No close historical matches found in training database.
      </div>
    );
  }

  return (
    <div className="similar-list">
      {questions.map((item, idx) => {
        const catColor = CATEGORY_COLOR_MAP[item.topic] || 'var(--primary-color)';

        return (
          <div key={idx} className="similar-card-item">
            <div className="similar-meta">
              <span className="similar-badge" style={{ backgroundColor: catColor }}>
                {item.topic}
              </span>
              <span className="similar-score">Match: {item.similarity_pct}</span>
            </div>
            <p className="similar-text">{item.question}</p>
          </div>
        );
      })}
    </div>
  );
}
