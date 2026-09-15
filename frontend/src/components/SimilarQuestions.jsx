import React from 'react';
import { ArrowUpRight } from 'lucide-react';

const CATEGORY_COLOR_MAP = {
  'Technical Knowledge': 'var(--cat-tech)',
  'Communication': 'var(--cat-comm)',
  'Problem Solving': 'var(--cat-prob)',
  'Leadership': 'var(--cat-lead)',
  'Role-Specific Skills': 'var(--cat-role)',
};

export default function SimilarQuestions({ questions = [], onSelectQuestion }) {
  if (!questions || questions.length === 0) {
    return (
      <div className="similar-empty-hint">
        No close matches found in the 54,000-question database.
      </div>
    );
  }

  return (
    <div className="similar-list">
      {questions.map((item, idx) => {
        const catColor = CATEGORY_COLOR_MAP[item.topic] || 'var(--primary-color)';

        return (
          <div
            key={idx}
            className="similar-card-item interactive"
            onClick={() => onSelectQuestion && onSelectQuestion(item.question)}
            title="Click to test this question"
          >
            <div className="similar-meta">
              <span className="similar-badge" style={{ backgroundColor: catColor }}>
                {item.topic}
              </span>
              <div className="similar-right-meta">
                <span className="similar-score">Similarity: {item.similarity_pct}</span>
                <span className="test-prompt-chip">
                  <span>Test</span>
                  <ArrowUpRight size={12} />
                </span>
              </div>
            </div>
            <p className="similar-text">{item.question}</p>
          </div>
        );
      })}
    </div>
  );
}
