import React from 'react';

const CATEGORY_COLOR_MAP = {
  'Technical Knowledge': 'var(--cat-tech)',
  'Communication': 'var(--cat-comm)',
  'Problem Solving': 'var(--cat-prob)',
  'Leadership': 'var(--cat-lead)',
  'Role-Specific Skills': 'var(--cat-role)',
};

export default function ProbabilityBars({ probabilities = {} }) {
  const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="probabilities-list">
      {sortedEntries.map(([topic, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const color = CATEGORY_COLOR_MAP[topic] || 'var(--primary-color)';

        return (
          <div key={topic} className="prob-item">
            <div className="prob-meta">
              <span>{topic}</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{pct}%</span>
            </div>
            <div className="prob-bar-track">
              <div
                className="prob-bar-fill"
                style={{
                  width: `${Math.max(Number(pct), 2)}%`,
                  backgroundColor: color,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
