import React from 'react';

const CATEGORY_COLOR_MAP = {
  'Technical Knowledge': 'var(--cat-tech)',
  'Communication': 'var(--cat-comm)',
  'Problem Solving': 'var(--cat-prob)',
  'Leadership': 'var(--cat-lead)',
  'Role-Specific Skills': 'var(--cat-role)',
};

const CATEGORY_ICONS = {
  'Technical Knowledge': '💻',
  'Communication': '💬',
  'Problem Solving': '🛠️',
  'Leadership': '👥',
  'Role-Specific Skills': '⚙️',
};

export default function ProbabilityBars({ probabilities = {} }) {
  const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const highestProb = sortedEntries.length > 0 ? sortedEntries[0][1] : 0;

  return (
    <div className="probabilities-list">
      {sortedEntries.map(([topic, prob], idx) => {
        const pct = (prob * 100).toFixed(1);
        const color = CATEGORY_COLOR_MAP[topic] || 'var(--primary-color)';
        const icon = CATEGORY_ICONS[topic] || '📌';
        const isWinner = idx === 0 && prob > 0;

        return (
          <div key={topic} className={`prob-item ${isWinner ? 'winner-prob-item' : ''}`}>
            <div className="prob-meta">
              <div className="prob-topic-label">
                <span className="prob-icon">{icon}</span>
                <span className={isWinner ? 'winner-topic-text' : ''}>{topic}</span>
                {isWinner && <span className="winner-pill">Top Match</span>}
              </div>
              <span className="prob-pct-badge" style={{ color: isWinner ? color : 'inherit' }}>
                {pct}%
              </span>
            </div>
            <div className="prob-bar-track">
              <div
                className="prob-bar-fill"
                style={{
                  width: `${Math.max(Number(pct), 2)}%`,
                  backgroundColor: color,
                  boxShadow: isWinner ? `0 0 10px ${color}66` : 'none',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
