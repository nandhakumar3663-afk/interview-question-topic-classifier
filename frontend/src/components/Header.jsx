import React from 'react';
import { Sun, Moon, Cpu, Activity } from 'lucide-react';

export default function Header({ isDark, onToggleTheme, isApiConnected, activeModel }) {
  return (
    <header className="app-header">
      <div className="logo-group">
        <div className="logo-badge">EMP</div>
        <div>
          <h1 className="brand-title">EMP-26 Question Classifier</h1>
          <p className="brand-subtitle">Interview Topic & Competency Categorization System</p>
        </div>
      </div>

      <div className="header-actions">
        <div className="status-pill">
          <span
            className="pulse-dot"
            style={{ backgroundColor: isApiConnected ? 'var(--success)' : 'var(--danger)' }}
          />
          <span>{isApiConnected ? 'Backend Online' : 'Connecting...'}</span>
        </div>

        <div className="status-pill">
          <Cpu size={14} color="var(--primary-color)" />
          <span>{activeModel === 'tfidf' ? 'TF-IDF Baseline' : 'SentenceTransformer'}</span>
        </div>

        <button
          className="btn-icon"
          onClick={onToggleTheme}
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          aria-label="Toggle Theme"
        >
          {isDark ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  );
}
