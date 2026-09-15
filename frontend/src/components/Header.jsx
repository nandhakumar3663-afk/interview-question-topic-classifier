import React, { useState } from 'react';
import { Sun, Moon, Cpu, HelpCircle, Sparkles } from 'lucide-react';
import HowItWorksModal from './HowItWorksModal';

export default function Header({ isDark, onToggleTheme, isApiConnected, activeModel }) {
  const [isGuideOpen, setIsGuideOpen] = useState(false);

  return (
    <>
      <header className="app-header">
        <div className="logo-group">
          <div className="logo-icon-wrapper">
            <img src="/logo.svg" alt="TopicSense AI Logo" className="brand-logo-img" />
            <div className="logo-glow-ring" />
          </div>
          <div>
            <div className="brand-title-row">
              <h1 className="brand-title">TopicSense AI</h1>
              <span className="brand-chip">EMP-26</span>
            </div>
            <p className="brand-subtitle">Interview Question Competency Classifier & Audit System</p>
          </div>
        </div>

        <div className="header-actions">
          {/* How It Works Button */}
          <button
            className="btn-help"
            onClick={() => setIsGuideOpen(true)}
            title="Learn how TopicSense AI works"
          >
            <HelpCircle size={15} />
            <span>How it works</span>
          </button>

          {/* API Status Pill */}
          <div className="status-pill" title={isApiConnected ? 'Connected to FastAPI backend' : 'Connecting to backend...'}>
            <span
              className="pulse-dot"
              style={{ backgroundColor: isApiConnected ? 'var(--success)' : 'var(--danger)' }}
            />
            <span>{isApiConnected ? 'API Live' : 'Offline'}</span>
          </div>

          {/* Model Pill */}
          <div className="status-pill model-pill" title={`Active engine: ${activeModel === 'tfidf' ? 'TF-IDF Pipeline' : 'SentenceTransformer'}`}>
            <Cpu size={14} color="var(--primary-color)" />
            <span>{activeModel === 'tfidf' ? 'TF-IDF (Fast)' : 'MiniLM (Semantic)'}</span>
          </div>

          {/* Dark/Light Toggle */}
          <button
            className="btn-icon theme-toggle-btn"
            onClick={onToggleTheme}
            title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle Theme"
          >
            {isDark ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      <HowItWorksModal isOpen={isGuideOpen} onClose={() => setIsGuideOpen(false)} />
    </>
  );
}
