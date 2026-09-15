import React, { useState, useEffect } from 'react';
import { Sparkles, Layers, BarChart3, BookOpen } from 'lucide-react';
import Header from './components/Header';
import SingleClassifier from './components/SingleClassifier';
import BatchClassifier from './components/BatchClassifier';
import MetricsDashboard from './components/MetricsDashboard';
import CategoryGuide from './components/CategoryGuide';
import { checkApiHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [isDark, setIsDark] = useState(() => {
    return localStorage.getItem('emp26_theme') === 'dark';
  });
  const [activeModel, setActiveModel] = useState('tfidf');
  const [threshold, setThreshold] = useState(0.65);
  const [isApiConnected, setIsApiConnected] = useState(false);

  // Apply theme to document element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('emp26_theme', isDark ? 'dark' : 'light');
  }, [isDark]);

  // Check health on mount and interval
  useEffect(() => {
    async function verifyHealth() {
      const res = await checkApiHealth();
      setIsApiConnected(res?.status === 'online');
    }
    verifyHealth();
    const interval = setInterval(verifyHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <Header
        isDark={isDark}
        onToggleTheme={() => setIsDark(!isDark)}
        isApiConnected={isApiConnected}
        activeModel={activeModel}
      />

      {/* Navigation Tabs */}
      <nav className="nav-tabs" role="tablist">
        <button
          className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          <Sparkles size={16} />
          <span>Interactive Classifier</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          <Layers size={16} />
          <span>Batch Testing</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          <BarChart3 size={16} />
          <span>Evaluation & Benchmarks</span>
        </button>

        <button
          className={`tab-btn ${activeTab === 'guide' ? 'active' : ''}`}
          onClick={() => setActiveTab('guide')}
        >
          <BookOpen size={16} />
          <span>Category Guide</span>
        </button>
      </nav>

      {/* Main Content Areas */}
      <main>
        {activeTab === 'single' && (
          <SingleClassifier
            activeModel={activeModel}
            onModelChange={setActiveModel}
            threshold={threshold}
            onThresholdChange={setThreshold}
          />
        )}

        {activeTab === 'batch' && (
          <BatchClassifier activeModel={activeModel} threshold={threshold} />
        )}

        {activeTab === 'metrics' && <MetricsDashboard />}

        {activeTab === 'guide' && <CategoryGuide />}
      </main>

      <footer
        style={{
          marginTop: '3.5rem',
          paddingTop: '1.5rem',
          borderTop: '1px solid var(--border-color)',
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
        }}
      >
        EMP-26 Lightweight Interview-Question Topic Classifier • React + FastAPI + Scikit-Learn +
        SentenceTransformers
      </footer>
    </div>
  );
}
