import React, { useState, useEffect } from 'react';
import {
  Send,
  AlertTriangle,
  CheckCircle,
  Sliders,
  RefreshCw,
  Sparkles,
  Info,
  Layers,
  ArrowRight,
  HelpCircle
} from 'lucide-react';
import ProbabilityBars from './ProbabilityBars';
import SimilarQuestions from './SimilarQuestions';
import { predictQuestion } from '../services/api';

const PRESETS = [
  {
    category: 'Technical Knowledge',
    icon: '💻',
    tag: 'Technical',
    color: 'var(--cat-tech)',
    text: 'How does database indexing work under the hood in PostgreSQL?',
  },
  {
    category: 'Communication',
    icon: '💬',
    tag: 'Communication',
    color: 'var(--cat-comm)',
    text: 'How do you explain technical debt to non-technical business executives?',
  },
  {
    category: 'Problem Solving',
    icon: '🛠️',
    tag: 'Problem Solving',
    color: 'var(--cat-prob)',
    text: 'Tell me about a time when you diagnosed and fixed a critical memory leak in production.',
  },
  {
    category: 'Leadership',
    icon: '👥',
    tag: 'Leadership',
    color: 'var(--cat-lead)',
    text: 'Describe a time when you mentored a junior software engineer struggling with code quality.',
  },
  {
    category: 'Role-Specific Skills',
    icon: '⚙️',
    tag: 'Role-Specific',
    color: 'var(--cat-role)',
    text: 'How do you configure Kubernetes Horizontal Pod Autoscalers based on Prometheus metrics?',
  },
  {
    category: 'Ambiguous Question',
    icon: '⚠️',
    tag: 'Review Demo',
    color: 'var(--warning)',
    text: 'What is your preferred philosophy when approaching software engineering teams?',
  },
];

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

export default function SingleClassifier({
  activeModel,
  onModelChange,
  threshold,
  onThresholdChange,
}) {
  const [question, setQuestion] = useState(
    'How does database indexing work under the hood in PostgreSQL?'
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleClassify = async (textToClassify = question) => {
    if (!textToClassify.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const data = await predictQuestion(textToClassify, activeModel, threshold);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to classify question. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // Run classification on initial load
  useEffect(() => {
    handleClassify();
  }, [activeModel]);

  const handlePresetClick = (presetText) => {
    setQuestion(presetText);
    handleClassify(presetText);
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleClassify();
    }
  };

  const currentTopic = result?.predicted_topic;
  const badgeColor = CATEGORY_COLOR_MAP[currentTopic] || 'var(--primary-color)';
  const currentIcon = CATEGORY_ICONS[currentTopic] || '🎯';

  return (
    <div className="classifier-container">
      {/* Friendly Introduction Hero */}
      <section className="hero-banner glass-panel">
        <div className="hero-content">
          <div className="hero-badge">
            <Sparkles size={14} />
            <span>AI-Powered Interview Analysis</span>
          </div>
          <h2 className="hero-title">Classify Any Interview Question in Milliseconds</h2>
          <p className="hero-subtitle">
            Determine whether a question evaluates <strong>Technical Knowledge</strong>, <strong>Communication</strong>, <strong>Problem Solving</strong>, <strong>Leadership</strong>, or <strong>Role-Specific Skills</strong> with calibrated certainty scoring.
          </p>
        </div>
        <div className="hero-stats-row">
          <div className="stat-card">
            <span className="stat-number">54,000</span>
            <span className="stat-label">Trained Dataset</span>
          </div>
          <div className="stat-card">
            <span className="stat-number">&lt; 5 ms</span>
            <span className="stat-label">Inference Latency</span>
          </div>
          <div className="stat-card">
            <span className="stat-number">5 Dimensions</span>
            <span className="stat-label">Core Competencies</span>
          </div>
        </div>
      </section>

      <div className="classifier-grid">
        {/* Left Column: Input Form */}
        <div className="glass-panel input-section">
          <div className="section-header-row">
            <div className="section-label">
              <span>Interview Question</span>
              <span className="char-counter">{question.length} characters</span>
            </div>
            {question.length > 0 && (
              <button
                type="button"
                className="btn-text-action"
                onClick={() => setQuestion('')}
              >
                Clear
              </button>
            )}
          </div>

          <textarea
            className="question-textarea"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={4}
            placeholder="Type or paste any interview question here (e.g. 'How do you resolve race conditions in multithreaded services?')..."
          />

          {/* User-Friendly Presets */}
          <div className="presets-container">
            <div className="presets-header">
              <span className="presets-label">⚡ Try a Sample Question:</span>
              <span className="presets-hint">Click any category to auto-test</span>
            </div>
            <div className="presets-cards-grid">
              {PRESETS.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  className={`preset-card-btn ${question === p.text ? 'active' : ''}`}
                  onClick={() => handlePresetClick(p.text)}
                  title={p.text}
                >
                  <div className="preset-card-header">
                    <span className="preset-icon">{p.icon}</span>
                    <span className="preset-tag" style={{ color: p.color }}>
                      {p.tag}
                    </span>
                  </div>
                  <div className="preset-card-text">{p.text}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Model & Review Controls */}
          <div className="controls-panel">
            <div className="controls-panel-title">
              <Sliders size={15} />
              <span>Model & Review Settings</span>
            </div>

            <div className="controls-grid">
              {/* Engine Selector */}
              <div className="control-item">
                <label htmlFor="model-select" className="control-label">
                  Classification Engine
                </label>
                <select
                  id="model-select"
                  className="custom-select"
                  value={activeModel}
                  onChange={(e) => onModelChange(e.target.value)}
                >
                  <option value="tfidf">⚡ TF-IDF Pipeline (Ultra-fast, &lt;5ms)</option>
                  <option value="embedding">🧠 SentenceTransformer (Semantic Embeddings)</option>
                </select>
                <span className="control-helper">
                  {activeModel === 'tfidf'
                    ? 'Best for instant keystrokes & high-throughput pipelines.'
                    : 'Best for nuanced, highly contextual semantic phrasing.'}
                </span>
              </div>

              {/* Threshold Presets & Slider */}
              <div className="control-item">
                <div className="control-label-row">
                  <label htmlFor="threshold-slider" className="control-label">
                    Audit Review Threshold: <strong>{Math.round(threshold * 100)}%</strong>
                  </label>
                  <span className="control-tooltip" title="Questions scoring below this confidence trigger a human review alert.">
                    <HelpCircle size={13} />
                  </span>
                </div>

                <div className="threshold-quick-buttons">
                  <button
                    type="button"
                    className={`btn-threshold-pill ${threshold === 0.5 ? 'active' : ''}`}
                    onClick={() => onThresholdChange(0.5)}
                  >
                    Lenient (50%)
                  </button>
                  <button
                    type="button"
                    className={`btn-threshold-pill ${threshold === 0.65 ? 'active' : ''}`}
                    onClick={() => onThresholdChange(0.65)}
                  >
                    Balanced (65%)
                  </button>
                  <button
                    type="button"
                    className={`btn-threshold-pill ${threshold === 0.75 ? 'active' : ''}`}
                    onClick={() => onThresholdChange(0.75)}
                  >
                    Strict (75%)
                  </button>
                </div>

                <div className="slider-container">
                  <input
                    id="threshold-slider"
                    type="range"
                    className="custom-slider"
                    min="0.40"
                    max="0.95"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Action Button */}
          <button
            className="btn-primary main-classify-btn"
            onClick={() => handleClassify()}
            disabled={loading || !question.trim()}
          >
            {loading ? (
              <>
                <RefreshCw className="spin" size={18} />
                <span>Analyzing Question Semantics...</span>
              </>
            ) : (
              <>
                <Send size={18} />
                <span>Classify Question Now</span>
                <span className="keyboard-shortcut">Ctrl+Enter</span>
              </>
            )}
          </button>

          {error && (
            <div className="review-box danger" style={{ marginTop: '1rem' }}>
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results */}
        <div className="results-section">
          {result ? (
            <div className="glass-panel result-card">
              <div className="result-card-header">
                <div>
                  <h3 className="card-title">Analysis Verdict</h3>
                  <p className="card-subtitle">Calibrated 5-class competency distribution</p>
                </div>
                <div className="model-tag-pill">
                  {result.model_used}
                </div>
              </div>

              {/* Main Winner Banner */}
              <div className="prediction-banner" style={{ borderColor: badgeColor }}>
                <div className="prediction-left">
                  <span className="topic-icon-large">{currentIcon}</span>
                  <div>
                    <span className="topic-badge-pill" style={{ backgroundColor: badgeColor }}>
                      {result.predicted_topic}
                    </span>
                    <div className="verdict-summary">
                      {result.needs_review ? (
                        <span className="verdict-review-text">
                          ⚠️ Ambiguous Signals: Confidence ({result.confidence_pct}) is below your {Math.round(threshold * 100)}% threshold.
                        </span>
                      ) : (
                        <span className="verdict-confident-text">
                          ✅ Clear Match: High confidence ({result.confidence_pct}) mapped to this competency.
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="confidence-circle-display">
                  <div className="confidence-circle-val">{result.confidence_pct}</div>
                  <div className="confidence-circle-label">Certainty</div>
                </div>
              </div>

              {/* Plain-English Explanation Callout */}
              {result.needs_review ? (
                <div className="review-box danger">
                  <AlertTriangle size={20} />
                  <div>
                    <strong>Human Review Recommended</strong>
                    <p>
                      This question scored <strong>{result.confidence_pct}</strong>, which is below the <strong>{Math.round(threshold * 100)}%</strong> threshold. It likely spans multiple domains (e.g. leadership + technical system design) and should be validated by an interviewer.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="review-box success">
                  <CheckCircle size={20} />
                  <div>
                    <strong>Verified High-Confidence Classification</strong>
                    <p>
                      Strong linguistic signals place this question squarely in <strong>{result.predicted_topic}</strong> with <strong>{result.confidence_pct}</strong> model certainty.
                    </p>
                  </div>
                </div>
              )}

              {/* Probability Breakdown */}
              <div className="result-subpanel">
                <div className="subpanel-title-row">
                  <h4>Category Probability Breakdown</h4>
                  <span className="subpanel-hint">Total 100% distribution</span>
                </div>
                <ProbabilityBars probabilities={result.probabilities} />
              </div>

              {/* Similar Questions */}
              <div className="result-subpanel">
                <div className="subpanel-title-row">
                  <h4>💡 Similar Questions in Database</h4>
                  <span className="subpanel-hint">Matched via vector cosine similarity</span>
                </div>
                <SimilarQuestions
                  questions={result.similar_questions}
                  onSelectQuestion={(q) => handlePresetClick(q)}
                />
              </div>
            </div>
          ) : (
            /* Empty State Placeholder */
            <div className="glass-panel empty-result-placeholder">
              <div className="empty-icon-circle">
                <Sparkles size={32} color="var(--primary-color)" />
              </div>
              <h3>Ready to Classify</h3>
              <p>Type an interview question on the left or select any sample to see instant competency breakdown and similar questions.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
