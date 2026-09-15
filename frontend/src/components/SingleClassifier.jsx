import React, { useState, useEffect } from 'react';
import { Send, AlertTriangle, CheckCircle, Sliders, RefreshCw, BookOpen } from 'lucide-react';
import ProbabilityBars from './ProbabilityBars';
import SimilarQuestions from './SimilarQuestions';
import { predictQuestion } from '../services/api';

const PRESETS = [
  {
    label: 'Postgres Indexing',
    category: 'Technical Knowledge',
    text: 'How does database indexing work under the hood in PostgreSQL?',
  },
  {
    label: 'Explaining Tech Debt',
    category: 'Communication',
    text: 'How do you explain technical debt to non-technical business executives?',
  },
  {
    label: 'Production Memory Leak',
    category: 'Problem Solving',
    text: 'Tell me about a time when you diagnosed and fixed a critical memory leak in production.',
  },
  {
    label: 'Junior Mentorship',
    category: 'Leadership',
    text: 'Describe a time when you mentored a junior software engineer struggling with code quality.',
  },
  {
    label: 'K8s Autoscaling',
    category: 'Role-Specific Skills',
    text: 'How do you configure Kubernetes Horizontal Pod Autoscalers based on custom Prometheus metrics?',
  },
  {
    label: 'Ambiguous Edge Case',
    category: 'General',
    text: 'What is your preferred routine for approaching a new engineering team project?',
  },
];

const CATEGORY_COLOR_MAP = {
  'Technical Knowledge': 'var(--cat-tech)',
  'Communication': 'var(--cat-comm)',
  'Problem Solving': 'var(--cat-prob)',
  'Leadership': 'var(--cat-lead)',
  'Role-Specific Skills': 'var(--cat-role)',
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

  return (
    <div className="classifier-grid">
      {/* Left Column: Input Form */}
      <div className="glass-panel input-section">
        <div className="section-label">
          <span>Interview Question</span>
          <span className="char-counter">{question.length} chars</span>
        </div>

        <textarea
          className="question-textarea"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Paste or type an interview question here (Press Ctrl+Enter to classify)..."
        />

        {/* Quick Presets */}
        <div className="presets-container">
          <div className="presets-label">⚡ Quick Presets</div>
          <div className="presets-chips">
            {PRESETS.map((p, idx) => (
              <button
                key={idx}
                type="button"
                className="preset-chip"
                onClick={() => handlePresetClick(p.text)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Model & Review Controls */}
        <div className="controls-grid">
          <div className="control-item">
            <label htmlFor="model-select">ML Classifier Engine</label>
            <select
              id="model-select"
              className="custom-select"
              value={activeModel}
              onChange={(e) => onModelChange(e.target.value)}
            >
              <option value="tfidf">TF-IDF Baseline (Fast, &lt;5ms)</option>
              <option value="embedding">SentenceTransformer (Semantic, MiniLM)</option>
            </select>
          </div>

          <div className="control-item">
            <label htmlFor="threshold-slider">Review Threshold</label>
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
              <span className="slider-value">{Math.round(threshold * 100)}%</span>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <button
          className="btn-primary"
          onClick={() => handleClassify()}
          disabled={loading || !question.trim()}
        >
          {loading ? (
            <>
              <RefreshCw className="spin" size={18} />
              Classifying Question...
            </>
          ) : (
            <>
              <Send size={18} />
              Classify Topic
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
        {result && (
          <div className="glass-panel result-card">
            <div className="card-title">Classification Result</div>

            {/* Top Prediction Banner */}
            <div className="prediction-banner">
              <div>
                <span className="category-tag" style={{ backgroundColor: badgeColor }}>
                  {result.predicted_topic}
                </span>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  Engine: {result.model_used}
                </div>
              </div>

              <div className="confidence-display">
                <div className="confidence-val">{result.confidence_pct}</div>
                <div className="confidence-label">Confidence</div>
              </div>
            </div>

            {/* Review Status Callout */}
            {result.needs_review ? (
              <div className="review-box danger">
                <AlertTriangle size={20} />
                <div>
                  <strong>Human Review Recommended</strong>
                  <div>
                    Confidence ({result.confidence_pct}) is below the {Math.round(threshold * 100)}%
                    quality threshold. This question exhibits cross-domain characteristics.
                  </div>
                </div>
              </div>
            ) : (
              <div className="review-box success">
                <CheckCircle size={20} />
                <div>
                  <strong>High Confidence Prediction</strong>
                  <div>
                    Score surpasses the {Math.round(threshold * 100)}% threshold with clear topic
                    signals.
                  </div>
                </div>
              </div>
            )}

            {/* Probability Breakdown */}
            <div style={{ marginTop: '1.5rem' }}>
              <div className="card-title">Category Probability Distribution</div>
              <ProbabilityBars probabilities={result.probabilities} />
            </div>

            {/* Bonus Feature: Similar Questions */}
            <div style={{ marginTop: '1.75rem' }}>
              <div className="card-title">💡 Semantically Similar Historical Questions</div>
              <SimilarQuestions questions={result.similar_questions} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
