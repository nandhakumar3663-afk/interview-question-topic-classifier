import React, { useState } from 'react';
import { Layers, Download, Play, AlertTriangle } from 'lucide-react';
import { predictBatch } from '../services/api';

const DEFAULT_BATCH_QUESTIONS = `How do you explain technical debt to non-technical business executives?
How does database indexing work under the hood in PostgreSQL?
Tell me about a time when you diagnosed and fixed a critical memory leak in production.
Describe a time when you mentored a junior engineer struggling with code quality.
How do you configure Kubernetes Horizontal Pod Autoscalers based on custom Prometheus metrics?
What is your general philosophy on code reviews?`;

const CATEGORY_COLOR_MAP = {
  'Technical Knowledge': 'var(--cat-tech)',
  'Communication': 'var(--cat-comm)',
  'Problem Solving': 'var(--cat-prob)',
  'Leadership': 'var(--cat-lead)',
  'Role-Specific Skills': 'var(--cat-role)',
};

export default function BatchClassifier({ activeModel, threshold }) {
  const [textInput, setTextInput] = useState(DEFAULT_BATCH_QUESTIONS);
  const [loading, setLoading] = useState(false);
  const [batchData, setBatchData] = useState(null);
  const [error, setError] = useState(null);

  const handleRunBatch = async () => {
    const lines = textInput
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 5);

    if (lines.length === 0) {
      setError('Please enter at least one valid question.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await predictBatch(lines, activeModel, threshold);
      setBatchData(data);
    } catch (err) {
      setError(err.message || 'Batch classification failed.');
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (!batchData || !batchData.results) return;

    const headers = ['Question', 'Predicted Topic', 'Confidence', 'Needs Review'];
    const rows = batchData.results.map((r) => [
      `"${r.question.replace(/"/g, '""')}"`,
      `"${r.predicted_topic}"`,
      r.confidence_pct,
      r.needs_review ? 'Yes' : 'No',
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'classified_interview_questions.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="glass-panel" style={{ padding: '1.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Bulk Question Classification</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Paste interview questions (one question per line) to classify in high-speed batch.
          </p>
        </div>

        {batchData && (
          <button
            type="button"
            className="btn-primary"
            style={{ width: 'auto', padding: '0.6rem 1.2rem', fontSize: '0.88rem' }}
            onClick={downloadCSV}
          >
            <Download size={16} />
            Export CSV
          </button>
        )}
      </div>

      <textarea
        className="question-textarea"
        style={{ minHeight: '130px', fontFamily: 'var(--font-mono)', fontSize: '0.88rem' }}
        value={textInput}
        onChange={(e) => setTextInput(e.target.value)}
        placeholder="Paste multiple questions here, one per line..."
      />

      <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem' }}>
        <button
          type="button"
          className="btn-primary"
          style={{ width: 'auto', padding: '0.75rem 1.8rem' }}
          onClick={handleRunBatch}
          disabled={loading}
        >
          <Play size={16} />
          {loading ? 'Processing Batch...' : 'Run Batch Analysis'}
        </button>
      </div>

      {error && (
        <div className="review-box danger" style={{ marginTop: '1rem' }}>
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {batchData && (
        <div style={{ marginTop: '2rem' }}>
          {/* Summary Metrics */}
          <div className="metrics-grid">
            <div className="metric-box">
              <div className="metric-box-title">Total Questions</div>
              <div className="metric-box-value">{batchData.total}</div>
            </div>
            <div className="metric-box">
              <div className="metric-box-title">Flagged for Review</div>
              <div className="metric-box-value" style={{ color: batchData.flagged_count > 0 ? 'var(--warning)' : 'var(--success)' }}>
                {batchData.flagged_count}
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-box-title">Avg Confidence</div>
              <div className="metric-box-value">{batchData.average_confidence_pct}</div>
            </div>
          </div>

          {/* Results Table */}
          <div style={{ overflowX: 'auto' }}>
            <table className="batch-table">
              <thead>
                <tr>
                  <th style={{ width: '55%' }}>Interview Question</th>
                  <th>Predicted Topic</th>
                  <th>Confidence</th>
                  <th>Review Status</th>
                </tr>
              </thead>
              <tbody>
                {batchData.results.map((r, i) => {
                  const catColor = CATEGORY_COLOR_MAP[r.predicted_topic] || 'var(--primary-color)';
                  return (
                    <tr key={i}>
                      <td>{r.question}</td>
                      <td>
                        <span
                          className="similar-badge"
                          style={{ backgroundColor: catColor, fontSize: '0.78rem' }}
                        >
                          {r.predicted_topic}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                        {r.confidence_pct}
                      </td>
                      <td>
                        {r.needs_review ? (
                          <span style={{ color: 'var(--danger)', fontWeight: 700, fontSize: '0.82rem' }}>
                            ⚠️ Flagged
                          </span>
                        ) : (
                          <span style={{ color: 'var(--success)', fontWeight: 700, fontSize: '0.82rem' }}>
                            ✓ Passed
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
