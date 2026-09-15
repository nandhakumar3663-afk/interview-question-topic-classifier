import React from 'react';
import { X, CheckCircle2, ShieldAlert, Cpu, Sparkles, BookOpen } from 'lucide-react';

export default function HowItWorksModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container glass-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <div className="modal-icon-badge">
              <Sparkles size={20} color="var(--primary-color)" />
            </div>
            <div>
              <h3>How TopicSense AI Works</h3>
              <p>EMP-26 Competency Classification & Quality Audit Guide</p>
            </div>
          </div>
          <button className="btn-icon" onClick={onClose} aria-label="Close guide">
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {/* Step 1 */}
          <div className="guide-step-card">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>5 Core Interview Competencies</h4>
              <p>
                Every engineering question tests one primary dimension. Our model categorizes questions into:
              </p>
              <div className="guide-tags-grid">
                <span className="guide-tag tech">💻 Technical Knowledge</span>
                <span className="guide-tag comm">💬 Communication</span>
                <span className="guide-tag prob">🛠️ Problem Solving</span>
                <span className="guide-tag lead">👥 Leadership</span>
                <span className="guide-tag role">⚙️ Role-Specific Skills</span>
              </div>
            </div>
          </div>

          {/* Step 2 */}
          <div className="guide-step-card">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Real-Time Confidence Scoring</h4>
              <p>
                The model computes a full probability distribution over all 5 topics. The winning topic's confidence percentage reflects how clearly the question maps to that competency.
              </p>
            </div>
          </div>

          {/* Step 3 */}
          <div className="guide-step-card">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Human Review Audit Shield</h4>
              <p>
                If confidence falls below your set threshold (default 65%), the system automatically flags it for <strong>Human Review</strong>. This prevents misclassifying nuanced or multi-part questions.
              </p>
            </div>
          </div>

          {/* Step 4 */}
          <div className="guide-step-card">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Semantic Similar Questions</h4>
              <p>
                Using vector cosine similarity, the system retrieves the closest verified questions from our 54,000-question dataset to give you instant context and benchmarking.
              </p>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-primary" onClick={onClose} style={{ width: '100%' }}>
            Got It, Let's Try It!
          </button>
        </div>
      </div>
    </div>
  );
}
