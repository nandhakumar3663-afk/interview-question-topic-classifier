import React, { useState, useEffect } from 'react';
import { getCategories } from '../services/api';

const FALLBACK_CATEGORIES = [
  {
    name: 'Technical Knowledge',
    color: 'var(--cat-tech)',
    description: 'Core CS principles, algorithms, data structures, concurrency, databases, and system architecture.',
    examples: [
      'How do indexes work under the hood in PostgreSQL?',
      'What is the difference between TCP and UDP?',
      'How do you resolve race conditions and deadlocks in multithreaded services?',
    ],
  },
  {
    name: 'Communication',
    color: 'var(--cat-comm)',
    description: 'Explaining technical concepts to non-technical stakeholders, handling disagreements, active listening, and cross-team alignment.',
    examples: [
      'How do you explain technical debt to non-technical business executives?',
      'Tell me about a difficult conversation you had with a peer and how you reached agreement.',
      'How do you deliver constructive code review feedback?',
    ],
  },
  {
    name: 'Problem Solving',
    color: 'var(--cat-prob)',
    description: 'Debugging critical production incidents, diagnosing bottlenecks, root cause analysis, and high-pressure trade-offs.',
    examples: [
      'Tell me about a time when you diagnosed and fixed a critical memory leak in production.',
      'How do you troubleshoot an intermittent issue that only reproduces under heavy traffic?',
      'Walk me through your debugging process when 504 Gateway Timeouts spike unexpectedly.',
    ],
  },
  {
    name: 'Leadership',
    color: 'var(--cat-lead)',
    description: 'Mentorship, setting engineering rigor and standards, managing underperformance, driving consensus, and strategic vision.',
    examples: [
      'Describe a time when you mentored a junior engineer struggling with delivery.',
      'How do you drive consensus on adopting a new technical paradigm when there is resistance?',
      'Tell me about an initiative you led from conception to delivery across teams.',
    ],
  },
  {
    name: 'Role-Specific Skills',
    color: 'var(--cat-role)',
    description: 'Domain-specialized workflows including Frontend/React, DevOps/Kubernetes/CI-CD, QA test automation, and Data Engineering.',
    examples: [
      'How do you configure Kubernetes Horizontal Pod Autoscalers based on Prometheus metrics?',
      'How do you avoid unnecessary re-renders in React applications?',
      'What is the difference between Blue-Green deployments and Canary deployments?',
    ],
  },
];

export default function CategoryGuide() {
  const [categories, setCategories] = useState(FALLBACK_CATEGORIES);

  useEffect(() => {
    async function load() {
      try {
        const res = await getCategories();
        if (res?.categories) {
          setCategories(res.categories);
        }
      } catch (err) {
        // Fallback already set
      }
    }
    load();
  }, []);

  return (
    <div>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 800 }}>5 Required Classification Competencies</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Detailed breakdown of topic domains and real-world evaluation questions.
        </p>
      </div>

      <div className="guide-grid">
        {categories.map((cat, idx) => (
          <div key={idx} className="guide-card">
            <div className="guide-card-header">
              <span className="guide-pill" style={{ backgroundColor: cat.color }}>
                {cat.name}
              </span>
            </div>

            <p className="guide-description">{cat.description}</p>

            <div style={{ marginTop: 'auto' }}>
              <div className="guide-examples-title">Sample Questions</div>
              {cat.examples?.map((ex, i) => (
                <div key={i} className="guide-example-item">
                  "{ex}"
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
