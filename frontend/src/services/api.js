/**
 * API client service for connecting React frontend to the FastAPI backend.
 */

// Support dynamic relative URLs in production and localhost:8000 in Vite dev mode
export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL !== undefined
    ? import.meta.env.VITE_API_BASE_URL
    : (typeof window !== 'undefined' && window.location.port === '5173')
    ? 'http://127.0.0.1:8000'
    : '';

export async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) throw new Error('API offline');
    return await res.json();
  } catch (err) {
    console.warn('API Health Check failed:', err);
    return { status: 'offline', error: err.message };
  }
}

export async function getCategories() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/categories`);
    if (!res.ok) throw new Error('Failed to fetch categories');
    return await res.json();
  } catch (err) {
    console.error('getCategories error:', err);
    throw err;
  }
}

export async function predictQuestion(question, modelType = 'tfidf', threshold = 0.65) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        model_type: modelType,
        threshold: parseFloat(threshold),
      }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Prediction failed');
    }
    return await res.json();
  } catch (err) {
    console.error('predictQuestion error:', err);
    throw err;
  }
}

export async function predictBatch(questions, modelType = 'tfidf', threshold = 0.65) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        questions,
        model_type: modelType,
        threshold: parseFloat(threshold),
      }),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Batch prediction failed');
    }
    return await res.json();
  } catch (err) {
    console.error('predictBatch error:', err);
    throw err;
  }
}

export async function getMetrics() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/metrics`);
    if (!res.ok) throw new Error('Failed to fetch metrics');
    return await res.json();
  } catch (err) {
    console.error('getMetrics error:', err);
    throw err;
  }
}
