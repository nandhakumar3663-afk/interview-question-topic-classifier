"""
FastAPI REST API server for the EMP-26 Interview Question Topic Classifier.
Exposes endpoints for single-question prediction, probability distribution,
batch inference, category metadata, and evaluation metrics.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    TOPIC_CATEGORIES,
    TOPIC_COLORS,
    REPORTS_DIR,
    EVALUATION_METRICS_PATH,
    TFIDF_MODEL_PATH,
    EMBEDDING_CLASSIFIER_PATH,
)
from src.predict import predict_topic, find_similar_questions

app = FastAPI(
    title="EMP-26 Topic Classifier API",
    description="REST API for classifying software engineering interview questions into 5 core competencies.",
    version="1.0.0",
)

# Enable CORS for local Vite dev server and general access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount reports directory for visual charts
if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


# Request & Response Schemas
class PredictRequest(BaseModel):
    question: str = Field(..., description="Interview question text to classify", min_length=2)
    model_type: str = Field(default="tfidf", description="Model: 'tfidf' or 'embedding'")
    threshold: float = Field(default=0.65, description="Confidence threshold for human review", ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    questions: List[str] = Field(..., description="List of interview questions")
    model_type: str = Field(default="tfidf", description="Model: 'tfidf' or 'embedding'")
    threshold: float = Field(default=0.65, description="Confidence threshold for human review", ge=0.0, le=1.0)


CATEGORY_METADATA = [
    {
        "id": "technical-knowledge",
        "name": "Technical Knowledge",
        "color": TOPIC_COLORS["Technical Knowledge"],
        "icon": "code",
        "description": "Core CS principles, algorithms, data structures, concurrency, networking, and distributed systems architecture.",
        "examples": [
            "How do indexes work under the hood in PostgreSQL?",
            "What is the difference between TCP and UDP?",
            "How do you resolve race conditions and deadlocks in multithreaded services?",
        ],
    },
    {
        "id": "communication",
        "name": "Communication",
        "color": TOPIC_COLORS["Communication"],
        "icon": "message-circle",
        "description": "Explaining technical concepts to non-technical stakeholders, handling disagreements, active listening, and cross-team alignment.",
        "examples": [
            "How do you explain technical debt to non-technical business executives?",
            "Tell me about a difficult conversation you had with a peer and how you reached agreement.",
            "How do you deliver constructive code review feedback?",
        ],
    },
    {
        "id": "problem-solving",
        "name": "Problem Solving",
        "color": TOPIC_COLORS["Problem Solving"],
        "icon": "alert-triangle",
        "description": "Debugging critical production incidents, diagnosing performance bottlenecks, root cause analysis, and high-pressure trade-offs.",
        "examples": [
            "Tell me about a time when you diagnosed and fixed a critical memory leak in production.",
            "How do you troubleshoot an intermittent issue that only reproduces under heavy traffic?",
            "Walk me through your debugging process when 504 Gateway Timeouts spike unexpectedly.",
        ],
    },
    {
        "id": "leadership",
        "name": "Leadership",
        "color": TOPIC_COLORS["Leadership"],
        "icon": "users",
        "description": "Mentorship, setting engineering rigor and standards, managing underperformance, driving consensus, and strategic vision.",
        "examples": [
            "Describe a time when you mentored a junior engineer struggling with delivery.",
            "How do you drive consensus on adopting a new technical paradigm when there is resistance?",
            "Tell me about an initiative you led from conception to delivery across teams.",
        ],
    },
    {
        "id": "role-specific-skills",
        "name": "Role-Specific Skills",
        "color": TOPIC_COLORS["Role-Specific Skills"],
        "icon": "layers",
        "description": "Domain-specialized workflows including Frontend/React optimization, DevOps/Kubernetes/CI-CD, QA test automation, and Data Engineering.",
        "examples": [
            "How do you configure Kubernetes Horizontal Pod Autoscalers based on Prometheus metrics?",
            "How do you avoid unnecessary re-renders in React applications?",
            "What is the difference between Blue-Green deployments and Canary deployments?",
        ],
    },
]


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "models": {
            "tfidf_available": TFIDF_MODEL_PATH.exists(),
            "embedding_available": EMBEDDING_CLASSIFIER_PATH.exists(),
        },
    }


@app.get("/api/categories")
def get_categories():
    return {
        "categories": CATEGORY_METADATA,
        "classes": TOPIC_CATEGORIES,
        "colors": TOPIC_COLORS,
    }


@app.post("/api/predict")
def predict_single(req: PredictRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = predict_topic(
            req.question,
            model_type=req.model_type,
            threshold=req.threshold,
        )
        # Retrieve similar questions
        similar_items = find_similar_questions(req.question, top_k=4)
        result["similar_questions"] = similar_items
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch")
def predict_batch(req: BatchPredictRequest):
    if not req.questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty.")

    results = []
    flagged_count = 0
    total_conf = 0.0

    for q in req.questions:
        res = predict_topic(q, model_type=req.model_type, threshold=req.threshold)
        if res.get("needs_review"):
            flagged_count += 1
        total_conf += res.get("confidence", 0.0)
        results.append({
            "question": q,
            "predicted_topic": res.get("predicted_topic"),
            "confidence": res.get("confidence"),
            "confidence_pct": res.get("confidence_pct"),
            "needs_review": res.get("needs_review"),
        })

    avg_conf = (total_conf / len(req.questions)) if req.questions else 0.0

    return {
        "total": len(req.questions),
        "flagged_count": flagged_count,
        "average_confidence": round(avg_conf, 4),
        "average_confidence_pct": f"{avg_conf * 100:.1f}%",
        "results": results,
    }


@app.get("/api/metrics")
def get_metrics():
    if not EVALUATION_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Evaluation metrics file not found.")

    with open(EVALUATION_METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

    return {
        "metrics": metrics_data,
        "charts": {
            "confusion_matrix": "/reports/confusion_matrix.png",
            "confusion_matrix_embedding": "/reports/confusion_matrix_embedding.png",
            "model_comparison": "/reports/model_comparison.png",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
