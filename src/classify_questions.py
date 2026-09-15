"""
Two-Stage Semantic Classification & Confidence Thresholding for EMP-26.
Stage 1: Fast candidate domain and heuristic prior filtering using metadata, tags, and linguistic cues.
Stage 2: Dense transformer semantic classification using sentence-transformers (all-MiniLM-L6-v2).

Enforces strict confidence thresholding:
- confidence >= 0.75: Accepted automatically
- 0.50 <= confidence < 0.75: Routed to review queue
- confidence < 0.50: Flagged as 'AMBIGUOUS' -> data/review/ambiguous_questions.csv
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

CLEANED_DIR = PROJECT_ROOT / CONFIG["paths"]["data_cleaned"]
LABELED_DIR = PROJECT_ROOT / CONFIG["paths"]["data_labeled"]
REVIEW_DIR = PROJECT_ROOT / CONFIG["paths"]["data_review"]
CACHE_DIR = PROJECT_ROOT / CONFIG["paths"]["data_cache"]

LABELED_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CONF_ACCEPT = CONFIG["thresholds"]["confidence_accept"]
CONF_REVIEW_MIN = CONFIG["thresholds"]["confidence_review_min"]
BATCH_SIZE = CONFIG["models"]["batch_size"]
EMBEDDING_MODEL_NAME = CONFIG["models"]["embedding_model"]

CATEGORIES = [
    "Technical Knowledge",
    "Communication",
    "Problem Solving",
    "Leadership",
    "Role-Specific Skills",
]

# Canonical prototypes defining the core semantic centroid of each category
CLASS_PROTOTYPES = {
    "Technical Knowledge": [
        "What is database indexing and how do B-tree indexes work under the hood?",
        "Explain the difference between synchronous and asynchronous processing in operating systems.",
        "What are the ACID properties in relational database management systems?",
        "How does virtual memory and paging work in modern computer architecture?",
        "Explain polymorphism, encapsulation, and inheritance in object-oriented programming.",
        "What is the difference between TCP and UDP networking transport protocols?",
        "Explain how memory allocation and garbage collection function in runtime environments.",
        "What is the CAP theorem in distributed systems design?",
    ],
    "Communication": [
        "How do you explain technical concepts and architecture to non-technical business stakeholders?",
        "Tell me about a difficult conversation you had with a product manager and how you reached consensus.",
        "How do you deliver constructive, actionable code review feedback to peer engineers?",
        "Describe a time when you had to persuade executive leadership to allocate budget for technical debt.",
        "How do you ensure clear communication across distributed cross-functional engineering teams?",
        "Describe an experience where a miscommunication caused a bug and how you resolved it.",
        "What strategies do you use for active listening when a colleague expresses strong disagreement?",
    ],
    "Problem Solving": [
        "Tell me about a time when you diagnosed and fixed a critical memory leak in production.",
        "Walk me through your step-by-step troubleshooting process when 504 gateway timeouts spike.",
        "How do you solve complex algorithmic problems and optimize time and space complexity?",
        "Describe a situation where a third-party dependency failed unexpectedly and how you recovered.",
        "How do you investigate intermittent race conditions and thread lock contention under high load?",
        "Tell me about a time your initial debugging hypothesis was incorrect and how you found root cause.",
        "How do you profile application bottlenecks when CPU utilization reaches 100 percent?",
    ],
    "Leadership": [
        "Describe a time when you mentored a struggling junior developer and supported their career growth.",
        "How do you resolve fierce architectural disagreements between senior engineers on your team?",
        "Tell me about an engineering initiative you led from conception to delivery across teams.",
        "How do you establish code quality standards, test rigor, and engineering excellence across a team?",
        "Describe how you manage an underperforming team member with empathy and constructive guidance.",
        "How do you keep team morale, ownership, and velocity high during difficult organizational restructuring?",
        "What is your philosophy on servant leadership, delegation, and empowerment versus micro-management?",
    ],
    "Role-Specific Skills": [
        "How do you configure Kubernetes Horizontal Pod Autoscalers based on custom Prometheus metrics?",
        "How do you avoid unnecessary component re-renders in React and optimize client-side bundle size?",
        "What are the differences between Blue-Green deployments and Canary rollouts in cloud DevOps?",
        "How do you design an idempotent ELT data pipeline using Apache Spark and Snowflake?",
        "How do you configure Terraform state locking and module hierarchies for multi-account AWS?",
        "What best practices do you follow for writing end-to-end integration tests using Playwright?",
        "How do you implement OAuth2 Authorization Code Flow with PKCE in modern web applications?",
    ],
}

_ENCODER = None
_CENTROIDS = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ENCODER = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _ENCODER


def get_class_centroids() -> Dict[str, np.ndarray]:
    """Compute normalized semantic centroid vectors for the 5 categories."""
    global _CENTROIDS
    if _CENTROIDS is not None:
        return _CENTROIDS

    encoder = get_encoder()
    _CENTROIDS = {}
    print("[Classifier] Initializing semantic class centroids from canonical prototypes...")
    for cat, examples in CLASS_PROTOTYPES.items():
        embs = encoder.encode(examples, normalize_embeddings=True, show_progress_bar=False)
        centroid = np.mean(embs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        _CENTROIDS[cat] = centroid
    return _CENTROIDS


def estimate_difficulty(text: str) -> str:
    """Heuristic difficulty estimation based on technical complexity keywords."""
    t = text.lower()
    if any(k in t for k in ["distributed", "deadlock", "under the hood", "architecture", "internals", "trade-off", "concurrency", "low-level", "zero-downtime"]):
        return "Hard"
    elif any(k in t for k in ["optimize", "troubleshoot", "difference between", "implement", "configure", "debug"]):
        return "Medium"
    else:
        return "Easy"


def estimate_domain(text: str, tags: str = "") -> str:
    """Assign specific engineering domain."""
    combined = (text + " " + str(tags)).lower()
    if any(k in combined for k in ["react", "css", "html", "vue", "frontend", "dom", "ui", "browser"]):
        return "Frontend Development"
    elif any(k in combined for k in ["kubernetes", "docker", "k8s", "devops", "ci/cd", "helm", "terraform", "ansible"]):
        return "DevOps"
    elif any(k in combined for k in ["aws", "cloud", "azure", "gcp", "serverless", "lambda", "s3"]):
        return "Cloud Engineering"
    elif any(k in combined for k in ["sql", "postgres", "database", "mongodb", "mysql", "redis", "indexing"]):
        return "Database Engineering"
    elif any(k in combined for k in ["spark", "kafka", "pipeline", "etl", "snowflake", "data engineering", "bigquery"]):
        return "Data Science"
    elif any(k in combined for k in ["test", "selenium", "playwright", "cypress", "qa", "unit test", "mock"]):
        return "Software Testing"
    elif any(k in combined for k in ["security", "oauth", "jwt", "owasp", "vulnerability", "encryption", "auth"]):
        return "Cybersecurity"
    elif any(k in combined for k in ["algorithm", "tree", "graph", "binary search", "dsa", "complexity", "big o"]):
        return "Backend Development"
    else:
        return "Full Stack Development"


def calculate_quality_score(text: str) -> float:
    """Calculate heuristic linguistic quality score (0.0 to 1.0)."""
    score = 0.5
    if 30 <= len(text) <= 300:
        score += 0.25
    if text.endswith("?"):
        score += 0.15
    if any(text.startswith(w) for w in ["How", "What", "Explain", "Describe", "Tell", "Can", "Why"]):
        score += 0.10
    return min(round(score, 2), 1.0)


def compute_heuristic_priors(text: str) -> np.ndarray:
    """Stage 1: Fast linguistic cues and domain keyword prior filtering."""
    t = text.lower()
    priors = np.zeros(5, dtype=np.float32)
    # 0: Technical Knowledge, 1: Communication, 2: Problem Solving, 3: Leadership, 4: Role-Specific Skills

    # Leadership
    if any(k in t for k in ["mentor", "lead ", "led a", "leadership", "spearhead", "underperforming", "initiative", "foster culture", "coached", "incident commander"]):
        priors[3] += 0.25

    # Communication
    if any(k in t for k in ["communicate", "articulate", "non-technical", "active listening", "code review feedback", "dispute", "persuade", "explain to", "rfc", "stakeholders"]):
        priors[1] += 0.25

    # Problem Solving
    if any(k in t for k in ["troubleshoot", "debug", "root cause", "diagnose", "memory leak", "race condition", "algorithm", "given an", "find the", "solve"]):
        priors[2] += 0.25

    # Technical Knowledge
    if any(k in t for k in ["what is ", "what are ", "explain how ", "difference between", "acid properties", "cap theorem", "polymorphism", "virtual memory", "garbage collection"]):
        priors[0] += 0.25

    # Role-Specific Skills
    if any(k in t for k in ["react", "kubernetes", "k8s", "docker", "terraform", "ci/cd", "playwright", "cypress", "aws", "postgres", "redis", "kafka", "graphql", "microservices"]):
        priors[4] += 0.25

    return priors


def classify_candidate_batch(texts: List[str]) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
    """
    Stage 2: Dense semantic classification combined with Stage 1 heuristic priors.
    """
    encoder = get_encoder()
    centroids = get_class_centroids()
    centroid_matrix = np.array([centroids[cat] for cat in CATEGORIES])  # (5, D)

    # Encode questions
    embs = encoder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)  # (N, D)
    
    # Cosine similarities
    semantic_sims = np.dot(embs, centroid_matrix.T)  # (N, 5)

    # Compute heuristic priors for each question
    heuristic_priors = np.array([compute_heuristic_priors(t) for t in texts])  # (N, 5)

    # Combined score
    combined_scores = semantic_sims + (0.4 * heuristic_priors)

    # Softmax with temperature
    temperature = 0.05
    exp_sims = np.exp(combined_scores / temperature)
    probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)  # (N, 5)

    predicted_labels = []
    confidences = []
    prob_dicts = []

    for i in range(len(texts)):
        p_row = probs[i]
        top_idx = int(np.argmax(p_row))
        conf = float(p_row[top_idx])
        label = CATEGORIES[top_idx]

        predicted_labels.append(label)
        confidences.append(round(conf, 4))
        prob_dicts.append({CATEGORIES[j]: round(float(p_row[j]), 4) for j in range(5)})

    return predicted_labels, confidences, prob_dicts


def classify_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full two-stage classification and confidence routing.
    """
    print(f"\n[Classification] Processing {len(df)} candidate questions...")
    texts = df["cleaned_question"].tolist()
    total = len(texts)
    
    batch_size = 500
    all_preds = []
    all_confs = []

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        preds, confs, _ = classify_candidate_batch(batch_texts)
        all_preds.extend(preds)
        all_confs.extend(confs)
        if (i // batch_size) % 5 == 0 or i + batch_size >= total:
            print(f"  - Classified {min(i + batch_size, total)} / {total} questions...")

    df["predicted_label"] = all_preds
    df["classification_confidence"] = all_confs

    # Route based on confidence thresholds
    accepted_mask = df["classification_confidence"] >= CONF_ACCEPT
    review_mask = (df["classification_confidence"] >= CONF_REVIEW_MIN) & (~accepted_mask)
    ambiguous_mask = df["classification_confidence"] < CONF_REVIEW_MIN

    print(f"\n[Thresholding Summary]")
    print(f"  - Accepted (>= {CONF_ACCEPT}): {accepted_mask.sum()} samples")
    print(f"  - Review Queue ({CONF_REVIEW_MIN} - {CONF_ACCEPT}): {review_mask.sum()} samples")
    print(f"  - Ambiguous (< {CONF_REVIEW_MIN}): {ambiguous_mask.sum()} samples")

    # Ambiguous & review records
    review_df = df[~accepted_mask].copy()
    review_df["reason"] = np.where(
        review_df["classification_confidence"] < CONF_REVIEW_MIN,
        "Low semantic confidence (<0.50)",
        "Moderate confidence review queue (0.50-0.75)"
    )
    ambiguous_path = REVIEW_DIR / "ambiguous_questions.csv"
    review_df[["cleaned_question", "predicted_label", "classification_confidence", "source", "reason"]].to_csv(ambiguous_path, index=False)
    print(f"  - Saved ambiguous and review items to {ambiguous_path}")

    # Format accepted master candidate records
    accepted_df = df[accepted_mask].copy()
    accepted_df["label"] = accepted_df["predicted_label"]
    accepted_df["is_synthetic"] = False
    accepted_df["difficulty"] = accepted_df["cleaned_question"].apply(estimate_difficulty)
    accepted_df["domain"] = [estimate_domain(q, t) for q, t in zip(accepted_df["cleaned_question"], accepted_df.get("raw_tags", ""))]
    accepted_df["quality_score"] = accepted_df["cleaned_question"].apply(calculate_quality_score)
    accepted_df["question"] = accepted_df["cleaned_question"]

    return accepted_df.reset_index(drop=True), review_df.reset_index(drop=True)


def run_classification():
    input_file = CLEANED_DIR / "deduplicated_candidates.parquet"
    if not input_file.exists():
        print(f"[Error] File not found at {input_file}. Run deduplicate stage first.")
        sys.exit(1)

    df = pd.read_parquet(input_file)
    accepted_df, _ = classify_dataset(df)

    output_path = LABELED_DIR / "candidates_labeled.parquet"
    accepted_df.to_parquet(output_path, index=False)
    print(f"\n[Saved] Labeled candidates: {len(accepted_df)} samples -> {output_path}")


if __name__ == "__main__":
    run_classification()
