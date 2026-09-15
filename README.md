# EMP-26: Interview Question Topic Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

An end-to-end lightweight, CPU-friendly Machine Learning system that classifies software engineering interview questions into 5 core competencies with confidence scoring, low-confidence human review flagging, semantic similar-question retrieval, and an interactive Streamlit application.

---

## 🎯 Classification Categories (5 Target Classes)

The classifier assigns any free-text interview question to exactly one of the 5 required categories:

| Category | Description | Sample Questions |
|---|---|---|
| **Technical Knowledge** | Core CS fundamentals, system design, databases, concurrency, networking, algorithms. | *"How do indexes work under the hood in PostgreSQL?"*<br>*"What is the difference between TCP and UDP?"* |
| **Communication** | Explaining technical concepts to non-technical stakeholders, feedback, conflict resolution. | *"How do you explain technical debt to non-technical executives?"*<br>*"How do you give constructive code review feedback?"* |
| **Problem Solving** | Production debugging, root cause analysis, incident response, troubleshooting under pressure. | *"Tell me about a time you diagnosed a severe memory leak in production."*<br>*"How do you troubleshoot intermittent 504 gateway errors?"* |
| **Leadership** | Mentorship, setting engineering culture, leading initiatives, managing underperformance. | *"Describe a time when you mentored a struggling junior developer."*<br>*"How do you drive consensus on architectural choices?"* |
| **Role-Specific Skills** | Domain-specialized tooling (Frontend/React, DevOps/Kubernetes/CI-CD, Data Engineering/Spark). | *"How do you configure Kubernetes Horizontal Pod Autoscaling?"*<br>*"How do you avoid unnecessary re-renders in React?"* |

---

## 🏗️ Architecture & Pipeline Flowchart

```
[1. User Question] 
       │
       ▼
[2. Data / Preprocessing] ──► Token Preservation (C++, CI/CD, K8s, O(1)) ──► Deduplication & Normalization
       │
       ▼
[3. Representation]      ──► TF-IDF Baseline (1-2 ngrams) OR Sentence Transformer (all-MiniLM-L6-v2)
       │
       ▼
[4. Classifier]          ──► Logistic Regression / Calibrated Probability Classifier
       │
       ▼
[5. Output & Delivery]   ──► Topic + Confidence Score (0-100%)
                         ──► Human Review Flag (if Confidence < Threshold)
                         ──► Similar Historical Questions (Cosine Similarity)
                         ──► Streamlit Web UI + Batch CSV Upload / Export
```

---

## 📁 Repository Structure

```
interview-question-topic-classifier/
+-- data/
|   +-- raw/
|   |   `-- raw_questions.csv             # 3,600 generated questions (720 per category)
|   +-- processed/
|   |   `-- labeled_questions.csv         # Cleaned and deduplicated dataset
|   `-- splits/
|       +-- train.csv                     # 70% stratified training partition
|       +-- validation.csv                # 15% stratified validation partition
|       `-- test.csv                      # 15% stratified test partition
+-- notebooks/
|   +-- 01_data_exploration.ipynb         # Class balance, token lengths, vocab distribution
|   +-- 02_preprocessing.ipynb            # Technical token preservation & cleaning demo
|   +-- 03_baseline_model.ipynb           # TF-IDF + Logistic Regression baseline
|   +-- 04_embedding_model.ipynb          # Sentence Transformers & calibrated classifier
|   `-- 05_evaluation.ipynb               # Test evaluation, confusion matrix & error analysis
+-- src/
|   +-- preprocessing.py                  # Token preservation, normalizer, cleaning functions
|   +-- train.py                          # Training pipeline for baseline and embedding models
|   +-- predict.py                        # Inference engine, review flagger & similar retrieval
|   +-- evaluate.py                       # Evaluation metrics, confusion matrix & charts
|   `-- utils.py                          # Paths, topic categories, constants, metrics helpers
+-- models/
|   +-- tfidf_model.joblib                # Serialized TF-IDF baseline model artifact
|   +-- embedding_classifier.joblib       # Serialized Sentence Transformer model artifact
|   `-- training_metadata.json            # Model parameters and validation metrics
+-- app/
|   `-- app.py                            # Interactive Streamlit Web UI
+-- reports/
|   +-- confusion_matrix.png              # Normalized confusion matrix heatmap
|   +-- model_comparison.png              # Benchmark metrics comparison bar chart
|   `-- evaluation_metrics.json           # JSON export of test metrics & reports
+-- tests/
|   `-- test_classifier.py                # Automated unit and integration tests
+-- requirements.txt                      # Project dependencies
`-- README.md                             # Comprehensive project documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Create virtual environment (Python 3.10+)
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset & Partitions (Data Layer)

Generates 3,600 balanced questions (720 per category), deduplicates, cleans, and creates 70/15/15 stratified splits:

```bash
python scripts/generate_dataset.py
```

### 3. Train Models (ML Layer)

Trains both the lightweight TF-IDF baseline and the semantic embedding model, saving artifacts to `models/`:

```bash
python src/train.py
```

### 4. Evaluate Models (Evaluation Layer)

Evaluates performance on the held-out 15% test set, generates confusion matrices and comparison charts in `reports/`:

```bash
python src/evaluate.py
```

### 5. Run Web Applications (Deployment Layer)

You have two interactive deployment interfaces available:

#### A. Modern React + FastAPI Web Application (Recommended)
Launch both the high-throughput FastAPI backend (port 8000) and the modern React frontend (port 5173):

```bash
# Unified launcher (starts backend + frontend)
python scripts/run_app.py
```

Or run them individually in separate terminals:
```bash
# Terminal 1: Start FastAPI REST API Server
python -m uvicorn server.api:app --host 127.0.0.1 --port 8000

# Terminal 2: Start React Frontend
cd frontend
npm run dev
```

Open `http://localhost:5173` to access the ultra-clean React dashboard featuring:
- **Interactive Classifier**: Real-time keystroke predictions, category badge gradients, animated confidence score rings.
- **Review Alert System**: Visual warning banners when confidence drops below user-selected review thresholds.
- **Similar Questions Explorer**: Cosine-similarity nearest neighbor retrieval with match percentages.
- **Bulk Batch Testing**: High-throughput multi-line question testing with real-time statistics and CSV export.
- **Model Evaluation Dashboard**: Embedded visual benchmark comparison and confusion matrix inspectors.
- **Competency Guide**: Comprehensive reference for all 5 topics with real-world examples.
- **Theme Switcher**: Polished light and dark mode toggling.

#### B. Streamlit Application
Launch the standalone Streamlit app on port 8501:

```bash
streamlit run app/app.py
```

---

## 💻 CLI Inference & Testing

Test single questions directly from the command line:

```bash
# Basic classification
python src/predict.py --question "How do you resolve race conditions and deadlocks in PostgreSQL?"

# With similar-question retrieval and customized review threshold
python src/predict.py --question "Describe a time when you had to manage an underperforming engineer." --threshold 0.70 --similar
```

---

## 🧪 Automated Testing

Run the automated test suite with pytest:

```bash
pytest tests/
```

---

## 📊 Practical Extras & Key Features

1. **Token Preservation**: Custom regex mappings preserve critical symbols like `C++`, `C#`, `.NET`, `CI/CD`, `k8s`, and `O(1)` so technical intent is never stripped during cleaning.
2. **Confidence Threshold & Review Flag**: Any prediction below the user-specified threshold (default 65%) is visually flagged for human review.
3. **Similar-Question Retrieval**: Vector cosine similarity over the training questions retrieves the 4 most relevant historical questions to provide interpretability and context.
4. **Batch CSV Processing**: Upload a CSV of unlabelled interview questions, run high-throughput batch classification, and download the tagged CSV.