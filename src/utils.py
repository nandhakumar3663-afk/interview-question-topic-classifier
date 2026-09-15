"""
Utility constants, helper functions, and shared path configurations
for the EMP-26 Interview Question Topic Classifier.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 5 Required Classification Categories
TOPIC_CATEGORIES: List[str] = [
    "Technical Knowledge",
    "Communication",
    "Problem Solving",
    "Leadership",
    "Role-Specific Skills",
]

# Color palette mapped to the 5 categories (consistent across plots and UI)
TOPIC_COLORS: Dict[str, str] = {
    "Technical Knowledge": "#2563EB",   # Vibrant Blue
    "Communication": "#0284C7",        # Ocean Sky
    "Problem Solving": "#7C3AED",      # Violet
    "Leadership": "#059669",           # Emerald Green
    "Role-Specific Skills": "#D97706", # Warm Amber
}

# Path definitions
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DATA_DIR = DATA_DIR / "splits"

RAW_QUESTIONS_PATH = RAW_DATA_DIR / "raw_questions.csv"
LABELED_QUESTIONS_PATH = PROCESSED_DATA_DIR / "labeled_questions.csv"
TRAIN_SPLIT_PATH = SPLITS_DATA_DIR / "train.csv"
VAL_SPLIT_PATH = SPLITS_DATA_DIR / "validation.csv"
TEST_SPLIT_PATH = SPLITS_DATA_DIR / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_model.joblib"
EMBEDDING_CLASSIFIER_PATH = MODELS_DIR / "embedding_classifier.joblib"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.png"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"


def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        NOTEBOOKS_DIR,
        APP_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filepath: Path) -> None:
    """Save serialized model object using joblib."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[Model Saved] Successfully saved model to {filepath}")


def load_model(filepath: Path) -> Any:
    """Load serialized model object from joblib file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    import __main__
    try:
        from src.train import EmbeddingClassifierWrapper
        __main__.EmbeddingClassifierWrapper = EmbeddingClassifierWrapper
    except Exception:
        pass
    return joblib.load(filepath)


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, labels=TOPIC_CATEGORIES, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "classification_report": report_dict,
    }
