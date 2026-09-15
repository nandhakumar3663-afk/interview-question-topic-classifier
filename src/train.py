"""
Training pipeline for EMP-26 Interview Question Topic Classifier.
Trains:
1. Baseline Model: TF-IDF Vectorizer + Logistic Regression.
2. Semantic Embedding Model: SentenceTransformer (all-MiniLM-L6-v2) + Calibrated Logistic Regression.
Saves model artifacts to models/ directory.
"""

import sys
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import (
    TOPIC_CATEGORIES,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
    TFIDF_MODEL_PATH,
    EMBEDDING_CLASSIFIER_PATH,
    TRAINING_METADATA_PATH,
    save_model,
    calculate_metrics,
    ensure_directories,
)
from src.preprocessing import clean_text


class EmbeddingClassifierWrapper:
    """
    Wrapper model that combines a SentenceTransformer encoder with a
    scikit-learn classifier (Logistic Regression with probability calibration).
    Provides scikit-learn compatible fit, predict, and predict_proba interfaces.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", C: float = 2.0, random_state: int = 42):
        self.model_name = model_name
        self.C = C
        self.random_state = random_state
        self.classes_ = np.array(TOPIC_CATEGORIES)
        self.encoder = None
        self.classifier = None
        self._is_fitted = False

    def _get_encoder(self):
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Embedding Encoder] Loading SentenceTransformer: {self.model_name}...")
                self.encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[Embedding Encoder Warning] Could not load SentenceTransformer ({e}).")
                raise e
        return self.encoder

    def encode(self, texts: list, show_progress: bool = True) -> np.ndarray:
        encoder = self._get_encoder()
        # Clean each text string
        cleaned_texts = [clean_text(t) for t in texts]
        embeddings = encoder.encode(
            cleaned_texts,
            batch_size=64,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def fit(self, texts: list, y: list):
        print(f"\n[EmbeddingClassifier] Generating embeddings for {len(texts)} training samples...")
        X_emb = self.encode(texts, show_progress=True)
        
        base_clf = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.classifier = CalibratedClassifierCV(estimator=base_clf, cv=3)
        print("[EmbeddingClassifier] Fitting calibrated logistic regression classifier...")
        self.classifier.fit(X_emb, y)
        self.classes_ = self.classifier.classes_
        self._is_fitted = True
        return self

    def predict(self, texts: list) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_emb = self.encode(texts, show_progress=False)
        return self.classifier.predict(X_emb)

    def predict_proba(self, texts: list) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_emb = self.encode(texts, show_progress=False)
        return self.classifier.predict_proba(X_emb)


def train_baseline_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train Baseline Model: TF-IDF (1-2 ngrams) + Logistic Regression.
    """
    print("\n" + "=" * 60)
    print("Training Baseline Model: TF-IDF + Logistic Regression")
    print("=" * 60)

    start_time = time.time()
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=(1, 2),
                min_df=2,
                max_features=12000,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=1.5,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])

    pipeline.fit(train_df["question"], train_df["label"])
    train_time = time.time() - start_time
    print(f"[Baseline] Training completed in {train_time:.2f}s")

    # Validation evaluation
    val_preds = pipeline.predict(val_df["question"])
    val_metrics = calculate_metrics(val_df["label"].tolist(), val_preds.tolist())
    val_metrics["training_time_seconds"] = train_time

    print(f"[Baseline Validation Accuracy]: {val_metrics['accuracy'] * 100:.2f}%")
    print(f"[Baseline Validation Macro F1]: {val_metrics['macro_f1']:.4f}")

    return pipeline, val_metrics


def train_embedding_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[EmbeddingClassifierWrapper, Dict[str, Any]]:
    """
    Train Final Semantic Candidate: SentenceTransformer + Calibrated Classifier.
    """
    print("\n" + "=" * 60)
    print("Training Semantic Embedding Model: all-MiniLM-L6-v2 + Calibrated Classifier")
    print("=" * 60)

    start_time = time.time()
    model = EmbeddingClassifierWrapper(model_name="all-MiniLM-L6-v2", C=2.0, random_state=42)
    model.fit(train_df["question"].tolist(), train_df["label"].tolist())
    train_time = time.time() - start_time
    print(f"[Embedding Model] Training completed in {train_time:.2f}s")

    # Validation evaluation
    val_preds = model.predict(val_df["question"].tolist())
    val_metrics = calculate_metrics(val_df["label"].tolist(), val_preds.tolist())
    val_metrics["training_time_seconds"] = train_time

    print(f"[Embedding Validation Accuracy]: {val_metrics['accuracy'] * 100:.2f}%")
    print(f"[Embedding Validation Macro F1]: {val_metrics['macro_f1']:.4f}")

    return model, val_metrics


def main():
    ensure_directories()

    if not TRAIN_SPLIT_PATH.exists() or not VAL_SPLIT_PATH.exists():
        print(f"[Error] Training splits not found. Please run scripts/generate_dataset.py first.")
        sys.exit(1)

    print(f"[Data Loader] Loading train set from {TRAIN_SPLIT_PATH}...")
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    val_df = pd.read_csv(VAL_SPLIT_PATH)
    print(f"  - Train size: {len(train_df)} rows")
    print(f"  - Validation size: {len(val_df)} rows")

    # 1. Train Baseline
    baseline_model, baseline_val_metrics = train_baseline_model(train_df, val_df)
    save_model(baseline_model, TFIDF_MODEL_PATH)

    # 2. Train Embedding Model
    try:
        embedding_model, embedding_val_metrics = train_embedding_model(train_df, val_df)
        save_model(embedding_model, EMBEDDING_CLASSIFIER_PATH)
    except Exception as e:
        print(f"\n[Warning] Embedding model training encountered: {e}")
        print("Sentence-transformers or torch might still be installing. You can run train.py once packages finish.")
        embedding_val_metrics = None

    # 3. Save training metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "classes": TOPIC_CATEGORIES,
        "baseline_val_metrics": baseline_val_metrics,
        "embedding_val_metrics": embedding_val_metrics,
    }
    with open(TRAINING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Metadata Saved] Written to {TRAINING_METADATA_PATH}")
    print("[Training Complete] All available models trained and serialized.")


if __name__ == "__main__":
    main()
