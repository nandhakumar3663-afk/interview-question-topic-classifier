"""
Inference and prediction module for EMP-26 Interview Question Topic Classifier.
Supports:
- Single question topic prediction with confidence probability distribution.
- Dynamic confidence thresholding with human review flagging.
- Bonus Feature: Cosine-similarity-based similar-question retrieval.
- CLI interface for fast terminal testing.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import (
    TOPIC_CATEGORIES,
    TFIDF_MODEL_PATH,
    EMBEDDING_CLASSIFIER_PATH,
    TRAIN_SPLIT_PATH,
    load_model,
)
from src.preprocessing import clean_text

# Cached instances for high-speed inference
_CACHED_MODELS = {}
_QUESTION_RETRIEVAL_CACHE = {}


def get_model(model_type: str = "tfidf"):
    """Load and cache requested model."""
    model_type = model_type.lower()
    if model_type in _CACHED_MODELS:
        return _CACHED_MODELS[model_type]

    if model_type in ("tfidf", "baseline"):
        if not TFIDF_MODEL_PATH.exists():
            raise FileNotFoundError(f"TF-IDF model not found at {TFIDF_MODEL_PATH}. Run src/train.py first.")
        model = load_model(TFIDF_MODEL_PATH)
        _CACHED_MODELS["tfidf"] = model
        return model

    elif model_type in ("embedding", "final", "semantic"):
        if not EMBEDDING_CLASSIFIER_PATH.exists():
            # Fallback to TF-IDF if embedding model hasn't been trained yet
            print(f"[Warning] Embedding model not found at {EMBEDDING_CLASSIFIER_PATH}. Falling back to TF-IDF.")
            return get_model("tfidf")
        model = load_model(EMBEDDING_CLASSIFIER_PATH)
        _CACHED_MODELS["embedding"] = model
        return model

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'tfidf' or 'embedding'.")


def get_retrieval_corpus():
    """Load historical question database and vectorizer for similarity retrieval."""
    if "corpus" in _QUESTION_RETRIEVAL_CACHE:
        return _QUESTION_RETRIEVAL_CACHE["corpus"], _QUESTION_RETRIEVAL_CACHE["vectorizer"], _QUESTION_RETRIEVAL_CACHE["matrix"]

    if not TRAIN_SPLIT_PATH.exists():
        return None, None, None

    df = pd.read_csv(TRAIN_SPLIT_PATH)
    vectorizer = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), max_features=10000)
    matrix = vectorizer.fit_transform(df["question"])

    _QUESTION_RETRIEVAL_CACHE["corpus"] = df
    _QUESTION_RETRIEVAL_CACHE["vectorizer"] = vectorizer
    _QUESTION_RETRIEVAL_CACHE["matrix"] = matrix
    return df, vectorizer, matrix


def predict_topic(
    question: str,
    model_type: str = "tfidf",
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    Predict the topic category for an interview question.
    Returns:
      - question: Original input
      - cleaned_question: Preprocessed normalized text
      - predicted_topic: Winning class name
      - confidence: Float in [0.0, 1.0]
      - confidence_pct: Percentage string (e.g. '88.5%')
      - probabilities: Dict mapping each category to probability float
      - needs_review: True if confidence < threshold
      - threshold: Threshold applied
      - model_used: Model identifier
    """
    if not question or not str(question).strip():
        return {
            "error": "Input question is empty or invalid.",
            "predicted_topic": "Unknown",
            "confidence": 0.0,
            "confidence_pct": "0.0%",
            "probabilities": {cat: 0.0 for cat in TOPIC_CATEGORIES},
            "needs_review": True,
            "threshold": threshold,
        }

    model = get_model(model_type)
    cleaned = clean_text(question)

    # Obtain probability distribution
    if hasattr(model, "predict_proba"):
        probs_array = model.predict_proba([cleaned])[0]
        classes = list(model.classes_)
    else:
        # Fallback if model has decision function or only predict
        pred_label = model.predict([cleaned])[0]
        probs_array = [1.0 if c == pred_label else 0.0 for c in TOPIC_CATEGORIES]
        classes = TOPIC_CATEGORIES

    # Map class probabilities
    prob_dict = {str(c): float(p) for c, p in zip(classes, probs_array)}
    # Ensure all 5 categories are present
    for cat in TOPIC_CATEGORIES:
        if cat not in prob_dict:
            prob_dict[cat] = 0.0

    # Best prediction
    best_class = max(prob_dict, key=prob_dict.get)
    best_confidence = prob_dict[best_class]
    needs_review = best_confidence < threshold

    return {
        "question": question,
        "cleaned_question": cleaned,
        "predicted_topic": best_class,
        "confidence": float(best_confidence),
        "confidence_pct": f"{best_confidence * 100:.1f}%",
        "probabilities": prob_dict,
        "needs_review": needs_review,
        "threshold": threshold,
        "model_used": "TF-IDF + Logistic Regression" if "tfidf" in model_type else "SentenceTransformer Embedding",
    }


def find_similar_questions(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Bonus Feature: Retrieve top_k most semantically similar historical
    questions from the training corpus using cosine similarity.
    """
    corpus_df, vectorizer, matrix = get_retrieval_corpus()
    if corpus_df is None or vectorizer is None or matrix is None:
        return []

    cleaned_query = clean_text(query)
    query_vec = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vec, matrix)[0]

    # Get top_k indices sorted descending
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sim_score = float(similarities[idx])
        if sim_score > 0.02:  # Filter out totally irrelevant matches
            row = corpus_df.iloc[idx]
            results.append({
                "question": row["question"],
                "topic": row["label"],
                "similarity_score": round(sim_score, 4),
                "similarity_pct": f"{sim_score * 100:.1f}%",
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Classify an interview question into one of 5 topics.")
    parser.add_argument("--question", "-q", type=str, required=True, help="Text of the interview question")
    parser.add_argument("--model", "-m", type=str, default="tfidf", choices=["tfidf", "embedding"], help="Model type to use")
    parser.add_argument("--threshold", "-t", type=float, default=0.65, help="Confidence threshold for human review")
    parser.add_argument("--similar", "-s", action="store_true", help="Retrieve similar historical questions")

    args = parser.parse_args()

    result = predict_topic(args.question, model_type=args.model, threshold=args.threshold)
    print("\n" + "=" * 60)
    print("EMP-26 Topic Classification Result")
    print("=" * 60)
    print(f"Question:         {result['question']}")
    print(f"Model Used:       {result['model_used']}")
    print(f"Predicted Topic:  {result['predicted_topic']}")
    print(f"Confidence:       {result['confidence_pct']} (score: {result['confidence']:.4f})")
    print(f"Needs Review:     {'[YES - Below Threshold]' if result['needs_review'] else '[NO - High Confidence]'}")
    print("\nProbability Distribution:")
    for cat, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob * 30)
        print(f"  {cat:<22} : {prob * 100:5.1f}% | {bar}")

    if args.similar:
        print("\nSimilar Historical Questions:")
        similar_items = find_similar_questions(args.question, top_k=4)
        if not similar_items:
            print("  No relevant historical questions found.")
        else:
            for i, item in enumerate(similar_items, 1):
                print(f"  {i}. [{item['similarity_pct']}] ({item['topic']}) {item['question']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
