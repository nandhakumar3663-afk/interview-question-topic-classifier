"""
Comprehensive evaluation script for EMP-26 Interview Question Topic Classifier.
Evaluates trained models on the held-out test split:
- Calculates Accuracy, Precision, Recall, F1 (Macro & Weighted).
- Produces per-class classification reports.
- Generates and saves Confusion Matrix heatmap (reports/confusion_matrix.png).
- Generates and saves Model Comparison visualization (reports/model_comparison.png).
- Performs error analysis on misclassified & low-confidence cases.
- Exports metrics to reports/evaluation_metrics.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import (
    TOPIC_CATEGORIES,
    TEST_SPLIT_PATH,
    TFIDF_MODEL_PATH,
    EMBEDDING_CLASSIFIER_PATH,
    CONFUSION_MATRIX_PATH,
    MODEL_COMPARISON_PATH,
    EVALUATION_METRICS_PATH,
    REPORTS_DIR,
    calculate_metrics,
    load_model,
    ensure_directories,
)
from src.preprocessing import clean_text


def plot_confusion_matrix(y_true: list, y_pred: list, model_name: str, output_path: Path):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=TOPIC_CATEGORIES)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TOPIC_CATEGORIES,
        yticklabels=TOPIC_CATEGORIES,
        cbar=False,
        ax=ax,
        linewidths=0.5,
    )

    # Annotate with percentages
    for i in range(len(TOPIC_CATEGORIES)):
        for j in range(len(TOPIC_CATEGORIES)):
            percentage = cm_norm[i, j] * 100
            ax.text(
                j + 0.5,
                i + 0.75,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                color="navy" if cm[i, j] < cm.max() / 2 else "white",
                fontsize=8,
            )

    plt.title(f"Confusion Matrix: {model_name} (Held-out Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Topic Label", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("True Topic Label", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to {output_path}")


def plot_model_comparison(results_summary: Dict[str, Dict[str, float]], output_path: Path):
    """Plot comparative bar chart across evaluation metrics for all tested models."""
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    metric_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    
    models = list(results_summary.keys())
    x = np.arange(len(metrics))
    width = 0.35 if len(models) == 2 else 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563EB", "#7C3AED", "#10B981"]

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        values = [results_summary[model].get(m, 0.0) * 100 for m in metrics]
        rects = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)], alpha=0.9, edgecolor="black")
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("EMP-26 Model Performance Comparison (Held-out Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved model comparison chart to {output_path}")


def analyze_errors(test_df: pd.DataFrame, y_pred: list, probs: np.ndarray, top_n: int = 8) -> pd.DataFrame:
    """Identify misclassified questions and calculate confidence gaps."""
    df_analysis = test_df.copy()
    df_analysis["predicted"] = y_pred
    df_analysis["is_correct"] = df_analysis["label"] == df_analysis["predicted"]
    df_analysis["max_confidence"] = np.max(probs, axis=1) if probs is not None else 1.0

    # Misclassified examples
    errors = df_analysis[~df_analysis["is_correct"]].copy()
    errors = errors.sort_values(by="max_confidence", ascending=False)
    return errors.head(top_n)


def main():
    ensure_directories()

    if not TEST_SPLIT_PATH.exists():
        print(f"[Error] Test set not found at {TEST_SPLIT_PATH}. Please generate dataset first.")
        sys.exit(1)

    print(f"[Evaluation] Loading test split from {TEST_SPLIT_PATH}...")
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    y_true = test_df["label"].tolist()
    questions = test_df["question"].tolist()
    print(f"  - Test dataset size: {len(test_df)} samples")

    results_summary = {}
    eval_details = {}

    # 1. Evaluate Baseline Model
    if TFIDF_MODEL_PATH.exists():
        print("\n" + "=" * 60)
        print("Evaluating Baseline Model (TF-IDF + Logistic Regression)")
        print("=" * 60)
        baseline_model = load_model(TFIDF_MODEL_PATH)
        y_pred_baseline = baseline_model.predict(questions).tolist()
        probs_baseline = baseline_model.predict_proba(questions) if hasattr(baseline_model, "predict_proba") else None

        metrics_baseline = calculate_metrics(y_true, y_pred_baseline)
        results_summary["TF-IDF Baseline"] = metrics_baseline
        eval_details["TF-IDF Baseline"] = {
            "metrics": metrics_baseline,
            "classification_report_str": classification_report(y_true, y_pred_baseline, labels=TOPIC_CATEGORIES),
        }

        print(f"Accuracy:        {metrics_baseline['accuracy'] * 100:.2f}%")
        print(f"Macro F1-Score:  {metrics_baseline['macro_f1']:.4f}")
        print(f"Weighted F1:     {metrics_baseline['weighted_f1']:.4f}")
        print("\nClass-wise Report:")
        print(eval_details["TF-IDF Baseline"]["classification_report_str"])

        # Plot confusion matrix for baseline
        plot_confusion_matrix(y_true, y_pred_baseline, "TF-IDF Baseline", CONFUSION_MATRIX_PATH)

        # Error analysis
        errors_baseline = analyze_errors(test_df, y_pred_baseline, probs_baseline)
        print("\nSample Misclassifications (TF-IDF Baseline):")
        for _, row in errors_baseline.head(4).iterrows():
            print(f"  True: [{row['label']}] | Pred: [{row['predicted']}] (Conf: {row['max_confidence']:.2f})")
            print(f"    Q: {row['question'][:90]}...")
    else:
        print(f"[Warning] Baseline model not found at {TFIDF_MODEL_PATH}")

    # 2. Evaluate Embedding Model (if available)
    if EMBEDDING_CLASSIFIER_PATH.exists():
        print("\n" + "=" * 60)
        print("Evaluating Semantic Model (SentenceTransformer + Calibrated Classifier)")
        print("=" * 60)
        try:
            emb_model = load_model(EMBEDDING_CLASSIFIER_PATH)
            y_pred_emb = emb_model.predict(questions).tolist()
            probs_emb = emb_model.predict_proba(questions)

            metrics_emb = calculate_metrics(y_true, y_pred_emb)
            results_summary["Sentence Transformer"] = metrics_emb
            eval_details["Sentence Transformer"] = {
                "metrics": metrics_emb,
                "classification_report_str": classification_report(y_true, y_pred_emb, labels=TOPIC_CATEGORIES),
            }

            print(f"Accuracy:        {metrics_emb['accuracy'] * 100:.2f}%")
            print(f"Macro F1-Score:  {metrics_emb['macro_f1']:.4f}")
            print(f"Weighted F1:     {metrics_emb['weighted_f1']:.4f}")
            print("\nClass-wise Report:")
            print(eval_details["Sentence Transformer"]["classification_report_str"])

            # Save embedding confusion matrix
            plot_confusion_matrix(y_true, y_pred_emb, "SentenceTransformer Embedding", REPORTS_DIR / "confusion_matrix_embedding.png")
        except Exception as e:
            print(f"[Warning] Could not evaluate embedding model: {e}")

    # 3. Model Comparison Visualization
    if results_summary:
        plot_model_comparison(results_summary, MODEL_COMPARISON_PATH)

    # 4. Save JSON metrics
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(eval_details, f, indent=2)
    print(f"\n[Evaluation Complete] Saved all metrics and charts to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
