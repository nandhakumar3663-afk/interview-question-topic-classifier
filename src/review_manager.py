"""
Review Manager for EMP-26 Dataset Generation Pipeline.
Parses, audits, and generates actionable reports for low-confidence (<0.50)
and borderline review queue (0.50-0.75) questions generated during classification.
Produces summary statistics, category confusions, and review recommendations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

REVIEW_DIR = PROJECT_ROOT / CONFIG["paths"]["data_review"]
REPORTS_DIR = PROJECT_ROOT / CONFIG["paths"]["reports"]
AMBIGUOUS_CSV = PROJECT_ROOT / CONFIG["paths"]["ambiguous_csv"]

REVIEW_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_review_queue(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Analyze ambiguous and borderline questions in the review queue.
    """
    target_path = csv_path or AMBIGUOUS_CSV
    if not target_path.exists():
        print(f"[Review Manager] No review file found at {target_path}")
        return {"status": "empty", "total_records": 0}

    df = pd.read_csv(target_path)
    total_records = len(df)
    if total_records == 0:
        print("[Review Manager] Review queue is empty.")
        return {"status": "empty", "total_records": 0}

    print("=" * 60)
    print("EMP-26 REVIEW QUEUE AUDIT REPORT")
    print("=" * 60)
    print(f"Total flagged items in review queue: {total_records}")

    # Breakdown by reason / confidence tier
    reason_counts = df["reason"].value_counts().to_dict() if "reason" in df.columns else {}
    print("\nBreakdown by Confidence Tier:")
    for reason, count in reason_counts.items():
        pct = (count / total_records) * 100
        print(f"  - {reason}: {count} ({pct:.1f}%)")

    # Breakdown by predicted label
    label_col = "predicted_label" if "predicted_label" in df.columns else "label"
    label_counts = df[label_col].value_counts().to_dict() if label_col in df.columns else {}
    print("\nPredicted Category Distribution in Review Queue:")
    for lbl, cnt in label_counts.items():
        pct = (cnt / total_records) * 100
        print(f"  - {lbl}: {cnt} ({pct:.1f}%)")

    # Breakdown by source
    source_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    print("\nSource Distribution in Review Queue:")
    for src, cnt in source_counts.items():
        pct = (cnt / total_records) * 100
        print(f"  - {src}: {cnt} ({pct:.1f}%)")

    # Confidence score statistics
    conf_col = "classification_confidence"
    conf_stats = {}
    if conf_col in df.columns:
        conf_stats = {
            "mean": round(float(df[conf_col].mean()), 4),
            "std": round(float(df[conf_col].std()), 4),
            "min": round(float(df[conf_col].min()), 4),
            "max": round(float(df[conf_col].max()), 4),
            "median": round(float(df[conf_col].median()), 4),
        }
        print(f"\nConfidence Score Stats: Mean={conf_stats['mean']}, Median={conf_stats['median']}, Min={conf_stats['min']}, Max={conf_stats['max']}")

    # Generate summary CSV
    summary_report = {
        "metric": [
            "Total In Review",
            "Low Confidence (<0.50)",
            "Moderate Confidence (0.50-0.75)",
            "Mean Confidence",
            "Median Confidence",
        ],
        "value": [
            total_records,
            sum(1 for c in df[conf_col] if c < 0.50) if conf_col in df.columns else 0,
            sum(1 for c in df[conf_col] if 0.50 <= c < 0.75) if conf_col in df.columns else 0,
            conf_stats.get("mean", 0),
            conf_stats.get("median", 0),
        ]
    }
    summary_df = pd.DataFrame(summary_report)
    summary_path = REPORTS_DIR / "review_queue_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Saved] Review queue audit report saved to {summary_path}")
    print("=" * 60)

    return {
        "status": "success",
        "total_records": total_records,
        "reason_breakdown": reason_counts,
        "label_breakdown": label_counts,
        "source_breakdown": source_counts,
        "confidence_stats": conf_stats,
    }


if __name__ == "__main__":
    analyze_review_queue()
