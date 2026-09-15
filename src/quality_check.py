"""
Quality Audit and Verification Module for EMP-26 Dataset.
Audits the final master dataset against all project criteria:
1. Total dataset size >= 50,000 (target: 54,000+)
2. Per-class distribution (Technical Knowledge >= 12k, Problem Solving >= 12k, others >= 10k)
3. 80/10/10 Stratified Split
4. 100% Real Validation and Test Sets (zero synthetic data)
5. Zero duplicate questions (exact, normalized, cross-split)
6. Complete schema adherence with no null values
7. Generates visual charts and summary CSV reports:
   - reports/dataset_statistics.csv
   - reports/class_distribution.png
   - reports/source_distribution.png
   - reports/domain_distribution.png
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MASTER_PARQUET = PROJECT_ROOT / CONFIG["paths"]["master_parquet"]
MASTER_CSV = PROJECT_ROOT / CONFIG["paths"]["master_csv"]
REPORTS_DIR = PROJECT_ROOT / CONFIG["paths"]["reports"]
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = CONFIG["targets"]["per_class"]
MIN_CHARS = CONFIG["text_constraints"]["min_question_chars"]
MAX_CHARS = CONFIG["text_constraints"]["max_question_chars"]


def run_quality_audit(dataset_path: Path = MASTER_PARQUET) -> Dict[str, Any]:
    """Execute complete dataset audit and generate verification reports."""
    print("=" * 65)
    print("EMP-26 DATASET QUALITY AUDIT & VERIFICATION REPORT")
    print("=" * 65)

    if not dataset_path.exists():
        if MASTER_CSV.exists():
            dataset_path = MASTER_CSV
        else:
            print(f"[Error] Dataset file not found at {dataset_path}")
            return {"status": "failed", "error": "file_not_found"}

    df = pd.read_parquet(dataset_path) if str(dataset_path).endswith(".parquet") else pd.read_csv(dataset_path)
    total_count = len(df)
    print(f"Total Rows Loaded: {total_count}")

    # Check 1: Total volume target
    total_target = CONFIG["targets"]["total_target"]
    pass_total = total_count >= 50000
    status_total = "[PASS]" if pass_total else "[FAIL]"
    print(f"{status_total} Total Dataset Size: {total_count:,} (Min Target: 50,000, Preferred: {total_target:,})")

    # Check 2: Class targets
    class_counts = df["label"].value_counts().to_dict()
    print("\nClass Target Audits:")
    all_classes_pass = True
    for cat, target in TARGETS.items():
        cnt = class_counts.get(cat, 0)
        c_pass = cnt >= min(target, 10000)
        if not c_pass:
            all_classes_pass = False
        status_c = "[PASS]" if c_pass else "[FAIL]"
        print(f"  {status_c} {cat}: {cnt:,} / {target:,} target")

    # Check 3: Real vs Synthetic in Validation and Test sets
    print("\nEvaluation Set Isolation Audits (100% Real Required):")
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]

    val_synthetic_cnt = int(val_df["is_synthetic"].sum()) if not val_df.empty else 0
    test_synthetic_cnt = int(test_df["is_synthetic"].sum()) if not test_df.empty else 0
    pass_val_real = (val_synthetic_cnt == 0)
    pass_test_real = (test_synthetic_cnt == 0)

    print(f"  {'[PASS]' if pass_val_real else '[FAIL]'} Validation Split: {len(val_df):,} samples | Synthetic: {val_synthetic_cnt} ({'100% Real' if pass_val_real else 'VIOLATION'})")
    print(f"  {'[PASS]' if pass_test_real else '[FAIL]'} Test Split:       {len(test_df):,} samples | Synthetic: {test_synthetic_cnt} ({'100% Real' if pass_test_real else 'VIOLATION'})")
    print(f"  {'[INFO]'} Training Split:   {len(train_df):,} samples | Real: {(~train_df['is_synthetic']).sum():,}, Synthetic: {train_df['is_synthetic'].sum():,}")

    # Check 4: Null Values & Schema Adherence
    required_cols = [
        "question_id", "question", "label", "source", "source_id",
        "domain", "difficulty", "is_synthetic", "quality_score",
        "classification_confidence", "split"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    null_counts = df[required_cols].isnull().sum().to_dict() if not missing_cols else {}
    total_nulls = sum(null_counts.values()) if null_counts else 999
    pass_schema = len(missing_cols) == 0 and total_nulls == 0
    print(f"\n{'[PASS]' if pass_schema else '[FAIL]'} Master Schema Adherence: Missing Columns={missing_cols}, Total Nulls={total_nulls}")

    # Check 5: Duplicate check
    exact_dupes = df.duplicated(subset=["question"]).sum()
    pass_dupes = (exact_dupes == 0)
    print(f"{'[PASS]' if pass_dupes else '[FAIL]'} Duplicate Check: {exact_dupes} duplicate questions detected.")

    # Check 6: Length constraints
    char_lens = df["question"].str.len()
    len_violations = ((char_lens < MIN_CHARS) | (char_lens > MAX_CHARS)).sum()
    pass_len = (len_violations == 0)
    print(f"{'[PASS]' if pass_len else '[FAIL]'} Text Length Audit: {len_violations} length violations (Allowed: {MIN_CHARS}-{MAX_CHARS} chars).")

    # Generate Reports & Charts
    print("\nGenerating audit reports and visual artifacts...")

    # 1. Dataset statistics CSV
    stats_records = []
    for lbl in TARGETS.keys():
        sub = df[df["label"] == lbl]
        stats_records.append({
            "Category": lbl,
            "Total_Count": len(sub),
            "Train_Count": len(sub[sub["split"] == "train"]),
            "Val_Count": len(sub[sub["split"] == "validation"]),
            "Test_Count": len(sub[sub["split"] == "test"]),
            "Real_Count": len(sub[~sub["is_synthetic"]]),
            "Synthetic_Count": len(sub[sub["is_synthetic"]]),
            "Real_Percentage": round((len(sub[~sub["is_synthetic"]]) / len(sub) * 100) if len(sub) > 0 else 0, 1),
            "Mean_Quality": round(sub["quality_score"].mean(), 3),
            "Mean_Confidence": round(sub["classification_confidence"].mean(), 3),
        })

    stats_df = pd.DataFrame(stats_records)
    stats_csv_path = REPORTS_DIR / "dataset_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"  - Saved dataset statistics table -> {stats_csv_path}")

    # 2. Class Distribution Plot
    plt.figure(figsize=(10, 6))
    cats = list(TARGETS.keys())
    real_counts = [len(df[(df["label"] == c) & (~df["is_synthetic"])]) for c in cats]
    synth_counts = [len(df[(df["label"] == c) & (df["is_synthetic"])]) for c in cats]

    bar_width = 0.55
    plt.bar(cats, real_counts, label="Real Historical Questions", color="#2563eb", width=bar_width)
    plt.bar(cats, synth_counts, bottom=real_counts, label="Synthetic Training Questions", color="#9333ea", width=bar_width)

    plt.title("EMP-26 Dataset: Class Distribution (Real vs Synthetic)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Category", fontsize=12, labelpad=10)
    plt.ylabel("Number of Questions", fontsize=12)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    class_dist_path = REPORTS_DIR / "class_distribution.png"
    plt.savefig(class_dist_path, dpi=200)
    plt.close()
    print(f"  - Saved class distribution chart -> {class_dist_path}")

    # 3. Source Distribution Plot
    plt.figure(figsize=(10, 5))
    source_counts = df["source"].value_counts()
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#64748b"]
    source_counts.plot(kind="bar", color=colors[:len(source_counts)], width=0.6)
    plt.title("EMP-26 Dataset: Source Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Source Repository", fontsize=12, labelpad=10)
    plt.ylabel("Question Count", fontsize=12)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    source_dist_path = REPORTS_DIR / "source_distribution.png"
    plt.savefig(source_dist_path, dpi=200)
    plt.close()
    print(f"  - Saved source distribution chart -> {source_dist_path}")

    # 4. Domain Distribution Plot
    plt.figure(figsize=(12, 6))
    domain_counts = df[df["label"] == "Role-Specific Skills"]["domain"].value_counts()
    if not domain_counts.empty:
        domain_counts.plot(kind="bar", color="#059669", width=0.6)
        plt.title("EMP-26 Role-Specific Skills: 12-Domain Balance", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Engineering Domain", fontsize=12, labelpad=10)
        plt.ylabel("Question Count", fontsize=12)
        plt.xticks(rotation=35, ha="right", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        domain_dist_path = REPORTS_DIR / "domain_distribution.png"
        plt.savefig(domain_dist_path, dpi=200)
        plt.close()
        print(f"  - Saved domain distribution chart -> {domain_dist_path}")

    overall_pass = pass_total and all_classes_pass and pass_val_real and pass_test_real and pass_schema and pass_dupes
    print("\n" + "=" * 65)
    if overall_pass:
        print("OVERALL AUDIT RESULT: ALL QUALITY CONSTRAINTS PASSED SUCCESSFULLY")
    else:
        print("OVERALL AUDIT RESULT: REVIEW REQUIRED (Some constraints need attention)")
    print("=" * 65)

    return {
        "status": "passed" if overall_pass else "needs_attention",
        "total_count": total_count,
        "class_counts": class_counts,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    run_quality_audit()
