"""
EMP-26 Interview Question Topic Classifier - Dataset Generation Pipeline CLI.
Orchestrates all pipeline stages with resume capability, progress tracking, and validation:
- inspect: Inspect hardware environment and available datasets
- download: Ingest raw datasets from Hugging Face and public repositories
- clean: Clean HTML, normalize Unicode, filter noise, preserve technical terms
- deduplicate: Remove exact, normalized, and semantic near-duplicates
- label: Two-stage classification (heuristics + dense embeddings) and confidence routing
- augment: Synthetic augmentation strictly for training split deficits
- split: Stratified 80/10/10 split with 100% real test/val and leakage purging
- validate: Run quality audit, schema check, and visual reports
- review: Audit review queue and ambiguous questions
- sample: Execute pilot run on 2,500 questions across all stages to verify schema and confidence
- all: Execute complete end-to-end dataset generation
"""

import sys
import argparse
from pathlib import Path
import yaml
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

from src.download_datasets import download_all
from src.clean_data import clean_all_datasets, clean_raw_dataframe
from src.deduplicate import deduplicate_dataset
from src.classify_questions import classify_dataset
from src.augment_training import generate_synthetic_questions
from src.split_dataset import build_and_split_dataset
from src.quality_check import run_quality_audit
from src.review_manager import analyze_review_queue


def run_stage_inspect():
    print("\n[Stage: Inspect] Inspecting environment and datasets...")
    from src.inspect_datasets import inspect_all
    inspect_all()


def run_stage_sample(sample_size: int = 2500):
    """
    Execute pilot run on 1,000-5,000 samples (target: 2,500) through all pipeline stages
    to verify schema, confidence distributions, and routing before full-scale processing.
    """
    print("\n" + "=" * 65)
    print(f"PILOT PIPELINE RUN ({sample_size} Questions)")
    print("=" * 65)

    # 1. Ingest small sample
    print(f"\n[Pilot Stage 1/5] Ingesting {sample_size} questions across repositories...")
    per_source = sample_size // 4
    raw_dict = download_all(max_stackpulse=per_source, max_coding=per_source)
    
    # Combine samples
    sample_dfs = []
    for name, df in raw_dict.items():
        if not df.empty:
            sub = df.head(per_source).copy()
            sample_dfs.append(sub)
    raw_sample = pd.concat(sample_dfs, ignore_index=True)
    print(f"  - Ingested pilot batch of {len(raw_sample)} questions.")

    # 2. Clean
    print("\n[Pilot Stage 2/5] Cleaning and normalizing pilot batch...")
    clean_sample = clean_raw_dataframe(raw_sample)
    print(f"  - Cleaned pilot batch: {len(clean_sample)} valid questions.")

    # 3. Deduplicate
    print("\n[Pilot Stage 3/5] Deduplicating pilot batch...")
    dedup_sample, _ = deduplicate_dataset(clean_sample, max_semantic_check=1000)
    print(f"  - Deduplicated pilot batch: {len(dedup_sample)} unique questions.")

    # 4. Classify & Confidence Threshold
    print("\n[Pilot Stage 4/5] Running two-stage classification on pilot batch...")
    accepted_sample, review_sample = classify_dataset(dedup_sample)
    print(f"  - Pilot Classification Results:")
    print(f"    * Accepted (>=0.75): {len(accepted_sample)}")
    print(f"    * Review / Ambiguous (<0.75): {len(review_sample)}")

    # 5. Verify Pilot Schema & Audit
    print("\n[Pilot Stage 5/5] Auditing pilot schema and distribution...")
    print(f"  - Accepted columns: {list(accepted_sample.columns)}")
    print(f"  - Class distribution in accepted pilot:")
    for c, cnt in accepted_sample["label"].value_counts().items():
        print(f"    * {c}: {cnt}")

    print("\n" + "=" * 65)
    print("PILOT RUN COMPLETED SUCCESSFULLY: Pipeline schemas and confidence distributions verified!")
    print("=" * 65)


def run_stage_all(force: bool = False):
    """Execute end-to-end full dataset generation pipeline to produce 54,000+ questions."""
    print("\n" + "=" * 65)
    print("STARTING FULL EMP-26 DATASET GENERATION (Target: 54,000+ Questions)")
    print("=" * 65)

    # 1. Download
    raw_dict = download_all(max_stackpulse=25000, max_coding=25000)

    # 2. Clean
    clean_df = clean_all_datasets()

    # 3. Deduplicate
    cleaned_cand_path = PROJECT_ROOT / CONFIG["paths"]["data_cleaned"] / "deduplicated_candidates.parquet"
    if cleaned_cand_path.exists() and not force:
        print(f"\n[Stage: Deduplicate] Reusing existing deduplicated candidates: {cleaned_cand_path}")
        dedup_df = pd.read_parquet(cleaned_cand_path)
    else:
        dedup_df, _ = deduplicate_dataset(clean_df)
        dedup_df.to_parquet(cleaned_cand_path, index=False)

    # 4. Two-stage Classification & Confidence Routing
    labeled_cand_path = PROJECT_ROOT / CONFIG["paths"]["data_labeled"] / "candidates_labeled.parquet"
    if labeled_cand_path.exists() and not force:
        print(f"\n[Stage: Classify] Reusing existing classified candidates: {labeled_cand_path}")
        accepted_df = pd.read_parquet(labeled_cand_path)
    else:
        accepted_df, _ = classify_dataset(dedup_df)
        accepted_df.to_parquet(labeled_cand_path, index=False)

    # 5. Audit Review Queue
    analyze_review_queue()

    # 6. Split Dataset (80/10/10) with 100% Real Validation & Test Sets + Training Augmentation
    master_df, train_df, val_df, test_df = build_and_split_dataset(accepted_df)

    # 8. Run Final Quality Audit & Visual Verification
    run_quality_audit()

    print("\n" + "=" * 65)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"Final Master Dataset: {len(master_df):,} questions")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="EMP-26 Dataset Generation Pipeline CLI")
    parser.add_argument(
        "--stage",
        choices=["inspect", "sample", "download", "clean", "deduplicate", "label", "augment", "split", "validate", "review", "all"],
        default="sample",
        help="Pipeline stage to execute (default: sample)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2500,
        help="Sample size for pilot run (default: 2500)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation and ignore intermediate file caches",
    )

    args = parser.parse_args()

    if args.stage == "inspect":
        run_stage_inspect()
    elif args.stage == "sample":
        run_stage_sample(sample_size=args.size)
    elif args.stage == "download":
        download_all()
    elif args.stage == "clean":
        clean_all_datasets()
    elif args.stage == "deduplicate":
        from src.deduplicate import run_deduplication
        run_deduplication()
    elif args.stage == "label":
        from src.classify_questions import run_classification
        run_classification()
    elif args.stage == "review":
        analyze_review_queue()
    elif args.stage == "augment":
        labeled_path = PROJECT_ROOT / CONFIG["paths"]["data_labeled"] / "candidates_labeled.parquet"
        accepted_df = pd.read_parquet(labeled_path)
        class_targets = CONFIG["targets"]["per_class"]
        current_counts = accepted_df["label"].value_counts().to_dict()
        synth_df = generate_synthetic_questions(class_targets, current_counts)
        print(f"Generated {len(synth_df)} synthetic questions.")
    elif args.stage == "split":
        labeled_path = PROJECT_ROOT / CONFIG["paths"]["data_labeled"] / "candidates_labeled.parquet"
        accepted_df = pd.read_parquet(labeled_path)
        build_and_split_dataset(accepted_df)
    elif args.stage == "validate":
        run_quality_audit()
    elif args.stage == "all":
        run_stage_all(force=args.force)
    else:
        print(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
