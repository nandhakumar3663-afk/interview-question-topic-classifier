"""
Deduplication and Semantic Near-Duplicate Filtering Module for EMP-26.
Performs:
1. Exact text duplicate removal
2. Normalized-text duplicate removal (case-insensitive, whitespace normalized)
3. Embedding-based semantic near-duplicate detection using all-MiniLM-L6-v2
   with cosine similarity threshold (default 0.90)
Caches embeddings in data/cache/ to avoid redundant recomputations.
Generates reports/duplicate_report.csv.
"""

import sys
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Set
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
CACHE_DIR = PROJECT_ROOT / CONFIG["paths"]["data_cache"]
REPORTS_DIR = PROJECT_ROOT / CONFIG["paths"]["reports"]

CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SIM_THRESHOLD = CONFIG["thresholds"]["near_duplicate_similarity"]
EMBEDDING_MODEL_NAME = CONFIG["models"]["embedding_model"]
BATCH_SIZE = CONFIG["models"]["batch_size"]

_ENCODER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Encoder] Initializing {EMBEDDING_MODEL_NAME} on {device.upper()}...")
        _ENCODER = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _ENCODER


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization for exact-duplicate detection."""
    import re
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_or_load_embeddings(texts: List[str], cache_key: str) -> np.ndarray:
    """Compute embeddings in batches and cache them to disk."""
    cache_file = CACHE_DIR / f"{cache_key}_embeddings.npy"
    if cache_file.exists():
        try:
            embs = np.load(cache_file)
            if len(embs) == len(texts):
                print(f"[Embedding Cache] Loaded {len(embs)} vectors from {cache_file.name}")
                return embs
        except Exception as e:
            print(f"[Embedding Cache Warning] Could not load cache: {e}. Recomputing...")

    encoder = get_encoder()
    print(f"[Encoding] Computing dense embeddings for {len(texts)} questions (batch size: {BATCH_SIZE})...")
    embs = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache_file, embs)
    print(f"[Embedding Cache] Saved {len(embs)} vectors to {cache_file}")
    return np.array(embs)


def deduplicate_dataset(df: pd.DataFrame, max_semantic_check: int = 15000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform multi-stage deduplication:
    1. Exact string deduplication
    2. Normalized string deduplication
    3. Cosine similarity near-duplicate filtering
    """
    initial_count = len(df)
    print(f"\n[Deduplication] Starting with {initial_count} questions...")

    # Stage 1: Exact matches
    df = df.drop_duplicates(subset=["cleaned_question"]).reset_index(drop=True)
    exact_removed = initial_count - len(df)
    print(f"  - Stage 1 (Exact): Removed {exact_removed} exact duplicates.")

    # Stage 2: Normalized matches
    df["norm_key"] = df["cleaned_question"].apply(normalize_for_dedup)
    before_norm = len(df)
    df = df.drop_duplicates(subset=["norm_key"]).reset_index(drop=True)
    norm_removed = before_norm - len(df)
    print(f"  - Stage 2 (Normalized): Removed {norm_removed} normalized duplicates.")

    # Stage 3: Semantic near-duplicate detection
    questions_list = df["cleaned_question"].tolist()
    total_q = len(questions_list)
    to_remove: Set[int] = set()
    near_dupe_records = []

    # Process in chunk blocks to prevent O(N^2) memory bottlenecks
    chunk_size = min(max_semantic_check, 4000)
    print(f"  - Stage 3 (Semantic): Scanning near-duplicates (Similarity >= {SIM_THRESHOLD})...")

    for start_idx in range(0, total_q, chunk_size):
        end_idx = min(start_idx + chunk_size, total_q)
        chunk_texts = questions_list[start_idx:end_idx]
        chunk_cache_key = f"chunk_{start_idx}_{end_idx}_{len(chunk_texts)}"
        
        chunk_embs = compute_or_load_embeddings(chunk_texts, chunk_cache_key)
        sim_matrix = cosine_similarity(chunk_embs)

        # Upper triangular comparisons
        for i in range(len(chunk_texts)):
            global_i = start_idx + i
            if global_i in to_remove:
                continue
            for j in range(i + 1, len(chunk_texts)):
                global_j = start_idx + j
                if global_j in to_remove:
                    continue
                score = float(sim_matrix[i, j])
                if score >= SIM_THRESHOLD:
                    to_remove.add(global_j)
                    near_dupe_records.append({
                        "original_question": chunk_texts[i],
                        "duplicate_question": chunk_texts[j],
                        "similarity_score": round(score, 4),
                    })

    semantic_removed = len(to_remove)
    print(f"  - Stage 3 (Semantic): Filtered {semantic_removed} semantic near-duplicates.")

    keep_indices = [idx for idx in range(total_q) if idx not in to_remove]
    deduped_df = df.iloc[keep_indices].drop(columns=["norm_key"]).reset_index(drop=True)

    # Save near duplicate report
    report_df = pd.DataFrame(near_dupe_records)
    report_path = REPORTS_DIR / "duplicate_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"  - Saved near-duplicate audit report to {report_path}")

    print(f"[Deduplication Complete] Final count: {len(deduped_df)} (Total removed: {initial_count - len(deduped_df)})\n")
    return deduped_df, report_df


def run_deduplication():
    input_file = CLEANED_DIR / "cleaned_candidates.parquet"
    if not input_file.exists():
        print(f"[Error] Cleaned file not found at {input_file}. Run clean stage first.")
        sys.exit(1)

    df = pd.read_parquet(input_file)
    deduped_df, _ = deduplicate_dataset(df)

    output_path = CLEANED_DIR / "deduplicated_candidates.parquet"
    deduped_df.to_parquet(output_path, index=False)
    print(f"[Saved] Deduplicated dataset: {len(deduped_df)} questions -> {output_path}")


if __name__ == "__main__":
    run_deduplication()
