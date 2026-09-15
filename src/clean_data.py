"""
Cleaning and Normalization Module for EMP-26.
Performs:
- HTML tag stripping and noise removal
- Code block normalization (preserving technical intent without code clutter)
- Unicode normalization and whitespace deduplication
- Preservation of technical keywords (C++, C#, .NET, CI/CD, K8s, Python, AWS, Docker, O(1), etc.)
- Length validation (15-600 characters)
"""

import sys
import os
import re
import html
import unicodedata
from pathlib import Path
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / CONFIG["paths"]["data_raw"]
CLEANED_DIR = PROJECT_ROOT / CONFIG["paths"]["data_cleaned"]
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = CONFIG["text_constraints"]["min_question_chars"]
MAX_CHARS = CONFIG["text_constraints"]["max_question_chars"]

# Technical tokens to preserve during cleaning
PROTECTED_TERMS = [
    (r"\bC\+\+(?!\+)", "TOKEN_CPPLUS"),
    (r"\bC#(?![#\w])", "TOKEN_CSHARP"),
    (r"(?:\b|\s)\.NET\b", " TOKEN_DOTNET"),
    (r"\bCI/CD\b", "TOKEN_CICD"),
    (r"\bK8s\b", "TOKEN_KUBERNETES"),
    (r"\bO\(([1nN]\^?[0-9]*|log\s*n|n\s*log\s*n)\)", r"TOKEN_BIG_O_\1"),
]

REVERSE_TERMS = [
    ("TOKEN_CPPLUS", "C++"),
    ("TOKEN_CSHARP", "C#"),
    ("TOKEN_DOTNET", ".NET"),
    ("TOKEN_CICD", "CI/CD"),
    ("TOKEN_KUBERNETES", "Kubernetes"),
]


def clean_question_text(raw_text: str) -> str:
    """Clean and normalize a raw interview question string."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(raw_text)

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove markdown code blocks and inline code ticks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Normalize Unicode (NFKD decomposition then ASCII/latin safe)
    text = unicodedata.normalize("NFKD", text)

    # Protect technical keywords
    for pattern, placeholder in PROTECTED_TERMS:
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

    # Remove weird punctuation and noisy characters while preserving standard question punctuation
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\w\s\-\?\.,\:\(\)\/\+#'\"]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # Restore technical tokens
    for placeholder, clean_val in REVERSE_TERMS:
        text = text.replace(placeholder, clean_val)
    text = re.sub(r"TOKEN_BIG_O_([a-zA-Z0-9\^]+)", r"O(\1)", text)

    # Format into proper sentence / question ending
    if text and not text.endswith(("?", ".", "!")):
        text += "?"

    # Capitalize first character
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text.strip()


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataframe and filter valid candidate questions."""
    if df.empty:
        return df

    initial_len = len(df)
    df = df.dropna(subset=["raw_question"]).copy()
    df["cleaned_question"] = df["raw_question"].astype(str).apply(clean_question_text)

    # Length filter
    df["char_len"] = df["cleaned_question"].str.len()
    df = df[(df["char_len"] >= MIN_CHARS) & (df["char_len"] <= MAX_CHARS)].copy()

    # Drop obvious non-questions or code dumps
    df = df[~df["cleaned_question"].str.contains(r"^[\d\W_]+$", regex=True)]

    # Deduplicate exact cleaned questions
    df = df.drop_duplicates(subset=["cleaned_question"]).reset_index(drop=True)

    print(f"[Cleaning] {initial_len} -> {len(df)} questions ({initial_len - len(df)} removed)")
    return df


def clean_all_datasets() -> pd.DataFrame:
    """Clean all ingested raw files and save candidate pool."""
    print("\n" + "=" * 60)
    print("DATA CLEANING & NORMALIZATION PIPELINE")
    print("=" * 60)

    all_dfs = []
    raw_files = list(RAW_DIR.glob("raw_*.parquet")) + list(RAW_DIR.glob("raw_*.csv"))
    for f in raw_files:
        print(f"[Processing] Cleaning {f.name}...")
        df_raw = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        if "raw_question" not in df_raw.columns and "question" in df_raw.columns:
            df_raw["raw_question"] = df_raw["question"]
        if "source" not in df_raw.columns:
            df_raw["source"] = "CuratedHistoricalQuestions"
        df_clean = clean_raw_dataframe(df_raw)
        if not df_clean.empty:
            all_dfs.append(df_clean)

    if not all_dfs:
        print("[Warning] No raw parquet files found in data/raw/. Please run download stage first.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["cleaned_question"]).reset_index(drop=True)

    output_path = CLEANED_DIR / "cleaned_candidates.parquet"
    combined_df.to_parquet(output_path, index=False)
    print(f"\n[Saved] Cleaned candidate pool: {len(combined_df)} samples -> {output_path}")
    print("=" * 60)
    return combined_df


if __name__ == "__main__":
    clean_all_datasets()
