"""
Text preprocessing and data normalization pipeline for interview questions.
Preserves critical technical tokens (e.g., C++, C#, .NET, CI/CD, k8s, O(1), etc.)
while cleaning punctuation, normalizing whitespace, and deduplicating rows.
"""

import re
from typing import List, Optional
import pandas as pd

# Mapping of technical keywords/symbols to safe placeholders during punctuation cleaning
TECH_TOKEN_REPLACEMENTS = [
    (r"\bC\+\+(?!\+)", "TOKEN_CPPLUS"),
    (r"\bC#(?![#\w])", "TOKEN_CSHARP"),
    (r"(?:\b|\s)\.NET\b", " TOKEN_DOTNET"),
    (r"\bCI/CD\b", "TOKEN_CICD"),
    (r"\bK8s\b", "TOKEN_KUBERNETES"),
    (r"\bO\(([1nN]\^?[0-9]*|log\s*n|n\s*log\s*n)\)", r"TOKEN_BIG_O_\1"),
    (r"\bnode\.js\b", "TOKEN_NODEJS"),
    (r"\bvue\.js\b", "TOKEN_VUEJS"),
    (r"\breact\.js\b", "TOKEN_REACTJS"),
    (r"\bnext\.js\b", "TOKEN_NEXTJS"),
]

# Reverse placeholder mapping back to clean, standardized text
REVERSE_TOKEN_MAPPINGS = [
    ("TOKEN_CPPLUS", "cplusplus"),
    ("TOKEN_CSHARP", "csharp"),
    ("TOKEN_DOTNET", "dotnet"),
    ("TOKEN_CICD", "cicd"),
    ("TOKEN_KUBERNETES", "kubernetes"),
    ("TOKEN_NODEJS", "nodejs"),
    ("TOKEN_VUEJS", "vuejs"),
    ("TOKEN_REACTJS", "reactjs"),
    ("TOKEN_NEXTJS", "nextjs"),
]


def preserve_technical_tokens(text: str) -> str:
    """Safeguard technical keywords containing special symbols."""
    for pattern, placeholder in TECH_TOKEN_REPLACEMENTS:
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text


def restore_technical_tokens(text: str) -> str:
    """Restore technical token placeholders into standardized recognizable tokens."""
    for placeholder, clean_token in REVERSE_TOKEN_MAPPINGS:
        text = text.replace(placeholder.lower(), clean_token)
    # Restore Big-O tokens
    text = re.sub(r"token_big_o_([a-z0-9\^]+)", r"big_o_\1", text)
    return text


def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize single interview question string.
    - Handles null/empty values
    - Preserves technical keywords
    - Lowercases text
    - Removes non-alphanumeric noise while retaining hyphenated/tokenized compounds
    - Normalizes multiple spaces into a single space
    """
    if text is None or not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    # Preserve special technical tokens
    text = preserve_technical_tokens(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URL or file paths if present
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Retain alphanumeric characters, underscores (used in tokens), and basic question spacing
    text = re.sub(r"[^\w\s\-_]", " ", text)

    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text).strip()

    # Restore placeholders to unified clean tokens
    text = restore_technical_tokens(text)

    return text


def clean_dataset(df: pd.DataFrame, text_col: str = "question", label_col: str = "label") -> pd.DataFrame:
    """
    Clean an entire dataset DataFrame:
    - Drops null rows
    - Cleans questions using domain-aware normalizer
    - Drops duplicates based on cleaned question
    - Strips label whitespaces
    - Filters out questions that became too short after cleaning (< 10 chars)
    """
    initial_count = len(df)
    
    # Drop rows missing question or label
    df = df.dropna(subset=[text_col, label_col]).copy()

    # Clean text
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df[label_col] = df[label_col].astype(str).str.strip()

    # Filter out empty or trivial questions
    df = df[df[text_col].str.len() >= 10].copy()

    # Remove duplicates
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    cleaned_count = len(df)
    print(f"[Preprocessing] Deduplication & cleaning: {initial_count} -> {cleaned_count} rows (removed {initial_count - cleaned_count} duplicates/invalid rows)")

    return df
