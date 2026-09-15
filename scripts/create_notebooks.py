"""
Generates the 5 project notebooks according to the EMP-26 specification:
- 01_data_exploration.ipynb
- 02_preprocessing.ipynb
- 03_baseline_model.ipynb
- 04_embedding_model.ipynb
- 05_evaluation.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.13"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def generate_01_exploration():
    cells = [
        md_cell("""# EMP-26: 01. Data Exploration & Distribution Analysis
This notebook explores the dataset of interview questions across the 5 target categories:
1. `Technical Knowledge`
2. `Communication`
3. `Problem Solving`
4. `Leadership`
5. `Role-Specific Skills`
"""),
        code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_path = Path("../data/processed/labeled_questions.csv")
df = pd.read_csv(data_path)
print(f"Total samples: {len(df)}")
df.head(10)"""),
        md_cell("## Class Balance Check\nVerify that all 5 classes are evenly represented."),
        code_cell("""plt.figure(figsize=(9, 4.5))
sns.countplot(data=df, x="label", palette="Blues_r", order=df["label"].value_counts().index)
plt.title("Class Distribution of Interview Questions", fontsize=13, fontweight="bold")
plt.xlabel("Topic Category", fontweight="bold")
plt.ylabel("Number of Questions", fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""),
        md_cell("## Question Length Distribution\nAnalyze word counts and character lengths per class."),
        code_cell("""df["char_count"] = df["question"].str.len()
df["word_count"] = df["question"].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=df, x="label", y="word_count", ax=axes[0], palette="Set2")
axes[0].set_title("Word Count Distribution by Category", fontweight="bold")
axes[0].tick_params(axis='x', rotation=25)

sns.histplot(df["char_count"], bins=30, kde=True, ax=axes[1], color="#2563EB")
axes[1].set_title("Overall Character Count Distribution", fontweight="bold")
plt.tight_layout()
plt.show()""")
    ]
    path = NOTEBOOKS_DIR / "01_data_exploration.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Generated {path}")


def generate_02_preprocessing():
    cells = [
        md_cell("""# EMP-26: 02. NLP Preprocessing & Token Preservation
Demonstrates text normalization and preservation of technical keywords (e.g. `C++`, `C#`, `CI/CD`, `k8s`, `O(n)`)."""),
        code_cell("""import sys
from pathlib import Path
sys.path.append(str(Path("..").resolve()))

from src.preprocessing import clean_text, preserve_technical_tokens, restore_technical_tokens

test_samples = [
    "How does CI/CD integration with K8s and Docker streamline deployment?",
    "Explain the memory management in C++ vs C# and .NET CLR.",
    "What is the time complexity: O(n log n) vs O(1) for hash table lookups?",
    "Can you describe a time you communicated with the VP of Engineering?"
]

for s in test_samples:
    print(f"Raw:     {s}")
    print(f"Cleaned: {clean_text(s)}\\n")"""),
        md_cell("## Split Verification\nVerify the stratified 70/15/15 partitions."),
        code_cell("""import pandas as pd

train_df = pd.read_csv("../data/splits/train.csv")
val_df = pd.read_csv("../data/splits/validation.csv")
test_df = pd.read_csv("../data/splits/test.csv")

print(f"Train split:      {len(train_df)} ({len(train_df)/len(train_df+val_df+test_df)*100:.1f}%)")
print(f"Validation split: {len(val_df)} ({len(val_df)/len(train_df+val_df+test_df)*100:.1f}%)")
print(f"Test split:       {len(test_df)} ({len(test_df)/len(train_df+val_df+test_df)*100:.1f}%)")""")
    ]
    path = NOTEBOOKS_DIR / "02_preprocessing.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Generated {path}")


def generate_03_baseline():
    cells = [
        md_cell("""# EMP-26: 03. Baseline Model (TF-IDF + Logistic Regression)
Constructs the lightweight CPU-friendly baseline pipeline with sublinear TF-IDF and Logistic Regression."""),
        code_cell("""import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path("..").resolve()))

from src.preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

train_df = pd.read_csv("../data/splits/train.csv")
val_df = pd.read_csv("../data/splits/validation.csv")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
    ("clf", LogisticRegression(C=1.5, max_iter=1000, random_state=42))
])

pipeline.fit(train_df["question"], train_df["label"])
val_preds = pipeline.predict(val_df["question"])

print(f"Validation Accuracy: {accuracy_score(val_df['label'], val_preds)*100:.2f}%")
print("\\nClassification Report:")
print(classification_report(val_df["label"], val_preds))""")
    ]
    path = NOTEBOOKS_DIR / "03_baseline_model.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Generated {path}")


def generate_04_embedding():
    cells = [
        md_cell("""# EMP-26: 04. Semantic Embedding Model (Sentence Transformers)
Evaluates dense semantic representations using `sentence-transformers` (`all-MiniLM-L6-v2`) paired with a calibrated linear classifier."""),
        code_cell("""import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path("..").resolve()))

from src.train import EmbeddingClassifierWrapper
from sklearn.metrics import classification_report, accuracy_score

train_df = pd.read_csv("../data/splits/train.csv")
val_df = pd.read_csv("../data/splits/validation.csv")

model = EmbeddingClassifierWrapper(model_name="all-MiniLM-L6-v2", C=2.0)
model.fit(train_df["question"].tolist(), train_df["label"].tolist())

val_preds = model.predict(val_df["question"].tolist())
print(f"Validation Accuracy: {accuracy_score(val_df['label'], val_preds)*100:.2f}%")
print("\\nClassification Report:")
print(classification_report(val_df["label"], val_preds))""")
    ]
    path = NOTEBOOKS_DIR / "04_embedding_model.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Generated {path}")


def generate_05_evaluation():
    cells = [
        md_cell("""# EMP-26: 05. Evaluation, Confusion Matrix & Error Analysis
Evaluates models against the held-out test partition and performs deep dive into misclassified and ambiguous questions."""),
        code_cell("""import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path("..").resolve()))

from src.utils import CONFUSION_MATRIX_PATH, MODEL_COMPARISON_PATH, EVALUATION_METRICS_PATH

if EVALUATION_METRICS_PATH.exists():
    with open(EVALUATION_METRICS_PATH) as f:
        data = json.load(f)
    print("Test Evaluation Metrics:")
    for m, details in data.items():
        print(f"\\n--- {m} ---")
        for k, v in details['metrics'].items():
            if k != "classification_report":
                print(f"  {k}: {v}")"""),
        md_cell("## Visualizations\nDisplaying confusion matrix and model benchmark comparisons."),
        code_cell("""from IPython.display import Image, display

if CONFUSION_MATRIX_PATH.exists():
    print("Confusion Matrix:")
    display(Image(filename=str(CONFUSION_MATRIX_PATH)))

if MODEL_COMPARISON_PATH.exists():
    print("Model Metric Comparison:")
    display(Image(filename=str(MODEL_COMPARISON_PATH)))""")
    ]
    path = NOTEBOOKS_DIR / "05_evaluation.ipynb"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Generated {path}")


def main():
    generate_01_exploration()
    generate_02_preprocessing()
    generate_03_baseline()
    generate_04_embedding()
    generate_05_evaluation()
    print("All 5 Jupyter notebooks generated successfully.")


if __name__ == "__main__":
    main()
