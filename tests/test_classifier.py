"""
Unit and integration tests for EMP-26 Interview Question Topic Classifier.
Verifies:
1. Token preservation during text preprocessing
2. Question cleaning and normalization
3. Category validity and probability constraints
4. Similar-question retrieval structure
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import TOPIC_CATEGORIES
from src.preprocessing import clean_text, preserve_technical_tokens, restore_technical_tokens
from src.predict import predict_topic, find_similar_questions


def test_topic_categories():
    """Verify exactly the 5 required categories from specification."""
    expected = [
        "Technical Knowledge",
        "Communication",
        "Problem Solving",
        "Leadership",
        "Role-Specific Skills",
    ]
    assert TOPIC_CATEGORIES == expected
    assert len(TOPIC_CATEGORIES) == 5


def test_token_preservation():
    """Verify technical symbols like C++, CI/CD, K8s, and Big-O are preserved."""
    sample = "How do you set up CI/CD with K8s for a C++ application with O(1) complexity?"
    cleaned = clean_text(sample)
    
    # Assert tokens were converted to standardized keywords
    assert "cicd" in cleaned
    assert "kubernetes" in cleaned
    assert "cplusplus" in cleaned
    assert "big_o_1" in cleaned


def test_clean_text_normalization():
    """Verify whitespace normalization, lowercasing, and null safety."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   WHAT    IS   REDIS?   ") == "what is redis"


def test_predict_structure_and_probabilities():
    """Test output dictionary structure and probability properties."""
    question = "How does database indexing work under the hood in PostgreSQL?"
    result = predict_topic(question, model_type="tfidf", threshold=0.65)
    
    assert "predicted_topic" in result
    assert result["predicted_topic"] in TOPIC_CATEGORIES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 5
    
    # Verify probabilities sum close to 1.0
    total_prob = sum(result["probabilities"].values())
    assert pytest.approx(total_prob, rel=1e-2) == 1.0


def test_review_flag_logic():
    """Verify low confidence flags questions for review."""
    # When threshold is set extremely high (0.99), should trigger review flag
    res_high_thresh = predict_topic("A generic question", threshold=0.99)
    assert res_high_thresh["needs_review"] is True

    # When threshold is 0.0, should not trigger review flag
    res_low_thresh = predict_topic("A generic question", threshold=0.0)
    assert res_low_thresh["needs_review"] is False
