"""
test_utils.py - Unit tests for SQuAD normalization and evaluation metrics.
"""

import pytest
from src.utils import (
    normalize_answer,
    compute_exact_match,
    compute_f1_score,
    metric_max_over_ground_truths,
    highlight_answer_in_context,
)


def test_normalize_answer():
    # Lowercase, punctuation removal, article removal, whitespace normalization
    assert normalize_answer("The United States of America.") == "united states of america"
    assert normalize_answer("  an apple, and   a banana!  ") == "apple and banana"
    assert normalize_answer("1885.") == "1885"


def test_exact_match():
    # Exact matches
    assert compute_exact_match("1885", "1885") == 100.0
    assert compute_exact_match("Stanford University", "the stanford university") == 100.0
    assert compute_exact_match("California.", "California") == 100.0

    # Non-matches
    assert compute_exact_match("1885", "1886") == 0.0
    assert compute_exact_match("Stanford", "Harvard") == 0.0


def test_f1_score():
    # Complete match -> 100%
    assert compute_f1_score("Saint Louis", "Saint Louis") == 100.0

    # Partial match
    # Prediction: "Leland Stanford", Truth: "Leland Stanford Junior"
    # Overlap: 2 tokens ("leland", "stanford")
    # Precision: 2/2 = 1.0, Recall: 2/3 = 0.667 -> F1: 2 * (1 * 2/3) / (1 + 2/3) = 80.0%
    f1 = compute_f1_score("Leland Stanford", "Leland Stanford Junior")
    assert abs(f1 - 80.0) < 1e-3

    # Disjoint match -> 0%
    assert compute_f1_score("1885", "1886") == 0.0


def test_metric_max_over_ground_truths():
    ground_truths = ["October 1885", "1885", "autumn of 1885"]
    prediction = "1885"
    best_em = metric_max_over_ground_truths(compute_exact_match, prediction, ground_truths)
    assert best_em == 100.0


def test_highlight_answer_in_context():
    context = "Stanford University was founded in 1885."
    highlighted = highlight_answer_in_context(context, 35, 39, format_type="markdown")
    assert "**__1885__**" in highlighted
