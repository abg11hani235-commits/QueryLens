"""
utils.py - Utility functions for ContextQA system.

Provides helpers for:
  - Random seed setting and reproducibility
  - Device selection and GPU memory tracking
  - YAML and JSON configuration loading/saving
  - SQuAD official string normalization and metric computations (Exact Match, F1)
  - Text span highlighting for terminal and Streamlit UI
"""

import os
import re
import string
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across all libraries for full experimental reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Utils] Reproducibility seed set to: {seed}")


def get_device() -> torch.device:
    """
    Detect and return the best available compute device (CUDA GPU or CPU).

    Returns:
        torch.device instance.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[Utils] Using CUDA GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Utils] CUDA not available. Falling back to CPU.")
    return device


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory allocation and reserved cache in Megabytes (MB).

    Returns:
        Dictionary with allocated_mb, reserved_mb, and max_allocated_mb.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        file_path: Path to the .yaml configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save serializable data to a JSON file.

    Args:
        data: Data structure to serialize.
        file_path: Target JSON file path.
        indent: JSON indentation level.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"[Utils] Saved JSON file to: {file_path}")


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load and deserialize a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Deserialized Python object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """
    Format a floating-point seconds duration into a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        String formatted as 'Xh Ym Zs' or 'Xs'.
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{seconds:.2f}s"


# ============================================================================
# SQuAD v1.1 Official Normalization and Metric Functions
# ============================================================================

def normalize_answer(s: str) -> str:
    """
    Official SQuAD v1.1 answer normalization:
    1. Lowercase text
    2. Remove punctuation
    3. Remove English articles ('a', 'an', 'the')
    4. Normalize whitespace

    Args:
        s: Raw input text.

    Returns:
        Normalized string.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute official Exact Match (EM) binary score between prediction and ground truth.

    Args:
        prediction: Predicted answer string.
        ground_truth: Ground truth reference answer.

    Returns:
        100.0 if normalized strings match exactly, else 0.0.
    """
    return 100.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute official Token-level F1 score between prediction and ground truth.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        prediction: Predicted answer string.
        ground_truth: Reference answer string.

    Returns:
        F1 score between 0.0 and 100.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 100.0 if pred_tokens == truth_tokens else 0.0

    # Multi-set intersection counts
    common: Dict[str, int] = {}
    truth_counts: Dict[str, int] = {}
    for tok in truth_tokens:
        truth_counts[tok] = truth_counts.get(tok, 0) + 1

    num_same = 0
    for tok in pred_tokens:
        if truth_counts.get(tok, 0) > 0:
            num_same += 1
            truth_counts[tok] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1 * 100.0


def metric_max_over_ground_truths(
    metric_fn: Any,
    prediction: str,
    ground_truths: List[str]
) -> float:
    """
    Compute the maximum metric score across multiple valid reference answers.

    In SQuAD evaluation, validation examples typically have 3 ground truth answers.
    The official score is the maximum score achieved against ANY ground truth.

    Args:
        metric_fn: Metric function (compute_exact_match or compute_f1_score).
        prediction: Predicted answer text.
        ground_truths: List of valid reference strings.

    Returns:
        Maximum score across all ground truths (0.0 to 100.0).
    """
    if not ground_truths:
        return 0.0
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores)


def highlight_answer_in_context(
    context: str,
    start_char: int,
    end_char: int,
    format_type: str = "html"
) -> str:
    """
    Highlight an answer span within the context text.

    Args:
        context: Original context text.
        start_char: Beginning character index of the answer.
        end_char: Ending character index of the answer.
        format_type: 'html' (using <mark>) or 'markdown' (using **bold**).

    Returns:
        Formatted context string with answer highlighted.
    """
    if start_char < 0 or end_char <= start_char or start_char >= len(context):
        return context

    # Clamp end_char to context length
    end_char = min(end_char, len(context))

    before = context[:start_char]
    answer = context[start_char:end_char]
    after = context[end_char:]

    if format_type == "html":
        return (
            f"{before}"
            f"<mark style='background-color: #ffd166; color: #111; padding: 2px 4px; border-radius: 4px; font-weight: bold;'>"
            f"{answer}</mark>{after}"
        )
    elif format_type == "markdown":
        return f"{before}**__{answer}__**{after} (chars [{start_char}:{end_char}])"
    else:
        return f"{before}[ANSWER: {answer}]{after}"
