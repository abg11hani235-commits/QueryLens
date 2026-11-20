"""
data_loader.py - SQuAD v1.1 Dataset Loading, Inspection, and Statistics.

This module provides modular, robust functions to load the SQuAD v1.1 dataset
from Hugging Face Datasets, compute text statistics (lengths, answer starts),
extract representative samples, and generate publication-quality exploratory
visualizations.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict, Dataset


def load_squad_dataset(
    dataset_name: str = "rajpurkar/squad",
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load SQuAD v1.1 dataset from Hugging Face Hub using caching.

    Args:
        dataset_name: Hugging Face dataset identifier (default: "squad").
        cache_dir: Optional directory to store cached data.

    Returns:
        DatasetDict containing 'train' and 'validation' splits.
    """
    print(f"[DataLoader] Loading '{dataset_name}' from Hugging Face Hub...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    print(f"[DataLoader] Successfully loaded splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  - {split_name}: {len(split_data):,} examples")
    return dataset


def get_dataset_overview(dataset: DatasetDict) -> Dict[str, Any]:
    """
    Extract structural overview, splits, and schema of the dataset.

    Args:
        dataset: The loaded DatasetDict.

    Returns:
        Dictionary containing splits, sample counts, and feature definitions.
    """
    overview = {
        "splits": {},
        "features": {}
    }
    for split_name in dataset.keys():
        split: Dataset = dataset[split_name]
        overview["splits"][split_name] = {
            "num_examples": len(split),
            "column_names": split.column_names,
        }
    first_split = next(iter(dataset.keys()))
    overview["features"] = {
        col: str(dataset[first_split].features[col])
        for col in dataset[first_split].column_names
    }
    return overview


def extract_sample_records(
    dataset: DatasetDict,
    split: str = "train",
    indices: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve formatted representative samples for human inspection.

    Args:
        dataset: The DatasetDict.
        split: Split name ("train" or "validation").
        indices: List of integer indices to inspect. Defaults to [0, 42, 100].

    Returns:
        List of structured sample dictionaries.
    """
    if indices is None:
        indices = [0, 42, 100]

    samples = []
    target_split = dataset[split]
    for idx in indices:
        if idx >= len(target_split):
            continue
        item = target_split[idx]
        sample_dict = {
            "index": idx,
            "id": item["id"],
            "title": item["title"],
            "context": item["context"],
            "question": item["question"],
            "answers": item["answers"]["text"],
            "answer_starts": item["answers"]["answer_start"],
        }
        samples.append(sample_dict)
    return samples


def compute_length_statistics(
    dataset: DatasetDict,
    split: str = "train"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Compute lengths of contexts, questions, and answers (both characters and words),
    along with answer_start positions.

    Args:
        dataset: The loaded DatasetDict.
        split: Dataset split to analyze.

    Returns:
        Tuple of:
          - DataFrame containing length columns for each example.
          - Summary dictionary containing descriptive statistics.
    """
    target_split = dataset[split]
    contexts = target_split["context"]
    questions = target_split["question"]
    answers = target_split["answers"]

    records = []
    for ctx, q, ans in zip(contexts, questions, answers):
        first_ans_text = ans["text"][0] if len(ans["text"]) > 0 else ""
        first_ans_start = ans["answer_start"][0] if len(ans["answer_start"]) > 0 else -1

        records.append({
            "context_char_len": len(ctx),
            "context_word_len": len(ctx.split()),
            "question_char_len": len(q),
            "question_word_len": len(q.split()),
            "answer_char_len": len(first_ans_text),
            "answer_word_len": len(first_ans_text.split()),
            "answer_start": first_ans_start,
        })

    df = pd.DataFrame(records)

    summary = {}
    for col in df.columns:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "25%": float(df[col].quantile(0.25)),
            "median": float(df[col].median()),
            "75%": float(df[col].quantile(0.75)),
            "95%": float(df[col].quantile(0.95)),
            "max": float(df[col].max()),
        }

    return df, summary


def generate_and_save_visualizations(
    df: pd.DataFrame,
    output_dir: str = "results"
) -> Dict[str, str]:
    """
    Generate EDA distribution plots and save them in the results directory.

    Plots generated:
      1. context_length_dist.png: Context word length distribution with 384-token threshold.
      2. question_answer_length_dist.png: Question and Answer word length distributions.
      3. answer_start_dist.png: Answer start character position distribution.
      4. dataset_overview_grid.png: Combined 4-panel dashboard.

    Args:
        df: DataFrame produced by compute_length_statistics.
        output_dir: Directory path where plots will be saved.

    Returns:
        Dictionary mapping plot name to saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    saved_plots = {}

    # Plot 1: Context Length Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        df["context_word_len"],
        bins=60,
        kde=True,
        color="#2b5c8f",
        ax=ax
    )
    median_ctx = df["context_word_len"].median()
    p95_ctx = df["context_word_len"].quantile(0.95)
    ax.axvline(median_ctx, color="darkorange", linestyle="--", linewidth=2, label=f"Median: {median_ctx:.0f} words")
    ax.axvline(p95_ctx, color="firebrick", linestyle=":", linewidth=2, label=f"95th Pct: {p95_ctx:.0f} words")
    ax.set_title("SQuAD v1.1 - Context Word Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Context Length (Words)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(frameon=True)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "context_length_distribution.png")
    fig.savefig(path1, dpi=300)
    plt.close(fig)
    saved_plots["context_length"] = path1

    # Plot 2: Question & Answer Word Length Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["question_word_len"], bins=30, kde=True, color="#1f77b4", ax=ax1)
    ax1.set_title("Question Word Length Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Words", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.axvline(df["question_word_len"].median(), color="darkred", linestyle="--",
                label=f"Median: {df['question_word_len'].median():.0f}")
    ax1.legend()

    sns.histplot(df["answer_word_len"], bins=25, kde=False, color="#2ca02c", ax=ax2)
    ax2.set_title("Answer Word Length Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Words", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_xlim(0, 30)
    ax2.axvline(df["answer_word_len"].median(), color="darkred", linestyle="--",
                label=f"Median: {df['answer_word_len'].median():.0f}")
    ax2.legend()
    plt.tight_layout()
    path2 = os.path.join(output_dir, "question_answer_length_distribution.png")
    fig.savefig(path2, dpi=300)
    plt.close(fig)
    saved_plots["question_answer_length"] = path2

    # Plot 3: Answer Start Character Position Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["answer_start"], bins=50, kde=True, color="#9467bd", ax=ax)
    ax.set_title("SQuAD v1.1 - Answer Start Position (Characters)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Answer Start Offset (Char Index)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    med_start = df["answer_start"].median()
    ax.axvline(med_start, color="black", linestyle="--", label=f"Median Offset: {med_start:.0f}")
    ax.legend()
    plt.tight_layout()
    path3 = os.path.join(output_dir, "answer_start_distribution.png")
    fig.savefig(path3, dpi=300)
    plt.close(fig)
    saved_plots["answer_start"] = path3

    # Plot 4: Comprehensive 4-panel Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 1. Context word lengths
    sns.histplot(df["context_word_len"], bins=50, kde=True, color="#3470a3", ax=axes[0, 0])
    axes[0, 0].set_title("Context Length (Words)", fontsize=12, fontweight="bold")
    axes[0, 0].axvline(df["context_word_len"].median(), color="red", linestyle="--",
                       label=f"Median: {df['context_word_len'].median():.0f}")
    axes[0, 0].legend()

    # 2. Question word lengths
    sns.histplot(df["question_word_len"], bins=30, kde=True, color="#2e8b57", ax=axes[0, 1])
    axes[0, 1].set_title("Question Length (Words)", fontsize=12, fontweight="bold")
    axes[0, 1].axvline(df["question_word_len"].median(), color="red", linestyle="--",
                       label=f"Median: {df['question_word_len'].median():.0f}")
    axes[0, 1].legend()

    # 3. Answer word lengths
    sns.histplot(df["answer_word_len"], bins=25, color="#d95f02", ax=axes[1, 0])
    axes[1, 0].set_title("Answer Length (Words, Zoomed <= 25)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlim(0, 25)
    axes[1, 0].axvline(df["answer_word_len"].median(), color="red", linestyle="--",
                       label=f"Median: {df['answer_word_len'].median():.0f}")
    axes[1, 0].legend()

    # 4. Answer start offsets
    sns.histplot(df["answer_start"], bins=50, kde=True, color="#7570b3", ax=axes[1, 1])
    axes[1, 1].set_title("Answer Start Character Offset", fontsize=12, fontweight="bold")
    axes[1, 1].axvline(df["answer_start"].median(), color="red", linestyle="--",
                       label=f"Median: {df['answer_start'].median():.0f}")
    axes[1, 1].legend()

    fig.suptitle("SQuAD v1.1 Dataset Exploration & Structural Analysis", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    path4 = os.path.join(output_dir, "dataset_overview_grid.png")
    fig.savefig(path4, dpi=300)
    plt.close(fig)
    saved_plots["overview_grid"] = path4

    print(f"[DataLoader] Saved {len(saved_plots)} visualization plots to '{output_dir}'.")
    return saved_plots
