"""
run_exploration.py - Execute Phase 1 SQuAD v1.1 exploration and generate visualizations.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_squad_dataset,
    get_dataset_overview,
    extract_sample_records,
    compute_length_statistics,
    generate_and_save_visualizations,
)
from src.utils import save_json


def main():
    print("=" * 60)
    print("🔍 PHASE 1: SQuAD v1.1 Dataset Exploration & Verification")
    print("=" * 60)

    # 1. Load Dataset
    dataset = load_squad_dataset("rajpurkar/squad")

    train_count = len(dataset["train"])
    val_count = len(dataset["validation"])
    print(f"\n[Verification] Actual Split Counts:")
    print(f"  - Training:   {train_count:,} (Expected: 87,599)")
    print(f"  - Validation: {val_count:,} (Expected: 10,570)")
    assert train_count == 87599, f"Mismatch in train count: {train_count}"
    assert val_count == 10570, f"Mismatch in validation count: {val_count}"
    print("  => Split verification: 100% MATCH!\n")

    # 2. Dataset Overview & Schema
    overview = get_dataset_overview(dataset)
    print("[Schema & Features]")
    for col, feature_type in overview["features"].items():
        print(f"  - {col}: {feature_type}")

    # 3. Extract Sample Records
    samples = extract_sample_records(dataset, split="train", indices=[0, 42, 100])
    print("\n[Representative Samples]")
    for s in samples:
        print(f"\n--- Example {s['index']} ({s['title']}) ---")
        print(f"Context: {s['context'][:180]}...")
        print(f"Question: {s['question']}")
        print(f"Answers: {s['answers']} (Starts: {s['answer_starts']})")

    # 4. Length & Position Statistics
    print("\n[Computing Statistical Distributions on Training Split...]")
    df, summary = compute_length_statistics(dataset, split="train")

    print("\n[Key Length Statistics (Words)]")
    for key in ["context_word_len", "question_word_len", "answer_word_len"]:
        s = summary[key]
        print(f"  - {key}: Mean={s['mean']:.1f}, Median={s['median']:.1f}, Min={s['min']}, Max={s['max']}, 95th-Pct={s['95%']:.1f}")

    print("\n[Answer Start Position (Characters)]")
    ans_start_s = summary["answer_start"]
    print(f"  - Mean={ans_start_s['mean']:.1f}, Median={ans_start_s['median']:.1f}, Min={ans_start_s['min']}, Max={ans_start_s['max']}")

    # 5. Save Visualizations
    vis_dir = "results/visualizations"
    plots = generate_and_save_visualizations(df, output_dir=vis_dir)
    # Also save in results/
    generate_and_save_visualizations(df, output_dir="results")

    # Save summary statistics JSON
    stats_path = Path("results/dataset_statistics.json")
    save_json(summary, stats_path)

    print("\n" + "=" * 60)
    print("✅ Phase 1 Exploration Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
