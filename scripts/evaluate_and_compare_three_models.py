"""
evaluate_and_compare_three_models.py - Comparative evaluation of the 3 DistilBERT QA models:
  1. Pretrained Baseline (distilbert-base-uncased-distilled-squad)
  2. 4k Fine-Tuned Model (models/distilbert_finetuned)
  3. Full-Dataset Fine-Tuned Model (models/distilbert_full_finetuned)

All evaluated on the exact same complete official SQuAD v1.1 validation split (10,570 examples).
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

from src.utils import set_seed, get_device, get_gpu_memory_usage, save_json
from src.preprocessing import prepare_validation_features
from src.evaluate import evaluate_qa_model, perform_error_analysis


def plot_comparison_chart(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sns.set_theme(style="whitegrid")

    palette = ["#0284c7", "#10b981", "#6366f1"]

    # 1. Exact Match & F1
    df_acc = df.melt(id_vars=["display_name"], value_vars=["exact_match", "f1_score"], var_name="Metric", value_name="Score")
    sns.barplot(data=df_acc, x="display_name", y="Score", hue="Metric", palette=["#2563eb", "#10b981"], ax=axes[0])
    axes[0].set_title("SQuAD v1.1 Accuracy (EM & F1)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Score (%)", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].set_ylim(0, 100)

    # 2. Latency
    sns.barplot(data=df, x="display_name", y="latency_ms", hue="display_name", palette=palette, legend=False, ax=axes[1])
    axes[1].set_title("Inference Latency per Sample (Lower is Better)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Milliseconds (ms)", fontsize=11)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=15)

    # 3. Training Samples Scale
    sns.barplot(data=df, x="display_name", y="training_samples", hue="display_name", palette=palette, legend=False, ax=axes[2])
    axes[2].set_title("Training Dataset Scale (Examples)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Number of Samples", fontsize=11)
    axes[2].set_xlabel("")
    axes[2].tick_params(axis='x', rotation=15)

    fig.suptitle("Comprehensive Benchmark Across DistilBERT QA Iterations (Full SQuAD Validation: 10,570 Examples)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Comparison] Saved comparison plot to: {output_path}")


def main():
    set_seed(42)
    device = get_device()
    output_dir = "results/full_training"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("🔬 Evaluating and Comparing All 3 Models on FULL SQuAD v1.1 Validation Set")
    print("=" * 65)

    # 1. Load Validation Dataset
    print("[Comparison] Loading SQuAD v1.1 validation split...")
    raw_eval = load_dataset("rajpurkar/squad")["validation"]
    print(f"[Comparison] Total validation examples: {len(raw_eval):,}")

    models_config = [
        {
            "id": "baseline",
            "path": "models/baseline",
            "display_name": "Pretrained Baseline (Full SQuAD)",
            "training_samples": 87599,
            "description": "DistilBERT distilled on full SQuAD v1.1 (87k)"
        },
        {
            "id": "finetuned_4k",
            "path": "models/distilbert_finetuned",
            "display_name": "Fine-Tuned 4k Subset (Ours)",
            "training_samples": 4000,
            "description": "DistilBERT fine-tuned on 4,000 SQuAD examples"
        },
        {
            "id": "full_finetuned",
            "path": "models/distilbert_full_finetuned",
            "display_name": "Full Fine-Tuned (Ours)",
            "training_samples": 87599,
            "description": "DistilBERT fine-tuned on full 87,599 SQuAD dataset"
        }
    ]

    # Preprocess validation features using shared tokenizer
    print("[Comparison] Preprocessing validation features...")
    shared_tokenizer = AutoTokenizer.from_pretrained(models_config[0]["path"])
    eval_features = raw_eval.map(
        lambda ex: prepare_validation_features(ex, shared_tokenizer, max_seq_length=384, doc_stride=128),
        batched=True,
        remove_columns=raw_eval.column_names,
        desc="Tokenizing full validation dataset",
    )
    print(f"[Comparison] Generated {len(eval_features):,} feature chunks.")

    results_summary = []

    for cfg in models_config:
        model_path = cfg["path"]
        name = cfg["display_name"]
        print(f"\n" + "-" * 55)
        print(f"👉 Evaluating: {name} ({model_path})")
        print("-" * 55)

        if not Path(model_path).exists():
            print(f"[Warning] Path '{model_path}' not found! Skipping...")
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        model.to(device)

        metrics, preds, total_time = evaluate_qa_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_features,
            raw_eval_dataset=raw_eval,
            batch_size=32,
            device=device,
        )

        sub_output = os.path.join(output_dir, f"eval_{cfg['id']}")
        os.makedirs(sub_output, exist_ok=True)
        save_json(metrics, os.path.join(sub_output, "metrics.json"))
        perform_error_analysis(raw_eval, preds, output_dir=sub_output)

        rec = {
            "model_id": cfg["id"],
            "display_name": name,
            "training_samples": cfg["training_samples"],
            "exact_match": metrics["exact_match"],
            "f1_score": metrics["f1"],
            "latency_ms": metrics["latency_ms_per_sample"],
            "total_inference_time_sec": metrics["inference_time_sec"],
        }
        results_summary.append(rec)

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_results = pd.DataFrame(results_summary)
    csv_path = os.path.join(output_dir, "three_models_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Comparison] Saved comparison CSV to: {csv_path}")

    chart_path = os.path.join(output_dir, "three_models_comparison.png")
    plot_comparison_chart(df_results, chart_path)

    # Print results table
    print("\n" + "=" * 65)
    print("🏆 FINAL COMPARISON ACROSS ALL 3 MODELS (Full SQuAD Val: 10,570 Samples):")
    print("=" * 65)
    print(df_results.to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
