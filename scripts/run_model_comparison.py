"""
run_model_comparison.py - Benchmark and compare DistilBERT, BERT-base, and RoBERTa-base.

Measures:
  - Exact Match (EM) & F1 on standard SQuAD validation subset
  - Number of trainable parameters
  - Disk model size (MB)
  - Mean inference latency (ms/sample) on RTX 3060 GPU
  - Peak GPU memory allocation (MB)
  - Generates: results/model_comparison.csv & results/model_comparison.png
"""

import os
import sys
import gc
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, get_device, get_gpu_memory_usage, compute_exact_match, compute_f1_score, metric_max_over_ground_truths
from src.preprocessing import prepare_validation_features
from src.evaluate import postprocess_qa_predictions


def benchmark_model_architecture(
    model_name: str,
    raw_eval_subset,
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """Benchmark parameter count, inference speed, VRAM, and SQuAD accuracy."""
    print(f"\n" + "-" * 50)
    print(f"🔬 Benchmarking Architecture: {model_name}")
    print("-" * 50)

    # Clean GPU memory before run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 1. Parameter Count
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6

    # 2. Preprocess validation subset
    eval_features = raw_eval_subset.map(
        lambda ex: prepare_validation_features(ex, tokenizer, max_seq_length=384, doc_stride=128),
        batched=True,
        remove_columns=raw_eval_subset.column_names,
        desc=f"Tokenizing for {model_name}"
    )

    # 3. Measure Inference Latency & Peak VRAM
    input_ids = torch.tensor(eval_features["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(eval_features["attention_mask"], dtype=torch.long)

    # Warmup pass
    with torch.no_grad():
        dummy_in = input_ids[:2].to(device)
        dummy_mask = attention_mask[:2].to(device)
        _ = model(input_ids=dummy_in, attention_mask=dummy_mask)

    all_start_logits = []
    all_end_logits = []

    start_time = time.time()
    num_chunks = len(input_ids)
    with torch.no_grad():
        for i in range(0, num_chunks, batch_size):
            b_ids = input_ids[i : i + batch_size].to(device)
            b_mask = attention_mask[i : i + batch_size].to(device)
            outputs = model(input_ids=b_ids, attention_mask=b_mask)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    total_eval_time = time.time() - start_time
    latency_per_sample = (total_eval_time / len(raw_eval_subset)) * 1000  # ms/sample

    # Peak VRAM
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # 4. Post-process and compute accuracy metrics
    preds = postprocess_qa_predictions(
        examples=raw_eval_subset,
        features=eval_features,
        raw_predictions=(all_start_logits, all_end_logits),
    )

    em_scores = []
    f1_scores = []
    for ex in raw_eval_subset:
        p = preds.get(ex["id"], "")
        gts = ex["answers"]["text"]
        em_scores.append(metric_max_over_ground_truths(compute_exact_match, p, gts))
        f1_scores.append(metric_max_over_ground_truths(compute_f1_score, p, gts))

    mean_em = float(np.mean(em_scores))
    mean_f1 = float(np.mean(f1_scores))

    # Free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    results = {
        "model_name": model_name,
        "parameters_m": round(num_params_m, 1),
        "exact_match": round(mean_em, 2),
        "f1_score": round(mean_f1, 2),
        "latency_ms": round(latency_per_sample, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }

    print(f"  - Parameters:     {results['parameters_m']}M")
    print(f"  - Exact Match:    {results['exact_match']}%")
    print(f"  - Token F1:       {results['f1_score']}%")
    print(f"  - Latency:        {results['latency_ms']} ms/sample")
    print(f"  - Peak VRAM:      {results['peak_vram_mb']} MB")
    return results


def plot_model_comparison(df: pd.DataFrame, output_path: str = "results/model_comparison.png"):
    """Generate 4-chart comparative visual dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    sns.set_theme(style="whitegrid")

    palette = ["#0284c7", "#10b981", "#6366f1", "#f43f5e"][:len(df)]

    # 1. Exact Match & F1
    df_acc = df.melt(id_vars=["model_name"], value_vars=["exact_match", "f1_score"], var_name="Metric", value_name="Score")
    sns.barplot(data=df_acc, x="model_name", y="Score", hue="Metric", palette=["#2563eb", "#10b981"], ax=axes[0, 0])
    axes[0, 0].set_title("Accuracy Metrics (EM & F1)", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("Percentage (%)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis='x', rotation=15)

    # 2. Parameter Count
    sns.barplot(data=df, x="model_name", y="parameters_m", palette=palette, ax=axes[0, 1])
    axes[0, 1].set_title("Model Parameter Count (Millions)", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("Parameters (M)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis='x', rotation=15)

    # 3. Inference Latency
    sns.barplot(data=df, x="model_name", y="latency_ms", palette=palette, ax=axes[1, 0])
    axes[1, 0].set_title("Inference Latency per Query (Lower is Better)", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Milliseconds (ms)")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis='x', rotation=15)

    # 4. Peak VRAM Consumption
    sns.barplot(data=df, x="model_name", y="peak_vram_mb", palette=palette, ax=axes[1, 1])
    axes[1, 1].set_title("Peak GPU VRAM Usage (RTX 3060)", fontsize=13, fontweight="bold")
    axes[1, 1].set_ylabel("Memory (MB)")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis='x', rotation=15)

    fig.suptitle("Comparative Evaluation: DistilBERT vs BERT vs RoBERTa", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Benchmark] Saved comparison plot to: {output_path}")


def main():
    set_seed(42)
    device = get_device()

    # Load SQuAD validation split
    raw_val = load_dataset("rajpurkar/squad")["validation"]
    # Standard 200-sample benchmark slice for fair, direct comparison
    benchmark_slice = raw_val.select(range(200))
    print(f"[Benchmark] Running comparative benchmark over {len(benchmark_slice)} validation examples.")

    models_to_compare = [
        "models/baseline",
        "models/distilbert_finetuned",
        "bert-base-uncased",
        "roberta-base",
    ]

    all_records = []
    for m in models_to_compare:
        rec = benchmark_model_architecture(m, benchmark_slice, device=device, batch_size=8)
        all_records.append(rec)

    df = pd.DataFrame(all_records)
    csv_path = "results/model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Benchmark] Saved metrics table to: {csv_path}")

    plot_model_comparison(df, "results/model_comparison.png")
    plot_model_comparison(df, "results/visualizations/model_comparison.png")


if __name__ == "__main__":
    main()
