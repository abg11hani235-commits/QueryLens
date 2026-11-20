"""
train.py - Fine-Tuning Pipeline for DistilBERT on SQuAD v1.1.

Optimized for:
  - NVIDIA RTX 3060 Laptop GPU (6GB VRAM)
  - FP16 Mixed Precision
  - Gradient Accumulation (effective batch size 16-32)
  - Hugging Face Trainer with AdamW, linear warmup, and checkpoint saving
  - Loss curves plotting (training & validation loss)
"""

import os
import sys
import argparse
import math
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset, Dataset

from src.utils import set_seed, get_device, load_yaml_config, save_json
from src.preprocessing import prepare_train_features, prepare_validation_features
from src.evaluate import evaluate_qa_model, perform_error_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer model on SQuAD v1.1")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to YAML training config")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="models/distilbert_finetuned", help="Directory to save final model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional subset of training examples (for quick test/debugging)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional subset of validation examples")
    return parser.parse_args()


def plot_loss_curves(log_history: list, output_path: str = "results/training_loss.png") -> None:
    """
    Plot and save training and validation loss curves from Trainer log history.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_steps = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_losses.append(entry["loss"])
            train_steps.append(entry["step"])
        if "eval_loss" in entry and "step" in entry:
            eval_losses.append(entry["eval_loss"])
            eval_steps.append(entry["step"])

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Validation Loss", color="#d62728", marker="o", linewidth=2)

    ax.set_title("DistilBERT Fine-Tuning Loss Curve (SQuAD v1.1)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    if train_losses or eval_losses:
        ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    # Also save copy in results/visualizations/
    viz_path = os.path.join(os.path.dirname(output_path), "visualizations", os.path.basename(output_path))
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    fig.savefig(viz_path, dpi=300)
    plt.close(fig)
    print(f"[Trainer] Saved training loss plot to: {output_path} and {viz_path}")


def main():
    args = parse_args()
    config = load_yaml_config(args.config) if os.path.exists(args.config) else {}

    # Extract configuration with CLI overrides
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    model_name = args.model_name or config.get("models_to_train", {}).get("distilbert", {}).get("model_name_or_path", "distilbert-base-uncased")
    output_dir = args.output_dir or config.get("experiment", {}).get("output_dir", "models/distilbert_finetuned")
    epochs = args.epochs or config.get("training", {}).get("num_train_epochs", 2)
    batch_size = args.batch_size or config.get("training", {}).get("per_device_train_batch_size", 8)
    lr = args.learning_rate or config.get("training", {}).get("learning_rate", 3e-5)
    grad_accum = config.get("training", {}).get("gradient_accumulation_steps", 2)
    max_seq_length = config.get("data", {}).get("max_seq_length", 384)
    doc_stride = config.get("data", {}).get("doc_stride", 128)
    fp16 = torch.cuda.is_available() and config.get("training", {}).get("fp16", True)

    print(f"\n" + "=" * 60)
    print(f"🚀 ContextQA - Starting Model Fine-Tuning:")
    print(f"  - Model:                  {model_name}")
    print(f"  - Device:                 {device}")
    print(f"  - FP16 Mixed Precision:   {fp16}")
    print(f"  - Batch Size:             {batch_size} (Grad Accum: {grad_accum}, Effective: {batch_size * grad_accum})")
    print(f"  - Learning Rate:          {lr}")
    print(f"  - Epochs:                 {epochs}")
    print(f"  - Sequence Length:        {max_seq_length} (Stride: {doc_stride})")
    print(f"  - Output Directory:       {output_dir}")
    print("=" * 60 + "\n")

    # 1. Load Tokenizer & Model
    print(f"[Trainer] Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # 2. Load SQuAD v1.1 Dataset
    dataset_name = config.get("data", {}).get("dataset_name", "rajpurkar/squad")
    raw_datasets = load_dataset(dataset_name)

    raw_train = raw_datasets["train"]
    raw_eval = raw_datasets["validation"]

    if args.max_train_samples:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
        print(f"[Trainer] Using SUBSET of {len(raw_train)} training samples for rapid testing.")
    if args.max_eval_samples:
        raw_eval = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))
        print(f"[Trainer] Using SUBSET of {len(raw_eval)} validation samples for evaluation.")

    # 3. Preprocess Dataset
    print("[Trainer] Preprocessing training features...")
    train_dataset = raw_train.map(
        lambda examples: prepare_train_features(
            examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            pad_to_max_length=True,
        ),
        batched=True,
        remove_columns=raw_train.column_names,
        desc="Tokenizing train dataset",
    )

    print("[Trainer] Preprocessing validation features...")
    eval_dataset = raw_eval.map(
        lambda examples: prepare_validation_features(
            examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            pad_to_max_length=True,
        ),
        batched=True,
        remove_columns=raw_eval.column_names,
        desc="Tokenizing validation dataset",
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=50,
        fp16=fp16,
        logging_steps=10,
        eval_strategy="no",  # We evaluate explicitly with official SQuAD post-processing
        save_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,  # 0 for Windows multiprocessing safety
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    # 5. Train
    print("[Trainer] Beginning training loop...")
    train_result = trainer.train()
    print(f"[Trainer] Training completed in {train_result.metrics.get('train_runtime', 0):.2f}s.")

    # 6. Save Final Model & Tokenizer
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Trainer] Model and tokenizer successfully saved to '{output_dir}'.")

    # 7. Plot Loss Curves
    plot_loss_curves(trainer.state.log_history, output_path="results/training_loss.png")

    # 8. Post-Training Evaluation on Validation Set
    print("\n[Trainer] Evaluating fine-tuned model on SQuAD validation set...")
    metrics, predictions, eval_time = evaluate_qa_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        raw_eval_dataset=raw_eval,
        batch_size=16,
        device=device,
    )

    # Add training metadata to metrics
    metrics["train_runtime_sec"] = train_result.metrics.get("train_runtime", 0)
    metrics["train_samples_per_second"] = train_result.metrics.get("train_samples_per_second", 0)
    metrics["train_loss"] = train_result.metrics.get("train_loss", 0)

    save_json(metrics, "results/metrics.json")
    perform_error_analysis(raw_eval, predictions, output_dir="results")

    print("[Trainer] Pipeline successfully finished!")


if __name__ == "__main__":
    main()
