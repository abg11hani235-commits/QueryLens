"""
train_full_squad.py - Full SQuAD v1.1 Fine-Tuning Pipeline for DistilBERT.

Trained on the complete SQuAD v1.1 dataset (87,599 examples).
Optimized for:
  - NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)
  - FP16 Mixed Precision
  - Gradient Accumulation (Effective Batch Size = 16)
  - Periodic Checkpoint Saving with Automatic Resume Support
  - Evaluation across complete official validation set (10,570 examples)
  - Results saved under results/full_training/
"""

import os
import sys
import argparse
import time
from pathlib import Path

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
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

from src.utils import set_seed, get_device, get_gpu_memory_usage, save_json
from src.preprocessing import prepare_train_features, prepare_validation_features
from src.evaluate import evaluate_qa_model, perform_error_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Full SQuAD v1.1 Fine-Tuning for DistilBERT")
    parser.add_argument("--model_name", type=str, default="models/distilbert_base", help="Base model path or HF ID")
    parser.add_argument("--output_dir", type=str, default="models/distilbert_full_finetuned", help="Directory for checkpoints and final model")
    parser.add_argument("--results_dir", type=str, default="results/full_training", help="Directory for metrics and reports")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device training batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Peak learning rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log metrics every X steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def plot_loss_curves(log_history: list, output_path: str) -> None:
    """Plot and save training loss curves from Trainer log history."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_losses = []
    train_steps = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_losses.append(entry["loss"])
            train_steps.append(entry["step"])

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="Training Loss", color="#0284c7", linewidth=2)
        ax.legend(frameon=True)

    ax.set_title("Full SQuAD v1.1 Fine-Tuning Loss Curve (DistilBERT)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Optimization Steps", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Trainer] Saved training loss plot to: {output_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    fp16 = torch.cuda.is_available()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("🔥 ContextQA - Full SQuAD v1.1 Fine-Tuning Execution")
    print(f"  - Base Checkpoint:       {args.model_name}")
    print(f"  - Device:                {device} (FP16: {fp16})")
    print(f"  - Per-Device Batch Size: {args.batch_size} (Grad Accum: {args.grad_accum}, Effective Batch: {args.batch_size * args.grad_accum})")
    print(f"  - Epochs:                {args.epochs}")
    print(f"  - Learning Rate:         {args.learning_rate}")
    print(f"  - Checkpoint Save Steps: Every {args.save_steps} steps")
    print(f"  - Output Checkpoints:    {args.output_dir}")
    print(f"  - Results Directory:     {args.results_dir}")
    print("=" * 65 + "\n")

    # 1. Load Tokenizer & Model
    print(f"[Trainer] Loading base model and tokenizer from '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # 2. Load Full SQuAD Dataset
    print("[Trainer] Loading complete SQuAD v1.1 dataset from Hugging Face...")
    raw_datasets = load_dataset("rajpurkar/squad")
    raw_train = raw_datasets["train"]
    raw_eval = raw_datasets["validation"]
    print(f"[Trainer] SQuAD Dataset loaded: {len(raw_train):,} train samples, {len(raw_eval):,} validation samples.")

    # 3. Preprocess Dataset
    print(f"[Trainer] Tokenizing complete training set ({len(raw_train):,} samples)...")
    train_dataset = raw_train.map(
        lambda examples: prepare_train_features(
            examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            pad_to_max_length=True,
        ),
        batched=True,
        remove_columns=raw_train.column_names,
        desc="Tokenizing full train dataset",
    )
    print(f"[Trainer] Total training feature chunks generated: {len(train_dataset):,}")

    print(f"[Trainer] Tokenizing complete validation set ({len(raw_eval):,} samples)...")
    eval_dataset = raw_eval.map(
        lambda examples: prepare_validation_features(
            examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            pad_to_max_length=True,
        ),
        batched=True,
        remove_columns=raw_eval.column_names,
        desc="Tokenizing full validation dataset",
    )
    print(f"[Trainer] Total validation feature chunks generated: {len(eval_dataset):,}")

    # Check for existing checkpoint to resume from
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f"[Trainer] Found existing checkpoint: '{last_checkpoint}'. Resuming training...")
    else:
        print("[Trainer] No previous checkpoint found. Starting fresh training run.")

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=200,
        fp16=fp16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    # 5. Train
    print("\n[Trainer] Starting training loop on GPU...")
    start_train_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    total_train_runtime = time.time() - start_train_time
    print(f"\n[Trainer] Training completed successfully in {total_train_runtime:.2f}s ({total_train_runtime/60:.1f} mins)!")

    # 6. Save Final Model & Tokenizer
    print(f"[Trainer] Saving final fine-tuned model and tokenizer to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[Trainer] Final model saved successfully.")

    # 7. Save Loss Curves and Log History
    loss_plot_path = os.path.join(args.results_dir, "training_loss.png")
    plot_loss_curves(trainer.state.log_history, output_path=loss_plot_path)

    log_history_path = os.path.join(args.results_dir, "training_log_history.json")
    save_json(trainer.state.log_history, log_history_path)

    # 8. Complete Official Validation Evaluation (10,570 samples)
    print("\n" + "=" * 65)
    print(f"📊 Starting Official Evaluation over COMPLETE Validation Set ({len(raw_eval):,} samples)...")
    print("=" * 65)

    metrics, predictions, eval_time = evaluate_qa_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        raw_eval_dataset=raw_eval,
        batch_size=32,
        device=device,
    )

    # Attach training metadata
    metrics["model_checkpoint"] = args.output_dir
    metrics["dataset"] = "SQuAD v1.1 (Full)"
    metrics["training_samples"] = len(raw_train)
    metrics["validation_samples"] = len(raw_eval)
    metrics["epochs"] = args.epochs
    metrics["effective_batch_size"] = args.batch_size * args.grad_accum
    metrics["learning_rate"] = args.learning_rate
    metrics["train_runtime_sec"] = round(total_train_runtime, 2)
    metrics["train_runtime_min"] = round(total_train_runtime / 60, 2)
    metrics["train_loss"] = round(float(train_result.metrics.get("train_loss", 0.0)), 4)
    metrics["peak_gpu_memory"] = get_gpu_memory_usage()

    metrics_path = os.path.join(args.results_dir, "metrics.json")
    save_json(metrics, metrics_path)

    # Error Analysis & Report
    perform_error_analysis(raw_eval, predictions, output_dir=args.results_dir)

    print("\n" + "=" * 65)
    print(f"✅ Full SQuAD v1.1 Training & Evaluation Finished!")
    print(f"  - Final Exact Match (EM): {metrics['exact_match']}%")
    print(f"  - Final Token-Level F1:   {metrics['f1']}%")
    print(f"  - Output Checkpoint:      {args.output_dir}")
    print(f"  - Full Results:           {args.results_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
