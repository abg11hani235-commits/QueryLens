"""
evaluate.py - Official SQuAD v1.1 Evaluation, Post-Processing, and Error Analysis.

Computes:
  - Exact Match (EM)
  - Token-level F1
  - Inference Latency (ms/sample)
  - Error Analysis across answer lengths, context lengths, and question types
  - Produces: metrics.json, predictions.csv, error_analysis.csv, evaluation_report.md
"""

import os
import sys
import time
import collections
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator

from src.utils import (
    normalize_answer,
    compute_exact_match,
    compute_f1_score,
    metric_max_over_ground_truths,
    save_json,
    get_device,
)
from src.preprocessing import prepare_validation_features


def postprocess_qa_predictions(
    examples,
    features,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> Dict[str, str]:
    """
    Post-process raw start and end logits from multiple sliding-window chunks
    into final predicted answer strings for each original SQuAD example.

    Args:
        examples: Original raw validation dataset split (containing 'id', 'context').
        features: Tokenized validation features (containing 'example_id', 'offset_mapping').
        raw_predictions: Tuple of (all_start_logits, all_end_logits).
        n_best_size: Number of top logits to consider per chunk.
        max_answer_length: Maximum allowed answer span length.

    Returns:
        Dictionary mapping example_id to predicted answer text.
    """
    all_start_logits, all_end_logits = raw_predictions

    # Build mapping from example_id to feature chunk indices
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    all_predictions = collections.OrderedDict()

    for example in tqdm(examples, desc="[Postprocessing] Extracting best spans"):
        example_id = example["id"]
        context = example["context"]
        feature_indices = example_to_features[example_id]

        valid_answers = []

        # Iterate over all sliding window features for this example
        for feat_idx in feature_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits = all_end_logits[feat_idx]
            offset_mapping = features[feat_idx]["offset_mapping"]

            # Sort candidate start and end token indices
            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1].tolist()
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Ignore out-of-scope spans
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Ignore invalid spans where end is before start
                    if end_index < start_index:
                        continue
                    # Ignore answers exceeding max length
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    valid_answers.append({
                        "score": float(start_logits[start_index] + end_logits[end_index]),
                        "text": context[start_char:end_char].strip(),
                    })

        # Select the best answer across all features
        if len(valid_answers) > 0:
            best_answer = max(valid_answers, key=lambda x: x["score"])
            all_predictions[example_id] = best_answer["text"]
        else:
            all_predictions[example_id] = ""

    return all_predictions


def evaluate_qa_model(
    model,
    tokenizer,
    eval_dataset,
    raw_eval_dataset,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> Tuple[Dict[str, float], Dict[str, str], float]:
    """
    Run evaluation loop over preprocessed validation features and compute SQuAD metrics.

    Returns:
        Tuple of (metrics_dict, predictions_dict, total_inference_time_seconds).
    """
    device = device or get_device()
    model.to(device)
    model.eval()

    # Prepare PyTorch DataLoader
    # Exclude non-tensor columns for collation
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    all_start_logits = []
    all_end_logits = []

    start_time = time.time()
    print(f"[Evaluation] Running inference over {len(eval_dataset):,} validation feature chunks...")

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            for batch in tqdm(eval_dataloader, desc="[Evaluation] Batches"):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                all_start_logits.append(outputs.start_logits.float().cpu().numpy())
                all_end_logits.append(outputs.end_logits.float().cpu().numpy())

    total_time = time.time() - start_time
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Post-process into final textual predictions
    predictions = postprocess_qa_predictions(
        examples=raw_eval_dataset,
        features=eval_dataset,
        raw_predictions=(all_start_logits, all_end_logits),
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
    )

    # Compute official SQuAD metrics
    em_scores = []
    f1_scores = []

    for example in raw_eval_dataset:
        example_id = example["id"]
        pred_text = predictions.get(example_id, "")
        ground_truths = example["answers"]["text"]

        em = metric_max_over_ground_truths(compute_exact_match, pred_text, ground_truths)
        f1 = metric_max_over_ground_truths(compute_f1_score, pred_text, ground_truths)

        em_scores.append(em)
        f1_scores.append(f1)

    exact_match = float(np.mean(em_scores))
    f1 = float(np.mean(f1_scores))
    latency_ms = (total_time / len(raw_eval_dataset)) * 1000

    metrics = {
        "exact_match": round(exact_match, 2),
        "f1": round(f1, 2),
        "num_samples": len(raw_eval_dataset),
        "inference_time_sec": round(total_time, 2),
        "latency_ms_per_sample": round(latency_ms, 2),
    }

    print("\n" + "=" * 50)
    print(f"📊 SQuAD v1.1 Evaluation Results:")
    print(f"  - Exact Match (EM): {metrics['exact_match']}%")
    print(f"  - Token F1 Score:   {metrics['f1']}%")
    print(f"  - Total Eval Time:  {metrics['inference_time_sec']}s")
    print(f"  - Latency:          {metrics['latency_ms_per_sample']} ms/example")
    print("=" * 50 + "\n")

    return metrics, predictions, total_time


def perform_error_analysis(
    raw_eval_dataset,
    predictions: Dict[str, str],
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Perform deep error analysis: categorize predictions into Exact, Partial, and Misses,
    and correlate performance with context length, question length, and answer length.

    Outputs:
      - results/predictions.csv
      - results/error_analysis.csv
      - results/evaluation_report.md
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for example in raw_eval_dataset:
        eid = example["id"]
        pred = predictions.get(eid, "")
        ground_truths = example["answers"]["text"]

        em = metric_max_over_ground_truths(compute_exact_match, pred, ground_truths)
        f1 = metric_max_over_ground_truths(compute_f1_score, pred, ground_truths)

        if em == 100.0:
            category = "Exact Match"
        elif f1 > 0.0:
            category = "Partial Match"
        else:
            category = "Complete Miss"

        first_gt = ground_truths[0] if ground_truths else ""
        records.append({
            "id": eid,
            "title": example.get("title", ""),
            "question": example["question"],
            "first_ground_truth": first_gt,
            "all_ground_truths": " | ".join(ground_truths),
            "prediction": pred,
            "exact_match": em,
            "f1_score": round(f1, 2),
            "error_category": category,
            "context_word_len": len(example["context"].split()),
            "question_word_len": len(example["question"].split()),
            "answer_word_len": len(first_gt.split()),
        })

    df = pd.DataFrame(records)

    # Save CSVs
    pred_path = os.path.join(output_dir, "predictions.csv")
    df[["id", "question", "all_ground_truths", "prediction", "exact_match", "f1_score"]].to_csv(
        pred_path, index=False, encoding="utf-8"
    )

    error_path = os.path.join(output_dir, "error_analysis.csv")
    df.to_csv(error_path, index=False, encoding="utf-8")

    # Generate Markdown Report
    cat_counts = df["error_category"].value_counts().to_dict()
    total = len(df)

    report_content = f"""# SQuAD v1.1 Error Analysis & Evaluation Report

## 1. Overall Performance Summary
- **Total Evaluated Examples**: {total:,}
- **Exact Match (EM)**: {df['exact_match'].mean():.2f}%
- **Token-Level F1 Score**: {df['f1_score'].mean():.2f}%

## 2. Prediction Breakdown
- **Exact Match**: {cat_counts.get('Exact Match', 0):,} ({cat_counts.get('Exact Match', 0)/total*100:.1f}%)
- **Partial Match**: {cat_counts.get('Partial Match', 0):,} ({cat_counts.get('Partial Match', 0)/total*100:.1f}%)
- **Complete Miss**: {cat_counts.get('Complete Miss', 0):,} ({cat_counts.get('Complete Miss', 0)/total*100:.1f}%)

## 3. Error Analysis by Answer Length
| Answer Length (Words) | Count | Mean EM (%) | Mean F1 (%) |
|---|---|---|---|
"""
    df["ans_len_bin"] = pd.cut(df["answer_word_len"], bins=[0, 2, 5, 10, 100], labels=["1-2 words", "3-5 words", "6-10 words", ">10 words"])
    for bin_name, group in df.groupby("ans_len_bin", observed=False):
        report_content += f"| {bin_name} | {len(group):,} | {group['exact_match'].mean():.2f}% | {group['f1_score'].mean():.2f}% |\n"

    report_content += """
## 4. Representative Qualitative Error Samples

### Partial Matches (Correct Concept, Boundary Imperfection)
"""
    partial_samples = df[df["error_category"] == "Partial Match"].head(3)
    for _, row in partial_samples.iterrows():
        report_content += f"- **Question**: {row['question']}\n"
        report_content += f"  - **Ground Truth**: `{row['all_ground_truths']}`\n"
        report_content += f"  - **Prediction**: `{row['prediction']}` (F1: {row['f1_score']}%)\n\n"

    report_content += """### Complete Misses (Incorrect Span Selected)
"""
    miss_samples = df[df["error_category"] == "Complete Miss"].head(3)
    for _, row in miss_samples.iterrows():
        report_content += f"- **Question**: {row['question']}\n"
        report_content += f"  - **Ground Truth**: `{row['all_ground_truths']}`\n"
        report_content += f"  - **Prediction**: `{row['prediction']}`\n\n"

    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[ErrorAnalysis] Saved predictions to '{pred_path}', error analysis to '{error_path}', and report to '{report_path}'.")
    return df


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate QA model on SQuAD v1.1 validation set")
    parser.add_argument("--model_path", type=str, default="models/distilbert_finetuned", help="Path or HF ID of QA model")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Number of validation samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save metrics and predictions")
    return parser.parse_args()


if __name__ == "__main__":
    from datasets import load_dataset
    args = parse_args()
    device = get_device()

    print(f"\n[Evaluation] Loading model and tokenizer from '{args.model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    model.to(device)

    print("[Evaluation] Loading SQuAD v1.1 validation split...")
    raw_eval = load_dataset("rajpurkar/squad")["validation"]
    if args.max_eval_samples:
        raw_eval = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))
        print(f"[Evaluation] Evaluating on {len(raw_eval)} samples...")

    eval_features = raw_eval.map(
        lambda ex: prepare_validation_features(ex, tokenizer, max_seq_length=384, doc_stride=128),
        batched=True,
        remove_columns=raw_eval.column_names,
        desc="Tokenizing validation set",
    )

    metrics, predictions, eval_time = evaluate_qa_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_features,
        raw_eval_dataset=raw_eval,
        batch_size=args.batch_size,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(metrics, os.path.join(args.output_dir, "metrics.json"))
    perform_error_analysis(raw_eval, predictions, output_dir=args.output_dir)
    print(f"[Evaluation] Completed! Metrics saved to {args.output_dir}.")
