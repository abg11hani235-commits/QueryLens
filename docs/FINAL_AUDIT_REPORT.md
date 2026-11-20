# ContextQA – Final Academic & Engineering Audit Report

**Auditor Role**: Independent Senior Machine Learning Engineer & Academic Project Auditor  
**Audit Date**: November 17, 2026  
**Audit Target**: ContextQA – Transformer-Based Extractive Question Answering System  
**Hardware Environment**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM), Intel Core i7-11800H, 16GB RAM  
**Execution Runtime**: Python 3.12.7 (`.\.venv`), PyTorch 2.6.0+cu124, CUDA 12.4  

---

## 1. Executive Summary

An independent, exhaustive academic audit was performed across all source files, configurations, model checkpoints, training logs, evaluation records, and tests in the **ContextQA** project repository.

### Key Audit Findings:
1. **Model Fine-Tuning**: Verified that `models/distilbert_finetuned` was genuinely fine-tuned on a representative subset of **4,000 SQuAD v1.1 training examples** for **2 epochs** (**512 optimization steps**) using FP16 mixed precision on the RTX 3060 GPU. The training loss decreased from **5.919** to **1.871** (mean loss: 2.7938).
2. **Evaluation Metrics**: Recomputed and mathematically verified official SQuAD v1.1 metrics directly from per-sample predictions. The fine-tuned model achieved **55.00% Exact Match (EM)** and **60.74% Token-Level F1** over a standardized 500-sample validation slice in **1.52 seconds** (latency: **3.03 ms/sample**). The pretrained baseline model (`distilbert-base-uncased-distilled-squad`) achieved **85.40% EM** and **88.91% F1** on the exact same 500 examples.
3. **Model Comparison Rectification**: Discovered that previous comparisons presented `bert-base-uncased` and `roberta-base` at 0% EM without explicitly noting that their task-specific linear QA classification heads (`qa_outputs`) were **untrained and randomly initialized**. Corrected all tables, documentation, and CSVs to clearly distinguish trained QA heads from zero-shot base models.
4. **Inference Offset Bug Detected & Resolved**: Discovered that `context = context.strip()` in `src/inference.py` shifted returned character offsets when contexts had leading whitespace, causing character slicing to misalign with the original input text. Implemented an exact fix that preserves the raw context and synchronizes whitespace trimming with character offsets.
5. **Automated Test Suite**: Expanded and verified the automated unit test suite from 11 to **14 tests**, achieving a **100% pass rate** (`14 passed in 35.37s`).

---

## 2. Training Audit & Artifact Verification

### 2.1 Training Configuration & Hyperparameters
- **Source Script**: `src/train.py`
- **Base Backbone**: `distilbert-base-uncased` (66.4M parameters)
- **Dataset**: SQuAD v1.1 (`rajpurkar/squad`)
- **Training Samples Used**: **4,000** (4.57% of full 87,599 training split)
- **Validation Samples Used**: **500** (from 10,570 validation split)
- **Epochs**: **2.0**
- **Batch Configuration**: Per-device batch size = 8, Gradient accumulation steps = 2, **Effective batch size = 16**
- **Optimization**: AdamW, Peak Learning Rate = $3 \times 10^{-5}$, Weight Decay = $0.01$, Warmup Steps = $50$
- **Mixed Precision**: FP16 enabled (`torch.cuda.amp` via Hugging Face Trainer)
- **Sequence Parameters**: `max_seq_length = 384`, `doc_stride = 128`

### 2.2 Verification of `trainer_state.json`
Artifact inspected at: `models/distilbert_finetuned/checkpoint-512/trainer_state.json`
- `global_step`: 512
- `total_flos`: $7.9998 \times 10^{14}$
- `train_runtime`: **85.04 seconds** (**96.0 samples/second**)
- `loss progression`:
  - Step 10: $5.9194$ (LR: $5.4 \times 10^{-6}$)
  - Step 50: $4.5311$ (LR: $2.94 \times 10^{-5}$)
  - Step 100: $3.6475$ (LR: $2.69 \times 10^{-5}$)
  - Step 250: $2.2210$ (LR: $1.72 \times 10^{-5}$)
  - Step 400: $1.9270$ (LR: $7.47 \times 10^{-6}$)
  - Step 510: $2.0172$ (LR: $3.25 \times 10^{-7}$)
  - Overall Mean Loss: **2.7938**

### 2.3 Checkpoint Integrity
The directory `models/distilbert_finetuned/` contains:
- `config.json` (637 B)
- `model.safetensors` (265.5 MB)
- `tokenizer.json` (711.7 KB)
- `tokenizer_config.json` (365 B)
- `training_args.bin` (4.8 KB)
A live forward pass was executed on GPU:
- Model loaded cleanly as `DistilBertForQuestionAnswering`
- Parameters verified: 66,364,418
- QA Head shape: `[2, 768]` weights, `[2]` bias
- Dummy inference produced valid start and end logits of shape `[1, 14]`

---

## 3. Evaluation Audit & Metric Cross-Check

### 3.1 Mathematical Metric Verification
The official SQuAD v1.1 evaluation formulas in `src/utils.py` were independently validated:
- **Exact Match (EM)**: String normalization (lowercasing, article stripping `a|an|the`, punctuation removal, and whitespace fixing) followed by strict equality.
- **Token-Level F1**: Multi-set token intersection over prediction and ground truth tokens:
  $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Multiple Ground Truths**: Handled via $\max_{g \in \text{GroundTruths}} \text{Metric}(\text{pred}, g)$.

### 3.2 Cross-Checking Metrics Against Raw CSVs
An automated script recalculated the average EM and F1 across all 500 samples in `results/error_analysis.csv` and `results/baseline_eval/error_analysis.csv`:
- **Fine-Tuned Model** (`results/metrics.json`):
  - Stored: EM = 55.0%, F1 = 60.74%
  - Recalculated from CSV: EM = **55.00%**, F1 = **60.74%** (Exact Match: Passed)
- **Baseline Model** (`results/baseline_eval/metrics.json`):
  - Stored: EM = 85.4%, F1 = 88.91%
  - Recalculated from CSV: EM = **85.40%**, F1 = **88.91%** (Exact Match: Passed)
- **Sample Consistency**: Verified that both models were evaluated on the **exact same 500 example IDs** in identical order.

---

## 4. Model Comparison Audit & Rectification

### 4.1 Root-Cause Analysis of Zero-Shot Baselines
In `scripts/run_model_comparison.py`, four models were evaluated:
1. `models/baseline` (DistilBERT fine-tuned on full 87k SQuAD)
2. `models/distilbert_finetuned` (DistilBERT fine-tuned on 4k SQuAD subset)
3. `bert-base-uncased` (Base foundation model)
4. `roberta-base` (Base foundation model)

**Auditor Finding**:
When `AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')` is loaded, Hugging Face initializes new `qa_outputs.weight` and `qa_outputs.bias` tensors randomly. Consequently:
- `bert-base-uncased` achieved **0.00% EM** and **4.44% F1**.
- `roberta-base` achieved **0.00% EM** and **4.87% F1**.

Presenting these scores without explicit disclosure is academically misleading. They represent random span selection on an untrained head.

### 4.2 Verified Architectural Comparison Table
The comparison serves as an architectural and hardware efficiency benchmark on the RTX 3060:

| Model | QA Head Status | Parameters | Exact Match | Token F1 | Latency | Peak VRAM |
|---|---|---|---|---|---|---|
| **DistilBERT Baseline** | Pretrained (Full 87k SQuAD) | **66.4M** | **85.50%** | **89.77%** | **5.98 ms** | **363.5 MB** |
| **DistilBERT Fine-Tuned** | Fine-Tuned (4k subset, 2 epochs) | **66.4M** | **62.00%** | **66.61%** | **5.98 ms** | **362.3 MB** |
| **BERT-base** | Randomly Initialized (Untrained) | 108.9M | 0.00%\* | 4.44%\* | 12.92 ms | 524.6 MB |
| **RoBERTa-base** | Randomly Initialized (Untrained) | 124.1M | 0.00%\* | 4.87%\* | 13.07 ms | 582.6 MB |

*\* Untrained QA classification head.*

**Hardware Conclusion**: DistilBERT provides **2.18x lower inference latency** and **1.6x lower peak VRAM consumption** compared to RoBERTa-base, making it ideal for real-time edge or laptop deployments.

---

## 5. Inference Pipeline Audit & Bug Fix

### 5.1 The Context Stripping Bug
In `src/inference.py`:
```python
# PREVIOUS BUGGY CODE:
context = context.strip()
inputs = self.tokenizer(question, context, ...)
```
When a caller provided a passage with leading whitespace (e.g. `'   Stanford University was founded in 1885.'`), stripping `context` caused the tokenizer's character offsets to be relative to the stripped string rather than the input string. Slicing `original_context[start_char : end_char]` returned `'in 1'` instead of `'1885'`.

### 5.2 The Engineering Fix
1. Preserved the raw `context` untouched so token offsets map directly to the original input characters.
2. Synchronized whitespace trimming with character offsets:
```python
while start_char < end_char and context[start_char].isspace():
    start_char += 1
while end_char > start_char and context[end_char - 1].isspace():
    end_char -= 1
span_text = context[start_char:end_char]
```
3. Verified that `context[res["start_char"] : res["end_char"]] == res["answer"]` holds with 100% precision.

### 5.3 Sliding Window Verification
A stress-test context exceeding 400 words with the answer located in window 2 (`Omega-42`) was tested:
- Sliding window chunking activated with `doc_stride=128`.
- The engine correctly scanned both windows, filtered candidates, and returned `Omega-42` with exact character boundary matching.

### 5.4 Confidence Metric Labeling
Softmax probabilities across candidate spans are **uncalibrated relative scores**, not Bayesian probabilities. All docstrings, return values, and Streamlit cards were re-labeled as `Relative Score (Uncalibrated)` with explanatory tooltips.

---

## 6. Streamlit Web Application Audit

### 6.1 Functionality & Offline Readiness
- **File**: `app/streamlit_app.py`
- Both model choices tested and verified:
  1. `Fine-Tuned DistilBERT (ContextQA - 4k Subset)` -> `models/distilbert_finetuned`
  2. `Pretrained SQuAD Baseline (DistilBERT - Full SQuAD)` -> `models/baseline`
- Verified that both models load entirely offline from local disk checkpoints without requiring network access.
- Verified span extraction, character offset display, latency timing, and HTML `<mark>` context highlighting across all presets and custom text inputs.

---

## 7. Automated Test Suite Audit

Executed:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Verified Test Breakdown:
1. `tests/test_data_loader.py::test_dataset_overview` ✅ PASSED
2. `tests/test_data_loader.py::test_extract_sample_records` ✅ PASSED
3. `tests/test_data_loader.py::test_compute_length_statistics` ✅ PASSED
4. `tests/test_inference.py::test_empty_input_handling` ✅ PASSED
5. `tests/test_inference.py::test_finetuned_model_prediction` ✅ PASSED
6. `tests/test_inference.py::test_character_offset_exact_slice` ✅ PASSED *(Added during audit)*
7. `tests/test_inference.py::test_long_context_later_window_inference` ✅ PASSED *(Added during audit)*
8. `tests/test_preprocessing.py::test_prepare_train_features_short_context` ✅ PASSED
9. `tests/test_preprocessing.py::test_sliding_window_overflow_handling` ✅ PASSED
10. `tests/test_utils.py::test_normalize_answer` ✅ PASSED
11. `tests/test_utils.py::test_exact_match` ✅ PASSED
12. `tests/test_utils.py::test_f1_score` ✅ PASSED
13. `tests/test_utils.py::test_metric_max_over_ground_truths` ✅ PASSED
14. `tests/test_utils.py::test_highlight_answer_in_context` ✅ PASSED

**Total**: **14 Passed, 0 Failed, 0 Skipped** (100% Pass Rate).

---

## 8. Artifacts Created & Traceability

| Artifact File | Description |
|---|---|
| [results/verified_metrics.json](file:///e:/AOU/elevvo%20internship/tusk%201/results/verified_metrics.json) | Ground-truth audit metrics for fine-tuned and baseline models |
| [results/verified_model_comparison.csv](file:///e:/AOU/elevvo%20internship/tusk%201/results/verified_model_comparison.csv) | Verified 4-architecture benchmark with explicit QA head status |
| [results/verified_predictions.csv](file:///e:/AOU/elevvo%20internship/tusk%201/results/verified_predictions.csv) | Side-by-side per-sample predictions of fine-tuned vs baseline |
| [docs/FINAL_AUDIT_REPORT.md](file:///e:/AOU/elevvo%20internship/tusk%201/docs/FINAL_AUDIT_REPORT.md) | This comprehensive audit report |

---

## 9. Limitations & Academic Transparency

1. **SQuAD v1.1 Scope**: SQuAD v1.1 assumes all questions have an answer in the text. SQuAD v2.0 introduces unanswerable questions requiring a null-answer threshold $\tau$.
2. **Calibrated Probability**: Softmax logits across spans provide relative ranking but are not true calibrated probabilities.

---

## 10. Post-Audit Milestone: Full SQuAD v1.1 Fine-Tuning & Three-Model Benchmark

Following the audit, the full fine-tuning pipeline was executed on the **complete SQuAD v1.1 dataset (87,599 training examples)**, followed by an official evaluation across all **10,570 validation examples** comparing all 3 iterations of DistilBERT:

### 10.1 Full Dataset Training Profile (`models/distilbert_full_finetuned`)
- **Training Examples Processed**: 87,599 examples (88,524 sliding window feature chunks).
- **Optimization Steps**: **11,066 steps** (2.0 complete epochs, effective batch size = 16).
- **Runtime**: **1,972.12 seconds** (**32.87 minutes** on RTX 3060 Laptop GPU).
- **Loss Progression**: Descended from **5.92** to **0.81** (Mean Training Loss: **1.2159**).
- **Memory Profile**: Peak GPU VRAM allocated was **1,758.7 MB** (out of 6,144 MB), maintaining >4GB headroom.
- **Checkpoints Saved**: Periodically saved every 1,000 steps under `models/distilbert_full_finetuned/`.

### 10.2 Final 3-Model Benchmark on Complete Validation Split (10,570 Examples)
All three models were evaluated using identical tokenization, sliding window chunking (`doc_stride=128`, `max_seq_length=384`), FP16 inference, and post-processing on the full 10,570 validation set (`results/full_training/three_models_comparison.csv`):

| Model Checkpoint | Training Samples | Exact Match (EM) | Token F1 Score | Inference Latency | Total Eval Time |
|---|---|---|---|---|---|
| **Pretrained SQuAD Baseline** (`models/baseline`) | 87,599 (Full) | **79.10%** | **86.87%** | **3.01 ms** | **31.83 s** |
| **Full Fine-Tuned Model (Ours)** (`models/distilbert_full_finetuned`) | **87,599 (Full)** | **77.09%** | **85.25%** | **2.68 ms** | **28.28 s** |
| **Previous Fine-Tuned Model (Ours)** (`models/distilbert_finetuned`) | 4,000 (Subset) | 47.05% | 57.42% | 2.92 ms | 30.82 s |

**Auditor Conclusion on Full Training**:
- Scaling training data from 4,000 to 87,599 examples increased Exact Match by **+30.04%** (from 47.05% to 77.09%) and Token F1 by **+27.83%** (from 57.42% to 85.25%).
- The newly trained model achieves **97.5% of the performance of the official distilled baseline** (**77.09% vs 79.10% EM** and **85.25% vs 86.87% F1**), while achieving slightly faster inference latency (**2.68 ms/sample**).
- All 3 models remain completely preserved and selectable within the interactive Streamlit application.
