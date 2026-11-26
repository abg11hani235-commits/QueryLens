# ContextQA – Project Progress Tracker

## Project Overview
- **Project**: ContextQA – Transformer-Based Extractive Question Answering System
- **Dataset**: SQuAD v1.1 (Stanford Question Answering Dataset - `rajpurkar/squad`)
- **Primary Model**: DistilBERT (`distilbert-base-uncased`)
- **Baseline Model**: `distilbert-base-uncased-distilled-squad`
- **Target Hardware**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM), 16GB RAM, Intel Core i7-11800H
- **Python Environment**: Dedicated Isolated `.venv` (Python 3.12.7)

---

## Phases Status

| Phase | Description | Status | Verified Output / Artifacts |
|---|---|---|---|
| **Phase 0** | Workspace & Hardware Inspection | 🟢 COMPLETED | RTX 3060 detected, CUDA 12.4 active, PyTorch 2.6.0+cu124 verified |
| **Phase 1** | Environment Setup & Dataset Exploration | 🟢 COMPLETED | `data_loader.py`, SQuAD v1.1 verified (87,599 train, 10,570 val), 4 plots in `results/visualizations/` |
| **Phase 2** | Data Preprocessing & Tokenization | 🟢 COMPLETED | `src/preprocessing.py`, offset mapping, sliding window, character-to-token alignment unit tests |
| **Phase 3** | Baseline Model Inference | 🟢 COMPLETED | `src/inference.py`, `scripts/test_baseline_inference.py`, offline checkpoint in `models/baseline/` |
| **Phase 4** | Model Fine-Tuning (4k Subset) | 🟢 COMPLETED | `src/train.py`, trained on 4k samples with FP16, checkpoints in `models/distilbert_finetuned/` |
| **Phase 4.1** | **Full SQuAD Fine-Tuning (87,599 Samples)** | 🟢 COMPLETED | `scripts/train_full_squad.py`, 2 epochs (11,066 steps), checkpoints in `models/distilbert_full_finetuned/` |
| **Phase 5** | Evaluation & Error Analysis | 🟢 COMPLETED | `src/evaluate.py`, evaluated on complete 10,570 validation set, `results/full_training/evaluation_report.md` |
| **Phase 6** | Three-Model Benchmark Comparison | 🟢 COMPLETED | `scripts/evaluate_and_compare_three_models.py`, side-by-side benchmark on complete 10,570 validation set |
| **Phase 7** | Interactive Streamlit App | 🟢 COMPLETED | `app/streamlit_app.py`, 3-model switcher, span highlighting, presets, 100% offline |

---

## Final Verified Benchmark Results (Complete SQuAD Validation Set: 10,570 Examples)

| Model Name | Training Data Scale | Epochs / Steps | Exact Match (EM) | Token F1 | Inference Latency | Total Eval Time |
|---|---|---|---|---|---|---|
| **Pretrained SQuAD Baseline** (`models/baseline`) | 87,599 samples | N/A (Pretrained) | **79.10%** | **86.87%** | **3.01 ms** | **31.83 s** |
| **Full Fine-Tuned Model (Ours)** (`models/distilbert_full_finetuned`) | **87,599 samples** | **2.0 epochs (11,066 steps)** | **77.09%** | **85.25%** | **2.68 ms** | **28.28 s** |
| **Previous Fine-Tuned (Ours)** (`models/distilbert_finetuned`) | 4,000 samples | 2.0 epochs (512 steps) | 47.05% | 57.42% | 2.92 ms | 30.82 s |

*All 3 models were evaluated on the exact same 10,570 validation questions with identical post-processing.*

### Full Training Execution Profile (`models/distilbert_full_finetuned`):
- **Training Samples Processed**: 87,599 examples (88,524 feature chunks with `doc_stride=128`).
- **Total Optimization Steps**: 11,066 steps across 2 complete epochs.
- **Hardware Profile**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM), FP16 Mixed Precision.
- **Training Runtime**: 1,972.12 seconds (**32.87 minutes**).
- **Average Training Speed**: ~45.0 feature chunks / second.
- **Peak VRAM Allocated**: **1,758.7 MB** (out of 6,144 MB VRAM, 28.6% utilization).
- **Mean Training Loss**: **1.2159** (loss dropped from 5.92 down to 0.81).
- **Checkpoints Saved**: `checkpoint-10000`, `checkpoint-11000`, `checkpoint-11066`.

---

## Verified Package Installation Versions
- `torch`: `2.6.0+cu124` (CUDA acceleration: True, Device: NVIDIA GeForce RTX 3060 Laptop GPU)
- `transformers`: `5.23.0`
- `datasets`: `3.1.0`
- `evaluate`: `0.4.6`
- `accelerate`: `1.15.0`
- `streamlit`: `1.67.0`
- `pandas`: `2.2.2`
- `numpy`: `1.26.4`
- `scipy`: `1.13.1`
- `scikit-learn`: `1.9.1`
- `matplotlib`: `3.11.2`
- `seaborn`: `0.13.2`
- `pytest`: `9.2.0`

---

## Automated Test Suite Verification
Executed: `python -m pytest tests/ -v`
Result: **15 passed in 16.43s** (100% Pass Rate):
1. `tests/test_data_loader.py::test_dataset_overview` PASSED
2. `tests/test_data_loader.py::test_extract_sample_records` PASSED
3. `tests/test_data_loader.py::test_compute_length_statistics` PASSED
4. `tests/test_inference.py::test_empty_input_handling` PASSED
5. `tests/test_inference.py::test_finetuned_model_prediction` PASSED
6. `tests/test_inference.py::test_character_offset_exact_slice` PASSED
7. `tests/test_inference.py::test_long_context_later_window_inference` PASSED
8. `tests/test_inference.py::test_full_finetuned_model_prediction` PASSED
9. `tests/test_preprocessing.py::test_prepare_train_features_short_context` PASSED
10. `tests/test_preprocessing.py::test_sliding_window_overflow_handling` PASSED
11. `tests/test_utils.py::test_normalize_answer` PASSED
12. `tests/test_utils.py::test_exact_match` PASSED
13. `tests/test_utils.py::test_f1_score` PASSED
14. `tests/test_utils.py::test_metric_max_over_ground_truths` PASSED
15. `tests/test_utils.py::test_highlight_answer_in_context` PASSED

---

## Hugging Face Deployment & Live Space
- **Space URL**: [https://huggingface.co/spaces/a7medanoo/QueryLens](https://huggingface.co/spaces/a7medanoo/QueryLens)
- **Model Repository**: [https://huggingface.co/a7medanoo/querylens-distilbert-squad](https://huggingface.co/a7medanoo/querylens-distilbert-squad)
- **Hardware Tier**: ZeroGPU (`zero-a10g`) - NVIDIA RTX PRO 6000
- **Application Framework**: Gradio 6.32.0 (with MCP server support)
- **Features**:
  - Interactive context & question inputs
  - Answer extraction with sliding-window span ranking
  - Dynamic HTML context highlighting
  - Real-time confidence score & character span indicators
  - Curated SQuAD v1.1 benchmark examples
  - Status: Verified live and fully operational (`HTTP 200`, `stage: RUNNING`)
