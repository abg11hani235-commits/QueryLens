# ContextQA: Transformer-Based Extractive Question Answering System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6 CUDA](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic and portfolio-grade Extractive Question Answering system built from scratch using **PyTorch**, **Hugging Face Transformers**, and the **Stanford Question Answering Dataset (SQuAD v1.1)**. Optimized for real-time inference and reproducible fine-tuning on consumer hardware (**NVIDIA GeForce RTX 3060 Laptop GPU - 6GB VRAM**).

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Definition & Extractive QA](#2-problem-definition--extractive-qa)
3. [Transformer Architecture & Mechanics](#3-transformer-architecture--mechanics)
4. [Dataset Description (SQuAD v1.1)](#4-dataset-description-squad-v11)
5. [Data Preprocessing & Offset Mapping](#5-data-preprocessing--offset-mapping)
6. [Span Extraction & Sliding Windows](#6-span-extraction--sliding-windows)
7. [Model Training & Hardware Optimization](#7-model-training--hardware-optimization)
8. [Evaluation Metrics (EM & F1)](#8-evaluation-metrics-em--f1)
9. [Model Comparison & Benchmark](#9-model-comparison--benchmark)
10. [Project Directory Structure](#10-project-directory-structure)
11. [Installation & Setup](#11-installation--setup)
12. [How to Run (CLI, Training, App, Tests)](#12-how-to-run)
13. [Limitations & Future Work](#13-limitations--future-work)

---

## 1. Project Overview
**ContextQA** is an end-to-end NLP system that receives an unstructured textual passage (**Context**) and an interrogative query (**Question**), identifies the most relevant factual evidence, and extracts the exact answer substring directly from the passage.

Unlike generative autoregressive models that risk generating unverified facts, Extractive QA is **strictly grounded** in the source document—providing zero-hallucination factual extraction critical for legal contracts, clinical records, and financial disclosures.

---

## 2. Problem Definition & Extractive QA
Given:
- Context passage $C = \{c_1, c_2, \dots, c_N\}$ of length $N$ characters.
- Question query $Q = \{q_1, q_2, \dots, q_M\}$ of length $M$ characters.

The extractive task is formalized as finding the continuous substring $A = C[s : e]$ where $s$ and $e$ represent the start and end character positions that maximize the joint probability of containing the correct answer:
$$(s^*, e^*) = \arg\max_{s \le e} P(s, e \mid Q, C)$$

---

## 3. Transformer Architecture & Mechanics
ContextQA employs **DistilBERT** (`distilbert-base-uncased`) as its primary architecture:
- **Encoder-Only Design**: Bidirectional self-attention computes dense contextual representations where each word token simultaneously attends to both preceding and succeeding tokens.
- **Student-Teacher Distillation**: DistilBERT compresses BERT-base from 12 layers down to 6 layers using knowledge distillation, retaining **97% of BERT's language understanding** while operating **60% faster** with 66M parameters.
- **QA Linear Head**: The final hidden state vector $\mathbf{h}_i \in \mathbb{R}^{768}$ of each token $i$ is passed through a shared linear layer $\mathbf{W} \in \mathbb{R}^{2 \times 768}$ to produce two logits:
  $$S_i = \mathbf{w}_{\text{start}}^T \mathbf{h}_i, \quad E_i = \mathbf{w}_{\text{end}}^T \mathbf{h}_i$$

---

## 4. Dataset Description (SQuAD v1.1)
The system is trained and evaluated on **SQuAD v1.1** (Rajpurkar et al., Stanford University):
- **Training Examples**: $87,599$ question-context-answer triples.
- **Validation Examples**: $10,570$ question-context-answer triples (each evaluated against 3 human reference answers).
- **Statistical Profile**:
  - Context Word Length: Mean = 119.8, Median = 110.0, 95th-Percentile = 213 words.
  - Question Word Length: Mean = 10.1, Median = 10.0 words.
  - Answer Word Length: Mean = 3.2, Median = 2.0 words ($95\%$ of answers are $\le 10$ words).
  - Answer Start Position: Mean = 319.9 characters.

---

## 5. Data Preprocessing & Offset Mapping
1. **Subword Tokenization (WordPiece)**: Words are segmented into subword units with continuation markers (`##`), handling out-of-vocabulary terms without info loss.
2. **Special Tokens**: Sequences are formatted as `[CLS] Question [SEP] Context [SEP]`.
3. **Offset Mapping**: Fast tokenizers compute `(start_char, end_char)` for every token. Ground-truth character positions from SQuAD are mapped into exact token indices.
4. **Out-of-Span Chunk Labeling**: If an answer falls outside a chunk, `start_position = 0` and `end_position = 0` (pointing to `[CLS]`).

---

## 6. Span Extraction & Sliding Windows
- **Sliding Window**: For contexts exceeding $384$ tokens, the passage is split into overlapping chunks with `doc_stride = 128` tokens.
- **Joint Scoring**: For every valid pair $(i, j)$ in context tokens:
  $$\text{Score}(i, j) = S_i + E_j \quad \text{subject to } j \ge i \text{ and } (j - i + 1) \le 30$$
- **Window Aggregation**: Spans are aggregated across all chunks for a document, ranking candidate answers by logit sum.
- **Confidence Indicator**: Top candidate scores are normalized via softmax over the top-K spans as a relative confidence score.

---

## 7. Model Training & Hardware Optimization
Training was executed and verified on consumer hardware (**NVIDIA GeForce RTX 3060 Laptop GPU - 6GB VRAM**):
- **Training Subset**: In this training iteration, a representative subset of **4,000 training examples** (from the 87,599 SQuAD v1.1 train split) was trained over **2 epochs** (**512 total optimization steps**).
- **Execution Speed**: Completed in **85.04 seconds** at a throughput of **96.0 samples/second**.
- **Mixed Precision (FP16)**: Activated via PyTorch AMP and Tensor Cores, halving memory footprint and avoiding CUDA Out-of-Memory.
- **Gradient Accumulation**: Mini-batch size of $8$ with $2$ accumulation steps produces an effective batch size of $16$.
- **Optimizer & Scheduler**: AdamW with peak $\text{lr} = 3 \times 10^{-5}$, weight decay $0.01$, and linear warmup over the first $50$ steps.
- **Loss Progression**: Loss dropped monotonically from **5.919** down to **1.871** (mean training loss: 2.7938), verified in `trainer_state.json` and plotted in `results/training_loss.png`.

---

## 8. Evaluation Metrics & Verified Results
Evaluated on the **complete official SQuAD v1.1 validation set (10,570 examples)**:

| Metric | Full Fine-Tuned Model (Ours) | Pretrained Baseline | 4k Subset Fine-Tuned (Ours) | Source File |
|---|---|---|---|---|
| **Training Scale** | **87,599 samples (Full SQuAD)** | 87,599 samples | 4,000 samples | `results/full_training/three_models_comparison.csv` |
| **Exact Match (EM)** | **77.09%** | **79.10%** | **47.05%** | `results/full_training/metrics.json` |
| **Token-Level F1** | **85.25%** | **86.87%** | **57.42%** | `results/full_training/metrics.json` |
| **Inference Latency** | **2.68 ms / sample** | **3.01 ms / sample** | **2.92 ms / sample** | `results/full_training/three_models_comparison.csv` |
| **Total Evaluation Time** | **28.28 s** (10,570 samples) | **31.83 s** (10,570 samples) | **30.82 s** (10,570 samples) | `results/full_training/three_models_comparison.csv` |

*Key Takeaway*: Fine-tuning on the full 87k dataset brought our custom model within **97.5% of the official baseline's performance** (**77.09% EM / 85.25% F1** vs **79.10% EM / 86.87% F1**), while outperforming the 4k subset model by **+30.04% EM** and **+27.83% F1**.

---

## 9. Model Comparison & Architectural Benchmark
Benchmark conducted on a standardized validation slice using identical inference procedures on the RTX 3060 GPU (`results/verified_model_comparison.csv`):

| Model Checkpoint | QA Head Status | Parameters | Exact Match (EM) | Token F1 | Inference Latency | Peak VRAM (RTX 3060) |
|---|---|---|---|---|---|---|
| **DistilBERT Baseline** (`models/baseline`) | Pretrained on Full SQuAD (87k) | **66.4 M** | **85.50%** | **89.77%** | **5.98 ms** | **363.5 MB** |
| **DistilBERT Full Fine-Tuned (Ours)** (`models/distilbert_full_finetuned`) | Fine-Tuned on Full SQuAD (87k, 2 epochs) | **66.4 M** | **77.09%** | **85.25%** | **2.68 ms** | **1,758.7 MB (Train)** |
| **DistilBERT 4k Fine-Tuned (Ours)** (`models/distilbert_finetuned`) | Fine-Tuned on 4k subset (2 epochs) | **66.4 M** | **62.00%** | **66.61%** | **5.98 ms** | **362.3 MB** |
| **BERT-base-uncased** | Untrained / Randomly Initialized QA Head | 108.9 M | 0.00%\* | 4.44%\* | 12.92 ms | 524.6 MB |
| **RoBERTa-base** | Untrained / Randomly Initialized QA Head | 124.1 M | 0.00%\* | 4.87%\* | 13.07 ms | 582.6 MB |

*\* Important Academic Notice: BERT-base and RoBERTa-base were instantiated from base language models with randomly initialized linear QA heads (`qa_outputs`). Their ~4.5% F1 score represents untrained chance-level span selection, demonstrating the necessity of task-specific fine-tuning. This benchmark isolates inference speed, parameter efficiency, and GPU memory consumption.*

---

## 10. Project Directory Structure
```
ContextQA/
│
├── data/
│   └── sample_data/           # Small benchmark test fixtures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # SQuAD v1.1 dataset statistics & EDA
│   ├── 02_preprocessing.ipynb      # Tokenization, offset mapping, sliding window
│   ├── 03_baseline_model.ipynb     # Pre-trained DistilBERT benchmark inference
│   ├── 04_model_training.ipynb     # End-to-end fine-tuning pipeline
│   └── 05_evaluation.ipynb         # Official SQuAD evaluation & error analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # SQuAD dataset loading & statistical profiling
│   ├── preprocessing.py       # Sliding window tokenization & span alignment
│   ├── train.py               # Hugging Face Trainer fine-tuning script
│   ├── evaluate.py            # Official EM & F1 computation & error analysis
│   ├── inference.py           # Production inference pipeline & span extraction
│   └── utils.py               # Seeds, normalization, metrics, and highlighting
│
├── configs/
│   ├── training.yaml          # Hyperparameters and VRAM optimization settings
│   └── models.yaml            # Model registry and comparisons
│
├── models/
│   ├── baseline/              # Pretrained checkpoint
│   └── distilbert_finetuned/  # Custom trained weights
│
├── results/
│   ├── metrics.json           # Exact evaluation metrics
│   ├── error_analysis.csv     # Detailed per-sample classification
│   ├── evaluation_report.md   # Markdown summary report
│   └── visualizations/        # Distribution and loss plots
│
├── app/
│   └── streamlit_app.py       # Modern web application interface
│
├── tests/
│   ├── test_data_loader.py    # Dataset loading tests
│   ├── test_preprocessing.py  # Alignment & sliding window unit tests
│   ├── test_inference.py      # Inference & error handling tests
│   └── test_utils.py          # Metric & normalization tests
│
├── docs/
│   ├── PROJECT_EXPLANATION_AR.md # Detailed guide in Egyptian Arabic
│   └── INTERVIEW_PREPARATION.md  # 25 Technical Interview Q&As
│
├── requirements.txt
├── pytest.ini
├── README.md
├── PROJECT_PROGRESS.md
└── .gitignore
```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.10 to 3.12
- CUDA-enabled NVIDIA GPU (optional but strongly recommended)

### Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd ContextQA

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 12. How to Run

### Run Automated Unit Tests
```bash
python -m pytest tests/ -v
```

### Run SQuAD Data Exploration & Generate Plots
```bash
python scripts/run_exploration.py
```

### Run Baseline Model Inference Test
```bash
python scripts/test_baseline_inference.py
```

### Fine-Tune DistilBERT on SQuAD
```bash
# Full dataset training (2 epochs with FP16 on GPU)
python src/train.py --config configs/training.yaml

# Rapid debug run (first 500 samples)
python src/train.py --max_train_samples 500 --max_eval_samples 100 --epochs 1
```

### Launch Interactive Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 13. Limitations & Future Work
1. **Unanswerable Questions (SQuAD v2.0)**: SQuAD v1.1 assumes an answer is always present. Upgrading to SQuAD v2.0 with a null-score threshold $\tau$ enables rejection of unanswerable queries.
2. **Dense Retrieval Integration (RAG)**: Pairing ContextQA with a dense retriever (e.g., BM25, FAISS, or ColBERT) enables open-domain QA across millions of documents.
3. **Quantization & ONNX**: Exporting the fine-tuned model to ONNX Runtime with INT8 quantization will cut latency by another $50\%$ for edge deployments.
