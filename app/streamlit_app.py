"""
streamlit_app.py - ContextQA Interactive Web Application.

A modern, responsive Streamlit dashboard for Extractive Question Answering.
Features:
  - Passage (Context) & Question inputs with preloaded academic examples
  - Model selection (Fine-Tuned DistilBERT vs Pretrained Baseline)
  - Real-time span extraction, start/end character offsets, and inference latency
  - Visual span highlighting in original context
  - Confidence scoring indicator with calibrated probability caveats
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import QAInferenceEngine
from src.utils import highlight_answer_in_context

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ContextQA - Extractive Question Answering",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Navy & Cyan Palette)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #38bdf8;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #0f172a;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #334155;
        font-size: 1.25rem;
        color: #38bdf8;
        font-weight: 700;
    }
    .context-display {
        background-color: #0f172a;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #334155;
        line-height: 1.6;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_engine(model_path: str):
    """Load and cache QAInferenceEngine to prevent reloading on each interaction."""
    return QAInferenceEngine(model_name_or_path=model_path)


# -----------------------------------------------------------------------------
# Preset Benchmark Examples
# -----------------------------------------------------------------------------
PRESET_EXAMPLES = {
    "Stanford University (Founding Date)": {
        "context": "Stanford University is a private research university in Stanford, California. It was founded in 1885 by Leland and Jane Stanford in memory of their only child, Leland Stanford Jr., who had died of typhoid fever at age 15 the previous year.",
        "question": "When was Stanford University founded?",
    },
    "Apollo 11 (Moon Mission)": {
        "context": "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC, and Armstrong became the first person to step onto the Moon's surface six hours and 39 minutes later.",
        "question": "Who was the commander of Apollo 11?",
    },
    "Transformers in Deep Learning (NLP Milestone)": {
        "context": "The Transformer is a deep learning architecture introduced in the 2017 seminal paper 'Attention Is All You Need' by Ashish Vaswani and colleagues at Google Brain. Unlike previous recurrent neural networks (RNNs) and LSTMs, Transformers rely entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        "question": "In what year was the Transformer architecture introduced?",
    },
}

# -----------------------------------------------------------------------------
# Sidebar: Configuration & Model Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("Settings & Models")

    # Detect Available Models
    available_models = {}
    full_finetuned_path = Path("models/distilbert_full_finetuned")
    if full_finetuned_path.exists() and ((full_finetuned_path / "model.safetensors").exists() or (full_finetuned_path / "pytorch_model.bin").exists()):
        available_models["Fine-Tuned DistilBERT (ContextQA - Full 87k SQuAD)"] = str(full_finetuned_path)

    finetuned_path = Path("models/distilbert_finetuned")
    if finetuned_path.exists() and ((finetuned_path / "model.safetensors").exists() or (finetuned_path / "pytorch_model.bin").exists()):
        available_models["Fine-Tuned DistilBERT (ContextQA - 4k Subset)"] = str(finetuned_path)

    baseline_path = Path("models/baseline")
    if baseline_path.exists() and ((baseline_path / "model.safetensors").exists() or (baseline_path / "pytorch_model.bin").exists()):
        available_models["Pretrained SQuAD Baseline (DistilBERT - Full SQuAD)"] = str(baseline_path)
    else:
        available_models["Pretrained SQuAD Baseline (DistilBERT - Full SQuAD)"] = "distilbert-base-uncased-distilled-squad"

    model_choice = st.selectbox(
        "Select Active Model:",
        options=list(available_models.keys()),
        index=0
    )
    active_model_path = available_models[model_choice]

    st.markdown("---")
    st.subheader("📚 Preset Examples")
    preset_choice = st.selectbox(
        "Load Sample Context & Question:",
        options=["(Custom Input)"] + list(PRESET_EXAMPLES.keys())
    )

    st.markdown("---")
    st.markdown("""
    **Architecture**: Transformer (Encoder)  
    **Task**: Extractive Question Answering  
    **Evaluation**: SQuAD v1.1 (EM & F1)  
    **Hardware**: RTX 3060 (6GB VRAM)
    """)

# -----------------------------------------------------------------------------
# Main Application Interface
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>ContextQA - Intelligent Passage QA</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Extract precise factual answers directly from arbitrary passages using fine-tuned Transformers.</div>", unsafe_allow_html=True)

# Handle Preset Selection
initial_context = ""
initial_question = ""
if preset_choice != "(Custom Input)":
    initial_context = PRESET_EXAMPLES[preset_choice]["context"]
    initial_question = PRESET_EXAMPLES[preset_choice]["question"]

# Input Forms
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("📖 Passage (Context)")
    context_input = st.text_area(
        label="Context Passage",
        value=initial_context,
        height=240,
        placeholder="Paste or type a paragraph of text here...",
        label_visibility="collapsed"
    )

    st.subheader("❓ Question")
    question_input = st.text_input(
        label="Question",
        value=initial_question,
        placeholder="Ask a question about the context above...",
        label_visibility="collapsed"
    )

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        predict_clicked = st.button("🚀 Extract Answer", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🔄 Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

with col_right:
    st.subheader("🎯 Extracted Result")

    if predict_clicked:
        if not context_input.strip() or not question_input.strip():
            st.warning("⚠️ Please provide both a Context passage and a Question.")
        else:
            with st.spinner("Extracting answer span with Transformer..."):
                start_time = time.time()
                try:
                    engine = get_model_engine(active_model_path)
                    prediction = engine.predict(question_input, context_input)
                    latency_ms = (time.time() - start_time) * 1000
                except Exception as e:
                    st.error(f"❌ Error during model execution: {str(e)}")
                    prediction = None

            if prediction and prediction["answer"]:
                ans = prediction["answer"]
                conf = prediction["confidence"]
                start_char = prediction["start_char"]
                end_char = prediction["end_char"]

                st.markdown(f"<div class='answer-box'>'{ans}'</div>", unsafe_allow_html=True)

                # Metrics Row
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="Relative Score (Uncalibrated)", value=f"{conf * 100:.1f}%")
                with m_col2:
                    st.metric(label="Character Span", value=f"[{start_char} : {end_char}]")
                with m_col3:
                    st.metric(label="Inference Latency", value=f"{latency_ms:.1f} ms")

                st.info("💡 **Relative Score Note**: The score is derived from relative softmax over candidate spans. In modern deep NLP, logits are uncalibrated; treat this as a relative ranking indicator rather than a true Bayesian probability.")

                # Highlighted Context
                st.subheader("🔍 Context with Highlighted Answer")
                highlighted_html = highlight_answer_in_context(
                    context_input, start_char, end_char, format_type="html"
                )
                st.markdown(f"<div class='context-display'>{highlighted_html}</div>", unsafe_allow_html=True)

                # Top Alternative Candidates
                if len(prediction.get("all_candidates", [])) > 1:
                    with st.expander("🔎 View Alternative Candidate Spans"):
                        for i, cand in enumerate(prediction["all_candidates"][1:], 2):
                            st.write(f"**#{i}**: `{cand['answer']}` — Span: `[{cand['start_char']}:{cand['end_char']}]`, Logit: `{cand['raw_score']:.2f}`, Relative Score: `{cand.get('confidence', 0)*100:.1f}%`")
            elif prediction:
                st.warning("⚠️ No confident answer span could be extracted from the context.")
    else:
        st.info("👉 Enter a passage and question on the left, or select an example from the sidebar, then click **Extract Answer**.")
