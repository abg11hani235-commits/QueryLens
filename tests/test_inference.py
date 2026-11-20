"""
test_inference.py - Unit tests for inference pipeline and span extraction.
"""

import pytest
from src.inference import QAInferenceEngine


def test_empty_input_handling():
    # Instantiate engine with dummy parameters for fast input validation check
    # Note: QAInferenceEngine.predict handles empty inputs before running any tensors
    class MockEngine(QAInferenceEngine):
        def __init__(self):
            pass

    engine = MockEngine()
    result = QAInferenceEngine.predict(engine, "", "Some context")
    assert result["answer"] == ""
    assert "error" in result

    result2 = QAInferenceEngine.predict(engine, "What is X?", "")
    assert result2["answer"] == ""
    assert "error" in result2


def test_finetuned_model_prediction():
    from pathlib import Path
    model_path = Path("models/distilbert_finetuned")
    if not (model_path / "model.safetensors").exists():
        pytest.skip("Fine-tuned model checkpoint not found")
    engine = QAInferenceEngine(model_name_or_path=str(model_path))
    res = engine.predict("When was Stanford founded?", "Stanford University was founded in 1885.")
    assert "1885" in res["answer"]
    assert res["start_char"] >= 0
    assert res["end_char"] > res["start_char"]


def test_character_offset_exact_slice():
    from pathlib import Path
    model_path = Path("models/baseline")
    if not (model_path / "model.safetensors").exists():
        pytest.skip("Baseline model checkpoint not found")
    engine = QAInferenceEngine(model_name_or_path=str(model_path))
    ctx = "   Stanford University was founded in 1885 by Leland Stanford.   "
    res = engine.predict("When was Stanford founded?", ctx)
    assert res["answer"] == "1885"
    assert ctx[res["start_char"] : res["end_char"]] == res["answer"]


def test_long_context_later_window_inference():
    from pathlib import Path
    model_path = Path("models/baseline")
    if not (model_path / "model.safetensors").exists():
        pytest.skip("Baseline model checkpoint not found")
    engine = QAInferenceEngine(model_name_or_path=str(model_path))
    filler = "This is introductory filler sentence number %d to test window overflow. " * 40
    long_ctx = (filler % tuple(range(40))) + " The project codename is Apollo-QA. " + (filler % tuple(range(40, 80)))
    res = engine.predict("What is the project codename?", long_ctx)
    assert "Apollo-QA" in res["answer"]
    assert long_ctx[res["start_char"] : res["end_char"]] == res["answer"]


def test_full_finetuned_model_prediction():
    from pathlib import Path
    model_path = Path("models/distilbert_full_finetuned")
    if not (model_path / "model.safetensors").exists():
        pytest.skip("Full fine-tuned model checkpoint not found")
    engine = QAInferenceEngine(model_name_or_path=str(model_path))
    ctx = "Stanford University was founded in 1885 by Leland Stanford."
    res = engine.predict("When was Stanford founded?", ctx)
    assert "1885" in res["answer"]
    assert ctx[res["start_char"] : res["end_char"]] == res["answer"]
