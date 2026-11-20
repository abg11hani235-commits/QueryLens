"""
test_preprocessing.py - Unit tests for tokenization, offset alignment, and sliding windows.
"""

from transformers import AutoTokenizer
from src.preprocessing import (
    prepare_train_features,
    prepare_validation_features,
    reconstruct_answer_text,
)


def test_prepare_train_features_short_context():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    context = "Stanford University is a private research university in California. It was founded in 1885."
    ans_text = "1885"
    ans_start = context.index(ans_text)

    examples = {
        "id": ["ex1"],
        "title": ["Stanford"],
        "context": [context],
        "question": ["When was Stanford University founded?"],
        "answers": [{"text": [ans_text], "answer_start": [ans_start]}],
    }

    features = prepare_train_features(examples, tokenizer, max_seq_length=128, doc_stride=32)

    assert "input_ids" in features
    assert "start_positions" in features
    assert "end_positions" in features

    start_tok = features["start_positions"][0]
    end_tok = features["end_positions"][0]

    # Target answer should not be mapped to [CLS] (index 0) because it's within context
    assert start_tok > 0
    assert end_tok >= start_tok

    # Check decoded tokens (end_tok is inclusive)
    predicted_tokens = features["input_ids"][0][start_tok : end_tok + 1]
    decoded_answer = tokenizer.decode(predicted_tokens).strip()
    assert decoded_answer == "1885"


def test_sliding_window_overflow_handling():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Create a long context where answer is near the end
    long_prefix = "Word " * 150  # ~150 words -> will exceed max_seq_length 64
    context = long_prefix + "The secret key is ALPHA1234. End of document."
    ans_text = "ALPHA1234"
    ans_start = context.index(ans_text)

    examples = {
        "id": ["overflow_ex"],
        "title": ["Secret"],
        "context": [context],
        "question": ["What is the secret key?"],
        "answers": [{"text": [ans_text], "answer_start": [ans_start]}],
    }

    # Small max_seq_length to force multiple chunks
    features = prepare_train_features(examples, tokenizer, max_seq_length=64, doc_stride=16)

    # Should generate more than 1 chunk
    num_chunks = len(features["input_ids"])
    assert num_chunks > 1

    # In early chunks where the answer has not appeared yet, start/end position must be [CLS] (index 0)
    assert features["start_positions"][0] == 0
    assert features["end_positions"][0] == 0

    # In at least one subsequent chunk, the answer should be correctly aligned
    found_valid_chunk = False
    for start_pos, end_pos, input_ids in zip(
        features["start_positions"], features["end_positions"], features["input_ids"]
    ):
        if start_pos > 0:
            extracted_tokens = input_ids[start_pos : end_pos + 1]
            decoded = tokenizer.decode(extracted_tokens).strip()
            if "alpha" in decoded.lower():
                found_valid_chunk = True
                break

    assert found_valid_chunk, "Sliding window failed to capture answer in subsequent chunk!"
