"""
preprocessing.py - Data Preprocessing, Tokenization, and Span Alignment for SQuAD.

Handles:
  1. Fast Tokenization of Question-Context pairs with truncation="only_second".
  2. Sliding window generation with return_overflowing_tokens=True and doc_stride.
  3. Precise Character-to-Token span alignment using offset_mapping and sequence_ids.
  4. Correct labeling of start_positions and end_positions (pointing to [CLS]=0 when out-of-span).
  5. Preparation of evaluation features preserving overflow-to-sample mappings for post-processing.
"""

from typing import Dict, Any, List, Optional, Tuple
from transformers import PreTrainedTokenizerFast


def prepare_train_features(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    pad_to_max_length: bool = True
) -> Dict[str, Any]:
    """
    Tokenize question-context pairs for training, generate sliding windows for
    overflowing contexts, and map character-level answer spans to token indices.

    For each feature chunk:
      - If the answer is completely contained inside the chunk, start_position and
        end_position point to the exact token boundaries.
      - If the answer is partially or completely outside the chunk, start_position
        and end_position are set to the [CLS] token index (index 0).

    Args:
        examples: Batch of dataset records (with 'question', 'context', 'answers').
        tokenizer: Fast Hugging Face tokenizer instance.
        max_seq_length: Maximum sequence length (tokens).
        doc_stride: Sliding window stride for overflowing contexts.
        pad_to_max_length: Whether to pad all features to max_seq_length.

    Returns:
        Dictionary of tokenized tensors and aligned start_positions & end_positions.
    """
    # Clean questions (remove excessive whitespace)
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    # Tokenize question-context pairs with sliding window
    tokenized_inputs = tokenizer(
        questions,
        contexts,
        max_length=max_seq_length,
        truncation="only_second",  # Truncate context, preserve question
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_inputs.sequence_ids(i)

        # Retrieve the original sample corresponding to this feature chunk
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # In SQuAD v1.1, all examples have at least one answer
        if len(answers["answer_start"]) == 0 or len(answers["text"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find the token boundaries of the context (where sequence_ids == 1)
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Check if the answer is completely inside the context window
        context_start_char = offsets[token_start_index][0]
        context_end_char = offsets[token_end_index][1]

        if not (context_start_char <= start_char and end_char <= context_end_char):
            # Answer is outside this window -> label points to [CLS]
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Answer is inside: find the exact start and end token indices
            current_token = token_start_index
            while current_token <= token_end_index and offsets[current_token][0] <= start_char:
                current_token += 1
            ans_token_start = current_token - 1

            current_token = token_end_index
            while current_token >= token_start_index and offsets[current_token][1] >= end_char:
                current_token -= 1
            ans_token_end = current_token + 1

            start_positions.append(ans_token_start)
            end_positions.append(ans_token_end)

    tokenized_inputs["start_positions"] = start_positions
    tokenized_inputs["end_positions"] = end_positions

    return tokenized_inputs


def prepare_validation_features(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    pad_to_max_length: bool = True
) -> Dict[str, Any]:
    """
    Tokenize question-context pairs for validation / evaluation.
    Preserves example_id and sets offsets outside the context to None so post-processing
    only extracts spans from valid context tokens.

    Args:
        examples: Batch of dataset records (with 'id', 'question', 'context').
        tokenizer: Fast Hugging Face tokenizer instance.
        max_seq_length: Maximum sequence length (tokens).
        doc_stride: Sliding window stride for overflowing contexts.
        pad_to_max_length: Whether to pad all features to max_seq_length.

    Returns:
        Tokenized features with offset_mapping and example_id.
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized_inputs = tokenizer(
        questions,
        contexts,
        max_length=max_seq_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(tokenized_inputs["input_ids"])):
        sample_index = sample_mapping[i]
        example_ids.append(examples["id"][sample_index])

        sequence_ids = tokenized_inputs.sequence_ids(i)
        offset = tokenized_inputs["offset_mapping"][i]

        # Mask offsets not in the context (sequence_ids != 1) with None
        tokenized_inputs["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(offset)
        ]

    tokenized_inputs["example_id"] = example_ids
    return tokenized_inputs


def reconstruct_answer_text(
    context: str,
    offset_mapping: List[Optional[Tuple[int, int]]],
    start_idx: int,
    end_idx: int
) -> str:
    """
    Safely extract character substring from context using predicted token offsets.

    Args:
        context: Original context text string.
        offset_mapping: Token offset tuples [(start_char, end_char), ...].
        start_idx: Starting token index.
        end_idx: Ending token index.

    Returns:
        Extracted answer string (empty string if invalid indices).
    """
    if (
        start_idx >= len(offset_mapping)
        or end_idx >= len(offset_mapping)
        or start_idx > end_idx
        or offset_mapping[start_idx] is None
        or offset_mapping[end_idx] is None
    ):
        return ""

    start_char = offset_mapping[start_idx][0]
    end_char = offset_mapping[end_idx][1]

    if start_char is None or end_char is None or end_char <= start_char:
        return ""

    return context[start_char:end_char]
