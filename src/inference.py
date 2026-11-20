"""
inference.py - Extractive Question Answering Inference Pipeline.

Supports:
  - Loading fine-tuned or baseline Hugging Face QA checkpoints
  - Sliding window inference for arbitrarily long contexts
  - Start and end logits computation and span extraction
  - Filtering invalid spans (end < start, length > max_answer_length)
  - Answer ranking across multiple sliding windows
  - Formatted output with confidence scoring and character offsets
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, PreTrainedTokenizerFast
from src.utils import get_device


class QAInferenceEngine:
    """
    Extractive QA Inference Engine supporting single-pass and sliding window inference.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased-distilled-squad",
        device: Optional[torch.device] = None,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_answer_length: int = 30,
        n_best_size: int = 20,
    ):
        """
        Initialize the QA model and tokenizer.

        Args:
            model_name_or_path: Local directory path or Hugging Face Hub model ID.
            device: Compute device (defaults to auto-detected CUDA or CPU).
            max_seq_length: Maximum sequence length in tokens.
            doc_stride: Overlap between consecutive sliding windows.
            max_answer_length: Maximum allowable answer length in tokens.
            n_best_size: Top candidate spans to evaluate per window.
        """
        self.model_name_or_path = model_name_or_path
        self.device = device or get_device()
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size

        print(f"[QAInferenceEngine] Loading model and tokenizer from '{model_name_or_path}'...")
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[QAInferenceEngine] Model ready on device: {self.device}")

    def predict(
        self,
        question: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Extract the best answer span for a given question and context.

        Args:
            question: Question string.
            context: Context paragraph string.

        Returns:
            Dictionary containing:
              - 'answer': Extracted answer substring.
              - 'start_char': Beginning character index in context.
              - 'end_char': Ending character index in context.
              - 'raw_score': Sum of start and end logits for the span.
              - 'confidence': Softmax-derived confidence indicator (0.0 to 1.0).
              - 'all_candidates': List of top candidate answers.
        """
        if not question or not context or not question.strip() or not context.strip():
            return {
                "answer": "",
                "start_char": -1,
                "end_char": -1,
                "raw_score": -9999.0,
                "confidence": 0.0,
                "all_candidates": [],
                "error": "Question or context is empty."
            }

        question = question.strip()
        # Note: Do NOT strip context so offsets match the exact character indices of the input string

        # Tokenize with sliding window for long contexts
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_length,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt"
        )

        offset_mapping = inputs.pop("offset_mapping")
        # Pop overflow mapping if present in tensor batch
        if "overflow_to_sample_mapping" in inputs:
            inputs.pop("overflow_to_sample_mapping")

        num_features = inputs["input_ids"].shape[0]

        # Send inputs to device
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()

        candidates: List[Dict[str, Any]] = []

        for feature_idx in range(num_features):
            feat_start_logits = start_logits[feature_idx].numpy()
            feat_end_logits = end_logits[feature_idx].numpy()
            offsets = offset_mapping[feature_idx].tolist()
            sequence_ids = inputs.sequence_ids(feature_idx)

            # Get candidate indices sorted by logit value
            best_start_indices = feat_start_logits.argsort()[-self.n_best_size:][::-1]
            best_end_indices = feat_end_logits.argsort()[-self.n_best_size:][::-1]

            for s_idx in best_start_indices:
                # Must be part of context (sequence_ids == 1)
                if sequence_ids[s_idx] != 1:
                    continue

                for e_idx in best_end_indices:
                    # Must be part of context
                    if sequence_ids[e_idx] != 1:
                        continue
                    # Must be a valid span
                    if e_idx < s_idx:
                        continue
                    # Must not exceed maximum answer length
                    span_length = e_idx - s_idx + 1
                    if span_length > self.max_answer_length:
                        continue

                    start_char = offsets[s_idx][0]
                    end_char = offsets[e_idx][1]

                    if start_char is None or end_char is None or end_char <= start_char:
                        continue

                    # Trim whitespace while adjusting character offsets synchronously
                    while start_char < end_char and context[start_char].isspace():
                        start_char += 1
                    while end_char > start_char and context[end_char - 1].isspace():
                        end_char -= 1

                    span_text = context[start_char:end_char]
                    if not span_text:
                        continue

                    score = float(feat_start_logits[s_idx] + feat_end_logits[e_idx])

                    candidates.append({
                        "answer": span_text,
                        "start_char": start_char,
                        "end_char": end_char,
                        "raw_score": score,
                        "feature_idx": feature_idx,
                    })

        if not candidates:
            return {
                "answer": "",
                "start_char": -1,
                "end_char": -1,
                "raw_score": -9999.0,
                "confidence": 0.0,
                "all_candidates": [],
            }

        # Sort all candidates across windows by raw logit score descending
        candidates.sort(key=lambda x: x["raw_score"], reverse=True)

        # Remove duplicate answers while keeping highest scoring span
        unique_candidates: List[Dict[str, Any]] = []
        seen_answers = set()
        for c in candidates:
            ans_key = (c["answer"].lower(), c["start_char"], c["end_char"])
            if ans_key not in seen_answers:
                seen_answers.add(ans_key)
                unique_candidates.append(c)

        top_candidates = unique_candidates[:self.n_best_size]

        # Compute softmax over top candidate scores for a normalized confidence indicator
        # Note: Clearly documented as an uncalibrated relative score, not true posterior probability
        scores_tensor = torch.tensor([c["raw_score"] for c in top_candidates], dtype=torch.float32)
        probs = torch.softmax(scores_tensor, dim=0).tolist()
        for idx, c in enumerate(top_candidates):
            c["confidence"] = round(probs[idx], 4)

        best_answer = top_candidates[0]
        return {
            "answer": best_answer["answer"],
            "start_char": best_answer["start_char"],
            "end_char": best_answer["end_char"],
            "raw_score": round(best_answer["raw_score"], 4),
            "confidence": best_answer["confidence"],
            "all_candidates": top_candidates[:5],
        }
