"""
test_baseline_inference.py - Test pretrained DistilBERT baseline on representative SQuAD examples.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import QAInferenceEngine
from src.utils import highlight_answer_in_context


def main():
    print("=" * 60)
    print("🤖 PHASE 3: Testing Baseline Model Inference")
    print("   Checkpoint: distilbert-base-uncased-distilled-squad")
    print("=" * 60)

    baseline_dir = Path("models/baseline")
    model_path = str(baseline_dir) if (baseline_dir / "model.safetensors").exists() else "distilbert-base-uncased-distilled-squad"
    engine = QAInferenceEngine(model_name_or_path=model_path)

    test_cases = [
        {
            "title": "Stanford University",
            "context": "Stanford University is a private research university in Stanford, California. It was founded in 1885 by Leland and Jane Stanford.",
            "question": "When was Stanford University founded?",
            "expected": "1885"
        },
        {
            "title": "Apollo 11",
            "context": "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.",
            "question": "Who was the commander of Apollo 11?",
            "expected": "Neil Armstrong"
        },
        {
            "title": "Long Context / Sliding Window Test",
            "context": "Introductory padding words. " * 80 + "Deep learning revolutionized NLP in recent years. The Transformer architecture was introduced in 2017 by Vaswani et al. in Attention Is All You Need. " + "Concluding remarks. " * 40,
            "question": "In what year was the Transformer architecture introduced?",
            "expected": "2017"
        }
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {tc['title']} ---")
        print(f"Question: {tc['question']}")
        print(f"Expected: {tc['expected']}")

        result = engine.predict(tc["question"], tc["context"])

        print(f"Predicted Answer: '{result['answer']}'")
        print(f"Span Offsets:     [{result['start_char']}:{result['end_char']}]")
        print(f"Raw Logit Score:  {result['raw_score']}")
        print(f"Confidence:       {result['confidence']}")

        highlighted = highlight_answer_in_context(tc["context"], result["start_char"], result["end_char"], format_type="markdown")
        print(f"Snippet:          ...{highlighted[max(0, result['start_char']-40) : min(len(highlighted), result['end_char']+40)]}...")

        is_match = tc["expected"].lower() in result["answer"].lower()
        print(f"Status:           {'✅ MATCH' if is_match else '❌ MISMATCH'}")

    print("\n" + "=" * 60)
    print("✅ Baseline Inference Tested Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
