"""
test_data_loader.py - Unit tests for data loading and dataset overview structures.
"""

from datasets import Dataset, DatasetDict
from src.data_loader import (
    get_dataset_overview,
    extract_sample_records,
    compute_length_statistics,
)


def _create_mock_dataset():
    data = {
        "id": ["sample_1", "sample_2"],
        "title": ["Stanford_University", "Stanford_University"],
        "context": [
            "Stanford University is in California. It was founded in 1885.",
            "Leland Stanford established the university in memory of his son.",
        ],
        "question": [
            "When was Stanford University founded?",
            "Who established the university?",
        ],
        "answers": [
            {"text": ["1885"], "answer_start": [56]},
            {"text": ["Leland Stanford"], "answer_start": [0]},
        ],
    }
    return DatasetDict({"train": Dataset.from_dict(data)})


def test_dataset_overview():
    ds = _create_mock_dataset()
    overview = get_dataset_overview(ds)
    assert "train" in overview["splits"]
    assert overview["splits"]["train"]["num_examples"] == 2
    assert "context" in overview["features"]
    assert "answers" in overview["features"]


def test_extract_sample_records():
    ds = _create_mock_dataset()
    samples = extract_sample_records(ds, split="train", indices=[0])
    assert len(samples) == 1
    assert samples[0]["question"] == "When was Stanford University founded?"
    assert samples[0]["answers"] == ["1885"]
    assert samples[0]["answer_starts"] == [56]


def test_compute_length_statistics():
    ds = _create_mock_dataset()
    df, summary = compute_length_statistics(ds, split="train")
    assert len(df) == 2
    assert "context_word_len" in df.columns
    assert "answer_start" in df.columns
    assert summary["context_word_len"]["min"] > 0
    assert summary["answer_word_len"]["max"] <= 5
