"""Calibration dataset management for expert tracing and quantization."""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class CalibrationDataset(Dataset):
    """Token-level calibration dataset for expert trace collection.

    Wraps a slice of the pretraining distribution (e.g., The Pile)
    and yields fixed-length token sequences for calibration passes.
    """

    def __init__(
        self,
        tokenized_data: torch.Tensor,
        seq_length: int = 2048,
    ):
        self.data = tokenized_data
        self.seq_length = seq_length
        self.num_sequences = len(self.data) // seq_length

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_length
        end = start + self.seq_length
        tokens = self.data[start:end]
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}


def load_calibration_data(
    dataset_name: str = "the_pile",
    split: str = "validation",
    num_tokens: int = 4_000_000,
    seq_length: int = 2048,
    seed: int = 42,
    tokenizer=None,
    cache_dir: Optional[str] = None,
) -> CalibrationDataset:
    """Load and tokenize calibration data from HuggingFace datasets.

    Args:
        dataset_name: Dataset identifier (e.g., 'the_pile', 'c4').
        split: Dataset split.
        num_tokens: Target number of tokens to collect.
        seq_length: Sequence length for calibration samples.
        seed: Random seed for reproducibility.
        tokenizer: Tokenizer instance. If None, must be provided externally.
        cache_dir: Optional cache directory for the dataset.

    Returns:
        CalibrationDataset ready for iteration.
    """
    from datasets import load_dataset

    # Map friendly names to HF dataset IDs
    dataset_map = {
        "the_pile": "EleutherAI/the_pile",
        "c4": "allenai/c4",
        "openwebtext": "openwebtext",
    }
    hf_name = dataset_map.get(dataset_name, dataset_name)

    ds = load_dataset(
        hf_name,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    # Tokenize and concatenate until we have enough tokens
    all_tokens = []
    total = 0
    for sample in ds:
        text = sample.get("text", sample.get("content", ""))
        if not text:
            continue
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].squeeze(0)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= num_tokens:
            break

    tokenized = torch.cat(all_tokens)[:num_tokens]
    return CalibrationDataset(tokenized, seq_length=seq_length)


def save_calibration_cache(
    dataset: CalibrationDataset, path: str | Path
):
    """Save tokenized calibration data for reuse across steps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"data": dataset.data, "seq_length": dataset.seq_length},
        path,
    )


def load_calibration_cache(path: str | Path) -> CalibrationDataset:
    """Load previously saved calibration data."""
    saved = torch.load(path, weights_only=True)
    return CalibrationDataset(saved["data"], seq_length=saved["seq_length"])
