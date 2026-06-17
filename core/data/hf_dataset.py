"""Hugging Face dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def load_dataset_from_path(data_path: str, split: str = "train") -> tuple[Dataset, str]:
    """Load a dataset from JSONL, a saved HF dataset directory, or a HF dataset id."""
    path = Path(data_path)
    if path.exists():
        if path.is_dir():
            loaded = load_from_disk(str(path))
            return _select_split(loaded, split), str(path)
        if path.suffix.lower() == ".jsonl":
            return _load_jsonl(path), str(path)
        if path.suffix.lower() == ".json":
            return load_dataset("json", data_files=str(path), split=split), str(path)
        raise ValueError(f"Unsupported dataset file type: {path}")

    loaded = load_dataset(data_path, split=split)
    if not isinstance(loaded, Dataset):
        raise TypeError(f"Expected Dataset for '{data_path}' split '{split}'.")
    return loaded, data_path


def load_hf_text_dataset(data_path: str, split: str = "train") -> Dataset:
    dataset, _ = load_dataset_from_path(data_path, split=split)
    return dataset


def load_hf_image_dataset(data_path: str, split: str = "train") -> Dataset:
    dataset, _ = load_dataset_from_path(data_path, split=split)
    return dataset


def load_hf_multimodal_dataset(data_path: str, split: str = "train") -> Dataset:
    dataset, _ = load_dataset_from_path(data_path, split=split)
    return dataset


def _load_jsonl(path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _select_split(loaded: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(loaded, Dataset):
        return loaded
    if split in loaded:
        return loaded[split]
    if "train" in loaded:
        return loaded["train"]
    first_split = next(iter(loaded.keys()))
    return loaded[first_split]
