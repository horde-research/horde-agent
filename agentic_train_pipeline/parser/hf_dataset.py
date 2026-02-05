
import os
from typing import Any, Tuple

from datasets import load_dataset, load_from_disk


def load_dataset_from_path(data_path: str, split: str = "train") -> Tuple[Any, str]:
    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
        resolved_id = data_path
    else:
        dataset = load_dataset(data_path, split=split)
        resolved_id = data_path
    if hasattr(dataset, "keys") and "train" in dataset:
        dataset = dataset["train"]
    return dataset, resolved_id


def load_hf_text_dataset(data_path: str, split: str = "train") -> Tuple[Any, str]:
    return load_dataset_from_path(data_path, split=split)


def load_hf_image_dataset(data_path: str, split: str = "train") -> Tuple[Any, str]:
    return load_dataset_from_path(data_path, split=split)


def load_hf_multimodal_dataset(data_path: str, split: str = "train") -> Tuple[Any, str]:
    return load_dataset_from_path(data_path, split=split)