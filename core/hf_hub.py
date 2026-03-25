"""Hugging Face Hub push/pull helpers.

Thin wrappers around ``huggingface_hub`` and ``datasets`` for:
  - Pushing / pulling SFT datasets
  - Pushing / pulling LoRA adapters

Authentication: reads HF_TOKEN from the environment. No need for
``huggingface-cli login`` — just set HF_TOKEN in .env.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or None


def _hf_username() -> str:
    """Return the HF username from env or the Hub whoami endpoint."""
    name = os.getenv("HF_USERNAME")
    if name:
        return name
    from huggingface_hub import HfApi
    info = HfApi(token=_hf_token()).whoami()
    return info["name"]


# ─── Datasets ─────────────────────────────────────────────────────────────────

def push_dataset(
    local_path: str,
    repo_name: str,
    *,
    private: bool = True,
    username: Optional[str] = None,
) -> str:
    """Push a local JSONL or save_to_disk dataset to HF Hub.

    Returns the full repo id (``username/repo_name``).
    """
    from datasets import Dataset, load_dataset, load_from_disk

    username = username or _hf_username()
    repo_id = f"{username}/{repo_name}"

    p = Path(local_path)
    if p.is_dir():
        ds = load_from_disk(str(p))
    elif p.suffix in (".jsonl", ".json"):
        ds = load_dataset("json", data_files=str(p), split="train")
    else:
        raise ValueError(f"Unsupported dataset path: {local_path}")

    if not isinstance(ds, Dataset):
        from datasets import DatasetDict
        if isinstance(ds, DatasetDict):
            ds = next(iter(ds.values()))

    logger.info("Pushing dataset (%d rows) → %s (private=%s)", len(ds), repo_id, private)
    ds.push_to_hub(repo_id, private=private, token=_hf_token())
    logger.info("Dataset pushed: https://huggingface.co/datasets/%s", repo_id)
    return repo_id


def pull_dataset(repo_id: str, split: str = "train"):
    """Pull a dataset from HF Hub. Returns a ``datasets.Dataset``."""
    from datasets import load_dataset

    logger.info("Pulling dataset from HF Hub: %s (split=%s)", repo_id, split)
    ds = load_dataset(repo_id, split=split, token=_hf_token())
    logger.info("Pulled %d rows.", len(ds))
    return ds


# ─── Adapters / Models ────────────────────────────────────────────────────────

def push_adapter(
    adapter_dir: str,
    repo_name: str,
    *,
    private: bool = True,
    username: Optional[str] = None,
) -> str:
    """Push a LoRA adapter directory to HF Hub.

    Returns the full repo id (``username/repo_name``).
    """
    from huggingface_hub import HfApi

    username = username or _hf_username()
    repo_id = f"{username}/{repo_name}"

    api = HfApi(token=_hf_token())
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    logger.info("Pushing adapter %s → %s", adapter_dir, repo_id)
    api.upload_folder(folder_path=adapter_dir, repo_id=repo_id, repo_type="model")
    logger.info("Adapter pushed: https://huggingface.co/%s", repo_id)
    return repo_id


def pull_adapter(
    repo_id: str,
    local_dir: Optional[str] = None,
) -> str:
    """Download a LoRA adapter from HF Hub. Returns local directory path."""
    from huggingface_hub import snapshot_download

    logger.info("Pulling adapter from HF Hub: %s", repo_id)
    path = snapshot_download(repo_id, local_dir=local_dir, token=_hf_token())
    logger.info("Adapter downloaded to: %s", path)
    return path
