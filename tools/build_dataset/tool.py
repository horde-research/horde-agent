"""Dataset building tool.

Prepares training-ready datasets from raw data.
"""

import shutil
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from tools.base_tool import BaseTool

from core.data.hf_dataset import load_dataset_from_path
from core.data.manifest import write_manifest
from core.data.modality import build_example_preview, infer_modality
from core.data.validation import validate_text_columns
from core.types.pipeline_types import DatasetSummary


class BuildDatasetTool(BaseTool):
    """
    Builds formatted datasets for training.
    
    Operations:
    - Train/validation/test split
    - Tokenization
    - Format conversion (SFT/GRPO/DPO specific)
    - Padding and truncation
    - Dataset saving in HF format
    
    Output format depends on training method:
    - SFT: (input, output) pairs
    - GRPO: (prompt, chosen, rejected) tuples
    - DPO: (prompt, chosen, rejected) tuples
    """
    
    def execute(self, data_path, config):
        """
        Build training dataset.
        
        Args:
            data_path (str): Path to evaluated data
            config (dict):
                - split_ratio: float
                - max_length: int
                - training_method: str (sft|grpo|dpo)
                
        Returns:
            dict: {
                'dataset_ref': dict,
                'dataset_summary': dict,
                'dataset_manifest_path': str
            }
        """
        source_split = str(config.get("split", "train"))
        train_split = str(config.get("train_split", "train"))
        validation_split = str(config.get("validation_split") or config.get("eval_split") or "validation")
        validation_ratio = _validation_ratio(config.get("validation_ratio", config.get("val_ratio", 0.1)))
        seed = int(config.get("seed", 42))
        run_dir = config.get("run_dir") or config.get("out_dir")
        if not run_dir:
            raise ValueError("BuildDatasetTool requires config['run_dir'] (or 'out_dir').")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        dataset, resolved_id = load_dataset_from_path(data_path, split=source_split)
        columns = list(getattr(dataset, "column_names", []))
        features = getattr(dataset, "features", {}) or {}
        modality_candidates = infer_modality(columns, features)
        example = build_example_preview(dataset[0]) if len(dataset) > 0 else {}
        text_columns = [c for c in columns if c in {"messages", "text", "prompt", "response", "instruction", "output"}]
        warnings = validate_text_columns(dataset, text_columns)
        group_key_column = str(config.get("group_key_column") or "group_key")
        split_dataset, split_warnings, split_metadata = _build_train_validation_splits(
            dataset,
            train_split=train_split,
            validation_split=validation_split,
            validation_ratio=validation_ratio,
            seed=seed,
            group_key_column=group_key_column,
        )
        warnings.extend(split_warnings)
        dataset_dir = Path(run_dir) / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        split_dataset.save_to_disk(str(dataset_dir))
        split_counts = {name: len(split_dataset[name]) for name in split_dataset.keys()}

        summary = DatasetSummary(
            data_path=str(dataset_dir),
            resolved_data_id=resolved_id,
            columns=columns,
            sample_count=len(dataset),
            split_counts=split_counts,
            split_strategy=split_metadata["split_strategy"],
            group_key_column=split_metadata.get("group_key_column"),
            split_group_counts=split_metadata["split_group_counts"],
            example=example,
            modality_candidates=modality_candidates,
            validation_warnings=warnings,
        )
        manifest_path = write_manifest(run_dir, summary.model_dump())

        return {
            "dataset_ref": {
                "kind": "hf",
                "data_path": str(dataset_dir),
                "split": train_split,
                "eval_split": validation_split,
                "resolved_id": resolved_id,
                "source_data_path": str(data_path),
                "source_split": source_split,
                "split_counts": split_counts,
                "split_strategy": split_metadata["split_strategy"],
                "group_key_column": split_metadata.get("group_key_column"),
                "split_group_counts": split_metadata["split_group_counts"],
            },
            "dataset_summary": summary.model_dump(),
            "dataset_manifest_path": manifest_path,
        }


def _validation_ratio(value: Any) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        ratio = 0.1
    return min(max(ratio, 0.0), 0.5)


def _build_train_validation_splits(
    dataset: Dataset,
    *,
    train_split: str,
    validation_split: str,
    validation_ratio: float,
    seed: int,
    group_key_column: str = "group_key",
) -> tuple[DatasetDict, list[str], dict[str, Any]]:
    warnings: list[str] = []
    total = len(dataset)
    row_split_metadata = {
        "split_strategy": "row",
        "group_key_column": None,
        "split_group_counts": {},
    }
    if total <= 0:
        return DatasetDict({train_split: dataset, validation_split: dataset}), ["dataset_empty"], row_split_metadata
    if total == 1 or validation_ratio <= 0.0:
        if total == 1:
            warnings.append("validation_reuses_train_single_sample")
        else:
            warnings.append("validation_disabled_reuses_train")
        return DatasetDict({train_split: dataset, validation_split: dataset}), warnings, row_split_metadata

    group_values = _group_values(dataset, group_key_column)
    if group_values:
        group_split = _build_group_train_validation_splits(
            dataset,
            group_values=group_values,
            train_split=train_split,
            validation_split=validation_split,
            validation_ratio=validation_ratio,
            seed=seed,
            group_key_column=group_key_column,
        )
        if group_split is not None:
            return group_split

    val_count = max(1, int(round(total * validation_ratio)))
    val_count = min(val_count, total - 1)
    split = dataset.train_test_split(test_size=val_count, seed=seed, shuffle=True)
    return DatasetDict({train_split: split["train"], validation_split: split["test"]}), warnings, row_split_metadata


def _group_values(dataset: Dataset, group_key_column: str) -> list[str] | None:
    if group_key_column not in set(getattr(dataset, "column_names", []) or []):
        return None
    values: list[str] = []
    for value in dataset[group_key_column]:
        normalized = str(value or "").strip()
        if not normalized:
            return None
        values.append(normalized)
    return values


def _build_group_train_validation_splits(
    dataset: Dataset,
    *,
    group_values: list[str],
    train_split: str,
    validation_split: str,
    validation_ratio: float,
    seed: int,
    group_key_column: str,
) -> tuple[DatasetDict, list[str], dict[str, Any]] | None:
    groups = sorted(set(group_values))
    if len(groups) <= 1:
        return None

    shuffled_groups = list(groups)
    random.Random(seed).shuffle(shuffled_groups)
    val_group_count = max(1, int(round(len(groups) * validation_ratio)))
    val_group_count = min(val_group_count, len(groups) - 1)
    validation_groups = set(shuffled_groups[:val_group_count])
    train_indices: list[int] = []
    validation_indices: list[int] = []
    train_groups: set[str] = set()
    actual_validation_groups: set[str] = set()

    for idx, group_key in enumerate(group_values):
        if group_key in validation_groups:
            validation_indices.append(idx)
            actual_validation_groups.add(group_key)
        else:
            train_indices.append(idx)
            train_groups.add(group_key)

    if not train_indices or not validation_indices:
        return None

    metadata = {
        "split_strategy": "group",
        "group_key_column": group_key_column,
        "split_group_counts": {
            train_split: len(train_groups),
            validation_split: len(actual_validation_groups),
        },
    }
    return (
        DatasetDict(
            {
                train_split: dataset.select(train_indices),
                validation_split: dataset.select(validation_indices),
            }
        ),
        [],
        metadata,
    )
