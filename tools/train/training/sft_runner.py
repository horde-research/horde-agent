"""SFT runner for a single iteration.

Copied from `agentic_train_pipeline/training/sft_runner.py` and adjusted for new package layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from core.ml.lora import save_lora_adapters
from core.types.pipeline_types import TrainConfig


def run_sft_iteration(
    model,
    tokenizer,
    dataset,
    trainer_cls,
    train_config: TrainConfig,
    out_dir: str,
    eval_ratio: float = 0.05,
) -> Tuple[str, Dict[str, str]]:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if len(dataset) > 10:
        split = dataset.train_test_split(test_size=eval_ratio, seed=train_config.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        out_dir=out_dir,
        config=train_config,
    )
    trainer.train()

    adapter_dir = str(out_dir_path / "adapters")
    save_lora_adapters(model, adapter_dir)

    log_paths = {
        "train_log": str(out_dir_path / "logs" / "train.log"),
        "metrics": str(out_dir_path / "logs" / "metrics.jsonl"),
    }
    return adapter_dir, log_paths

