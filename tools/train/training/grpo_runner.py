"""GRPO runner for a single RL phase."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import Dataset

from core.data.modality import extract_text_input_output
from core.ml.lora import load_lora_adapters, save_lora_adapters
from core.types.pipeline_types import GRPOConfig
from tools.train.trainers.static_sft_trainer import JsonlMetricsCallback
from tools.train.training.grpo_rewards import LLMJudgeReward


def _build_prompt_dataset(dataset) -> Dataset:
    rows = []
    for example in dataset:
        prompt, reference = extract_text_input_output(dict(example))
        if prompt:
            rows.append({"prompt": prompt, "reference": reference})
    if not rows:
        raise ValueError("GRPO requires at least one row with a non-empty prompt.")
    return Dataset.from_list(rows)


def run_grpo_iteration(
    *,
    model,
    tokenizer,
    dataset,
    grpo_config: GRPOConfig,
    out_dir: str,
    previous_adapter_path: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    try:
        from trl import GRPOConfig as TrlGRPOConfig
        from trl import GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            "GRPO training requires the 'trl' package. Install requirements.txt or requirements-cpu.txt."
        ) from exc

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if previous_adapter_path:
        model = load_lora_adapters(model, previous_adapter_path, is_trainable=True)

    prompt_dataset = _build_prompt_dataset(dataset)
    metrics_path = logs_dir / "metrics.jsonl"
    judge_log_path = logs_dir / "judge_rewards.jsonl"
    reward_func = LLMJudgeReward(
        log_path=str(judge_log_path),
        batch_size=grpo_config.judge_batch_size,
        batch_delay_seconds=grpo_config.judge_batch_delay,
    )

    args = TrlGRPOConfig(
        output_dir=str(out_dir_path / "hf_outputs"),
        learning_rate=grpo_config.lr,
        per_device_train_batch_size=grpo_config.batch_size,
        gradient_accumulation_steps=grpo_config.grad_accum,
        max_steps=grpo_config.max_steps,
        num_generations=grpo_config.num_generations,
        max_prompt_length=grpo_config.max_prompt_length,
        max_completion_length=grpo_config.max_completion_length,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=grpo_config.seed,
    )

    callback = JsonlMetricsCallback(str(metrics_path))
    trainer_kwargs = {
        "model": model,
        "args": args,
        "reward_funcs": reward_func,
        "train_dataset": prompt_dataset,
        "callbacks": [callback],
    }
    trainer_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()

    adapter_dir = str(out_dir_path / "adapters")
    save_lora_adapters(trainer.model, adapter_dir)

    log_paths = {
        "train_log": str(logs_dir / "train.log"),
        "metrics": str(metrics_path),
        "judge_rewards": str(judge_log_path),
    }
    Path(log_paths["train_log"]).touch()
    return adapter_dir, log_paths
