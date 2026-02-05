"""Static SFT trainer using HuggingFace Trainer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments

from agentic_train_pipeline.parser.modality import format_text_for_sft
from agentic_train_pipeline.types import TrainConfig


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: str) -> None:
        self.metrics_path = metrics_path
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    def _write(self, logs: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(logs) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs = dict(logs)
            logs["step"] = state.global_step
            self._write(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            metrics = dict(metrics)
            metrics["step"] = state.global_step
            self._write(metrics)


class StaticSFTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        out_dir: str,
        config: TrainConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.out_dir = Path(out_dir)
        self.config = config

        self.logs_dir = self.out_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "train.log"
        self.metrics_path = self.logs_dir / "metrics.jsonl"

        self._setup_logger()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"sft.{id(self)}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def _tokenize(self, example: Dict[str, Any]) -> Dict[str, Any]:
        text = format_text_for_sft(example)
        tokens = self.tokenizer(
            text,
            max_length=self.config.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def train(self) -> Dict[str, Any]:
        self.logger.info("Starting static SFT training")

        tokenized_train = self.train_dataset.map(self._tokenize, remove_columns=self.train_dataset.column_names)
        tokenized_eval = None
        if self.eval_dataset is not None:
            tokenized_eval = self.eval_dataset.map(self._tokenize, remove_columns=self.eval_dataset.column_names)

        eval_strategy_value = "steps" if tokenized_eval is not None else "no"
        args_kwargs = dict(
            output_dir=str(self.out_dir / "hf_outputs"),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_accum,
            learning_rate=self.config.lr,
            max_steps=self.config.max_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            eval_steps=self.config.eval_steps,
            logging_steps=max(1, self.config.eval_steps // 2),
            save_strategy="no",
            report_to=[],
            seed=self.config.seed,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        )
        # Support both new and old Transformers argument names.
        if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
            args_kwargs["evaluation_strategy"] = eval_strategy_value
        else:
            args_kwargs["eval_strategy"] = eval_strategy_value
        args = TrainingArguments(**args_kwargs)

        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        callback = JsonlMetricsCallback(str(self.metrics_path))
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            callbacks=[callback],
        )

        train_result = trainer.train()
        self.logger.info("Training finished")
        return train_result.metrics
