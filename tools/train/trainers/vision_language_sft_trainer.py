"""Vision-language SFT trainer for image chat examples."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image
from transformers import Trainer, TrainerCallback, TrainingArguments

from core.types.pipeline_types import TrainConfig


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: str) -> None:
        self.metrics_path = metrics_path
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    def _write(self, logs: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(logs, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            payload = dict(logs)
            payload["step"] = state.global_step
            self._write(payload)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            payload = dict(metrics)
            payload["step"] = state.global_step
            self._write(payload)


class VisionLanguageSFTCollator:
    """Convert image-chat rows into processor tensors for HF Trainer."""

    def __init__(self, processor, *, max_seq_len: int) -> None:
        self.processor = processor
        self.max_seq_len = max_seq_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts: list[str] = []
        images: list[Image.Image] = []
        for feature in features:
            messages = feature.get("messages")
            if not isinstance(messages, list):
                raise ValueError("Image SFT rows must contain a 'messages' list.")
            image_paths = _extract_image_paths(messages)
            if not image_paths:
                raise ValueError("Image SFT row has no image content item.")
            if len(image_paths) > 1:
                raise ValueError("Only one image per SFT row is supported in the first image trainer.")
            prompts.append(_apply_chat_template(self.processor, messages))
            images.append(_load_image(image_paths[0]))

        batch = _processor_call(
            self.processor,
            prompts=prompts,
            images=images,
            max_seq_len=self.max_seq_len,
        )
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Processor output must include input_ids.")
        labels = input_ids.clone()
        pad_token_id = _pad_token_id(self.processor)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch


class VisionLanguageSFTTrainer:
    """LoRA SFT trainer for single-image vision-language models."""

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
        self.processor = tokenizer
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
        self.logger = logging.getLogger(f"vlm_sft.{id(self)}")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

    def train(self) -> Dict[str, Any]:
        self.logger.info("Starting vision-language SFT training")

        eval_strategy_value = "steps" if self.eval_dataset is not None else "no"
        use_mps = (
            not torch.cuda.is_available()
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
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
            remove_unused_columns=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        )
        ta_vars = TrainingArguments.__init__.__code__.co_varnames
        if use_mps and "use_mps_device" in ta_vars:
            args_kwargs["use_mps_device"] = True
        if "evaluation_strategy" in ta_vars:
            args_kwargs["evaluation_strategy"] = eval_strategy_value
        else:
            args_kwargs["eval_strategy"] = eval_strategy_value
        args = TrainingArguments(**args_kwargs)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=VisionLanguageSFTCollator(self.processor, max_seq_len=self.config.max_seq_len),
            callbacks=[JsonlMetricsCallback(str(self.metrics_path))],
        )

        train_result = trainer.train()
        self.logger.info("Vision-language SFT training finished")
        return train_result.metrics


def _extract_image_paths(messages: Iterable[Dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image_path = str(item.get("image") or item.get("path") or "").strip()
                if image_path:
                    paths.append(image_path)
    return paths


def _apply_chat_template(processor, messages: list[dict[str, Any]]) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    raise NotImplementedError("The selected processor/tokenizer does not support chat templates.")


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _processor_call(processor, *, prompts: list[str], images: list[Image.Image], max_seq_len: int):
    kwargs = {
        "text": prompts,
        "images": images,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_seq_len,
    }
    try:
        return processor(**kwargs)
    except TypeError:
        kwargs.pop("truncation", None)
        kwargs.pop("max_length", None)
        return processor(**kwargs)


def _pad_token_id(processor) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    if getattr(processor, "pad_token_id", None) is not None:
        return int(processor.pad_token_id)
    return None
