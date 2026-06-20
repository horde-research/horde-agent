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
        prompt_prefix_lengths: list[int] = []
        for feature in features:
            messages = _normalize_messages(feature.get("messages"))
            if not isinstance(messages, list):
                raise ValueError("Image SFT rows must contain a 'messages' list.")
            image_paths = _extract_image_paths(messages)
            if not image_paths:
                raise ValueError("Image SFT row has no image content item.")
            if len(image_paths) > 1:
                raise ValueError("Only one image per SFT row is supported in the first image trainer.")
            prompts.append(_apply_chat_template(self.processor, messages))
            image = _load_image(image_paths[0])
            images.append(image)
            prompt_prefix_lengths.append(_assistant_prefix_token_count(self.processor, messages, image))

        batch = _processor_call(
            self.processor,
            prompts=prompts,
            images=images,
        )
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Processor output must include input_ids.")
        labels = input_ids.clone()
        pad_token_id = _pad_token_id(self.processor)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        for row_idx, prefix_len in enumerate(prompt_prefix_lengths):
            labels[row_idx, : min(prefix_len, labels.shape[-1])] = -100
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
        train_dataset = self._filter_overlong_examples(self.train_dataset, split_name="train")
        eval_dataset = self._filter_overlong_examples(self.eval_dataset, split_name="validation")

        eval_strategy_value = "steps" if eval_dataset is not None else "no"
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
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageSFTCollator(self.processor, max_seq_len=self.config.max_seq_len),
            callbacks=[JsonlMetricsCallback(str(self.metrics_path))],
        )

        train_result = trainer.train()
        self.logger.info("Vision-language SFT training finished")
        return train_result.metrics

    def _filter_overlong_examples(self, dataset, *, split_name: str):
        if dataset is None:
            return None
        if len(dataset) == 0:
            return dataset

        kept_indices: list[int] = []
        dropped = 0
        for idx in range(len(dataset)):
            feature = dataset[idx]
            try:
                if self._example_fits(feature):
                    kept_indices.append(idx)
                else:
                    dropped += 1
            except Exception as exc:
                dropped += 1
                self.logger.warning("Dropping %s example %s: %s", split_name, idx, exc)

        if kept_indices and len(kept_indices) == len(dataset):
            return dataset

        if not kept_indices:
            raise ValueError(f"All {split_name} examples exceed max_seq_len={self.config.max_seq_len} or failed preprocessing.")

        self.logger.info(
            "Filtered %s split for multimodal length: kept=%s dropped=%s max_seq_len=%s",
            split_name,
            len(kept_indices),
            dropped,
            self.config.max_seq_len,
        )
        return dataset.select(kept_indices)

    def _example_fits(self, feature: Dict[str, Any]) -> bool:
        messages = _normalize_messages(feature.get("messages"))
        if not isinstance(messages, list):
            raise ValueError("Image SFT rows must contain a 'messages' list.")
        image_paths = _extract_image_paths(messages)
        if not image_paths:
            raise ValueError("Image SFT row has no image content item.")
        if len(image_paths) > 1:
            raise ValueError("Only one image per SFT row is supported in the first image trainer.")
        prompt = _apply_chat_template(self.processor, messages)
        image = _load_image(image_paths[0])
        batch = _processor_call(self.processor, prompts=[prompt], images=[image])
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Processor output must include input_ids.")
        return int(input_ids.shape[-1]) <= int(self.config.max_seq_len)


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


def _normalize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        content = message.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            visual_items: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    visual_items = []
                    text_parts = []
                    break
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        text_parts.append(text)
                elif item_type in {"image", "video"}:
                    visual_item = {key: value for key, value in item.items() if value is not None}
                    visual_items.append(visual_item)
                else:
                    visual_items = []
                    text_parts = []
                    break
            if visual_items or text_parts:
                message = dict(message)
                if visual_items:
                    message["content"] = [*visual_items, *[{"type": "text", "text": text} for text in text_parts]]
                else:
                    message["content"] = " ".join(text_parts).strip()
        normalized.append(message)
    return normalized


def _assistant_prefix_token_count(processor, messages: list[dict[str, Any]], image: Image.Image) -> int:
    prefix_messages = _messages_with_empty_assistant_content(messages)
    prefix_prompt = _apply_chat_template(processor, prefix_messages)
    prefix_batch = _processor_call(processor, prompts=[prefix_prompt], images=[image])
    input_ids = prefix_batch.get("input_ids")
    if input_ids is None:
        return 0
    attention_mask = prefix_batch.get("attention_mask")
    pad_token_id = _pad_token_id(processor)
    return _nonpad_length(input_ids[0], attention_mask[0] if attention_mask is not None else None, pad_token_id)


def _messages_with_empty_assistant_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        current = dict(message)
        if str(current.get("role") or "").strip().lower() == "assistant":
            current["content"] = ""
            prefix.append(current)
            break
        prefix.append(current)
    return prefix


def _nonpad_length(input_ids: torch.Tensor, attention_mask: torch.Tensor | None, pad_token_id: int | None) -> int:
    if attention_mask is not None:
        return int(attention_mask.to(dtype=torch.long).sum().item())
    if pad_token_id is not None:
        return int((input_ids != pad_token_id).to(dtype=torch.long).sum().item())
    return int(input_ids.shape[-1])


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


def _processor_call(processor, *, prompts: list[str], images: list[Image.Image]):
    kwargs = {
        "text": prompts,
        "images": images,
        "return_tensors": "pt",
        "padding": True,
    }
    try:
        return processor(**kwargs)
    except TypeError:
        return processor(**kwargs)


def _pad_token_id(processor) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    if getattr(processor, "pad_token_id", None) is not None:
        return int(processor.pad_token_id)
    return None
