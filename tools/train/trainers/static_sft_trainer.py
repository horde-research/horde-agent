"""Static SFT trainer using HuggingFace Trainer.

Copied from `agentic_train_pipeline/training/static_sft_trainer.py` and adjusted for new package layout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator

from core.data.modality import _content_to_text, extract_text_input_output, format_text_for_sft
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
        text, assistant_spans = _format_text_with_assistant_spans(example, tokenizer=self.tokenizer)
        try:
            tokens = self.tokenizer(
                text,
                max_length=self.config.max_seq_len,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
            )
            offsets = tokens.pop("offset_mapping", None)
        except TypeError:
            tokens = self.tokenizer(
                text,
                max_length=self.config.max_seq_len,
                truncation=True,
                padding="max_length",
            )
            offsets = None
        tokens["labels"] = _assistant_only_labels(
            tokens["input_ids"],
            offsets,
            assistant_spans,
            attention_mask=tokens.get("attention_mask"),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        return tokens

    def train(self) -> Dict[str, Any]:
        self.logger.info("Starting static SFT training")

        tokenized_train = self.train_dataset.map(self._tokenize, remove_columns=self.train_dataset.column_names)
        tokenized_eval = None
        if self.eval_dataset is not None:
            tokenized_eval = self.eval_dataset.map(self._tokenize, remove_columns=self.eval_dataset.column_names)

        eval_strategy_value = "steps" if tokenized_eval is not None else "no"
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
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        )
        ta_vars = TrainingArguments.__init__.__code__.co_varnames
        if use_mps and "use_mps_device" in ta_vars:
            args_kwargs["use_mps_device"] = True
        # Support both new and old Transformers argument names.
        if "evaluation_strategy" in ta_vars:
            args_kwargs["evaluation_strategy"] = eval_strategy_value
        else:
            args_kwargs["eval_strategy"] = eval_strategy_value
        args = TrainingArguments(**args_kwargs)

        data_collator = default_data_collator
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


def _format_text_with_assistant_spans(example: Dict[str, Any], *, tokenizer: Any = None) -> tuple[str, list[tuple[int, int]]]:
    messages = example.get("messages")
    if isinstance(messages, list):
        rendered = _render_chat_template(tokenizer, messages, add_generation_prompt=False)
        if rendered:
            spans = _assistant_spans_in_rendered_chat(rendered, messages)
            if spans:
                return rendered, spans
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            if parts:
                parts.append("\n")
                cursor += 1
            role = str(message.get("role", "user"))
            content = _content_to_text(message.get("content", ""))
            prefix = f"<|{role}|>\n"
            parts.append(prefix)
            cursor += len(prefix)
            start = cursor
            parts.append(content)
            cursor += len(content)
            if role.strip().lower() == "assistant" and content.strip():
                spans.append((start, cursor))
        return "".join(parts), spans
    if any(
        example.get(key) is not None
        for key in ("prompt", "instruction", "question", "input", "response", "output", "answer", "label")
    ):
        prompt, reference = extract_text_input_output(example)
        text = f"{prompt}\n{reference}".strip()
        if reference:
            start = text.rfind(str(reference))
            return text, [(start, start + len(str(reference)))] if start >= 0 else []
        return text, []
    text = format_text_for_sft(example)
    return text, [(0, len(text))] if text.strip() else []


def _render_chat_template(tokenizer: Any, messages: list[Any], *, add_generation_prompt: bool) -> str:
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return ""
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return ""
    return rendered if isinstance(rendered, str) else ""


def _assistant_spans_in_rendered_chat(rendered: str, messages: list[Any]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        content = _content_to_text(message.get("content", ""))
        if not content:
            continue
        start = rendered.find(content, cursor)
        if start < 0:
            return []
        end = start + len(content)
        spans.append((start, end))
        cursor = end
    return spans


def _assistant_only_labels(
    input_ids: Any,
    offsets: Any,
    assistant_spans: list[tuple[int, int]],
    *,
    attention_mask: Any = None,
    pad_token_id: int | None = None,
) -> list[int]:
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    labels = list(ids)
    if not assistant_spans or offsets is None:
        return _mask_padding(labels, attention_mask=attention_mask, pad_token_id=pad_token_id)
    normalized_offsets = [tuple(offset) for offset in offsets]
    for idx, (start, end) in enumerate(normalized_offsets):
        keep = start != end and any(start < span_end and end > span_start for span_start, span_end in assistant_spans)
        if not keep:
            labels[idx] = -100
    return _mask_padding(labels, attention_mask=attention_mask, pad_token_id=pad_token_id)


def _mask_padding(labels: list[int], *, attention_mask: Any = None, pad_token_id: int | None = None) -> list[int]:
    mask = list(attention_mask) if attention_mask is not None else None
    for idx, value in enumerate(labels):
        if (mask is not None and idx < len(mask) and int(mask[idx]) == 0) or (
            pad_token_id is not None and value == pad_token_id
        ):
            labels[idx] = -100
    return labels
