"""Inference helpers for small evaluation samples.

Copied from `agentic_train_pipeline/eval/inference.py` and adjusted for new package layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from core.data.modality import extract_text_input_output


def run_inference(
    model,
    tokenizer,
    dataset,
    out_dir: str,
    max_samples: int = 64,
    max_new_tokens: int = 128,
) -> str:
    out_path = Path(out_dir) / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = dataset.select(range(min(len(dataset), max_samples)))
    results: List[Dict[str, Any]] = []

    model.eval()
    for idx, example in enumerate(sample):
        prompt, reference = extract_text_input_output(example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated_ids = output_ids[0][prompt_len:] if prompt_len else output_ids[0]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not pred_text:
            pred_text = _strip_prompt_echo(tokenizer.decode(output_ids[0], skip_special_tokens=True), prompt)
        results.append(
            {
                "id": idx,
                "input": prompt,
                "prediction": pred_text,
                "reference": reference,
            }
        )

    with out_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(out_path)


def run_image_inference(
    model,
    processor,
    dataset,
    out_dir: str,
    max_samples: int = 64,
    max_new_tokens: int = 128,
) -> str:
    out_path = Path(out_dir) / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = dataset.select(range(min(len(dataset), max_samples)))
    results: List[Dict[str, Any]] = []

    model.eval()
    for idx, example in enumerate(sample):
        prompt, reference, image_path = extract_image_input_output(example)
        if not image_path:
            results.append(
                {
                    "id": idx,
                    "input": prompt,
                    "prediction": "",
                    "reference": reference,
                    "image_path": None,
                    "error": "missing_image",
                }
            )
            continue
        try:
            image = _load_image(image_path)
            rendered_prompt = _render_image_prompt(processor, example.get("messages"), prompt)
            inputs = _processor_call(
                processor,
                prompt=rendered_prompt,
                image=image,
                max_seq_len=getattr(getattr(processor, "tokenizer", None), "model_max_length", 2048),
            )
            inputs = {key: value.to(_model_input_device(model)) for key, value in inputs.items() if hasattr(value, "to")}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            generated_ids = output_ids[0][prompt_len:] if prompt_len else output_ids[0]
            pred_text = _decode_processor(processor, generated_ids).strip()
            if not pred_text:
                pred_text = _strip_prompt_echo(_decode_processor(processor, output_ids[0]), rendered_prompt)
            results.append(
                {
                    "id": idx,
                    "input": prompt,
                    "prediction": pred_text,
                    "reference": reference,
                    "image_path": image_path,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": idx,
                    "input": prompt,
                    "prediction": "",
                    "reference": reference,
                    "image_path": image_path,
                    "error": f"image_eval_failed:{type(exc).__name__}:{exc}",
                }
            )

    with out_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(out_path)


def extract_image_input_output(example: Dict[str, Any]) -> Tuple[str, str, str | None]:
    messages = example.get("messages")
    if not isinstance(messages, list):
        prompt, reference = extract_text_input_output(example)
        return prompt, reference, None
    prompt = ""
    reference = ""
    image_path: str | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        text_parts, image_paths = _content_parts(content)
        if role == "user":
            prompt = "\n".join(text_parts).strip()
            image_path = image_path or (image_paths[0] if image_paths else None)
        elif role == "assistant":
            reference = "\n".join(text_parts).strip()
    return prompt, reference, image_path


def _content_parts(content: Any) -> tuple[list[str], list[str]]:
    if isinstance(content, str):
        return [content], []
    texts: list[str] = []
    images: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                texts.append(str(item))
                continue
            if item.get("type") == "text":
                texts.append(str(item.get("text") or ""))
            elif item.get("type") == "image":
                image_path = str(item.get("image") or item.get("path") or "").strip()
                if image_path:
                    images.append(image_path)
            elif item.get("text"):
                texts.append(str(item["text"]))
    return [text for text in texts if text], images


def _strip_prompt_echo(generated: str, prompt: str) -> str:
    generated = str(generated or "").strip()
    prompt = str(prompt or "").strip()
    if prompt and generated.startswith(prompt):
        return generated[len(prompt) :].strip()
    return generated


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _render_image_prompt(processor, messages: Any, fallback_prompt: str) -> str:
    user_messages = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user_messages.append(message)
                break
    if user_messages and hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if user_messages and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    return fallback_prompt


def _processor_call(processor, *, prompt: str, image: Image.Image, max_seq_len: int):
    kwargs = {
        "text": [prompt],
        "images": [image],
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


def _decode_processor(processor, token_ids) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(token_ids, skip_special_tokens=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    raise ValueError("Processor/tokenizer does not provide decode().")


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
