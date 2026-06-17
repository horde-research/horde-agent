"""Modality inference and text formatting helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


TEXT_COLUMNS = {"messages", "text", "prompt", "response", "instruction", "output", "question", "answer"}
IMAGE_COLUMNS = {"image", "images", "image_path", "img", "file_path"}


def infer_modality(columns: Iterable[str], features: Dict[str, Any] | None = None) -> List[str]:
    column_set = {str(column) for column in columns}
    modalities: list[str] = []
    if column_set & IMAGE_COLUMNS:
        modalities.append("image")
    if column_set & TEXT_COLUMNS:
        modalities.append("text")
    features = features or {}
    if not modalities and any("image" in str(feature).lower() for feature in features.values()):
        modalities.append("image")
    return modalities or ["text"]


def build_example_preview(example: Dict[str, Any], *, max_chars: int = 500) -> Dict[str, Any]:
    return {str(key): _preview_value(value, max_chars=max_chars) for key, value in dict(example).items()}


def format_text_for_sft(example: Dict[str, Any]) -> str:
    if isinstance(example.get("messages"), list):
        parts = []
        for message in example["messages"]:
            role = str(message.get("role", "user"))
            content = _content_to_text(message.get("content", ""))
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)
    if example.get("text") is not None:
        return str(example["text"])
    prompt, reference = extract_text_input_output(example)
    return f"{prompt}\n{reference}".strip()


def extract_text_input_output(example: Dict[str, Any]) -> Tuple[str, str]:
    messages = example.get("messages")
    if isinstance(messages, list):
        prompt = ""
        reference = ""
        for message in messages:
            role = message.get("role")
            content = _content_to_text(message.get("content", ""))
            if role == "user":
                prompt = content
            elif role == "assistant":
                reference = content
        return prompt, reference

    prompt = (
        example.get("prompt")
        or example.get("instruction")
        or example.get("question")
        or example.get("input")
        or example.get("text")
        or ""
    )
    reference = (
        example.get("response")
        or example.get("output")
        or example.get("answer")
        or example.get("label")
        or ""
    )
    return str(prompt), str(reference)


def _preview_value(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, dict):
        return {str(k): _preview_value(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [_preview_value(item, max_chars=max_chars) for item in value[:3]]
    text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return value


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("text"):
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)
