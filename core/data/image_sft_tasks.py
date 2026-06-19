from __future__ import annotations

from collections.abc import Iterable
from typing import Any


IMAGE_SFT_TASKS = ("caption", "vqa", "ocr", "reason", "instruct_follow")
_ALIASES = {
    "captions": "caption",
    "captioning": "caption",
    "qa": "vqa",
    "visual_qa": "vqa",
    "visual-question-answering": "vqa",
    "visual_question_answering": "vqa",
    "reasoning": "reason",
    "visual_reasoning": "reason",
    "instruction": "instruct_follow",
    "instruction_following": "instruct_follow",
    "instruct": "instruct_follow",
}
_ALL_VALUES = {"all", "full", "multi", "multitask", "multi_task"}


def normalize_image_sft_tasks(value: Any) -> list[str]:
    if value in (None, "", []):
        return ["caption"]
    raw_items: list[Any]
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, Iterable):
        raw_items = list(value)
    else:
        raw_items = [value]

    tasks: list[str] = []
    for item in raw_items:
        normalized = str(item or "").strip().lower().replace("-", "_")
        if not normalized:
            continue
        if normalized in _ALL_VALUES:
            return list(IMAGE_SFT_TASKS)
        normalized = _ALIASES.get(normalized, normalized)
        if normalized not in IMAGE_SFT_TASKS:
            valid = ", ".join(IMAGE_SFT_TASKS)
            raise ValueError(f"Unsupported image SFT task '{item}'. Valid tasks: {valid}, all.")
        if normalized not in tasks:
            tasks.append(normalized)
    return tasks or ["caption"]
