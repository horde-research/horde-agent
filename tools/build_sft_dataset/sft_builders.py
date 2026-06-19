"""
Builds chat-format SFT training examples from parsed annotations.

Image examples include the image path (vision-language format).
Text examples are STANDALONE — the source text is NOT included.
"""

import json
from typing import Any, Dict, List

from core.data.image_sft_tasks import normalize_image_sft_tasks
from .schemas import ImageAnnotation, ImageCaptionAnnotation, TextAnnotation


# ─── parsers ───────────────────────────────────────────────────────────────────

def parse_image_annotation(payload: Dict[str, Any], tasks: Any = None) -> ImageAnnotation | ImageCaptionAnnotation:
    normalized_tasks = normalize_image_sft_tasks(tasks)
    if normalized_tasks == ["caption"]:
        return ImageCaptionAnnotation.model_validate(payload)
    return ImageAnnotation.model_validate(payload)


def parse_text_annotation(payload: Dict[str, Any]) -> TextAnnotation:
    return TextAnnotation.model_validate(payload)


# ─── message helpers ───────────────────────────────────────────────────────────

def _text_msg(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _image_msg(image_path: str) -> Dict[str, Any]:
    return {"type": "image", "image": image_path}


# ─── image SFT examples ───────────────────────────────────────────────────────

def build_image_sft_examples(
    annotation: ImageAnnotation | ImageCaptionAnnotation,
    image_path: str,
    metadata: Dict[str, Any] | None = None,
    tasks: Any = None,
) -> List[Dict[str, Any]]:
    """Convert an ImageAnnotation into a list of chat-format SFT examples."""
    examples: List[Dict[str, Any]] = []
    metadata = _clean_metadata(metadata or {})
    normalized_tasks = normalize_image_sft_tasks(tasks)

    def _add(instruction: str, answer: str) -> None:
        examples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [_text_msg(instruction), _image_msg(image_path)],
                },
                {
                    "role": "assistant",
                    "content": [_text_msg(answer)],
                },
            ],
            **metadata,
        })

    if "caption" in normalized_tasks:
        _add("Describe this image in detail.", annotation.caption.text)

    if "vqa" in normalized_tasks and hasattr(annotation, "vqa"):
        for qa in annotation.vqa:
            _add(qa.question, qa.answer)

    if (
        "ocr" in normalized_tasks
        and hasattr(annotation, "ocr")
        and annotation.ocr.instruction
        and annotation.ocr.answer
    ):
        _add(annotation.ocr.instruction, annotation.ocr.answer)

    if "reason" in normalized_tasks and hasattr(annotation, "reason"):
        _add(annotation.reason.instruction, annotation.reason.answer)

    if "instruct_follow" in normalized_tasks and hasattr(annotation, "instruct_follow"):
        _add(annotation.instruct_follow.instruction, annotation.instruct_follow.answer)

    return examples


# ─── text SFT examples (STANDALONE — no source text in training) ──────────────

def build_text_sft_examples(
    annotation: TextAnnotation,
    metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Convert a TextAnnotation into standalone SFT examples.

    The source text is intentionally NOT included — each example is
    a self-contained (question, answer) pair that embeds all knowledge.
    """
    examples: List[Dict[str, Any]] = []
    metadata = _clean_metadata(metadata or {})

    def _add_single_turn(question: str, answer: str) -> None:
        examples.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            **metadata,
        })

    # knowledge QA — 5 standalone expert Q&A pairs
    for qa in annotation.knowledge_qa:
        _add_single_turn(qa.question, qa.answer)

    # detailed explanation — in-depth expert response
    _add_single_turn(
        annotation.detailed_explanation.instruction,
        annotation.detailed_explanation.response,
    )

    # analytical reasoning — multi-fact inference
    _add_single_turn(
        annotation.analytical_reasoning.instruction,
        annotation.analytical_reasoning.response,
    )

    # conversational exchange — multi-turn dialogue
    conv = annotation.conversational_exchange
    examples.append({
        "messages": [
            {"role": "user", "content": conv.opening_question},
            {"role": "assistant", "content": conv.opening_response},
            {"role": "user", "content": conv.follow_up_question},
            {"role": "assistant", "content": conv.follow_up_response},
        ],
        **metadata,
    })

    return examples


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): value for key, value in metadata.items() if value not in (None, "", [], {})}


# ─── serialisation ─────────────────────────────────────────────────────────────

def serialize_examples(examples: List[Dict[str, Any]]) -> str:
    return json.dumps(examples, ensure_ascii=False)
