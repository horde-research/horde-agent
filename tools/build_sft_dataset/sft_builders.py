"""
Builds chat-format SFT training examples from parsed annotations.

Image examples include the image path (vision-language format).
Text examples are STANDALONE — the source text is NOT included.
"""

import json
from typing import Any, Dict, List

from .schemas import ImageAnnotation, TextAnnotation


# ─── parsers ───────────────────────────────────────────────────────────────────

def parse_image_annotation(payload: Dict[str, Any]) -> ImageAnnotation:
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
    annotation: ImageAnnotation,
    image_path: str,
) -> List[Dict[str, Any]]:
    """Convert an ImageAnnotation into a list of chat-format SFT examples."""
    examples: List[Dict[str, Any]] = []

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
        })

    # caption
    _add("Describe this image in detail.", annotation.caption.text)

    # VQA (3-7 pairs)
    for qa in annotation.vqa:
        _add(qa.question, qa.answer)

    # OCR (only if present)
    if annotation.ocr.instruction and annotation.ocr.answer:
        _add(annotation.ocr.instruction, annotation.ocr.answer)

    # reasoning
    _add(annotation.reason.instruction, annotation.reason.answer)

    # instruction following
    _add(annotation.instruct_follow.instruction, annotation.instruct_follow.answer)

    return examples


# ─── text SFT examples (STANDALONE — no source text in training) ──────────────

def build_text_sft_examples(
    annotation: TextAnnotation,
) -> List[Dict[str, Any]]:
    """Convert a TextAnnotation into standalone SFT examples.

    The source text is intentionally NOT included — each example is
    a self-contained (question, answer) pair that embeds all knowledge.
    """
    examples: List[Dict[str, Any]] = []

    def _add_single_turn(question: str, answer: str) -> None:
        examples.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
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
    })

    return examples


# ─── serialisation ─────────────────────────────────────────────────────────────

def serialize_examples(examples: List[Dict[str, Any]]) -> str:
    return json.dumps(examples, ensure_ascii=False)
