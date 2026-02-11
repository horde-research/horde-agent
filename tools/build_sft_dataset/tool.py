"""
SFT dataset generation tool.

Generates supervised fine-tuning datasets from images or text
using the unified core.llm.LLMClient for annotation.
"""

import json
import logging
import os
from typing import Any, Dict, Iterable, List

from core.llm import LLMClient
from tools.base_tool import BaseTool
from tools.build_sft_dataset.agents import ImageAnnotationAgent, TextAnnotationAgent
from tools.build_sft_dataset.loaders import load_images, load_texts_from_dir, load_texts_from_jsonl
from tools.build_sft_dataset.sft_builders import (
    build_image_sft_examples,
    build_text_sft_examples,
    parse_image_annotation,
    parse_text_annotation,
)
from tools.build_sft_dataset.types import ImageItem, TextItem

logger = logging.getLogger(__name__)


def _write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class BuildSftDatasetTool(BaseTool):
    """
    Generates SFT training data from images or text using LLM annotation.

    Supports two modes:
    - image: annotate images → build vision-language SFT examples
    - text:  annotate text   → build text SFT examples
    """

    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SFT dataset.

        Args:
            config: {
                'mode': str ('image' or 'text'),
                'provider': str (openai|gemini|xai, falls back to env),
                'model': str (falls back to env),
                'api_key': str (falls back to env),
                'input_dir': str (directory of images or texts),
                'input_jsonl': str (JSONL file for text items),
                'text_field': str (default 'text'),
                'image_exts': list[str],
                'output_annotations': str (output path),
                'output_sft': str (output path),
                'target_language': str (default 'English'),
                'batch_size': int,
                'batch_delay': float,
            }

        Returns:
            dict with stats and output paths.
        """
        mode = config.get("mode")
        if mode not in {"image", "text"}:
            raise ValueError("config['mode'] must be 'image' or 'text'")

        target_language = config.get("target_language", "English")
        batch_size = config.get("batch_size", 5)
        batch_delay = config.get("batch_delay", 1.0)
        output_annotations = config.get("output_annotations", "annotations.jsonl")
        output_sft = config.get("output_sft", "sft.jsonl")

        # Build unified LLM client
        client = LLMClient.from_env(
            provider=config.get("provider"),
            model=config.get("model"),
            api_key=config.get("api_key"),
        )

        # Step 1: Load items
        items = self._load_items(mode, config)
        items_map = {item.item_id: item for item in items}
        logger.info("Loaded %d items for mode=%s.", len(items), mode)

        # Step 2: Annotate with LLM
        if mode == "image":
            agent = ImageAnnotationAgent(
                client=client,
                target_language=target_language,
                batch_size=batch_size,
                batch_delay=batch_delay,
            )
        else:
            agent = TextAnnotationAgent(
                client=client,
                target_language=target_language,
                batch_size=batch_size,
                batch_delay=batch_delay,
            )

        annotations, failures = agent.annotate(items)
        success_count = sum(1 for a in annotations if a.get("success"))
        logger.info("Annotation: %d success, %d failures.", success_count, len(failures))

        # Step 3: Build SFT examples
        examples = self._build_examples(mode, annotations, items_map)
        logger.info("Built %d SFT examples.", len(examples))

        # Step 4: Save outputs
        _write_jsonl(output_annotations, annotations)
        _write_jsonl(output_sft, examples)
        logger.info("Saved annotations → %s", output_annotations)
        logger.info("Saved SFT examples → %s", output_sft)

        return {
            "mode": mode,
            "num_items": len(items),
            "num_annotations": success_count,
            "num_examples": len(examples),
            "num_failures": len(failures),
            "annotations_path": output_annotations,
            "sft_path": output_sft,
        }

    def _load_items(self, mode: str, config: Dict[str, Any]) -> list:
        if mode == "image":
            input_dir = config.get("input_dir")
            if not input_dir:
                raise ValueError("config['input_dir'] required for image mode")
            exts = config.get("image_exts", [".jpg", ".jpeg", ".png", ".webp"])
            return load_images(input_dir, exts)
        else:
            input_jsonl = config.get("input_jsonl")
            input_dir = config.get("input_dir")
            text_field = config.get("text_field", "text")
            if input_jsonl:
                return load_texts_from_jsonl(input_jsonl, text_field)
            elif input_dir:
                return load_texts_from_dir(input_dir)
            else:
                raise ValueError("config['input_jsonl'] or config['input_dir'] required for text mode")

    def _build_examples(
        self,
        mode: str,
        annotations: List[Dict[str, Any]],
        items_map: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for annotation in annotations:
            if not annotation.get("success"):
                continue
            item_id = annotation.get("id")
            item = items_map.get(item_id)
            if not item:
                continue
            if mode == "image":
                parsed = parse_image_annotation(annotation["data"])
                if isinstance(item, ImageItem):
                    examples.extend(build_image_sft_examples(parsed, item.image_path))
            elif mode == "text":
                parsed = parse_text_annotation(annotation["data"])
                if isinstance(item, TextItem):
                    examples.extend(build_text_sft_examples(parsed))
        return examples
