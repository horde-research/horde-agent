"""
SFT dataset generation tool.

Generates supervised fine-tuning datasets from images or text
using the unified core.llm.LLMClient for annotation.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List

from core.llm import LLMClient
from core.data.image_sft_tasks import normalize_image_sft_tasks
from tools.base_tool import BaseTool
from tools.build_sft_dataset.agents import ImageAnnotationAgent, TextAnnotationAgent
from tools.build_sft_dataset.loaders import (
    load_images,
    load_images_from_manifest,
    load_texts_from_dir,
    load_texts_from_jsonl,
)
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
                'image_manifest': str (optional images.json with taxonomy metadata),
                'input_jsonl': str (JSONL file for text items),
                'text_field': str (default 'text'),
                'image_exts': list[str],
                'output_annotations': str (output path),
                'output_sft': str (output path),
                'target_language': str (default 'English'),
                'prompt_preset': str (default 'default'),
                'image_tasks': list[str] or comma-separated str (image mode, default caption),
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
        prompt_preset = config.get("prompt_preset", "default")
        image_tasks = normalize_image_sft_tasks(config.get("image_tasks") or config.get("image_sft_tasks"))
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
                prompt_preset=prompt_preset,
                image_tasks=image_tasks,
            )
        else:
            agent = TextAnnotationAgent(
                client=client,
                target_language=target_language,
                batch_size=batch_size,
                batch_delay=batch_delay,
                prompt_preset=prompt_preset,
            )

        if mode == "text" and config.get("reuse_annotations"):
            annotations, failures, reuse_summary = self._annotate_text_with_cache(
                agent,
                items,
                config,
                target_language=target_language,
                prompt_preset=prompt_preset,
            )
        else:
            annotations, failures = agent.annotate(items)
            reuse_summary = {"enabled": False}
        success_count = sum(1 for a in annotations if a.get("success"))
        logger.info("Annotation: %d success, %d failures.", success_count, len(failures))

        # Step 3: Build SFT examples
        examples = self._build_examples(mode, annotations, items_map, image_tasks=image_tasks)
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
            "prompt_preset": prompt_preset,
            "image_tasks": image_tasks if mode == "image" else None,
            "annotation_reuse": reuse_summary,
            "annotation_cache_path": reuse_summary.get("cache_path"),
        }

    def _load_items(self, mode: str, config: Dict[str, Any]) -> list:
        if mode == "image":
            manifest = config.get("image_manifest") or config.get("input_manifest")
            exts = config.get("image_exts", [".jpg", ".jpeg", ".png", ".webp"])
            if manifest:
                return load_images_from_manifest(manifest, exts)
            input_dir = config.get("input_dir")
            if not input_dir:
                raise ValueError("config['input_dir'] or config['image_manifest'] required for image mode")
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
        *,
        image_tasks: List[str] | None = None,
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
                parsed = parse_image_annotation(annotation["data"], tasks=image_tasks)
                if isinstance(item, ImageItem):
                    examples.extend(
                        build_image_sft_examples(
                            parsed,
                            item.image_path,
                            metadata=_image_metadata(item),
                            tasks=image_tasks,
                        )
                    )
            elif mode == "text":
                parsed = parse_text_annotation(annotation["data"])
                if isinstance(item, TextItem):
                    examples.extend(build_text_sft_examples(parsed, metadata=_text_metadata(item)))
        return examples

    def _annotate_text_with_cache(
        self,
        agent: TextAnnotationAgent,
        items: List[TextItem],
        config: Dict[str, Any],
        *,
        target_language: str,
        prompt_preset: str,
    ) -> tuple[List[Dict[str, Any]], List[TextItem], Dict[str, Any]]:
        cache_path = str(config.get("annotation_cache_path") or "").strip()
        signature = _annotation_cache_signature(
            config.get("annotation_cache_metadata"),
            target_language=target_language,
            prompt_preset=prompt_preset,
            provider=config.get("provider"),
            model=config.get("model"),
        )
        cache = _load_annotation_cache(cache_path)
        cached_by_item_id: Dict[str, Dict[str, Any]] = {}
        missing_items: List[TextItem] = []

        for item in items:
            cache_key = _text_annotation_cache_key(item, signature)
            entry = cache.get(cache_key)
            cached_annotation = entry.get("annotation") if isinstance(entry, dict) else None
            if isinstance(cached_annotation, dict) and cached_annotation.get("success"):
                cached_by_item_id[item.item_id] = _annotation_for_current_item(cached_annotation, item, cache_key)
            else:
                missing_items.append(item)

        if missing_items:
            new_annotations, failures = agent.annotate(missing_items)
        else:
            new_annotations, failures = [], []

        new_by_item_id = {str(annotation.get("id")): annotation for annotation in new_annotations}
        for item in missing_items:
            annotation = new_by_item_id.get(item.item_id)
            if not annotation or not annotation.get("success"):
                continue
            cache_key = _text_annotation_cache_key(item, signature)
            cache[cache_key] = {
                "schema_version": "text_annotation_cache.v1",
                "cache_key": cache_key,
                "signature": signature,
                "source_id": item.source_id,
                "source_url": item.source_url,
                "group_key": item.group_key,
                "collection_iteration": item.collection_iteration,
                "annotation": dict(annotation),
            }

        annotations: List[Dict[str, Any]] = []
        for item in items:
            if item.item_id in cached_by_item_id:
                annotations.append(cached_by_item_id[item.item_id])
            elif item.item_id in new_by_item_id:
                annotations.append(new_by_item_id[item.item_id])

        if cache_path:
            _write_annotation_cache(cache_path, cache.values())

        summary = {
            "enabled": True,
            "cache_path": cache_path or None,
            "num_items": len(items),
            "num_reused_annotations": len(cached_by_item_id),
            "num_llm_annotation_requests": len(missing_items),
            "num_new_successful_annotations": sum(1 for item in missing_items if new_by_item_id.get(item.item_id, {}).get("success")),
            "num_cache_entries": len(cache),
            "signature": signature,
        }
        logger.info(
            "Annotation cache: reused %d/%d, requested %d new annotations.",
            summary["num_reused_annotations"],
            summary["num_items"],
            summary["num_llm_annotation_requests"],
        )
        return annotations, failures, summary


def _text_metadata(item: TextItem) -> Dict[str, Any]:
    return {
        "group_key": item.group_key or item.source_url or item.source_id or item.item_id,
        "source_id": item.source_id or item.item_id,
        "source_url": item.source_url,
        "source_query": item.source_query,
        "source_excerpt": item.source_excerpt,
        "collection_iteration": item.collection_iteration,
    }


def _image_metadata(item: ImageItem) -> Dict[str, Any]:
    return {
        "group_key": item.group_key or item.image_path,
        "source_url": item.source_url,
        "source_image_url": item.source_image_url,
        "source_query": item.source_query,
        "source_excerpt": item.source_excerpt,
    }


def _annotation_cache_signature(
    explicit: Any,
    *,
    target_language: str,
    prompt_preset: str,
    provider: Any,
    model: Any,
) -> Dict[str, Any]:
    signature = dict(explicit) if isinstance(explicit, dict) else {}
    signature.setdefault("schema_version", "text_annotation.v1")
    signature.setdefault("target_language", target_language)
    signature.setdefault("prompt_preset", prompt_preset)
    signature.setdefault("provider", provider)
    signature.setdefault("model", model)
    return {str(key): signature[key] for key in sorted(signature)}


def _text_annotation_cache_key(item: TextItem, signature: Dict[str, Any]) -> str:
    text_hash = hashlib.sha256(_normalize_text(item.text).encode("utf-8")).hexdigest()
    identity = item.source_url or item.group_key or item.source_id or item.item_id
    payload = {
        "identity": identity,
        "text_hash": text_hash,
        "signature": signature,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _annotation_for_current_item(annotation: Dict[str, Any], item: TextItem, cache_key: str) -> Dict[str, Any]:
    current = dict(annotation)
    current["id"] = item.item_id
    current["cache_key"] = cache_key
    current["reused_from_cache"] = True
    return current


def _load_annotation_cache(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("cache_key"):
                cache[str(row["cache_key"])] = row
    return cache


def _write_annotation_cache(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ordered = sorted(rows, key=lambda row: str(row.get("cache_key") or ""))
    with open(path, "w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
