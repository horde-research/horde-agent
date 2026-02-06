"""Image annotation agent — uses core.llm.LLMClient for Gemini/OpenAI/xAI."""

import logging
from typing import Any, Dict, List, Tuple

from core.llm import LLMClient, LLMRequest
from ..prompts import build_image_prompt
from ..sft_builders import parse_image_annotation
from ..types import ImageItem

logger = logging.getLogger(__name__)


class ImageAnnotationAgent:
    """Annotates images via the unified LLM client with batching + retries."""

    def __init__(
        self,
        client: LLMClient,
        target_language: str = "English",
        batch_size: int = 5,
        batch_delay: float = 1.0,
    ) -> None:
        self.client = client
        self.target_language = target_language
        self.batch_size = batch_size
        self.batch_delay = batch_delay

    def annotate(
        self, items: List[ImageItem],
    ) -> Tuple[List[Dict[str, Any]], List[ImageItem]]:
        """
        Annotate a list of images.

        Returns:
            (annotations, failures) where annotations is a list of
            ``{"id": ..., "success": bool, "data"|"error": ...}`` dicts.
        """
        logger.info("Annotating %d image items.", len(items))

        # build one LLMRequest per image
        prompt_cache: Dict[str, str] = {}
        requests: List[LLMRequest] = []
        for item in items:
            hint = item.topic_hint or ""
            if hint not in prompt_cache:
                prompt_cache[hint] = build_image_prompt(
                    target_language=self.target_language,
                    topic_hint=item.topic_hint,
                )
            requests.append(
                LLMRequest(
                    request_id=item.item_id,
                    user_message=prompt_cache[hint],
                    images=[item.image_path],
                )
            )

        responses = self.client.generate_json_batch_sync(
            requests,
            batch_size=self.batch_size,
            batch_delay_seconds=self.batch_delay,
        )

        annotations: List[Dict[str, Any]] = []
        failures: List[ImageItem] = []
        item_map = {item.item_id: item for item in items}

        for resp in responses:
            if not resp.success:
                annotations.append({"id": resp.request_id, "success": False, "error": resp.error})
                if resp.request_id in item_map:
                    failures.append(item_map[resp.request_id])
                continue
            try:
                parsed = parse_image_annotation(resp.data)
                annotations.append({"id": resp.request_id, "success": True, "data": parsed.model_dump()})
            except Exception as exc:
                annotations.append({"id": resp.request_id, "success": False, "error": f"schema: {exc}"})
                if resp.request_id in item_map:
                    failures.append(item_map[resp.request_id])

        ok = sum(1 for a in annotations if a.get("success"))
        logger.info("Image annotation done: %d/%d succeeded, %d failures.", ok, len(items), len(failures))
        return annotations, failures
