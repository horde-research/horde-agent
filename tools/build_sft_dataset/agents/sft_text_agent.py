"""Text annotation agent — uses core.llm.LLMClient with Pydantic schema injection."""

import logging
from typing import Any, Dict, List, Tuple

from core.llm import LLMClient, LLMRequest, format_instructions
from ..prompts import build_text_prompt
from ..schemas import TextAnnotation
from ..sft_builders import parse_text_annotation
from ..types import TextItem

logger = logging.getLogger(__name__)


class TextAnnotationAgent:
    """Annotates text items via the unified LLM client.

    Uses format_instructions(TextAnnotation) to auto-inject the Pydantic
    schema into the prompt, so the LLM knows exactly what JSON to produce.
    """

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

        # Build prompt once — task instructions + Pydantic schema
        self._prompt = (
            build_text_prompt(target_language=self.target_language)
            + format_instructions(TextAnnotation)
        )

    def annotate(
        self, items: List[TextItem],
    ) -> Tuple[List[Dict[str, Any]], List[TextItem]]:
        """
        Annotate a list of text items.

        Returns:
            (annotations, failures) where annotations is a list of
            ``{"id": ..., "success": bool, "data"|"error": ...}`` dicts.
        """
        logger.info("Annotating %d text items.", len(items))

        requests: List[LLMRequest] = []
        for item in items:
            requests.append(
                LLMRequest(
                    request_id=item.item_id,
                    user_message=f"{self._prompt}\n\nSOURCE TEXT:\n{item.text}",
                )
            )

        responses = self.client.generate_json_batch_sync(
            requests,
            batch_size=self.batch_size,
            batch_delay_seconds=self.batch_delay,
        )

        annotations: List[Dict[str, Any]] = []
        failures: List[TextItem] = []
        item_map = {item.item_id: item for item in items}

        for resp in responses:
            if not resp.success:
                annotations.append({"id": resp.request_id, "success": False, "error": resp.error})
                if resp.request_id in item_map:
                    failures.append(item_map[resp.request_id])
                continue
            try:
                parsed = parse_text_annotation(resp.data)
                annotations.append({"id": resp.request_id, "success": True, "data": parsed.model_dump()})
            except Exception as exc:
                annotations.append({"id": resp.request_id, "success": False, "error": f"schema: {exc}"})
                if resp.request_id in item_map:
                    failures.append(item_map[resp.request_id])

        ok = sum(1 for a in annotations if a.get("success"))
        logger.info("Text annotation done: %d/%d succeeded, %d failures.", ok, len(items), len(failures))
        return annotations, failures
