"""OpenAI Responses API client with JSON parsing."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

from agentic_train_pipeline.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_S,
)


def _extract_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    if hasattr(response, "output"):
        chunks: List[str] = []
        for item in response.output:
            if hasattr(item, "content"):
                for part in item.content:
                    if getattr(part, "type", None) == "output_text":
                        chunks.append(part.text)
        if chunks:
            return "\n".join(chunks)
    return ""


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class OpenAIJsonClient:
    def __init__(self) -> None:
        base_url = OPENAI_BASE_URL if OPENAI_BASE_URL else None
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url, timeout=OPENAI_TIMEOUT_S)

    def request_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        for attempt in range(OPENAI_MAX_RETRIES + 1):
            response = self.client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = _extract_text(response)
            try:
                return _parse_json(text)
            except Exception:
                if attempt >= OPENAI_MAX_RETRIES:
                    raise
                user_prompt = user_prompt + "\n\nReturn strict JSON only."
        raise RuntimeError("Failed to get JSON response")
