"""
Unified async LLM client with batch processing, retries, and multi-provider support.

Supported providers:
  - openai  → https://api.openai.com/v1/chat/completions
  - gemini  → https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  - xai     → https://api.x.ai/v1/chat/completions   (OpenAI-compatible)

Usage:
    from core.llm import LLMClient, LLMRequest

    client = LLMClient.from_env()                       # reads .env
    client = LLMClient("gemini", "gemini-2.5-flash", "key...")

    # single (sync)
    resp = client.generate_json_sync(LLMRequest(request_id="1", user_message="hi"))

    # batch  (sync wrapper over async)
    resps = client.generate_json_batch_sync(requests, batch_size=5, batch_delay=1.5)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# ─── provider registry ────────────────────────────────────────────────────────

PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "format": "openai",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "format": "openai",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "format": "gemini",
    },
}


# ─── data classes ──────────────────────────────────────────────────────────────

@dataclass
class LLMRequest:
    """A single request to any LLM provider."""
    request_id: str
    user_message: str = ""
    system_prompt: Optional[str] = None
    images: Optional[List[str]] = None           # file paths
    generation_config: Optional[Dict[str, Any]] = None  # provider-specific overrides


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_text: Optional[str] = None


# ─── helpers ───────────────────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    """Base-64-encode a local image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from *text*.

    Handles raw JSON, markdown fenced blocks, and text wrapped around JSON.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences (```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        inner_lines = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner_lines.append(line)
        try:
            return json.loads("\n".join(inner_lines))
        except json.JSONDecodeError:
            pass

    # Find first { … last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON object found in LLM response (len={len(text)}).")


# ─── Pydantic schema → format instructions ────────────────────────────────────

def format_instructions(response_model: Type[T]) -> str:
    """Generate prompt instructions from a Pydantic model's JSON Schema.

    Append the returned string to the end of your user_message so the LLM
    knows exactly what JSON structure to produce.  The schema is derived
    automatically from the Pydantic model — no manual sync needed.

    Usage::

        from core.llm import format_instructions
        from my_schemas import MyOutputModel

        prompt = my_task_prompt + format_instructions(MyOutputModel)
    """
    schema = response_model.model_json_schema()
    # Remove noisy $defs key for readability — inline definitions are
    # already resolved by Pydantic's schema generator in most cases,
    # but we keep them if present since models with nested types need them.
    schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
    return (
        "\n\n━━━ OUTPUT FORMAT ━━━\n"
        "You MUST respond with a SINGLE valid JSON object that conforms to "
        "the following JSON Schema. Return ONLY the JSON — no markdown "
        "fences, no commentary, no extra text.\n\n"
        f"{schema_str}"
    )


def parse_response(data: Dict[str, Any], response_model: Type[T]) -> T:
    """Validate an LLM JSON response against a Pydantic model.

    Raises ``pydantic.ValidationError`` on schema mismatch.
    """
    return response_model.model_validate(data)


# ─── client ────────────────────────────────────────────────────────────────────

class LLMClient:
    """Unified async/sync LLM client with batch processing and retries."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.3,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> None:
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported: {', '.join(PROVIDERS)}."
            )
        if not api_key:
            raise ValueError("api_key must not be empty.")

        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds

        cfg = PROVIDERS[provider]
        self._base_url: str = cfg["base_url"]
        self._format: str = cfg["format"]          # "openai" or "gemini"

    # ── factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, **overrides) -> "LLMClient":
        """Create a client from environment variables.

        Env vars:
            LLM_PROVIDER  – openai | gemini | xai  (default: gemini)
            LLM_MODEL     – model name              (default: gemini-2.5-flash)
            LLM_API_KEY   – API key                 (required)
        """
        provider = overrides.get("provider") or os.getenv("LLM_PROVIDER", "gemini")
        model = overrides.get("model") or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        api_key = overrides.get("api_key") or os.getenv("LLM_API_KEY", "")
        if not api_key:
            raise ValueError(
                "LLM_API_KEY environment variable is not set. "
                "Set it in .env or pass api_key= explicitly."
            )
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=overrides.get("temperature", 0.3),
            timeout_seconds=overrides.get("timeout_seconds", 120.0),
            max_retries=overrides.get("max_retries", 3),
            retry_delay_seconds=overrides.get("retry_delay_seconds", 2.0),
        )

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"

    # ── payload builders ───────────────────────────────────────────────────

    def _build_openai_payload(self, req: LLMRequest) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})

        if req.images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": req.user_message}]
            for img_path in req.images:
                b64 = _encode_image(img_path)
                mime = _guess_mime(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": req.user_message})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if req.generation_config:
            payload.update(req.generation_config)
        return payload

    def _build_gemini_payload(self, req: LLMRequest) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": req.user_message}]
        if req.images:
            for img_path in req.images:
                b64 = _encode_image(img_path)
                mime = _guess_mime(img_path)
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.temperature,
            },
        }
        if req.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": req.system_prompt}]}
        if req.generation_config:
            payload["generationConfig"].update(req.generation_config)
        return payload

    # ── response extractors ────────────────────────────────────────────────

    @staticmethod
    def _extract_openai_text(body: Dict[str, Any]) -> str:
        choices = body.get("choices", [])
        if not choices:
            raise ValueError("No choices in OpenAI response.")
        return choices[0].get("message", {}).get("content", "").strip()

    @staticmethod
    def _extract_gemini_text(body: Dict[str, Any]) -> str:
        candidates = body.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response.")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if "text" in p).strip()

    # ── low-level send with retries ────────────────────────────────────────

    async def _send_with_retries(
        self,
        http: httpx.AsyncClient,
        req: LLMRequest,
    ) -> Dict[str, Any]:
        """Send one request with exponential-backoff retries."""
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self._format == "openai":
                    url = f"{self._base_url}/chat/completions"
                    payload = self._build_openai_payload(req)
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                    resp = await http.post(url, json=payload, headers=headers)
                else:
                    url = f"{self._base_url}/models/{self.model}:generateContent"
                    payload = self._build_gemini_payload(req)
                    resp = await http.post(
                        url, json=payload, params={"key": self.api_key}
                    )

                resp.raise_for_status()
                body = resp.json()

                text = (
                    self._extract_openai_text(body)
                    if self._format == "openai"
                    else self._extract_gemini_text(body)
                )
                return _extract_json(text)

            except Exception as exc:
                last_exc = exc
                delay = self.retry_delay * attempt

                # respect Retry-After on 429
                if isinstance(exc, httpx.HTTPStatusError):
                    sc = exc.response.status_code
                    if sc in (400, 401, 403, 404):
                        # non-retryable client errors
                        raise
                    if sc == 429:
                        ra = exc.response.headers.get("Retry-After")
                        if ra:
                            try:
                                delay = max(delay, float(ra))
                            except ValueError:
                                pass
                        delay = max(delay, 5.0)

                logger.warning(
                    "[%s] Request '%s' attempt %d/%d failed: %s",
                    self.provider,
                    req.request_id,
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Request '{req.request_id}' failed after {self.max_retries} retries: {last_exc}"
        ) from last_exc

    # ── public async API ───────────────────────────────────────────────────

    async def generate_json(self, request: LLMRequest) -> LLMResponse:
        """Send a single request and return parsed JSON."""
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            try:
                data = await self._send_with_retries(http, request)
                return LLMResponse(
                    request_id=request.request_id,
                    success=True,
                    data=data,
                )
            except Exception as exc:
                return LLMResponse(
                    request_id=request.request_id,
                    success=False,
                    error=str(exc),
                )

    async def generate_json_batch(
        self,
        requests: Iterable[LLMRequest],
        *,
        batch_size: int = 5,
        batch_delay_seconds: float = 1.5,
    ) -> List[LLMResponse]:
        """Send many requests concurrently in batches, with delays between batches."""
        requests_list = list(requests)
        all_responses: List[LLMResponse] = []

        logger.info(
            "[%s] Batch: %d requests, batch_size=%d, delay=%.1fs",
            self.provider, len(requests_list), batch_size, batch_delay_seconds,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            for batch_start in range(0, len(requests_list), batch_size):
                batch = requests_list[batch_start : batch_start + batch_size]
                logger.info(
                    "[%s] Batch %d–%d of %d",
                    self.provider,
                    batch_start + 1,
                    batch_start + len(batch),
                    len(requests_list),
                )

                tasks = [self._send_with_retries(http, req) for req in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for req, result in zip(batch, results):
                    if isinstance(result, Exception):
                        all_responses.append(LLMResponse(
                            request_id=req.request_id,
                            success=False,
                            error=str(result),
                        ))
                    else:
                        all_responses.append(LLMResponse(
                            request_id=req.request_id,
                            success=True,
                            data=result,
                        ))

                # delay between batches (skip after last)
                if batch_start + batch_size < len(requests_list):
                    await asyncio.sleep(batch_delay_seconds)

        ok = sum(1 for r in all_responses if r.success)
        logger.info("[%s] Batch done: %d/%d succeeded.", self.provider, ok, len(all_responses))
        return all_responses

    # ── public sync wrappers ───────────────────────────────────────────────

    def generate_json_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper around :meth:`generate_json`."""
        return asyncio.run(self.generate_json(request))

    def generate_json_batch_sync(
        self,
        requests: Iterable[LLMRequest],
        *,
        batch_size: int = 5,
        batch_delay_seconds: float = 1.5,
    ) -> List[LLMResponse]:
        """Synchronous wrapper around :meth:`generate_json_batch`."""
        return asyncio.run(
            self.generate_json_batch(
                requests,
                batch_size=batch_size,
                batch_delay_seconds=batch_delay_seconds,
            )
        )
