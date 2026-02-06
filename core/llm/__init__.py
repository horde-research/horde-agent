"""Unified LLM client for the entire project.

Supports OpenAI, Google Gemini, and xAI (Grok) with a single interface.
Configure via environment variables or constructor args.
"""

from core.llm.client import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    format_instructions,
    parse_response,
)

__all__ = [
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "format_instructions",
    "parse_response",
]
