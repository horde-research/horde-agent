"""Global configuration for OpenAI and pipeline defaults."""

from typing import Optional

OPENAI_API_KEY: str = ""
OPENAI_MODEL: str = "gpt-4o"
OPENAI_BASE_URL: Optional[str] = None
OPENAI_TIMEOUT_S: int = 60
OPENAI_MAX_RETRIES: int = 2
DEFAULT_SEED: int = 42
