"""Global configuration for OpenAI and pipeline defaults.

Copied from `agentic_train_pipeline/config.py` and adjusted for new package layout.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

OPENAI_API_KEY: str = os.getenv("OPENAI_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", None)
OPENAI_TIMEOUT_S: int = int(os.getenv("OPENAI_TIMEOUT_S", "60"))
OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
DEFAULT_SEED: int = 42

