from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from tools.generate_taxonomy.tool import GenerateTaxonomyTool


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


country = os.getenv("COUNTRY", "Kazakhstan")
config = {
    "provider": os.getenv("LLM_PROVIDER"),
    "model": os.getenv("LLM_MODEL"),
    "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
    "temperature": _env_float("LLM_TEMPERATURE", 0.7),
    "batch_size": _env_int("TAXONOMY_BATCH_SIZE", _env_int("LLM_BATCH_SIZE", 3)),
    "batch_delay": _env_float("TAXONOMY_BATCH_DELAY", _env_float("LLM_BATCH_DELAY", 0.0)),
    "enable_taxonomy_mini_loop": _env_bool("ENABLE_TAXONOMY_MINI_LOOP", True),
    "taxonomy_min_categories": _env_int("TAXONOMY_MIN_CATEGORIES", 8),
    "taxonomy_max_categories": _env_int("TAXONOMY_MAX_CATEGORIES", 20),
    "taxonomy_min_subcategories": _env_int("TAXONOMY_MIN_SUBCATEGORIES", 4),
    "taxonomy_max_subcategories": _env_int("TAXONOMY_MAX_SUBCATEGORIES", 12),
    "taxonomy_min_queries": _env_int("TAXONOMY_MIN_QUERIES", 8),
    "taxonomy_max_queries": _env_int("TAXONOMY_MAX_QUERIES", 20),
    "taxonomy_repair_attempts": _env_int("TAXONOMY_REPAIR_ATTEMPTS", 2),
}

result = GenerateTaxonomyTool().execute(country, config)

print(json.dumps(result["taxonomy_quality"], indent=2, ensure_ascii=False))
