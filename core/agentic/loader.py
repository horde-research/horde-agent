"""Prompt loader for stage templates.

Copied from `agentic_train_pipeline/prompts/loader.py` and adjusted for new package layout.
"""

from pathlib import Path
from typing import Dict

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str, **kwargs) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    template = path.read_text(encoding="utf-8")
    return template.format(**kwargs)


def load_all_prompts() -> Dict[str, str]:
    prompts = {}
    for path in PROMPT_DIR.glob("*.txt"):
        prompts[path.name] = path.read_text(encoding="utf-8")
    return prompts

