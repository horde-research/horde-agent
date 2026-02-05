"""Dataset modality inference and text formatting helpers."""

from typing import Any, Dict, List, Optional, Tuple

from datasets import Image as HfImage
from datasets import Value


def infer_modality(columns: List[str], features: Dict[str, Any]) -> List[str]:
    has_image = False
    has_text = False

    for name, feature in features.items():
        if isinstance(feature, HfImage):
            has_image = True
        if isinstance(feature, Value) and feature.dtype in {"string", "large_string"}:
            has_text = True
        if name in {"text", "prompt", "response", "instruction", "output"}:
            has_text = True

    if has_image and has_text:
        return ["multimodal"]
    if has_image:
        return ["image"]
    if has_text:
        return ["text"]
    return ["text"]


def _truncate(value: Any, max_len: int = 200) -> Any:
    if isinstance(value, str):
        if len(value) > max_len:
            return value[:max_len] + "..."
        return value
    return value


def extract_text_input_output(example: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    if "text" in example:
        return str(example.get("text", "")), None
    if "prompt" in example and "response" in example:
        return str(example.get("prompt", "")), str(example.get("response", ""))
    if "instruction" in example and "output" in example:
        return str(example.get("instruction", "")), str(example.get("output", ""))
    for key, value in example.items():
        if isinstance(value, str):
            return value, None
    return "", None


def format_text_for_sft(example: Dict[str, Any]) -> str:
    prompt, response = extract_text_input_output(example)
    if response is None:
        return prompt
    return f"""### Prompt:
{prompt}

### Response:
{response}"""


def build_example_preview(example: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _truncate(v) for k, v in example.items()}