"""HuggingFace model loader utilities.

Copied from `agentic_train_pipeline/models/hf_loader.py` and adjusted for new package layout.
"""

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_hf_causal_lm(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    torch_dtype = _select_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        use_safetensors=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_hf_vision_placeholder(model_id: str):
    raise NotImplementedError(
        "Image model loading is not implemented yet. "
        "Provide a text dataset and use a causal LM."
    )


def load_hf_multimodal_placeholder(model_id: str):
    raise NotImplementedError(
        "Multimodal model loading is not implemented yet. "
        "Provide a text dataset and use a causal LM."
    )

