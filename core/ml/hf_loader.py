"""HuggingFace model loader utilities.

Copied from `agentic_train_pipeline/models/hf_loader.py` and adjusted for new package layout.
"""

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _accelerator_kind() -> str:
    """Primary device for training/inference: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_accelerator(kind: str) -> torch.dtype:
    if kind == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # MPS does not support bfloat16; float32 is the reliable choice for weights.
    return torch.float32


def load_hf_causal_lm(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kind = _accelerator_kind()
    torch_dtype = _dtype_for_accelerator(kind)

    if kind == "mps":
        # device_map="auto" + Accelerate can place bfloat16 checkpoints on MPS → TypeError.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
            use_safetensors=True,
        )
        model = model.to("mps")
    elif kind == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            use_safetensors=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=None,
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

