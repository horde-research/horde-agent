"""Built-in registry entries."""

from typing import Dict

from agentic_train_pipeline.parser.hf_dataset import (
    load_hf_multimodal_dataset,
    load_hf_image_dataset,
    load_hf_text_dataset,
)
from agentic_train_pipeline.models.hf_loader import (
    load_hf_causal_lm,
    load_hf_multimodal_placeholder,
    load_hf_vision_placeholder,
)
from agentic_train_pipeline.training.static_sft_trainer import StaticSFTTrainer
from agentic_train_pipeline.registry.registry import Registry

ATTN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
MLP_TARGET_MODULES = ["up_proj", "down_proj", "gate_proj", "c_fc"]

LORA_PRESETS: Dict[str, Dict[str, object]] = {
    "lora_attn_small": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ATTN_TARGET_MODULES,
    },
    "lora_attn_medium": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ATTN_TARGET_MODULES,
    },
    "lora_attn_mlp_small": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ATTN_TARGET_MODULES + MLP_TARGET_MODULES,
    },
}

DEFAULT_HF_MODEL_ID = "gpt4o-mini"


def build_registry() -> Registry:
    return Registry(
        dataset_loaders={
            "hf_text_default": load_hf_text_dataset,
            "hf_image_default": load_hf_image_dataset,
            "hf_multimodal_default": load_hf_multimodal_dataset,
        },
        model_loaders={
            "hf_causal_lm_default": load_hf_causal_lm,
            "hf_vision_default": load_hf_vision_placeholder,
            "hf_multimodal_default": load_hf_multimodal_placeholder,
        },
        lora_presets=LORA_PRESETS,
        trainers={"static_sft_default": StaticSFTTrainer},
        default_hf_model_id=DEFAULT_HF_MODEL_ID,
    )
