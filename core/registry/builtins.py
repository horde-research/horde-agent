"""Built-in registry entries.

Copied from `agentic_train_pipeline/registry/builtins.py` and adjusted for new package layout.
"""

from typing import Dict

from core.data.hf_dataset import (
    load_hf_image_dataset,
    load_hf_multimodal_dataset,
    load_hf_text_dataset,
)
from core.ml.hf_loader import (
    load_hf_causal_lm,
    load_hf_multimodal_placeholder,
    load_hf_vision_placeholder,
)
from core.registry.registry import Registry

# NOTE: This trainer class will be defined under tools/train and referenced here during cutover.
# During the migration, we keep the string key stable: "static_sft_default".
from tools.train.trainers.static_sft_trainer import StaticSFTTrainer

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

# Kept identical to current behavior (may be overridden by CLI/config).
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

