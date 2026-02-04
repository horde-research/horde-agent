"""LoRA attachment utilities."""

from dataclasses import dataclass
from typing import List

from peft import LoraConfig, get_peft_model


@dataclass
class LoraPreset:
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]


def attach_lora(model, preset: LoraPreset):
    for param in model.parameters():
        param.requires_grad = False

    config = LoraConfig(
        r=preset.r,
        lora_alpha=preset.alpha,
        lora_dropout=preset.dropout,
        target_modules=preset.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)

    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    _verify_only_lora_trainable(peft_model)
    return peft_model


def _verify_only_lora_trainable(model) -> None:
    non_lora = []
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            non_lora.append(name)
    if non_lora:
        raise RuntimeError(
            "Base model parameters are trainable. LoRA-only training required: "
            + ", ".join(non_lora)
        )


def save_lora_adapters(model, out_dir: str) -> str:
    import os

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir


def preset_from_dict(preset_dict: dict) -> LoraPreset:
    return LoraPreset(
        r=int(preset_dict["r"]),
        alpha=int(preset_dict["alpha"]),
        dropout=float(preset_dict["dropout"]),
        target_modules=list(preset_dict["target_modules"]),
    )
