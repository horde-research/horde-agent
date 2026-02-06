"""Registry for dataset loaders, model loaders, LoRA presets, and trainers.

Copied from `agentic_train_pipeline/registry/registry.py` and adjusted for new package layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, ConfigDict


class RegistrySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_loader_keys: List[str]
    model_loader_keys: List[str]
    lora_preset_keys: List[str]
    trainer_keys: List[str]
    default_hf_model_id: str


@dataclass
class Registry:
    dataset_loaders: Dict[str, Callable[..., Any]]
    model_loaders: Dict[str, Callable[..., Any]]
    lora_presets: Dict[str, Dict[str, Any]]
    trainers: Dict[str, Callable[..., Any]]
    default_hf_model_id: str

    def snapshot(self) -> RegistrySnapshot:
        return RegistrySnapshot(
            dataset_loader_keys=sorted(self.dataset_loaders.keys()),
            model_loader_keys=sorted(self.model_loaders.keys()),
            lora_preset_keys=sorted(self.lora_presets.keys()),
            trainer_keys=sorted(self.trainers.keys()),
            default_hf_model_id=self.default_hf_model_id,
        )

    def get_dataset_loader(self, key: str) -> Callable[..., Any]:
        return self.dataset_loaders[key]

    def get_model_loader(self, key: str) -> Callable[..., Any]:
        return self.model_loaders[key]

    def get_lora_preset(self, key: str) -> Dict[str, Any]:
        return self.lora_presets[key]

    def get_trainer(self, key: str) -> Callable[..., Any]:
        return self.trainers[key]

