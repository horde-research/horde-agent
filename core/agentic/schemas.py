"""Pydantic schemas for decision outputs.

Copied from `agentic_train_pipeline/agent/schemas.py`.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class ModalityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    modality: Literal["text", "image", "multimodal"]
    confidence: float
    rationale: str


class ComponentSelectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_loader_key: str
    model_loader_key: str
    lora_preset_key: str
    trainer_key: str
    hf_model_id: str
    primary_metric: str
    rationale: str


class TrainingAdjustmentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    should_retry: bool
    lr_multiplier: float
    batch_size_delta: int
    grad_accum_delta: int
    max_steps_delta: int
    switch_lora_preset_key: Optional[str]
    stop_reason: Optional[str]
    rationale: str


class ErrorAnalysisDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cluster_labels: List[str]
    root_causes: List[str]
    data_fixes: List[str]
    next_training_actions: List[str]

