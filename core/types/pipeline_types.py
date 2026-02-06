"""Shared types for the pipeline.

Copied from `agentic_train_pipeline/types.py` and adjusted for new package layout.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class DatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_path: str
    resolved_data_id: str
    columns: List[str]
    sample_count: int
    example: Dict[str, Any]
    modality_candidates: List[str]
    validation_warnings: List[str]


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_steps: int = 200
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_seq_len: int = 512
    eval_steps: int = 50
    seed: int = 42


class MetricsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int
    best_eval_loss: Optional[float]
    last_train_loss: Optional[float]
    last_eval_loss: Optional[float]


class IterationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iter_idx: int
    config: TrainConfig
    metrics: MetricsSummary
    adapter_path: str
    log_paths: Dict[str, str]


class FailureClusterPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clusters: List[Dict[str, Any]]

