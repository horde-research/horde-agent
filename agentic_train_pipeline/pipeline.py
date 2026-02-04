"""Pipeline orchestration for agentic LoRA SFT."""

from __future__ import annotations

from typing import Optional

from agentic_train_pipeline.langgraph_pipeline import run_langgraph_pipeline


def run_pipeline(
    data_path: str,
    task: str,
    out_dir: str,
    max_iters: int,
    hf_model_id_override: Optional[str] = None,
    search_trials: int = 0,
) -> str:
    return run_langgraph_pipeline(
        data_path=data_path,
        task=task,
        out_dir=out_dir,
        max_iters=max_iters,
        hf_model_id_override=hf_model_id_override,
        search_trials=search_trials,
    )
