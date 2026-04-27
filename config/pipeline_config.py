"""Single source of truth for all pipeline configuration.

Every default lives here. No hidden .get() defaults elsewhere.
Load from .env via ``PipelineConfig.from_env()``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ── Run ───────────────────────────────────────────────────────────────────
    mode: str = "full"
    run_dir: str = "/Users/sasha/Desktop/hugging_face/horde-agent/run_full_res_2"
    country: str

    # ── LLM (from .env) ──────────────────────────────────────────────────────
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_batch_size: int = 5
    llm_batch_delay: float = 1.5

    # ── Serper (data collection) ─────────────────────────────────────────────
    serper_api_key: str = ""
    serper_results_per_query: int = 10
    serper_top_results: int = 5
    serper_concurrency: int = 50
    max_queries: Optional[int] = None

    # ── Image collection (optional) ─────────────────────────────────────────
    collect_images: bool = False
    image_min_width: int = 300
    image_min_height: int = 300
    image_context_size: int = 500

    # ── SFT annotation ───────────────────────────────────────────────────────
    sft_mode: str = "text"
    sft_target_language: str

    # ── Training ─────────────────────────────────────────────────────────────
    hf_model_id: str
    trainer_key: str = "static_sft_default"
    lora_preset_key: str = "lora_attn_small"
    model_loader_key: str = "hf_causal_lm_default"
    max_iters: int = 1
    max_steps: int = 200
    seed: int = 42
    train_lr: float = 2e-4
    train_batch_size: int = 4
    train_grad_accum: int = 4
    train_warmup_ratio: float = 0.03
    train_weight_decay: float = 0.0
    train_max_seq_len: int = 512
    train_eval_steps: int = 50
    max_samples: Optional[int] = None
    search_trials: int = 0

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_split: str = "train"
    eval_max_samples: int = 64
    eval_max_new_tokens: int = 128

    # ── Hugging Face Hub ─────────────────────────────────────────────────────
    hf_token: str = ""
    hf_username: str = ""
    hf_dataset_repo: str = ""
    hf_adapter_repo: str = ""

    # ── Workflow mode: start from existing data ──────────────────────────────
    data_path: Optional[str] = None

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None, **overrides) -> "PipelineConfig":
        """Build config from .env, with optional overrides."""
        load_dotenv(dotenv_path=dotenv_path)

        env_values = {
            "country": os.getenv("COUNTRY"),
            "run_dir": os.getenv("RUN_DIR"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_model": os.getenv("LLM_MODEL"),
            "llm_api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            "llm_temperature": os.getenv("LLM_TEMPERATURE"),
            "serper_api_key": os.getenv("SERPER_API_KEY") or "",
            "serper_results_per_query": os.getenv("SERPER_RESULTS_PER_QUERY"),
            "serper_top_results": os.getenv("SERPER_TOP_RESULTS"),
            "max_queries": os.getenv("MAX_QUERIES"),
            "collect_images": os.getenv("COLLECT_IMAGES"),
            "image_min_width": os.getenv("IMAGE_MIN_WIDTH"),
            "image_min_height": os.getenv("IMAGE_MIN_HEIGHT"),
            "image_context_size": os.getenv("IMAGE_CONTEXT_SIZE"),
            "sft_target_language": os.getenv("SFT_TARGET_LANGUAGE"),
            "hf_model_id": os.getenv("HF_MODEL_ID"),
            "max_iters": os.getenv("MAX_ITERS"),
            "max_steps": os.getenv("MAX_STEPS"),
            "max_samples": os.getenv("MAX_SAMPLES"),
            "train_batch_size": os.getenv("TRAIN_BATCH_SIZE"),
            "train_max_seq_len": os.getenv("TRAIN_MAX_SEQ_LEN"),
            "eval_max_samples": os.getenv("EVAL_MAX_SAMPLES"),
            "hf_username": os.getenv("HF_USERNAME") or "",
            "hf_token": os.getenv("HF_TOKEN") or "",
            "hf_dataset_repo": os.getenv("HF_DATASET_REPO") or "",
            "hf_adapter_repo": os.getenv("HF_ADAPTER_REPO") or "",
        }
        # Only set values that are actually present in env
        env_values = {k: v for k, v in env_values.items() if v}

        return cls(**(env_values | overrides))

    def train_config_dict(self) -> dict:
        """Return the TrainConfig-compatible dict."""
        return {
            "lr": self.train_lr,
            "batch_size": self.train_batch_size,
            "grad_accum": self.train_grad_accum,
            "max_steps": self.max_steps,
            "warmup_ratio": self.train_warmup_ratio,
            "weight_decay": self.train_weight_decay,
            "max_seq_len": self.train_max_seq_len,
            "eval_steps": self.train_eval_steps,
            "seed": self.seed,
        }
