"""Single source of truth for all pipeline configuration.

Every default lives here. No hidden .get() defaults elsewhere.
Load from .env via ``PipelineConfig.from_env()``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, model_validator


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ── Run ───────────────────────────────────────────────────────────────────
    mode: str = "full"
    run_dir: str = "/Users/sasha/Desktop/hugging_face/horde-agent/run_full_res_2"
    country: str = ""
    resume_confirm_completed: bool = False
    restart_from_stage: Optional[str] = None
    fresh_run: bool = False

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
    max_queries_per_category: Optional[int] = None

    # ── Image collection (optional) ─────────────────────────────────────────
    collect_images: bool = False
    image_min_width: int = 300
    image_min_height: int = 300
    image_context_size: int = 500
    image_collection_mode: str = "serper"
    image_search_results_per_query: int = 10
    enable_image_taxonomy: bool = True
    image_taxonomy_queries_per_slot: int = 4
    image_taxonomy_max_slots: Optional[int] = None
    image_dedup_enable: bool = False
    image_dedup_threshold: float = 0.90
    image_dedup_model_path: str = "models/sscd_disc_mixup.torchscript.pt"
    image_dedup_model_url: str = (
        "https://dl.fbaipublicfiles.com/sscd-copy-detection/"
        "sscd_disc_mixup.torchscript.pt"
    )
    image_dedup_batch_size: int = 32
    image_dedup_max_reported_pairs: int = 100
    image_dedup_device: Optional[str] = None

    # ── Agentic coverage assessment ─────────────────────────────────────────
    coverage_min_text_samples: int = 3
    coverage_min_samples_per_query: float = 0.0
    coverage_max_added_queries: int = 12
    coverage_min_image_slot_ratio: float = 0.2
    coverage_min_images_per_slot: int = 1
    coverage_max_image_query_specs: int = 12

    # ── Text quality diagnostics ─────────────────────────────────────────────
    text_quality_enable_embeddings: bool = False
    text_quality_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    text_quality_embedding_threshold: float = 0.93
    text_quality_max_embedding_items: int = 256
    text_quality_shingle_threshold: float = 0.85
    text_quality_max_shingle_items: int = 1000
    text_quality_max_reported_pairs: int = 50
    text_filter_enable: bool = True
    text_filter_min_chars: int = 300
    text_filter_min_words: int = 40
    text_filter_min_unique_word_ratio: float = 0.15
    text_filter_shingle_threshold: float = 0.90
    text_filter_max_near_duplicate_items: int = 1000
    text_filter_max_reported_rows: int = 50

    # ── SFT annotation ───────────────────────────────────────────────────────
    # Primary modality switch for examples that flow into dataset/train.
    # `sft_mode` is kept as a backward-compatible alias.
    training_modality: str = "text"
    sft_mode: str = "text"
    sft_target_language: str
    sft_prompt_preset: str = "default"
    sft_reuse_annotations: bool = True
    source_eval_enable: bool = True
    source_eval_ratio: float = 0.1
    source_eval_max_items: int = 8

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
    dataset_val_ratio: float = 0.1

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_split: str = "validation"
    eval_max_samples: int = 64
    eval_max_new_tokens: int = 128
    eval_compare_base_model: bool = True
    eval_enable_llm_judge: bool = False
    eval_judge_max_samples: int = 32
    eval_judge_batch_size: int = 3
    eval_judge_batch_delay: float = 1.0

    # ── CPU-safe debug flow ─────────────────────────────────────────────────
    # Keep taxonomy/collection/SFT/dataset real, but replace GPU-heavy stages.
    debug_stub_train: bool = False
    debug_stub_eval: bool = False
    debug_eval_failure_rate: float = 0.0

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
            "mode": os.getenv("MODE"),
            "country": os.getenv("COUNTRY"),
            "run_dir": os.getenv("RUN_DIR"),
            "resume_confirm_completed": os.getenv("RESUME_CONFIRM_COMPLETED"),
            "restart_from_stage": os.getenv("RESTART_FROM_STAGE"),
            "fresh_run": os.getenv("FRESH_RUN"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_model": os.getenv("LLM_MODEL"),
            "llm_api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            "llm_temperature": os.getenv("LLM_TEMPERATURE"),
            "serper_api_key": os.getenv("SERPER_API_KEY") or "",
            "serper_results_per_query": os.getenv("SERPER_RESULTS_PER_QUERY"),
            "serper_top_results": os.getenv("SERPER_TOP_RESULTS"),
            "max_queries": os.getenv("MAX_QUERIES"),
            "max_queries_per_category": os.getenv("MAX_QUERIES_PER_CATEGORY"),
            "collect_images": os.getenv("COLLECT_IMAGES"),
            "image_min_width": os.getenv("IMAGE_MIN_WIDTH"),
            "image_min_height": os.getenv("IMAGE_MIN_HEIGHT"),
            "image_context_size": os.getenv("IMAGE_CONTEXT_SIZE"),
            "image_collection_mode": os.getenv("IMAGE_COLLECTION_MODE"),
            "image_search_results_per_query": os.getenv("IMAGE_SEARCH_RESULTS_PER_QUERY"),
            "enable_image_taxonomy": os.getenv("ENABLE_IMAGE_TAXONOMY"),
            "image_taxonomy_queries_per_slot": os.getenv("IMAGE_TAXONOMY_QUERIES_PER_SLOT"),
            "image_taxonomy_max_slots": os.getenv("IMAGE_TAXONOMY_MAX_SLOTS"),
            "image_dedup_enable": os.getenv("IMAGE_DEDUP_ENABLE"),
            "image_dedup_threshold": os.getenv("IMAGE_DEDUP_THRESHOLD"),
            "image_dedup_model_path": os.getenv("IMAGE_DEDUP_MODEL_PATH"),
            "image_dedup_model_url": os.getenv("IMAGE_DEDUP_MODEL_URL"),
            "image_dedup_batch_size": os.getenv("IMAGE_DEDUP_BATCH_SIZE"),
            "image_dedup_max_reported_pairs": os.getenv("IMAGE_DEDUP_MAX_REPORTED_PAIRS"),
            "image_dedup_device": os.getenv("IMAGE_DEDUP_DEVICE"),
            "coverage_min_text_samples": os.getenv("COVERAGE_MIN_TEXT_SAMPLES"),
            "coverage_min_samples_per_query": os.getenv("COVERAGE_MIN_SAMPLES_PER_QUERY"),
            "coverage_max_added_queries": os.getenv("COVERAGE_MAX_ADDED_QUERIES"),
            "coverage_min_image_slot_ratio": os.getenv("COVERAGE_MIN_IMAGE_SLOT_RATIO"),
            "coverage_min_images_per_slot": os.getenv("COVERAGE_MIN_IMAGES_PER_SLOT"),
            "coverage_max_image_query_specs": os.getenv("COVERAGE_MAX_IMAGE_QUERY_SPECS"),
            "text_quality_enable_embeddings": os.getenv("TEXT_QUALITY_ENABLE_EMBEDDINGS"),
            "text_quality_embedding_model": os.getenv("TEXT_QUALITY_EMBEDDING_MODEL"),
            "text_quality_embedding_threshold": os.getenv("TEXT_QUALITY_EMBEDDING_THRESHOLD"),
            "text_quality_max_embedding_items": os.getenv("TEXT_QUALITY_MAX_EMBEDDING_ITEMS"),
            "text_quality_shingle_threshold": os.getenv("TEXT_QUALITY_SHINGLE_THRESHOLD"),
            "text_quality_max_shingle_items": os.getenv("TEXT_QUALITY_MAX_SHINGLE_ITEMS"),
            "text_quality_max_reported_pairs": os.getenv("TEXT_QUALITY_MAX_REPORTED_PAIRS"),
            "text_filter_enable": os.getenv("TEXT_FILTER_ENABLE"),
            "text_filter_min_chars": os.getenv("TEXT_FILTER_MIN_CHARS"),
            "text_filter_min_words": os.getenv("TEXT_FILTER_MIN_WORDS"),
            "text_filter_min_unique_word_ratio": os.getenv("TEXT_FILTER_MIN_UNIQUE_WORD_RATIO"),
            "text_filter_shingle_threshold": os.getenv("TEXT_FILTER_SHINGLE_THRESHOLD"),
            "text_filter_max_near_duplicate_items": os.getenv("TEXT_FILTER_MAX_NEAR_DUPLICATE_ITEMS"),
            "text_filter_max_reported_rows": os.getenv("TEXT_FILTER_MAX_REPORTED_ROWS"),
            "training_modality": os.getenv("TRAINING_MODALITY"),
            "sft_mode": os.getenv("SFT_MODE"),
            "sft_target_language": os.getenv("SFT_TARGET_LANGUAGE"),
            "sft_prompt_preset": os.getenv("SFT_PROMPT_PRESET"),
            "sft_reuse_annotations": os.getenv("SFT_REUSE_ANNOTATIONS"),
            "source_eval_enable": os.getenv("SOURCE_EVAL_ENABLE"),
            "source_eval_ratio": os.getenv("SOURCE_EVAL_RATIO"),
            "source_eval_max_items": os.getenv("SOURCE_EVAL_MAX_ITEMS"),
            "hf_model_id": os.getenv("HF_MODEL_ID"),
            "max_iters": os.getenv("MAX_ITERS"),
            "max_steps": os.getenv("MAX_STEPS"),
            "max_samples": os.getenv("MAX_SAMPLES"),
            "dataset_val_ratio": os.getenv("DATASET_VAL_RATIO"),
            "train_batch_size": os.getenv("TRAIN_BATCH_SIZE"),
            "train_max_seq_len": os.getenv("TRAIN_MAX_SEQ_LEN"),
            "eval_split": os.getenv("EVAL_SPLIT"),
            "eval_max_samples": os.getenv("EVAL_MAX_SAMPLES"),
            "eval_compare_base_model": os.getenv("EVAL_COMPARE_BASE_MODEL"),
            "eval_enable_llm_judge": os.getenv("EVAL_ENABLE_LLM_JUDGE"),
            "eval_judge_max_samples": os.getenv("EVAL_JUDGE_MAX_SAMPLES"),
            "eval_judge_batch_size": os.getenv("EVAL_JUDGE_BATCH_SIZE"),
            "eval_judge_batch_delay": os.getenv("EVAL_JUDGE_BATCH_DELAY"),
            "debug_stub_train": os.getenv("DEBUG_STUB_TRAIN"),
            "debug_stub_eval": os.getenv("DEBUG_STUB_EVAL"),
            "debug_eval_failure_rate": os.getenv("DEBUG_EVAL_FAILURE_RATE"),
            "hf_username": os.getenv("HF_USERNAME") or "",
            "hf_token": os.getenv("HF_TOKEN") or "",
            "hf_dataset_repo": os.getenv("HF_DATASET_REPO") or "",
            "hf_adapter_repo": os.getenv("HF_ADAPTER_REPO") or "",
        }
        # Only set values that are actually present in env
        env_values = {k: v for k, v in env_values.items() if v}

        return cls(**(env_values | overrides))

    @model_validator(mode="before")
    @classmethod
    def _sync_training_modality_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        training_modality = values.get("training_modality")
        sft_mode = values.get("sft_mode")
        if training_modality is None and sft_mode is not None:
            values["training_modality"] = sft_mode
        elif sft_mode is None and training_modality is not None:
            values["sft_mode"] = training_modality
        elif training_modality is not None and sft_mode is not None and str(training_modality) != str(sft_mode):
            raise ValueError("training_modality and sft_mode must match when both are set.")
        return values

    @model_validator(mode="after")
    def _validate_training_modality(self) -> "PipelineConfig":
        normalized = str(self.training_modality).strip().lower()
        if normalized not in {"text", "image"}:
            raise ValueError("training_modality must be one of: text, image.")
        self.training_modality = normalized
        self.sft_mode = normalized
        if not 0.0 <= float(self.dataset_val_ratio) <= 0.5:
            raise ValueError("dataset_val_ratio must be between 0.0 and 0.5.")
        if int(self.coverage_min_text_samples) < 1:
            raise ValueError("coverage_min_text_samples must be >= 1.")
        if float(self.coverage_min_samples_per_query) < 0.0:
            raise ValueError("coverage_min_samples_per_query must be >= 0.")
        if not 0.0 <= float(self.coverage_min_image_slot_ratio) <= 1.0:
            raise ValueError("coverage_min_image_slot_ratio must be between 0.0 and 1.0.")
        if not 0.0 <= float(self.image_dedup_threshold) <= 1.0:
            raise ValueError("image_dedup_threshold must be between 0.0 and 1.0.")
        if int(self.image_dedup_batch_size) < 1:
            raise ValueError("image_dedup_batch_size must be >= 1.")
        if int(self.image_dedup_max_reported_pairs) < 0:
            raise ValueError("image_dedup_max_reported_pairs must be >= 0.")
        if not 0.0 <= float(self.text_quality_embedding_threshold) <= 1.0:
            raise ValueError("text_quality_embedding_threshold must be between 0.0 and 1.0.")
        if not 0.0 <= float(self.text_quality_shingle_threshold) <= 1.0:
            raise ValueError("text_quality_shingle_threshold must be between 0.0 and 1.0.")
        if int(self.text_quality_max_embedding_items) < 0:
            raise ValueError("text_quality_max_embedding_items must be >= 0.")
        if int(self.text_quality_max_shingle_items) < 0:
            raise ValueError("text_quality_max_shingle_items must be >= 0.")
        if int(self.text_quality_max_reported_pairs) < 0:
            raise ValueError("text_quality_max_reported_pairs must be >= 0.")
        if int(self.text_filter_min_chars) < 0:
            raise ValueError("text_filter_min_chars must be >= 0.")
        if int(self.text_filter_min_words) < 0:
            raise ValueError("text_filter_min_words must be >= 0.")
        if not 0.0 <= float(self.text_filter_min_unique_word_ratio) <= 1.0:
            raise ValueError("text_filter_min_unique_word_ratio must be between 0.0 and 1.0.")
        if not 0.0 <= float(self.text_filter_shingle_threshold) <= 1.0:
            raise ValueError("text_filter_shingle_threshold must be between 0.0 and 1.0.")
        if int(self.text_filter_max_near_duplicate_items) < 0:
            raise ValueError("text_filter_max_near_duplicate_items must be >= 0.")
        if int(self.text_filter_max_reported_rows) < 0:
            raise ValueError("text_filter_max_reported_rows must be >= 0.")
        if not 0.0 <= float(self.source_eval_ratio) <= 0.5:
            raise ValueError("source_eval_ratio must be between 0.0 and 0.5.")
        if int(self.source_eval_max_items) < 0:
            raise ValueError("source_eval_max_items must be >= 0.")
        return self

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
