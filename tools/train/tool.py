"""
Training tool.

Orchestrates model training using specified method (SFT/GRPO/DPO).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.base_tool import BaseTool

from core.data.hf_dataset import load_dataset_from_path
from core.ml.lora import attach_lora, preset_from_dict
from core.registry.builtins import build_registry
from core.types.pipeline_types import IterationRecord, TrainConfig
from tools.train.training.log_parser import parse_metrics
from tools.train.training.sft_runner import run_sft_iteration

class TrainTool(BaseTool):
    """
    Main training orchestrator.
    
    Delegates to specific trainer based on method:
    - SFT: Supervised Fine-Tuning
    - GRPO: Group Relative Policy Optimization
    - DPO: Direct Preference Optimization
    
    Handles:
    - Trainer initialization
    - Training loop execution
    - Checkpointing
    - Reporting callbacks
    """
    
    def __init__(self, reporting_callback=None):
        """
        Initialize training tool.
        
        Args:
            reporting_callback: Function to call with training metrics
        """
        super().__init__()
        self.reporting_callback = reporting_callback
    
    def execute(self, dataset_paths, config):
        """
        Train model using specified method.
        
        Args:
            dataset_paths (dict): Paths to train/val/test datasets
            config (dict):
                - method: str (sft|grpo|dpo)
                - model: str (base model)
                - epochs: int
                - batch_size: int
                - learning_rate: float
                - lora_r: int (optional)
                - lora_alpha: int (optional)
                
        Returns:
            dict: {
                'adapter_path': str,
                'log_paths': dict,
                'metrics': dict,
                'iteration_record': dict
            }
        """
        method = (config.get("method") or "sft").lower()
        if method != "sft":
            raise NotImplementedError("Only SFT is implemented in the migrated training stack.")

        run_dir = config.get("run_dir") or config.get("out_dir")
        if not run_dir:
            raise ValueError("TrainTool requires config['run_dir'] (or 'out_dir').")

        # Dataset can be passed either as a dataset_ref dict or legacy dicts.
        data_path: Optional[str] = None
        split: str = "train"
        max_samples: Optional[int] = config.get("max_samples")

        if isinstance(dataset_paths, dict) and "data_path" in dataset_paths:
            data_path = str(dataset_paths["data_path"])
            split = str(dataset_paths.get("split", "train"))
        elif isinstance(dataset_paths, dict) and "train_path" in dataset_paths:
            # Legacy workflow: treat train_path as a datasets.load_from_disk path.
            data_path = str(dataset_paths["train_path"])
            split = "train"
        else:
            raise ValueError("TrainTool expects dataset_paths to include 'data_path' (preferred) or 'train_path'.")

        hf_model_id = config.get("hf_model_id") or config.get("model")
        if not hf_model_id:
            raise ValueError("TrainTool requires config['hf_model_id'] (or legacy 'model').")

        iter_idx = int(config.get("iter_idx", 0))
        trainer_key = config.get("trainer_key", "static_sft_default")
        lora_preset_key = config.get("lora_preset_key", "lora_attn_small")
        model_loader_key = config.get("model_loader_key", "hf_causal_lm_default")

        train_config = TrainConfig.model_validate(config.get("train_config") or {})
        if config.get("max_steps") is not None:
            train_config = train_config.model_copy(update={"max_steps": int(config["max_steps"])})

        registry = build_registry()
        model_loader = registry.get_model_loader(model_loader_key)
        trainer_cls = registry.get_trainer(trainer_key)
        lora_preset_dict = registry.get_lora_preset(lora_preset_key)

        dataset, _ = load_dataset_from_path(data_path, split=split)
        if max_samples is not None:
            max_samples_int = int(max_samples)
            if max_samples_int > 0 and len(dataset) > max_samples_int:
                dataset = dataset.select(range(max_samples_int))

        model, tokenizer = model_loader(hf_model_id)
        model = attach_lora(model, preset_from_dict(lora_preset_dict))

        iter_dir = Path(run_dir) / "iterations" / f"iter_{iter_idx}"
        adapter_path, log_paths = run_sft_iteration(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            trainer_cls=trainer_cls,
            train_config=train_config,
            out_dir=str(iter_dir),
        )
        metrics = parse_metrics(log_paths["metrics"])

        record = IterationRecord(
            iter_idx=iter_idx,
            config=train_config,
            metrics=metrics,
            adapter_path=adapter_path,
            log_paths=log_paths,
        )

        # Update iterations.json incrementally (best-effort).
        iterations_path = Path(run_dir) / "iterations.json"
        existing: list[Any] = []
        if iterations_path.exists():
            try:
                existing = json.loads(iterations_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        existing.append(record.model_dump())
        iterations_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        return {
            "adapter_path": adapter_path,
            "log_paths": log_paths,
            "metrics": metrics.model_dump(),
            "iteration_record": record.model_dump(),
        }
