"""
Model evaluation tool.

Evaluates trained model on test set and benchmark tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from tools.base_tool import BaseTool

from core.data.hf_dataset import load_dataset_from_path
from core.ml.hf_loader import load_hf_causal_lm, load_hf_image_text_model
from core.ml.lora import load_lora_adapters
from tools.eval_model.eval.error_analysis import cluster_failures
from tools.eval_model.eval.failures import collect_failures_with_metrics
from tools.eval_model.eval.inference import run_image_inference, run_inference
from tools.eval_model.eval.llm_judge import disabled_judge_summary, run_llm_judge
from tools.eval_model.eval.train_health import evaluate_training_health


class EvalModelTool(BaseTool):
    """
    Evaluates trained LLM.
    
    Metrics:
    - Perplexity
    - Loss
    - Task-specific metrics (accuracy, BLEU, etc.)
    - Generation quality samples
    
    Can run on:
    - Held-out test set
    - External benchmarks
    """
    
    def execute(self, model_path, test_dataset_path, config):
        """
        Evaluate model.
        
        Args:
            model_path (str): Path to trained model
            test_dataset_path (str): Path to test dataset
            config (dict):
                - metrics: list[str]
                - num_samples: int (for generation)
                
        Returns:
            dict: {
                'predictions_path': str,
                'failures_path': str,
                'cluster_preview': dict
            }
        """
        run_dir = config.get("run_dir") or config.get("out_dir")
        if not run_dir:
            raise ValueError("EvalModelTool requires config['run_dir'] (or 'out_dir').")
        training_modality = str(config.get("training_modality") or config.get("sft_mode") or "text").strip().lower()
        if training_modality not in {"text", "image"}:
            raise NotImplementedError("Only text and single-image evaluation are implemented.")

        hf_model_id = config.get("hf_model_id") or config.get("model")
        if not hf_model_id:
            raise ValueError("EvalModelTool requires config['hf_model_id'] (or legacy 'model').")

        # `model_path` is treated as LoRA adapter directory produced by training.
        adapter_dir = model_path

        dataset, _ = load_dataset_from_path(test_dataset_path, split=config.get("split", "train"))
        max_samples = int(config.get("max_samples", 64))
        max_new_tokens = int(config.get("max_new_tokens", 128))
        train_health = evaluate_training_health(
            config.get("train_log_paths"),
            expected_steps=int(config["max_steps"]) if config.get("max_steps") is not None else None,
        )

        if training_modality == "image":
            base_model, processor = load_hf_image_text_model(hf_model_id)
            model = load_lora_adapters(base_model, adapter_dir)
            predictions_path = run_image_inference(
                model=model,
                processor=processor,
                dataset=dataset,
                out_dir=run_dir,
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
            )
        else:
            base_model, tokenizer = load_hf_causal_lm(hf_model_id)
            model = load_lora_adapters(base_model, adapter_dir)
            predictions_path = run_inference(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                out_dir=run_dir,
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
            )

        failures_path, deterministic_metrics = collect_failures_with_metrics(predictions_path, run_dir)
        cluster_preview = cluster_failures(failures_path, run_dir)
        if _as_bool(config.get("eval_enable_llm_judge", False)):
            judge_summary = run_llm_judge(
                predictions_path,
                run_dir,
                modality=training_modality,
                target_language=str(config.get("target_language") or config.get("sft_target_language") or ""),
                provider=config.get("llm_provider"),
                model=config.get("llm_model"),
                api_key=config.get("llm_api_key"),
                max_samples=int(config.get("eval_judge_max_samples", min(max_samples, 32))),
                batch_size=int(config.get("eval_judge_batch_size", config.get("llm_batch_size", 3))),
                batch_delay=float(config.get("eval_judge_batch_delay", config.get("llm_batch_delay", 1.0))),
            )
        else:
            judge_summary = disabled_judge_summary(run_dir)

        metrics = {
            **deterministic_metrics,
            "training_health": train_health,
            "judge": judge_summary,
            "training_modality": training_modality,
        }
        eval_metrics_path = Path(run_dir) / "eval_metrics.json"
        eval_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "predictions_path": predictions_path,
            "failures_path": failures_path,
            "cluster_preview": cluster_preview,
            "metrics": metrics,
            "eval_metrics_path": str(eval_metrics_path),
            "training_health": train_health,
            "judge_summary": judge_summary,
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
