"""
Model evaluation tool.

Evaluates trained model on test set and benchmark tasks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tools.base_tool import BaseTool

from core.data.hf_dataset import load_dataset_from_path
from core.ml.hf_loader import load_hf_causal_lm
from core.ml.lora import load_lora_adapters
from tools.eval_model.eval.error_analysis import cluster_failures
from tools.eval_model.eval.failures import collect_failures
from tools.eval_model.eval.inference import run_inference


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

        hf_model_id = config.get("hf_model_id") or config.get("model")
        if not hf_model_id:
            raise ValueError("EvalModelTool requires config['hf_model_id'] (or legacy 'model').")

        # `model_path` is treated as LoRA adapter directory produced by training.
        adapter_dir = model_path

        dataset, _ = load_dataset_from_path(test_dataset_path, split=config.get("split", "train"))
        max_samples = int(config.get("max_samples", 64))
        max_new_tokens = int(config.get("max_new_tokens", 128))

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
        failures_path = collect_failures(predictions_path, run_dir)
        cluster_preview = cluster_failures(failures_path, run_dir)

        return {
            "predictions_path": predictions_path,
            "failures_path": failures_path,
            "cluster_preview": cluster_preview,
        }
