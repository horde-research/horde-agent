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
        compare_base = _as_bool(config.get("eval_compare_base_model", True))
        train_health = evaluate_training_health(
            config.get("train_log_paths"),
            expected_steps=int(config["max_steps"]) if config.get("max_steps") is not None else None,
        )

        base_eval: Dict[str, Any] | None = None
        if training_modality == "image":
            base_model, processor = load_hf_image_text_model(hf_model_id)
            if compare_base:
                base_predictions_path = run_image_inference(
                    model=base_model,
                    processor=processor,
                    dataset=dataset,
                    out_dir=str(Path(run_dir) / "base"),
                    max_samples=max_samples,
                    max_new_tokens=max_new_tokens,
                )
                base_eval = _finalize_predictions(
                    base_predictions_path,
                    str(Path(run_dir) / "base"),
                    config,
                    training_modality=training_modality,
                    max_samples=max_samples,
                    label="base",
                )
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
            if compare_base:
                base_predictions_path = run_inference(
                    model=base_model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    out_dir=str(Path(run_dir) / "base"),
                    max_samples=max_samples,
                    max_new_tokens=max_new_tokens,
                )
                base_eval = _finalize_predictions(
                    base_predictions_path,
                    str(Path(run_dir) / "base"),
                    config,
                    training_modality=training_modality,
                    max_samples=max_samples,
                    label="base",
                )
            model = load_lora_adapters(base_model, adapter_dir)
            predictions_path = run_inference(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                out_dir=run_dir,
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
            )

        adapter_eval = _finalize_predictions(
            predictions_path,
            run_dir,
            config,
            training_modality=training_modality,
            max_samples=max_samples,
            label="adapter",
        )
        failures_path = adapter_eval["failures_path"]
        deterministic_metrics = adapter_eval["metrics"]
        cluster_preview = adapter_eval["cluster_preview"]
        judge_summary = adapter_eval["judge"]
        lift_summary = _build_lift_summary(base_eval, adapter_eval)
        lift_summary_path = Path(run_dir) / "lift_summary.json"
        lift_summary_path.write_text(json.dumps(lift_summary, indent=2, ensure_ascii=False), encoding="utf-8")

        metrics = {
            **deterministic_metrics,
            "training_health": train_health,
            "judge": judge_summary,
            "training_modality": training_modality,
            "base_eval": base_eval,
            "adapter_eval": adapter_eval,
            "lift": lift_summary,
        }
        eval_metrics_path = Path(run_dir) / "eval_metrics.json"
        eval_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        result = {
            "predictions_path": predictions_path,
            "failures_path": failures_path,
            "cluster_preview": cluster_preview,
            "metrics": metrics,
            "eval_metrics_path": str(eval_metrics_path),
            "training_health": train_health,
            "judge_summary": judge_summary,
            "lift_summary": lift_summary,
            "lift_summary_path": str(lift_summary_path),
        }
        if base_eval:
            result.update(
                {
                    "base_predictions_path": base_eval.get("predictions_path"),
                    "base_failures_path": base_eval.get("failures_path"),
                    "base_eval_metrics_path": base_eval.get("eval_metrics_path"),
                    "base_judge_summary": base_eval.get("judge"),
                }
            )
        return result


def _finalize_predictions(
    predictions_path: str,
    run_dir: str,
    config: Dict[str, Any],
    *,
    training_modality: str,
    max_samples: int,
    label: str,
) -> Dict[str, Any]:
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
    component_metrics = {
        **deterministic_metrics,
        "judge": judge_summary,
        "training_modality": training_modality,
        "label": label,
    }
    metrics_path = Path(run_dir) / "eval_metrics.json"
    metrics_path.write_text(json.dumps(component_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "label": label,
        "predictions_path": predictions_path,
        "failures_path": failures_path,
        "cluster_preview": cluster_preview,
        "metrics": deterministic_metrics,
        "judge": judge_summary,
        "eval_metrics_path": str(metrics_path),
    }


def _build_lift_summary(base_eval: Dict[str, Any] | None, adapter_eval: Dict[str, Any]) -> Dict[str, Any]:
    if not base_eval:
        return {"enabled": False, "reason": "base_eval_disabled"}
    base_metrics = base_eval.get("metrics") or {}
    adapter_metrics = adapter_eval.get("metrics") or {}
    base_judge = base_eval.get("judge") or {}
    adapter_judge = adapter_eval.get("judge") or {}
    return {
        "enabled": True,
        "failure_rate_delta": _delta(adapter_metrics, base_metrics, "failure_rate"),
        "quality_score_delta": _delta(adapter_judge, base_judge, "quality_score"),
        "major_failure_rate_delta": _delta(adapter_judge, base_judge, "major_failure_rate"),
        "unsupported_grounding_rate_delta": _delta(adapter_judge, base_judge, "unsupported_grounding_rate"),
        "base": {
            "failure_rate": base_metrics.get("failure_rate"),
            "judge_quality_score": base_judge.get("quality_score"),
            "judge_major_failure_rate": base_judge.get("major_failure_rate"),
            "unsupported_grounding_rate": base_judge.get("unsupported_grounding_rate"),
        },
        "adapter": {
            "failure_rate": adapter_metrics.get("failure_rate"),
            "judge_quality_score": adapter_judge.get("quality_score"),
            "judge_major_failure_rate": adapter_judge.get("major_failure_rate"),
            "unsupported_grounding_rate": adapter_judge.get("unsupported_grounding_rate"),
        },
    }


def _delta(left: Dict[str, Any], right: Dict[str, Any], key: str) -> float | None:
    try:
        if left.get(key) is None or right.get(key) is None:
            return None
        return float(left[key]) - float(right[key])
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
