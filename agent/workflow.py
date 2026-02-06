"""Workflow execution engine.

Runs the workflow as a sequence of tool calls in two modes:
- `workflow`: fixed steps, no LLM decisions
- `minimal_agentic`: current agentic pipeline behavior (component selection + iterative adjustments)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.agentic.agent import Agent
from core.agentic.config import DEFAULT_SEED
from core.registry.builtins import build_registry
from core.types.pipeline_types import TrainConfig
from tools.build_dataset.tool import BuildDatasetTool
from tools.eval_model.tool import EvalModelTool
from tools.reporting.tool import ReportingTool
from tools.train.tool import TrainTool
from tools.train.training.log_parser import read_log_tail
from tools.train.training.tuning import TuningBounds, apply_adjustments, generate_random_candidates

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Executes a workflow by invoking tools and passing artifacts/refs."""

    def __init__(self, tools: Dict[str, Any], config: Dict[str, Any]) -> None:
        self.tools = tools
        self.config = config

    def run(self) -> Dict[str, Any]:
        mode = (self.config.get("mode") or "workflow").lower()
        if mode == "workflow":
            return self._run_workflow()
        if mode in {"minimal_agentic", "agentic"}:
            return self._run_minimal_agentic()
        raise ValueError(f"Unknown mode: {mode}")

    def _run_workflow(self) -> Dict[str, Any]:
        run_dir = self._require("run_dir")
        data_path = self._require("data_path")

        build_dataset: BuildDatasetTool = self.tools["build_dataset"]
        train: TrainTool = self.tools["train"]
        eval_model: EvalModelTool = self.tools["eval_model"]
        reporting: ReportingTool = self.tools["reporting"]

        dataset_out = build_dataset.execute(
            data_path,
            {"run_dir": run_dir, **(self.config.get("build_dataset") or {})},
        )

        hf_model_id = self.config.get("hf_model_id") or self.config.get("model")
        if not hf_model_id:
            raise ValueError("workflow mode requires config['hf_model_id'] (or legacy 'model').")

        trainer_key = self.config.get("trainer_key", "static_sft_default")
        lora_preset_key = self.config.get("lora_preset_key", "lora_attn_small")
        model_loader_key = self.config.get("model_loader_key", "hf_causal_lm_default")

        max_iters = int(self.config.get("max_iters", 1))
        train_config = TrainConfig(
            seed=int(self.config.get("seed", DEFAULT_SEED)),
            max_steps=int(self.config.get("max_steps", 200)),
        )
        if isinstance(self.config.get("train_config"), dict):
            train_config = train_config.model_copy(update=self.config["train_config"])

        iterations: List[Dict[str, Any]] = []
        last_adapter_path: Optional[str] = None

        for iter_idx in range(max_iters):
            train_out = train.execute(
                dataset_out["dataset_ref"],
                {
                    "method": "sft",
                    "run_dir": run_dir,
                    "iter_idx": iter_idx,
                    "hf_model_id": hf_model_id,
                    "trainer_key": trainer_key,
                    "lora_preset_key": lora_preset_key,
                    "model_loader_key": model_loader_key,
                    "train_config": train_config.model_dump(),
                    "max_samples": self.config.get("max_samples"),
                },
            )
            iterations.append(train_out["iteration_record"])
            last_adapter_path = train_out["adapter_path"]

        if not last_adapter_path:
            raise RuntimeError("No adapter produced by training.")

        eval_out = eval_model.execute(
            last_adapter_path,
            data_path,
            {
                "run_dir": run_dir,
                "hf_model_id": hf_model_id,
                "split": self.config.get("eval_split", "train"),
                "max_samples": self.config.get("eval_max_samples", 64),
                "max_new_tokens": self.config.get("eval_max_new_tokens", 128),
            },
        )

        report_path = reporting.finalize(
            {
                "dataset_summary": dataset_out["dataset_summary"],
                "component_selection": {
                    "dataset_loader_key": "hf_text_default",
                    "model_loader_key": model_loader_key,
                    "lora_preset_key": lora_preset_key,
                    "trainer_key": trainer_key,
                    "hf_model_id": hf_model_id,
                    "primary_metric": "eval_loss",
                    "rationale": "workflow mode selection",
                },
                "iterations": iterations,
                "failures_path": eval_out["failures_path"],
                "cluster_preview": eval_out["cluster_preview"],
                "error_analysis": {},
            }
        )

        return {
            "mode": "workflow",
            "run_dir": run_dir,
            "adapter_path": last_adapter_path,
            "predictions_path": eval_out["predictions_path"],
            "failures_path": eval_out["failures_path"],
            "cluster_preview": eval_out["cluster_preview"],
            "report_path": report_path,
        }

    def _run_minimal_agentic(self) -> Dict[str, Any]:
        run_dir = self._require("run_dir")
        data_path = self._require("data_path")

        build_dataset: BuildDatasetTool = self.tools["build_dataset"]
        train: TrainTool = self.tools["train"]
        eval_model: EvalModelTool = self.tools["eval_model"]
        reporting: ReportingTool = self.tools["reporting"]

        dataset_out = build_dataset.execute(
            data_path,
            {"run_dir": run_dir, **(self.config.get("build_dataset") or {})},
        )

        registry = build_registry()
        snapshot = registry.snapshot()
        agent = Agent(run_dir, snapshot)

        modality_decision = agent.decide_modality(dataset_out["dataset_summary"])
        component_selection = agent.select_components(dataset_out["dataset_summary"], modality_decision.modality)

        hf_model_id = component_selection.hf_model_id
        if self.config.get("hf_model_id_override"):
            hf_model_id = str(self.config["hf_model_id_override"])
        if not hf_model_id:
            hf_model_id = snapshot.default_hf_model_id

        max_iters = int(self.config.get("max_iters", 1))
        max_steps_override = self.config.get("max_steps_override")

        train_config = TrainConfig(seed=int(self.config.get("seed", DEFAULT_SEED)))
        train_config = train_config.model_copy(
            update={"max_steps": int(max_steps_override)} if max_steps_override is not None else {"max_steps": 200}
        )
        if isinstance(self.config.get("train_config"), dict):
            train_config = train_config.model_copy(update=self.config["train_config"])

        bounds = TuningBounds()

        # Optional random search (delegating to TrainTool using isolated trial dirs)
        search_trials = int(self.config.get("search_trials", 0) or 0)
        if search_trials > 0:
            best_config = train_config
            best_score: Optional[float] = None
            candidates = generate_random_candidates(train_config, bounds, search_trials, DEFAULT_SEED)
            for idx, candidate in enumerate(candidates):
                trial_dir = Path(run_dir) / "search" / f"trial_{idx}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                trial_out = train.execute(
                    dataset_out["dataset_ref"],
                    {
                        "method": "sft",
                        "run_dir": str(trial_dir),
                        "iter_idx": 0,
                        "hf_model_id": hf_model_id,
                        "trainer_key": component_selection.trainer_key,
                        "lora_preset_key": component_selection.lora_preset_key,
                        "model_loader_key": component_selection.model_loader_key,
                        "train_config": candidate.model_dump(),
                        "max_samples": self.config.get("max_samples"),
                    },
                )
                metrics = trial_out["metrics"]
                score = metrics.get("best_eval_loss") if metrics.get("best_eval_loss") is not None else metrics.get("last_train_loss")
                if score is None:
                    continue
                if best_score is None or float(score) < best_score:
                    best_score = float(score)
                    best_config = candidate
            train_config = best_config

        iterations: List[Dict[str, Any]] = []
        last_adapter_path: Optional[str] = None

        lora_preset_key = component_selection.lora_preset_key

        for iter_idx in range(max_iters):
            train_out = train.execute(
                dataset_out["dataset_ref"],
                {
                    "method": "sft",
                    "run_dir": run_dir,
                    "iter_idx": iter_idx,
                    "hf_model_id": hf_model_id,
                    "trainer_key": component_selection.trainer_key,
                    "lora_preset_key": lora_preset_key,
                    "model_loader_key": component_selection.model_loader_key,
                    "train_config": train_config.model_dump(),
                    "max_samples": self.config.get("max_samples"),
                },
            )
            iterations.append(train_out["iteration_record"])
            last_adapter_path = train_out["adapter_path"]

            log_tail = read_log_tail(train_out["log_paths"]["train_log"])
            decision = agent.suggest_training_adjustments(
                metrics_summary=train_out["metrics"],
                log_tail=log_tail,
                bounds=bounds.__dict__,
            )
            if not decision.should_retry:
                break
            if decision.switch_lora_preset_key and decision.switch_lora_preset_key != lora_preset_key:
                lora_preset_key = decision.switch_lora_preset_key
            train_config = apply_adjustments(train_config, decision, bounds)

        if not last_adapter_path:
            raise RuntimeError("No adapter produced by training.")

        eval_out = eval_model.execute(
            last_adapter_path,
            data_path,
            {
                "run_dir": run_dir,
                "hf_model_id": hf_model_id,
                "split": self.config.get("eval_split", "train"),
                "max_samples": self.config.get("eval_max_samples", 64),
                "max_new_tokens": self.config.get("eval_max_new_tokens", 128),
            },
        )

        failure_overview: Dict[str, Any] = {"failures_path": eval_out["failures_path"]}
        try:
            failure_overview["total_failures"] = sum(
                1 for _ in open(eval_out["failures_path"], "r", encoding="utf-8")
            )
        except Exception:
            failure_overview["total_failures"] = None

        error_analysis = agent.analyze_errors(failure_overview, eval_out["cluster_preview"]).model_dump()

        report_path = reporting.finalize(
            {
                "dataset_summary": dataset_out["dataset_summary"],
                "component_selection": component_selection.model_dump() | {"hf_model_id": hf_model_id},
                "iterations": iterations,
                "failures_path": eval_out["failures_path"],
                "cluster_preview": eval_out["cluster_preview"],
                "error_analysis": error_analysis,
            }
        )

        return {
            "mode": "minimal_agentic",
            "run_dir": run_dir,
            "adapter_path": last_adapter_path,
            "component_selection": component_selection.model_dump() | {"hf_model_id": hf_model_id},
            "predictions_path": eval_out["predictions_path"],
            "failures_path": eval_out["failures_path"],
            "cluster_preview": eval_out["cluster_preview"],
            "error_analysis": error_analysis,
            "report_path": report_path,
        }

    def _require(self, key: str) -> Any:
        value = self.config.get(key)
        if value is None or value == "":
            raise ValueError(f"Missing required config key: {key}")
        return value
