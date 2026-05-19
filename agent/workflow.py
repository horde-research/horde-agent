"""Workflow runner — executes the pipeline using PipelineConfig.

Every parameter is read from PipelineConfig. No hidden defaults.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PipelineConfig
from core.agentic.agent import Agent, DEFAULT_SEED
from core.registry.builtins import build_registry
from core.types.pipeline_types import TrainConfig
from tools.build_dataset.tool import BuildDatasetTool
from tools.build_sft_dataset.tool import BuildSftDatasetTool
from tools.collect_data.tool import CollectDataTool
from tools.eval_model.tool import EvalModelTool
from tools.generate_taxonomy.tool import GenerateTaxonomyTool
from tools.reporting.tool import ReportingTool
from tools.train.tool import TrainTool
from tools.train.training.log_parser import read_log_tail
from tools.train.training.tuning import TuningBounds, apply_adjustments, generate_random_candidates

logger = logging.getLogger(__name__)


def _push_to_hf_hub_if_configured(cfg: PipelineConfig, *, dataset_path: str | None = None, adapter_path: str | None = None) -> Dict[str, str]:
    """Push dataset/adapter to HF Hub if repo names are configured. Returns repo IDs."""
    import os
    from core.hf_hub import push_dataset, push_adapter

    if cfg.hf_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = cfg.hf_token

    pushed: Dict[str, str] = {}
    username = cfg.hf_username or None

    if dataset_path and cfg.hf_dataset_repo:
        try:
            repo_id = push_dataset(dataset_path, cfg.hf_dataset_repo, username=username)
            pushed["dataset_repo_id"] = repo_id
            logger.info("Dataset pushed to HF Hub: %s", repo_id)
        except Exception as exc:
            logger.error("Failed to push dataset to HF Hub: %s", exc)

    if adapter_path and cfg.hf_adapter_repo:
        try:
            repo_id = push_adapter(adapter_path, cfg.hf_adapter_repo, username=username)
            pushed["adapter_repo_id"] = repo_id
            logger.info("Adapter pushed to HF Hub: %s", repo_id)
        except Exception as exc:
            logger.error("Failed to push adapter to HF Hub: %s", exc)

    return pushed


class WorkflowRunner:
    """Executes a workflow by invoking tools and passing artifacts."""

    def __init__(self, tools: Dict[str, Any], cfg: PipelineConfig) -> None:
        self.tools = tools
        self.cfg = cfg

    def run(self) -> Dict[str, Any]:
        mode = self.cfg.mode.lower()
        if mode == "full":
            return self._run_full_pipeline()
        if mode == "workflow":
            return self._run_workflow()
        if mode in {"minimal_agentic", "agentic"}:
            return self._run_minimal_agentic()
        raise ValueError(f"Unknown mode: {mode}")

    # ── full pipeline ─────────────────────────────────────────────────────────

    def _run_full_pipeline(self) -> Dict[str, Any]:
        cfg = self.cfg
        generate_taxonomy: GenerateTaxonomyTool = self.tools["generate_taxonomy"]
        collect_data: CollectDataTool = self.tools["collect_data"]
        build_sft: BuildSftDatasetTool = self.tools["build_sft_dataset"]
        build_dataset: BuildDatasetTool = self.tools["build_dataset"]
        train: TrainTool = self.tools["train"]
        eval_model: EvalModelTool = self.tools["eval_model"]
        reporting: ReportingTool = self.tools["reporting"]

        # Step 1 — Taxonomy
        logger.info("Step 1/7: Generating taxonomy for '%s'...", cfg.country)
        taxonomy_out = generate_taxonomy.execute(cfg.country, {
            "batch_size": cfg.llm_batch_size,
            "batch_delay": cfg.llm_batch_delay,
        })

        all_queries: List[str] = []
        for sub_dict in taxonomy_out["category_subcategory_queries"].values():
            for q_list in sub_dict.values():
                all_queries.extend(q_list)
        if not all_queries:
            raise RuntimeError("Taxonomy generation produced zero search queries.")
        logger.info("Taxonomy produced %d search queries.", len(all_queries))

        if cfg.max_queries and len(all_queries) > cfg.max_queries:
            logger.info("Limiting to %d queries (from %d).", cfg.max_queries, len(all_queries))
            all_queries = all_queries[:cfg.max_queries]

        # Step 2 — Collect data
        logger.info("Step 2/7: Collecting data with %d search queries...", len(all_queries))
        collect_dir = str(Path(cfg.run_dir) / "collect")
        collect_out = collect_data.execute({
            "queries": all_queries,
            "run_dir": collect_dir,
            "google_results_per_query": cfg.serper_results_per_query,
            "top_results": cfg.serper_top_results,
            "concurrency": cfg.serper_concurrency,
        })
        raw_data_path = collect_out["data_path"]
        logger.info("Collected %d samples -> %s", collect_out["num_samples"], raw_data_path)

        # Step 3 — SFT annotation
        logger.info("Step 3/7: Building SFT dataset from collected text...")
        sft_dir = Path(cfg.run_dir) / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)
        collected_jsonl = str(sft_dir / "collected_texts.jsonl")
        _export_hf_dataset_to_jsonl(raw_data_path, collected_jsonl)

        sft_out = build_sft.execute({
            "mode": cfg.sft_mode,
            "input_jsonl": collected_jsonl,
            "text_field": "text",
            "output_annotations": str(sft_dir / "annotations.jsonl"),
            "output_sft": str(sft_dir / "sft.jsonl"),
            "target_language": cfg.sft_target_language,
            "batch_size": cfg.llm_batch_size,
            "batch_delay": cfg.llm_batch_delay,
        })
        sft_path = sft_out["sft_path"]
        logger.info("Built %d SFT examples -> %s", sft_out["num_examples"], sft_path)

        # Push SFT dataset to HF Hub
        _push_to_hf_hub_if_configured(cfg, dataset_path=sft_path)

        # Step 4 — Build HF dataset
        logger.info("Step 4/7: Loading SFT dataset for training...")
        dataset_out = build_dataset.execute(sft_path, {"run_dir": cfg.run_dir})

        # Steps 5-7 — Train, evaluate, report
        result = self._train_eval_report(
            dataset_out=dataset_out,
            train=train,
            eval_model=eval_model,
            reporting=reporting,
            extra_report_data={
                "taxonomy": {
                    "country": cfg.country,
                    "num_categories": len(taxonomy_out["categories"]),
                    "num_queries": len(all_queries),
                },
                "collection": {
                    "num_samples": collect_out["num_samples"],
                    "raw_data_path": raw_data_path,
                },
                "sft_generation": {
                    "num_examples": sft_out["num_examples"],
                    "num_failures": sft_out["num_failures"],
                },
            },
            mode_name="full",
        )

        # Push trained adapter to HF Hub
        adapter_path = result.get("adapter_path")
        if adapter_path:
            hub_info = _push_to_hf_hub_if_configured(cfg, adapter_path=adapter_path)
            result.update(hub_info)

        return result

    # ── workflow mode: start from existing data_path ──────────────────────────

    def _run_workflow(self) -> Dict[str, Any]:
        cfg = self.cfg
        if not cfg.data_path:
            raise ValueError("Workflow mode requires data_path in config.")

        build_dataset: BuildDatasetTool = self.tools["build_dataset"]
        train: TrainTool = self.tools["train"]
        eval_model: EvalModelTool = self.tools["eval_model"]
        reporting: ReportingTool = self.tools["reporting"]

        dataset_out = build_dataset.execute(cfg.data_path, {"run_dir": cfg.run_dir})

        result = self._train_eval_report(
            dataset_out=dataset_out,
            train=train,
            eval_model=eval_model,
            reporting=reporting,
            extra_report_data={},
            mode_name="workflow",
        )

        adapter_path = result.get("adapter_path")
        if adapter_path:
            hub_info = _push_to_hf_hub_if_configured(cfg, adapter_path=adapter_path)
            result.update(hub_info)

        return result

    # ── shared train → eval → report ─────────────────────────────────────────

    def _train_eval_report(
        self,
        *,
        dataset_out: Dict[str, Any],
        train: TrainTool,
        eval_model: EvalModelTool,
        reporting: ReportingTool,
        extra_report_data: Dict[str, Any],
        mode_name: str,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        train_config = TrainConfig(**cfg.train_config_dict())

        data_path = dataset_out["dataset_ref"]["data_path"]
        iterations: List[Dict[str, Any]] = []
        last_adapter_path: Optional[str] = None

        for iter_idx in range(cfg.max_iters):
            logger.info("Training iteration %d/%d...", iter_idx + 1, cfg.max_iters)
            train_out = train.execute(
                dataset_out["dataset_ref"],
                {
                    "method": "sft",
                    "run_dir": cfg.run_dir,
                    "iter_idx": iter_idx,
                    "hf_model_id": cfg.hf_model_id,
                    "trainer_key": cfg.trainer_key,
                    "lora_preset_key": cfg.lora_preset_key,
                    "model_loader_key": cfg.model_loader_key,
                    "train_config": train_config.model_dump(),
                    "max_samples": cfg.max_samples,
                    "dataset_slice": {"part": "sft", "sft_train_fraction": cfg.sft_train_fraction},
                },
            )
            iterations.append(train_out["iteration_record"])
            last_adapter_path = train_out["adapter_path"]

        if not last_adapter_path:
            raise RuntimeError("No adapter produced by training.")

        if cfg.enable_grpo:
            logger.info("Running GRPO phase after SFT...")
            grpo_out = train.execute(
                dataset_out["dataset_ref"],
                {
                    "method": "grpo",
                    "run_dir": cfg.run_dir,
                    "iter_idx": len(iterations),
                    "hf_model_id": cfg.hf_model_id,
                    "lora_preset_key": cfg.lora_preset_key,
                    "model_loader_key": cfg.model_loader_key,
                    "grpo_config": cfg.grpo_config_dict(),
                    "max_samples": cfg.max_samples,
                    "dataset_slice": {"part": "rl", "sft_train_fraction": cfg.sft_train_fraction},
                    "previous_adapter_path": last_adapter_path,
                },
            )
            iterations.append(grpo_out["iteration_record"])
            last_adapter_path = grpo_out["adapter_path"]

        logger.info("Evaluating model...")
        eval_out = eval_model.execute(
            last_adapter_path,
            data_path,
            {
                "run_dir": cfg.run_dir,
                "hf_model_id": cfg.hf_model_id,
                "split": cfg.eval_split,
                "max_samples": cfg.eval_max_samples,
                "max_new_tokens": cfg.eval_max_new_tokens,
            },
        )

        logger.info("Generating report...")
        report_path = reporting.finalize({
            "dataset_summary": dataset_out["dataset_summary"],
            "component_selection": {
                "dataset_loader_key": "hf_text_default",
                "model_loader_key": cfg.model_loader_key,
                "lora_preset_key": cfg.lora_preset_key,
                "trainer_key": cfg.trainer_key,
                "hf_model_id": cfg.hf_model_id,
                "primary_metric": "reward" if cfg.enable_grpo else "eval_loss",
                "rationale": f"{mode_name} mode selection",
            },
            "iterations": iterations,
            "failures_path": eval_out["failures_path"],
            "cluster_preview": eval_out["cluster_preview"],
            "error_analysis": extra_report_data,
        })

        return {
            "mode": mode_name,
            "run_dir": cfg.run_dir,
            "adapter_path": last_adapter_path,
            "predictions_path": eval_out["predictions_path"],
            "failures_path": eval_out["failures_path"],
            "cluster_preview": eval_out["cluster_preview"],
            "report_path": report_path,
            **extra_report_data,
        }

    # ── minimal_agentic mode ─────────────────────────────────────────────────

    def _run_minimal_agentic(self) -> Dict[str, Any]:
        cfg = self.cfg
        if not cfg.data_path:
            raise ValueError("Agentic mode requires data_path in config.")

        build_dataset: BuildDatasetTool = self.tools["build_dataset"]
        train: TrainTool = self.tools["train"]
        eval_model: EvalModelTool = self.tools["eval_model"]
        reporting: ReportingTool = self.tools["reporting"]

        dataset_out = build_dataset.execute(cfg.data_path, {"run_dir": cfg.run_dir})

        registry = build_registry()
        snapshot = registry.snapshot()
        agent = Agent(cfg.run_dir, snapshot)

        modality_decision = agent.decide_modality(dataset_out["dataset_summary"])
        component_selection = agent.select_components(
            dataset_out["dataset_summary"], modality_decision.modality,
        )

        hf_model_id = cfg.hf_model_id
        if hf_model_id == PipelineConfig.model_fields["hf_model_id"].default:
            hf_model_id = component_selection.hf_model_id or snapshot.default_hf_model_id

        train_config = TrainConfig(**cfg.train_config_dict())

        bounds = TuningBounds()

        if cfg.search_trials > 0:
            best_config = train_config
            best_score: Optional[float] = None
            candidates = generate_random_candidates(train_config, bounds, cfg.search_trials, cfg.seed)
            for idx, candidate in enumerate(candidates):
                trial_dir = Path(cfg.run_dir) / "search" / f"trial_{idx}"
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
                        "max_samples": cfg.max_samples,
                        "dataset_slice": {"part": "sft", "sft_train_fraction": cfg.sft_train_fraction},
                    },
                )
                metrics = trial_out["metrics"]
                score = metrics.get("best_eval_loss") or metrics.get("last_train_loss")
                if score is None:
                    continue
                if best_score is None or float(score) < best_score:
                    best_score = float(score)
                    best_config = candidate
            train_config = best_config

        iterations: List[Dict[str, Any]] = []
        last_adapter_path: Optional[str] = None
        lora_preset_key = component_selection.lora_preset_key

        for iter_idx in range(cfg.max_iters):
            train_out = train.execute(
                dataset_out["dataset_ref"],
                {
                    "method": "sft",
                    "run_dir": cfg.run_dir,
                    "iter_idx": iter_idx,
                    "hf_model_id": hf_model_id,
                    "trainer_key": component_selection.trainer_key,
                    "lora_preset_key": lora_preset_key,
                    "model_loader_key": component_selection.model_loader_key,
                    "train_config": train_config.model_dump(),
                    "max_samples": cfg.max_samples,
                    "dataset_slice": {"part": "sft", "sft_train_fraction": cfg.sft_train_fraction},
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

        if cfg.enable_grpo:
            grpo_out = train.execute(
                dataset_out["dataset_ref"],
                {
                    "method": "grpo",
                    "run_dir": cfg.run_dir,
                    "iter_idx": len(iterations),
                    "hf_model_id": hf_model_id,
                    "lora_preset_key": lora_preset_key,
                    "model_loader_key": component_selection.model_loader_key,
                    "grpo_config": cfg.grpo_config_dict(),
                    "max_samples": cfg.max_samples,
                    "dataset_slice": {"part": "rl", "sft_train_fraction": cfg.sft_train_fraction},
                    "previous_adapter_path": last_adapter_path,
                },
            )
            iterations.append(grpo_out["iteration_record"])
            last_adapter_path = grpo_out["adapter_path"]

        # Push adapter to HF Hub
        hub_info = _push_to_hf_hub_if_configured(cfg, adapter_path=last_adapter_path)

        eval_out = eval_model.execute(
            last_adapter_path,
            cfg.data_path,
            {
                "run_dir": cfg.run_dir,
                "hf_model_id": hf_model_id,
                "split": cfg.eval_split,
                "max_samples": cfg.eval_max_samples,
                "max_new_tokens": cfg.eval_max_new_tokens,
            },
        )

        failure_overview: Dict[str, Any] = {"failures_path": eval_out["failures_path"]}
        try:
            failure_overview["total_failures"] = sum(
                1 for _ in open(eval_out["failures_path"], "r", encoding="utf-8")
            )
        except Exception:
            failure_overview["total_failures"] = None

        error_analysis = agent.analyze_errors(
            failure_overview, eval_out["cluster_preview"],
        ).model_dump()

        report_path = reporting.finalize({
            "dataset_summary": dataset_out["dataset_summary"],
            "component_selection": component_selection.model_dump() | {"hf_model_id": hf_model_id},
            "iterations": iterations,
            "failures_path": eval_out["failures_path"],
            "cluster_preview": eval_out["cluster_preview"],
            "error_analysis": error_analysis,
        })

        return {
            "mode": "minimal_agentic",
            "run_dir": cfg.run_dir,
            "adapter_path": last_adapter_path,
            "component_selection": component_selection.model_dump() | {"hf_model_id": hf_model_id},
            "predictions_path": eval_out["predictions_path"],
            "failures_path": eval_out["failures_path"],
            "cluster_preview": eval_out["cluster_preview"],
            "error_analysis": error_analysis,
            "report_path": report_path,
            **hub_info,
        }


def _export_hf_dataset_to_jsonl(hf_dataset_path: str, jsonl_path: str) -> None:
    import json
    from datasets import load_from_disk

    ds = load_from_disk(hf_dataset_path)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
