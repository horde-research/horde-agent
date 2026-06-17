"""CLI entrypoint for the Horde Agent pipeline.

Everything is configured via .env (loaded by PipelineConfig.from_env()).
CLI arguments are optional overrides.

Usage:
    # Run with everything from .env (full pipeline by default)
    python -m agent.main

    # Override country and output dir
    python -m agent.main --country "Kazakhstan" --out_dir ./output/run1

    # Workflow mode: start from existing dataset
    python -m agent.main --data_path ./my_sft.jsonl --out_dir ./output/run2

    # Agentic mode
    python -m agent.main --data_path ./my_data --mode minimal_agentic
"""

from __future__ import annotations

import argparse
import logging
import sys

from agent.orchestrator import Orchestrator
from core.agentic.action_space import FULL_GRAPH_ACTIONS


_TOOL_NAME_BY_LOGGER = {
    "__main__": "PipelineCLI",
    "agent.main": "PipelineCLI",
    "agent.workflow": "WorkflowRunner",
    "agent.orchestrator": "Orchestrator",
    "core.agentic.langgraph_runtime": "AgenticRuntime",
    "core.agentic.observability": "LangSmithObserver",
    "core.agentic.state_store": "AgenticStateStore",
    "core.agentic.recovery": "AgenticRecoveryPlanner",
    "core.llm.client": "LLMClient",
    "core.hf_hub": "HuggingFaceHub",
    "tools.collect_data.tool": "CollectDataTool",
    "tools.collect_data.image_search": "SerperImageSearchCollector",
    "tools.collect_data.images": "HtmlImageCollector",
    "tools.generate_taxonomy.tool": "GenerateTaxonomyTool",
    "tools.generate_taxonomy.agents.category_agent": "CategoryAgent",
    "tools.generate_taxonomy.agents.subcategory_agent": "SubcategoryAgent",
    "tools.generate_taxonomy.agents.keyword_agent": "KeywordAgent",
    "tools.build_sft_dataset.tool": "BuildSftDatasetTool",
    "tools.build_sft_dataset.agents.sft_text_agent": "SftTextAgent",
    "tools.build_sft_dataset.agents.sft_image_agent": "SftImageAgent",
    "tools.build_dataset.tool": "BuildDatasetTool",
    "tools.train.tool": "TrainTool",
    "tools.train.trainers.static_sft_trainer": "StaticSFTTrainer",
    "tools.train.trainers.vision_language_sft_trainer": "VisionLanguageSFTTrainer",
    "tools.eval_model.tool": "EvalModelTool",
    "tools.reporting.tool": "ReportingTool",
    "tools.reporting.report": "ReportingTool",
}

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "hpack",
)


def _tool_name_from_logger(logger_name: str) -> str:
    if logger_name in _TOOL_NAME_BY_LOGGER:
        return _TOOL_NAME_BY_LOGGER[logger_name]
    if logger_name.startswith("sft."):
        return "StaticSFTTrainer"
    if logger_name.startswith("vlm_sft."):
        return "VisionLanguageSFTTrainer"
    if logger_name.startswith("tools.") and logger_name.endswith(".tool"):
        parts = logger_name.split(".")
        if len(parts) >= 3:
            return f"{_snake_to_pascal(parts[1])}Tool"
    return "-"


def _snake_to_pascal(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_") if part)


def _install_tool_log_record_factory() -> None:
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_horde_tool_name_factory", False):
        return

    def record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        record.tool_name = _tool_name_from_logger(record.name)
        return record

    record_factory._horde_tool_name_factory = True
    logging.setLogRecordFactory(record_factory)


def _setup_logging(level: str) -> None:
    _install_tool_log_record_factory()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(name)s - %(tool_name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Horde Agent - LLM training pipeline")

    parser.add_argument("--country", type=str, default=None, help="Country or culture name (overrides .env COUNTRY)")
    parser.add_argument("--data_path", type=str, default=None, help="Existing dataset path or HF repo ID (skips taxonomy & collection)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (overrides default run_dir)")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["full", "workflow", "minimal_agentic", "full_agentic"],
        help="Execution mode (auto-detected if omitted)",
    )
    parser.add_argument("--max_queries", type=int, default=None, help="Limit taxonomy search queries")
    parser.add_argument("--max_queries_per_category", type=int, default=None, help="Limit selected taxonomy queries per top-level category")
    parser.add_argument("--max_iters", type=int, default=None, help="Max training iterations")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to N samples")
    parser.add_argument("--dataset-val-ratio", type=float, default=None, help="Fraction of built SFT examples held out for validation")
    parser.add_argument("--eval-split", type=str, default=None, help="Dataset split used by evaluation")
    parser.add_argument("--hf_model_id", default=None, help="Base HuggingFace model id for training")
    parser.add_argument(
        "--training-modality",
        choices=["text", "image"],
        default=None,
        help="Which collected modality is converted to SFT and used for training",
    )
    parser.add_argument("--search_trials", type=int, default=None, help="Random search trials before training")
    parser.add_argument("--debug-stub-train", action="store_true", help="Skip real training and emit a valid dummy adapter")
    parser.add_argument("--debug-stub-eval", action="store_true", help="Skip real model evaluation and emit happy-path eval artifacts")
    parser.add_argument("--eval-enable-llm-judge", action="store_true", help="Use configured LLM to judge validation predictions")
    parser.add_argument(
        "--resume-confirm-completed",
        action="store_true",
        help="Reuse completed stages from an existing full_agentic run without pausing for confirmation",
    )
    parser.add_argument(
        "--restart-from-stage",
        choices=[stage.value for stage in FULL_GRAPH_ACTIONS],
        default=None,
        help="Clear the named stage and all downstream stages before resuming full_agentic",
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Clear all completed full_agentic stages in the existing run directory before running",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    overrides: dict = {}

    if args.country:
        overrides["country"] = args.country
    if args.data_path:
        overrides["data_path"] = args.data_path
    if args.out_dir:
        overrides["run_dir"] = args.out_dir
    if args.max_iters is not None:
        overrides["max_iters"] = args.max_iters
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.max_samples is not None:
        overrides["max_samples"] = args.max_samples
    if args.dataset_val_ratio is not None:
        overrides["dataset_val_ratio"] = args.dataset_val_ratio
    if args.eval_split is not None:
        overrides["eval_split"] = args.eval_split
    if args.max_queries is not None:
        overrides["max_queries"] = args.max_queries
    if args.max_queries_per_category is not None:
        overrides["max_queries_per_category"] = args.max_queries_per_category
    if args.hf_model_id:
        overrides["hf_model_id"] = args.hf_model_id
    if args.training_modality:
        overrides["training_modality"] = args.training_modality
    if args.search_trials is not None:
        overrides["search_trials"] = args.search_trials
    if args.debug_stub_train:
        overrides["debug_stub_train"] = True
    if args.debug_stub_eval:
        overrides["debug_stub_eval"] = True
    if args.eval_enable_llm_judge:
        overrides["eval_enable_llm_judge"] = True
    if args.resume_confirm_completed:
        overrides["resume_confirm_completed"] = True
    if args.restart_from_stage:
        overrides["restart_from_stage"] = args.restart_from_stage
    if args.fresh_run:
        overrides["fresh_run"] = True

    # Auto-detect mode
    if args.mode:
        overrides["mode"] = args.mode
    elif args.data_path:
        overrides["mode"] = "workflow"

    logger.info("Loading config from .env%s...",
                f" + {len(overrides)} CLI overrides" if overrides else "")
    result = Orchestrator(None, **overrides).run()
    report_path = result.get("report_path") or (result.get("artifacts") or {}).get("report_path")
    if report_path:
        logger.info("Pipeline complete. Report: %s", report_path)
    else:
        logger.info(
            "Pipeline stopped without report. termination_reason=%s blockers=%s completed_stages=%s",
            result.get("termination_reason"),
            result.get("blockers"),
            result.get("completed_stages"),
        )


if __name__ == "__main__":
    main()
