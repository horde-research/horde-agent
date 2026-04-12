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
from agent.crewai_orchestrator import CrewAIOrchestrator


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Horde Agent - LLM training pipeline")

    parser.add_argument("--country", type=str, default=None, help="Country or culture name (overrides .env COUNTRY)")
    parser.add_argument("--data_path", type=str, default=None, help="Existing dataset path or HF repo ID (skips taxonomy & collection)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (overrides default run_dir)")
    parser.add_argument("--mode", default=None, choices=["full", "workflow", "minimal_agentic"], help="Execution mode (auto-detected if omitted)")
    parser.add_argument("--max_iters", type=int, default=None, help="Max training iterations")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to N samples")
    parser.add_argument("--hf_model_id", default=None, help="Base HuggingFace model id for training")
    parser.add_argument("--search_trials", type=int, default=None, help="Random search trials before training")
    parser.add_argument(
        "--orchestrator",
        default="native",
        choices=["native", "crewai"],
        help="Orchestration backend: native WorkflowRunner or CrewAI wrapper.",
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
    if args.hf_model_id:
        overrides["hf_model_id"] = args.hf_model_id
    if args.search_trials is not None:
        overrides["search_trials"] = args.search_trials

    # Auto-detect mode
    if args.mode:
        overrides["mode"] = args.mode
    elif args.data_path:
        overrides["mode"] = "workflow"

    logger.info("Loading config from .env%s...",
                f" + {len(overrides)} CLI overrides" if overrides else "")
    if args.orchestrator == "crewai":
        logger.info("Using CrewAI orchestrator backend.")
        result = CrewAIOrchestrator(None, **overrides).run()
    else:
        result = Orchestrator(None, **overrides).run()
    logger.info("Pipeline complete. Report: %s", result.get("report_path"))


if __name__ == "__main__":
    main()
