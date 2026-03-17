"""CLI entrypoint for the Horde Agent pipeline.

Usage examples:

    # Full pipeline: country → taxonomy → collect → sft → train → eval → report
    python -m agent.main --country "Kazakhstan" --out_dir ./output/run1

    # Workflow mode: start from existing dataset
    python -m agent.main --data_path ./my_sft.jsonl --out_dir ./output/run2

    # Agentic mode: LLM-driven decisions
    python -m agent.main --data_path ./my_data --out_dir ./output/run3 --mode minimal_agentic
"""

from __future__ import annotations

import argparse
import logging
import sys

from agent.orchestrator import Orchestrator


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Horde Agent - LLM training pipeline")

    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument(
        "--country",
        type=str,
        help="Country or culture name (triggers full pipeline: taxonomy → collect → sft → train)",
    )
    entry.add_argument(
        "--data_path",
        type=str,
        help="HF dataset id, local dataset path, or JSONL file (skips taxonomy & collection)",
    )

    parser.add_argument("--out_dir", required=True, help="Output directory (run_dir)")
    parser.add_argument("--max_iters", type=int, default=1, help="Max training iterations")

    parser.add_argument(
        "--mode",
        default=None,
        choices=["full", "workflow", "minimal_agentic"],
        help="Execution mode (auto-detected from --country / --data_path if omitted)",
    )

    parser.add_argument("--hf_model_id", default=None, help="Base HuggingFace model id for training")
    parser.add_argument("--hf_model_id_override", default=None, help="Override HF model id (minimal_agentic)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to N samples")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--max_steps_override", type=int, default=None, help="Override max training steps (minimal_agentic)")
    parser.add_argument("--search_trials", type=int, default=0, help="Random search trials before training")

    parser.add_argument("--provider", default=None, help="LLM provider for taxonomy/SFT generation (openai|gemini|xai)")
    parser.add_argument("--llm_model", default=None, help="LLM model name for taxonomy/SFT generation")
    parser.add_argument("--batch_size", type=int, default=None, help="LLM batch size for taxonomy/SFT generation")
    parser.add_argument("--batch_delay", type=float, default=None, help="Delay between LLM batches (seconds)")

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

    # Auto-detect mode from entry point
    if args.mode:
        mode = args.mode
    elif args.country:
        mode = "full"
    else:
        mode = "workflow"

    cfg: dict = {
        "mode": mode,
        "run_dir": args.out_dir,
        "max_iters": args.max_iters,
        "search_trials": args.search_trials,
        "max_samples": args.max_samples,
    }

    if args.country:
        cfg["country"] = args.country
    if args.data_path:
        cfg["data_path"] = args.data_path
    if args.hf_model_id:
        cfg["hf_model_id"] = args.hf_model_id
    if args.hf_model_id_override:
        cfg["hf_model_id_override"] = args.hf_model_id_override
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.max_steps_override is not None:
        cfg["max_steps_override"] = args.max_steps_override

    # LLM config for taxonomy / SFT generation tools
    if args.provider:
        cfg["provider"] = args.provider
    if args.llm_model:
        cfg["model"] = args.llm_model
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.batch_delay is not None:
        cfg["batch_delay"] = args.batch_delay

    logger.info("Starting pipeline in '%s' mode -> %s", mode, args.out_dir)
    result = Orchestrator(cfg).run()
    logger.info("Pipeline complete. Report: %s", result.get("report_path"))


if __name__ == "__main__":
    main()
