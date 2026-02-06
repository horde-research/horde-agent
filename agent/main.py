"""CLI entrypoint for the migrated training workflow.

Replaces `python -m agentic_train_pipeline.main` after cutover.
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
    parser = argparse.ArgumentParser(description="Horde Agent - migrated training workflow")
    parser.add_argument("--data_path", required=True, help="HF dataset id or local dataset path")
    parser.add_argument("--out_dir", required=True, help="Output directory (run_dir)")
    parser.add_argument("--max_iters", type=int, default=3, help="Max training iterations")

    # Keep old flag names where possible
    parser.add_argument("--task", default="sft_text", help="Task name (currently unused)")
    parser.add_argument("--hf_model_id", default=None, help="Base HF model id")
    parser.add_argument("--hf_model_id_override", default=None, help="Override HF model id (minimal_agentic)")
    parser.add_argument("--search_trials", type=int, default=0, help="Random search trials before training")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to N samples (for testing)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps (workflow mode)")
    parser.add_argument("--max_steps_override", type=int, default=None, help="Override max training steps (minimal_agentic)")
    parser.add_argument(
        "--mode",
        default="minimal_agentic",
        choices=["workflow", "minimal_agentic"],
        help="Execution mode",
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

    cfg = {
        "mode": args.mode,
        "data_path": args.data_path,
        "run_dir": args.out_dir,
        "max_iters": args.max_iters,
        "search_trials": args.search_trials,
        "max_samples": args.max_samples,
    }

    if args.hf_model_id:
        cfg["hf_model_id"] = args.hf_model_id
    if args.hf_model_id_override:
        cfg["hf_model_id_override"] = args.hf_model_id_override

    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.max_steps_override is not None:
        cfg["max_steps_override"] = args.max_steps_override

    result = Orchestrator(cfg).run()
    logging.getLogger(__name__).info("Completed. Report: %s", result.get("report_path"))


if __name__ == "__main__":
    main()

