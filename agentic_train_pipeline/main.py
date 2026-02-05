"""CLI entrypoint."""

import argparse
import logging
import sys

from agentic_train_pipeline.pipeline import run_pipeline


def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic LoRA SFT pipeline")
    parser.add_argument("--data_path", required=True, help="HF dataset id or local dataset path")
    parser.add_argument("--task", default="sft_text", help="Task name (default: sft_text)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--max_iters", type=int, default=3, help="Max training iterations")
    parser.add_argument("--hf_model_id", default=None, help="Override HF model id")
    parser.add_argument("--search_trials", type=int, default=0, help="Random search trials before training")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset to N samples (for testing)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("HORDE AGENT - Agentic LLM Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Max iterations: {args.max_iters}")
    logger.info(f"Model override: {args.hf_model_id or 'Auto-select'}")
    logger.info(f"Search trials: {args.search_trials}")
    if args.max_samples:
        logger.info(f"Dataset limit: {args.max_samples} samples (TESTING MODE)")
    if args.max_steps:
        logger.info(f"Max training steps: {args.max_steps}")
    logger.info("=" * 80)
    
    try:
        report_path = run_pipeline(
            data_path=args.data_path,
            task=args.task,
            out_dir=args.out_dir,
            max_iters=args.max_iters,
            hf_model_id_override=args.hf_model_id,
            search_trials=args.search_trials,
            max_samples=args.max_samples,
            max_steps_override=args.max_steps,
        )
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Final report: {report_path}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED!")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
