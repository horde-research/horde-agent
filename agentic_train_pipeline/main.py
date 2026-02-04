"""CLI entrypoint."""

import argparse

from agentic_train_pipeline.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic LoRA SFT pipeline")
    parser.add_argument("--data_path", required=True, help="HF dataset id or local dataset path")
    parser.add_argument("--task", default="sft_text", help="Task name (default: sft_text)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--max_iters", type=int, default=3, help="Max training iterations")
    parser.add_argument("--hf_model_id", default=None, help="Override HF model id")
    parser.add_argument("--search_trials", type=int, default=0, help="Random search trials before training")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        task=args.task,
        out_dir=args.out_dir,
        max_iters=args.max_iters,
        hf_model_id_override=args.hf_model_id,
        search_trials=args.search_trials,
    )


if __name__ == "__main__":
    main()
