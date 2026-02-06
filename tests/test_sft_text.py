#!/usr/bin/env python3
"""
Test script for BuildSftDatasetTool (text mode).

Usage:
    python scripts/test_sft_text.py

This will:
1. Load text files from test_data/texts/
2. Annotate them using the LLM (knowledge distillation)
3. Build standalone SFT examples (no source text in training)
4. Save annotations and SFT examples to test_outputs/
5. Print comprehensive logs
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

from shared.logging_config import setup_logging
from tools.build_sft_dataset.tool import BuildSftDatasetTool

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR = project_root/ "tests" / "test_data" / "texts"
OUTPUT_DIR = project_root / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ANNOTATIONS_PATH = OUTPUT_DIR / f"sft_text_annotations_{TIMESTAMP}.jsonl"
SFT_PATH = OUTPUT_DIR / f"sft_text_{TIMESTAMP}.jsonl"


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST: BuildSftDatasetTool (TEXT mode)")
    logger.info("=" * 80)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output annotations: {ANNOTATIONS_PATH}")
    logger.info(f"Output SFT examples: {SFT_PATH}")
    logger.info("")

    if not INPUT_DIR.exists():
        logger.error(f"❌ Input directory does not exist: {INPUT_DIR}")
        logger.error("Please create it and add some .txt files")
        return

    tool = BuildSftDatasetTool()

    config = {
        "mode": "text",
        "input_dir": str(INPUT_DIR),
        "output_annotations": str(ANNOTATIONS_PATH),
        "output_sft": str(SFT_PATH),
        "target_language": "English",
        "batch_size": 3,
        "batch_delay": 1.5,
    }

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("")

    # Execute
    logger.info("Executing SFT dataset generation (knowledge distillation)...")
    result = tool.execute(config)

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Items loaded: {result['num_items']}")
    logger.info(f"Annotations succeeded: {result['num_annotations']}")
    logger.info(f"Annotations failed: {result['num_failures']}")
    logger.info(f"SFT examples generated: {result['num_examples']}")
    logger.info("")
    logger.info(f"✓ Annotations saved to: {result['annotations_path']}")
    logger.info(f"✓ SFT examples saved to: {result['sft_path']}")
    logger.info("")

    # Show sample SFT examples
    if result["num_examples"] > 0:
        logger.info("Sample SFT examples (first 3):")
        with open(result["sft_path"], "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i > 3:
                    break
                example = json.loads(line)
                user_msg = example["messages"][0]["content"]
                assistant_msg = example["messages"][1]["content"]
                logger.info(f"  Example {i}:")
                logger.info(f"    User: {user_msg[:150]}...")
                logger.info(f"    Assistant: {assistant_msg[:150]}...")
                logger.info("")

    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
