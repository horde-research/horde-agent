#!/usr/bin/env python3
"""
Test script for BuildSftDatasetTool (image mode).

Usage:
    python scripts/test_sft_images.py

This will:
1. Load images from test_data/images/
2. Annotate them using the LLM
3. Build SFT examples
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

from shared.logging_config import setup_logging
from tools.build_sft_dataset.tool import BuildSftDatasetTool

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR = project_root / "test_data" / "images"
OUTPUT_DIR = project_root / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ANNOTATIONS_PATH = OUTPUT_DIR / f"sft_images_annotations_{TIMESTAMP}.jsonl"
SFT_PATH = OUTPUT_DIR / f"sft_images_{TIMESTAMP}.jsonl"


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST: BuildSftDatasetTool (IMAGE mode)")
    logger.info("=" * 80)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output annotations: {ANNOTATIONS_PATH}")
    logger.info(f"Output SFT examples: {SFT_PATH}")
    logger.info("")

    if not INPUT_DIR.exists():
        logger.error(f"❌ Input directory does not exist: {INPUT_DIR}")
        logger.error("Please create it and add some test images (.jpg, .png, .webp)")
        return

    tool = BuildSftDatasetTool()

    config = {
        "mode": "image",
        "input_dir": str(INPUT_DIR),
        "image_exts": [".jpg", ".jpeg", ".png", ".webp"],
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
    logger.info("Executing SFT dataset generation...")
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
        logger.info("Sample SFT examples (first 2):")
        with open(result["sft_path"], "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i > 2:
                    break
                example = json.loads(line)
                user_content = example["messages"][0]["content"]
                assistant_content = example["messages"][1]["content"]
                logger.info(f"  Example {i}:")
                logger.info(f"    User: {str(user_content)[:100]}...")
                logger.info(f"    Assistant: {str(assistant_content)[:100]}...")
                logger.info("")

    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
