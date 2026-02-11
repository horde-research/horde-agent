#!/usr/bin/env python3
"""
Test script for GenerateTaxonomyTool with early stopping.

Usage:
    python scripts/test_generate_taxonomy.py

This will:
1. Generate categories for a test country/culture
2. Process only the first N categories (early stopping)
3. Generate subcategories for those categories
4. Generate keywords for all subcategories
5. Save results to test_outputs/taxonomy_<timestamp>.json
6. Print comprehensive logs
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
from tools.generate_taxonomy.tool import GenerateTaxonomyTool

# ─── Configuration ────────────────────────────────────────────────────────────

TEST_COUNTRY = "Kazakhstan"
EARLY_STOP_CATEGORIES = 2  # Process only first 2 categories
EARLY_STOP_SUBCATEGORIES = None  # None = process all subcategories

OUTPUT_DIR = project_root / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("TEST: GenerateTaxonomyTool with early stopping")
    logger.info("=" * 80)
    logger.info(f"Country/Culture: {TEST_COUNTRY}")
    logger.info(f"Early stop: process first {EARLY_STOP_CATEGORIES} categories")
    logger.info("")

    tool = GenerateTaxonomyTool()

    # Execute taxonomy generation
    logger.info("Step 1: Generating categories...")
    result = tool.execute(
        country_or_culture=TEST_COUNTRY,
        config={
            "batch_size": 3,
            "batch_delay": 1.0,
        },
    )

    categories = result["categories"]
    logger.info(f"✓ Generated {len(categories)} categories total")
    logger.info("")

    # Early stopping: keep only first N categories
    if EARLY_STOP_CATEGORIES and len(categories) > EARLY_STOP_CATEGORIES:
        logger.info(f"⚠ EARLY STOPPING: Keeping only first {EARLY_STOP_CATEGORIES} categories")
        categories = categories[:EARLY_STOP_CATEGORIES]
        # Filter subcategories dict to match
        category_subcategories = {
            cat["name"]: result["category_subcategories"].get(cat["name"], [])
            for cat in categories
        }
        category_subcategory_keywords = {
            cat["name"]: result["category_subcategory_keywords"].get(cat["name"], {})
            for cat in categories
        }
    else:
        category_subcategories = result["category_subcategories"]
        category_subcategory_keywords = result["category_subcategory_keywords"]

    # Print summary
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Categories processed: {len(categories)}")
    logger.info("")

    total_subcategories = 0
    total_keywords = 0

    for cat in categories:
        cat_name = cat["name"]
        subs = category_subcategories.get(cat_name, [])
        total_subcategories += len(subs)
        logger.info(f"Category: {cat_name}")
        logger.info(f"  Description: {cat.get('description', 'N/A')[:80]}...")
        logger.info(f"  Subcategories: {len(subs)}")
        for sub in subs:
            sub_name = sub["name"]
            keywords = category_subcategory_keywords.get(cat_name, {}).get(sub_name, [])
            total_keywords += len(keywords)
            logger.info(f"    - {sub_name}: {len(keywords)} keywords")
            if keywords:
                logger.info(f"      Sample keywords: {', '.join(keywords[:3])}...")
        logger.info("")

    logger.info("=" * 80)
    logger.info("TOTALS")
    logger.info("=" * 80)
    logger.info(f"Categories: {len(categories)}")
    logger.info(f"Subcategories: {total_subcategories}")
    logger.info(f"Keywords: {total_keywords}")
    logger.info("")

    # Save full results
    output_data = {
        "country_or_culture": TEST_COUNTRY,
        "timestamp": datetime.now().isoformat(),
        "early_stop_categories": EARLY_STOP_CATEGORIES,
        "categories": categories,
        "category_subcategories": category_subcategories,
        "category_subcategory_keywords": category_subcategory_keywords,
        "stats": {
            "num_categories": len(categories),
            "num_subcategories": total_subcategories,
            "num_keywords": total_keywords,
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"taxonomy_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved full results to: {output_path}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)

    return output_data


if __name__ == "__main__":
    main()
