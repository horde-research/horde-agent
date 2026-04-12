"""
Collect Kazakhstan cultural SFT data and save to disk.

Pipeline:
  1. GenerateTaxonomyTool  — categories → subcategories → search queries
  2. CollectDataTool        — scrape web pages via Serper API
  3. BuildSftDatasetTool    — LLM annotation → SFT pairs (JSONL)

Output:
  <run_dir>/
    taxonomy.json       — generated topic taxonomy
    raw/                — scraped pages (one JSON per query)
    sft/sft.jsonl       — final SFT dataset

Usage:
  # Full run (requires API keys)
  python experiments/collect_kazakhstan.py --run_dir ./output/kz_run

  # Dry-run (no API keys, writes fake data to disk)
  python experiments/collect_kazakhstan.py --dry_run

Requirements:
  LLM_API_KEY or OPENAI_API_KEY
  SERPER_API_KEY
  pip install httpx aiohttp pydantic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_kazakhstan")

# Fixed LLM settings for this script (as requested): always OpenAI gpt-4o.
OPENAI_API_KEY_FIXED = "[REDACTED_OPENAI_KEY]"
LLM_PROVIDER_FIXED = "openai"
LLM_MODEL_FIXED = "gpt-4o"

def _fallback_queries(country: str) -> List[str]:
    """Fallback queries when taxonomy generation fails or returns no queries."""
    c = country.strip() or "Kazakhstan"
    return [
        f"{c} culture traditions",
        f"{c} traditional food",
        f"{c} folklore",
        f"{c} music dombra",
        f"{c} nauryz",
        f"{c} history nomad",
        f"{c} tourism landmarks",
        f"{c} crafts and art",
        f"{c} national clothing",
        f"{c} customs and rituals",
        f"{c} қазақ мәдениеті",
        f"{c} қазақтың дәстүрлері",
        f"{c} ұлттық тағамдар",
        f"{c} домбыра күй",
        f"{c} Наурыз мерекесі",
        f"{c} киіз үй тарихы",
        f"{c} салт-дәстүр",
        f"{c} мәдени мұра",
    ]

# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    country: str,
    run_dir: str,
    language: str = "Kazakh",
    max_queries: Optional[int] = None,
    max_samples: Optional[int] = None,
    serper_results_per_query: int = 5,
    serper_top_results: int = 3,
) -> Path:
    """
    Run the full data collection pipeline and return the path to sft.jsonl.
    """
    from tools.generate_taxonomy.tool import GenerateTaxonomyTool
    from tools.collect_data.tool import CollectDataTool
    from tools.build_sft_dataset.tool import BuildSftDatasetTool

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Taxonomy ─────────────────────────────────────────────────────
    # GenerateTaxonomyTool.execute(country_or_culture, config=None)
    # Returns: {
    #   "categories": [{"name": ..., "description": ...}, ...],
    #   "category_subcategories": {cat_name: [sub_name, ...], ...},
    #   "category_subcategory_queries": {cat_name: {sub_name: [query, ...], ...}, ...}
    # }
    logger.info("Step 1/3 — Generating taxonomy for %s...", country)
    taxonomy: Dict[str, Any]
    try:
        taxonomy = GenerateTaxonomyTool().execute(
            country_or_culture=country,
            config={
                "provider": LLM_PROVIDER_FIXED,
                "model": LLM_MODEL_FIXED,
                "api_key": OPENAI_API_KEY_FIXED,
            },
        )
    except Exception as exc:
        logger.warning("Taxonomy generation failed: %s", exc)
        taxonomy = {
            "categories": [],
            "category_subcategories": {},
            "category_subcategory_queries": {},
            "error": str(exc),
        }

    taxonomy_path = run_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Taxonomy saved → %s", taxonomy_path)

    queries: List[str] = []
    for cat_queries in taxonomy.get("category_subcategory_queries", {}).values():
        for sub_queries in cat_queries.values():
            queries.extend(sub_queries)

    # Fallback when LLM taxonomy fails or returns empty
    if not queries:
        queries = _fallback_queries(country)
        logger.warning(
            "Taxonomy produced 0 queries. Using %d fallback queries.",
            len(queries),
        )

    # Deduplicate while preserving order
    queries = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))

    if max_queries:
        queries = queries[:max_queries]

    logger.info("Total search queries: %d", len(queries))
    if not queries:
        raise RuntimeError("No queries available after fallback/deduplication.")

    # ── Step 2: Collect ──────────────────────────────────────────────────────
    # CollectDataTool.execute(config: dict)
    # config keys: queries, run_dir, google_results_per_query, top_results, size
    # Returns: {"data_path": str, "num_samples": int, ...}
    logger.info("Step 2/3 — Collecting web data (%d queries)...", len(queries))
    raw_dir = run_path / "raw"

    collect_config: Dict[str, Any] = {
        "queries": queries,
        "run_dir": str(raw_dir),
        "google_results_per_query": serper_results_per_query,
        "top_results": serper_top_results,
    }
    if max_samples:
        collect_config["size"] = max_samples

    collect_result = CollectDataTool().execute(collect_config)
    logger.info("Collected %d pages → %s", collect_result["num_samples"], collect_result["data_path"])

    # CollectDataTool stores an HF Dataset on disk (.arrow files).
    # BuildSftDatasetTool(text mode) expects input_jsonl or a directory of .txt files.
    # Convert HF dataset -> JSONL with {"text": ...} rows before annotation.
    collected_jsonl = run_path / "sft" / "collected_texts.jsonl"
    collected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _export_hf_dataset_to_jsonl(collect_result["data_path"], str(collected_jsonl))
    logger.info("Exported collected HF dataset to JSONL → %s", collected_jsonl)

    # ── Step 3: Build SFT dataset ────────────────────────────────────────────
    # BuildSftDatasetTool.execute(config: dict)
    # config keys: mode, input_dir (HF dataset dir from step 2), output_sft,
    #              target_language, batch_size, batch_delay
    # Returns: {"sft_path": str, "num_examples": int, ...}
    logger.info("Step 3/3 — Annotating to SFT pairs...")
    sft_dir = run_path / "sft"
    sft_dir.mkdir(exist_ok=True)
    sft_path = sft_dir / "sft.jsonl"
    annotations_path = sft_dir / "annotations.jsonl"

    sft_config: Dict[str, Any] = {
        "mode": "text",
        "input_jsonl": str(collected_jsonl),
        "text_field": "text",
        "output_sft": str(sft_path),
        "output_annotations": str(annotations_path),
        "target_language": language,
        "provider": LLM_PROVIDER_FIXED,
        "model": LLM_MODEL_FIXED,
        "api_key": OPENAI_API_KEY_FIXED,
    }
    sft_result = BuildSftDatasetTool().execute(sft_config)
    logger.info(
        "SFT dataset saved → %s  (%d examples)", sft_path, sft_result["num_examples"]
    )
    return sft_path


def _export_hf_dataset_to_jsonl(hf_dataset_path: str, jsonl_path: str) -> None:
    import json
    from datasets import load_from_disk

    ds = load_from_disk(hf_dataset_path)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Dry-run (no API keys)
# ════════════════════════════════════════════════════════════════════════════

_DRY_RUN_EXAMPLES = [
    {"messages": [
        {"role": "user", "content": "What is beshbarmak and what is its cultural significance?"},
        {"role": "assistant", "content": (
            "Beshbarmak (бешбармақ) is the national dish of Kazakhstan, made from boiled horse "
            "or lamb meat served over flat noodles. The name means 'five fingers' in Kazakh, "
            "referring to the tradition of eating with hands. It is served at weddings, funerals, "
            "and celebrations, symbolising communal unity."
        )},
    ]},
    {"messages": [
        {"role": "user", "content": "How is kumis prepared?"},
        {"role": "assistant", "content": (
            "Kumis (қымыз) is a fermented beverage made from raw mare's milk, prepared in a "
            "leather vessel called a saba. It has 1–3% alcohol content and is considered a "
            "health drink central to Kazakh nomadic hospitality."
        )},
    ]},
    {"messages": [
        {"role": "user", "content": "What is the dombra?"},
        {"role": "assistant", "content": (
            "The dombra (домбыра) is a two-stringed plucked instrument and the national "
            "instrument of Kazakhstan. It is central to the küy (instrumental composition) "
            "tradition performed by akyns (oral poet-musicians)."
        )},
    ]},
    {"messages": [
        {"role": "user", "content": "Explain the tradition of Nauryz in Kazakhstan."},
        {"role": "assistant", "content": (
            "Nauryz (Наурыз) is the Persian New Year celebrated on March 21–22, marking the "
            "spring equinox. Celebrations include traditional games, dombra music, communal "
            "feasts with nauryz-kozhe (a soup of seven ingredients), and ceremonial yurts. "
            "The number seven symbolises prosperity in Kazakh culture."
        )},
    ]},
    {"messages": [
        {"role": "user", "content": "Why has the yurt remained a cultural symbol in Kazakhstan?"},
        {"role": "assistant", "content": (
            "The yurt (киіз үй) served as the primary dwelling of Kazakh nomads for over a "
            "millennium. Its circular design symbolises the nomadic worldview. Even as over 55% "
            "of Kazakhstanis live in cities, the yurt appears at every state ceremony and Nauryz "
            "celebration as a deliberate act of cultural preservation."
        )},
    ]},
    {"messages": [
        {"role": "user", "content": "What are the main traditional Kazakh clothing items?"},
        {"role": "assistant", "content": (
            "Traditional clothing includes the shapan (long embroidered robe), takiya (skull cap), "
            "saukele (tall ceremonial bridal headdress), and kymeshek (head covering for married "
            "women). Garments were made from wool, felt, and leather suited to steppe life."
        )},
        {"role": "user", "content": "What is the significance of the saukele specifically?"},
        {"role": "assistant", "content": (
            "The saukele is a conical wedding headdress up to 70 cm tall, decorated with gold, "
            "silver, feathers, and precious stones. It is worn by the bride on her wedding day "
            "and indicates the family's social status. Crafting one required months of work."
        )},
    ]},
]


def run_dry(run_dir: str) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # Fake taxonomy
    taxonomy = {
        "categories": [
            {"name": "cuisine", "subcategories": [
                {"name": "traditional dishes", "keywords": ["beshbarmak Kazakhstan", "қазақ тағамдары"]},
            ]},
            {"name": "music", "subcategories": [
                {"name": "instruments", "keywords": ["dombra Kazakh instrument", "домбыра музыка"]},
            ]},
            {"name": "celebrations", "subcategories": [
                {"name": "holidays", "keywords": ["Nauryz celebration", "Наурыз мерекесі"]},
            ]},
        ]
    }
    taxonomy_path = run_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[dry_run] Taxonomy written → %s", taxonomy_path)

    # SFT dataset
    sft_dir = run_path / "sft"
    sft_dir.mkdir(exist_ok=True)
    sft_path = sft_dir / "sft.jsonl"

    with open(sft_path, "w", encoding="utf-8") as f:
        for ex in _DRY_RUN_EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("[dry_run] SFT dataset written → %s  (%d examples)", sft_path, len(_DRY_RUN_EXAMPLES))
    return sft_path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Kazakh cultural SFT data and save to JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run_dir",
        default="./output/kz_run",
        help="Output directory (default: ./output/kz_run).",
    )
    parser.add_argument(
        "--country",
        default="Kazakhstan",
        help="Country/culture for taxonomy (default: Kazakhstan).",
    )
    parser.add_argument(
        "--language",
        default="Kazakh",
        help="Target language for SFT pairs (default: Kazakh).",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Max search queries to send to Serper (default: all).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max scraped pages to annotate (default: all).",
    )
    parser.add_argument(
        "--serper_results",
        type=int,
        default=5,
        help="Serper results per query (default: 5).",
    )
    parser.add_argument(
        "--serper_top",
        type=int,
        default=3,
        help="Top pages to scrape per query (default: 3).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write fake data to disk without any API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        sft_path = run_dry(args.run_dir)
    else:
        _check_env_vars()
        sft_path = run_pipeline(
            country=args.country,
            run_dir=args.run_dir,
            language=args.language,
            max_queries=args.max_queries,
            max_samples=args.max_samples,
            serper_results_per_query=args.serper_results,
            serper_top_results=args.serper_top,
        )

    print(f"\nDataset saved: {sft_path.resolve()}")
    print(f"Lines:         {sum(1 for _ in open(sft_path, encoding='utf-8'))}")


def _check_env_vars() -> None:
    missing = []
    if not OPENAI_API_KEY_FIXED.strip():
        missing.append("OPENAI_API_KEY_FIXED (in this file)")
    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")
    if missing:
        logger.error("Missing required environment variables:\n  %s", "\n  ".join(missing))
        logger.error("Set them in your shell or copy env.example → .env and run: source .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
