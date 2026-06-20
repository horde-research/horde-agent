"""
Taxonomy generation tool.

Generates categories -> subcategories -> search queries from a country/culture name.
All LLM calls are batched via core.llm.LLMClient.
"""

import logging
from typing import Any, Dict, Optional

from core.llm import LLMClient
from tools.base_tool import BaseTool
from tools.generate_taxonomy.agents import CategoryAgent, SubcategoryAgent, QueryAgent
from tools.generate_taxonomy.image_taxonomy import build_image_taxonomy
from tools.generate_taxonomy.quality import (
    build_taxonomy_quality,
    infer_culture_profile,
    validate_categories,
    validate_query_groups,
    validate_subcategories,
)

logger = logging.getLogger(__name__)


class GenerateTaxonomyTool(BaseTool):
    """
    Generates a taxonomy (categories -> subcategories -> search queries) from
    a country or culture name.

    Pipeline:
        1. CategoryAgent    -- extract top-level categories           (1 request)
        2. SubcategoryAgent -- break each category into subcategories (batched)
        3. QueryAgent       -- generate search queries per subcategory (batched)
    """

    def execute(
        self,
        country_or_culture: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not country_or_culture or not country_or_culture.strip():
            raise ValueError("country_or_culture is required and must not be empty.")

        config = config or {}
        focus = str(config.get("focus") or "").strip()
        batch_size = config.get("batch_size", 5)
        batch_delay = config.get("batch_delay", 1.5)
        enable_mini_loop = bool(config.get("enable_taxonomy_mini_loop", True))
        repair_attempt_limit = int(config.get("taxonomy_repair_attempts", 2))
        min_categories = int(config.get("taxonomy_min_categories", 2))
        max_categories = int(config.get("taxonomy_max_categories", 20))
        min_subcategories = int(config.get("taxonomy_min_subcategories", 1))
        max_subcategories = int(config.get("taxonomy_max_subcategories", 12))
        min_queries = int(config.get("taxonomy_min_queries", 1))
        max_queries = int(config.get("taxonomy_max_queries", 20))
        enable_image_taxonomy = bool(config.get("enable_image_taxonomy", True))
        image_queries_per_slot = int(config.get("image_taxonomy_queries_per_slot", 4))
        image_max_slots = config.get("image_taxonomy_max_slots")
        image_max_slots = int(image_max_slots) if image_max_slots else None

        client = LLMClient.from_env(
            provider=config.get("provider"),
            model=config.get("model"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.7),
        )
        category_agent = CategoryAgent(client)
        subcategory_agent = SubcategoryAgent(client)
        query_agent = QueryAgent(client)
        culture_profile = infer_culture_profile(country_or_culture)
        repair_attempts = []

        # Step 1 -- categories (single request)
        logger.info("Step 1/3: Extracting categories for '%s'...", country_or_culture)
        categories = category_agent.extract_categories(country_or_culture, focus=focus)
        category_report = validate_categories(
            categories,
            min_categories=min_categories,
            max_categories=max_categories,
        )
        if enable_mini_loop:
            for attempt_idx in range(repair_attempt_limit):
                if category_report["passed"]:
                    break
                previous_count = len(categories)
                categories = category_agent.refine_categories(
                    country_or_culture,
                    categories,
                    category_report,
                    culture_profile,
                    focus=focus,
                )
                repair_attempts.append(
                    {
                        "stage": "categories",
                        "attempt": attempt_idx + 1,
                        "before_count": previous_count,
                        "after_count": len(categories),
                        "issues": list(category_report["blocking_issues"]),
                    }
                )
                category_report = validate_categories(
                    categories,
                    min_categories=min_categories,
                    max_categories=max_categories,
                )
        logger.info("Extracted %d categories.", len(categories))

        # Step 2 -- subcategories (batched)
        logger.info("Step 2/3: Generating subcategories for %d categories...", len(categories))
        category_subcategories = subcategory_agent.generate_for_categories(
            categories, country_or_culture,
            batch_size=batch_size, batch_delay=batch_delay, focus=focus,
        )
        subcategory_report = validate_subcategories(
            categories,
            category_subcategories,
            min_subcategories_per_category=min_subcategories,
            max_subcategories_per_category=max_subcategories,
        )
        if enable_mini_loop:
            for attempt_idx in range(repair_attempt_limit):
                failed_categories = list(subcategory_report["failed_categories"])
                if not failed_categories:
                    break
                for category in categories:
                    category_name = category["name"]
                    if category_name not in failed_categories:
                        continue
                    existing = category_subcategories.get(category_name, [])
                    repaired = subcategory_agent.repair_for_category(
                        category,
                        existing,
                        country_or_culture,
                        subcategory_report["per_category"].get(category_name, {}),
                        culture_profile,
                        focus=focus,
                    )
                    category_subcategories[category_name] = repaired
                repair_attempts.append(
                    {
                        "stage": "subcategories",
                        "attempt": attempt_idx + 1,
                        "repaired_categories": failed_categories,
                    }
                )
                subcategory_report = validate_subcategories(
                    categories,
                    category_subcategories,
                    min_subcategories_per_category=min_subcategories,
                    max_subcategories_per_category=max_subcategories,
                )
        total_subs = sum(len(v) for v in category_subcategories.values())
        logger.info("Generated %d subcategories total.", total_subs)

        # Step 3 -- search queries (batched)
        logger.info("Step 3/3: Generating search queries for %d subcategories...", total_subs)
        category_subcategory_queries = query_agent.generate_for_subcategories(
            categories, category_subcategories, country_or_culture,
            batch_size=batch_size, batch_delay=batch_delay, focus=focus,
        )
        query_report = validate_query_groups(
            categories,
            category_subcategories,
            category_subcategory_queries,
            min_queries_per_subcategory=min_queries,
            max_queries_per_subcategory=max_queries,
            culture_aliases=culture_profile.get("common_aliases", []),
        )
        if enable_mini_loop:
            for attempt_idx in range(repair_attempt_limit):
                failed_query_groups = list(query_report["failed_query_groups"])
                if not failed_query_groups:
                    break
                for failed_group in failed_query_groups:
                    category_name = failed_group["category"]
                    subcategory_name = failed_group["subcategory"]
                    category = _find_by_name(categories, category_name)
                    subcategory = _find_by_name(category_subcategories.get(category_name, []), subcategory_name)
                    if not category or not subcategory:
                        continue
                    existing = category_subcategory_queries.get(category_name, {}).get(subcategory_name, [])
                    repaired = query_agent.repair_for_subcategory(
                        category,
                        subcategory,
                        existing,
                        country_or_culture,
                        query_report["per_group"].get(f"{category_name}||{subcategory_name}", {}),
                        culture_profile,
                        focus=focus,
                    )
                    category_subcategory_queries.setdefault(category_name, {})[subcategory_name] = repaired
                repair_attempts.append(
                    {
                        "stage": "queries",
                        "attempt": attempt_idx + 1,
                        "repaired_query_groups": failed_query_groups,
                    }
                )
                query_report = validate_query_groups(
                    categories,
                    category_subcategories,
                    category_subcategory_queries,
                    min_queries_per_subcategory=min_queries,
                    max_queries_per_subcategory=max_queries,
                    culture_aliases=culture_profile.get("common_aliases", []),
                )
        total_queries = sum(
            len(qs) for sub_dict in category_subcategory_queries.values()
            for qs in sub_dict.values()
        )
        logger.info("Generated %d search queries total.", total_queries)
        taxonomy_quality = build_taxonomy_quality(
            culture_profile=culture_profile,
            category_report=category_report,
            subcategory_report=subcategory_report,
            query_report=query_report,
            repair_attempts=repair_attempts,
        )
        image_taxonomy = (
            build_image_taxonomy(
                country_or_culture,
                culture_profile,
                queries_per_slot=image_queries_per_slot,
                max_slots=image_max_slots,
                focus=focus,
            )
            if enable_image_taxonomy
            else None
        )

        output = {
            "categories": categories,
            "category_subcategories": category_subcategories,
            "category_subcategory_queries": category_subcategory_queries,
            "taxonomy_quality": taxonomy_quality,
        }
        if focus:
            output["focus"] = focus
        if image_taxonomy:
            output["image_taxonomy"] = image_taxonomy
        return output


def _find_by_name(items, name):
    for item in items:
        if item.get("name") == name:
            return item
    return None
