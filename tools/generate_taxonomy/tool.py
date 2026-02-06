"""
Taxonomy generation tool.

Generates categories → subcategories → keywords from a country/culture name.
All LLM calls are batched via core.llm.LLMClient.
"""

import logging
from typing import Any, Dict, Optional

from core.llm import LLMClient
from tools.base_tool import BaseTool
from tools.generate_taxonomy.agents import CategoryAgent, SubcategoryAgent, KeywordAgent

logger = logging.getLogger(__name__)


class GenerateTaxonomyTool(BaseTool):
    """
    Generates a taxonomy (categories → subcategories → keywords) from
    a country or culture name.

    Pipeline:
        1. CategoryAgent    — extract top-level categories          (1 request)
        2. SubcategoryAgent — break each category into subcategories (batched)
        3. KeywordAgent     — generate search keywords per subcategory (batched)
    """

    def execute(
        self,
        country_or_culture: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate taxonomy.

        Args:
            country_or_culture: Required. e.g. "Kazakhstan", "Japanese culture".
            config: Optional overrides:
                - provider:   str  (openai | gemini | xai)
                - model:      str
                - api_key:    str
                - batch_size: int  (default 5)
                - batch_delay: float (default 1.5)

        Returns:
            dict: {
                'categories': list[dict],
                'category_subcategories': dict[str, list[dict]],
                'category_subcategory_keywords': dict[str, dict[str, list[str]]],
            }
        """
        if not country_or_culture or not country_or_culture.strip():
            raise ValueError("country_or_culture is required and must not be empty.")

        config = config or {}
        batch_size = config.get("batch_size", 5)
        batch_delay = config.get("batch_delay", 1.5)

        client = LLMClient.from_env(
            provider=config.get("provider"),
            model=config.get("model"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.7),
        )

        category_agent = CategoryAgent(client)
        subcategory_agent = SubcategoryAgent(client)
        keyword_agent = KeywordAgent(client)

        # Step 1 — categories (single request)
        logger.info("Step 1/3: Extracting categories for '%s'...", country_or_culture)
        categories = category_agent.extract_categories(country_or_culture)
        logger.info("Extracted %d categories.", len(categories))

        # Step 2 — subcategories (batched)
        logger.info("Step 2/3: Generating subcategories for %d categories...", len(categories))
        category_subcategories = subcategory_agent.generate_for_categories(
            categories, country_or_culture,
            batch_size=batch_size, batch_delay=batch_delay,
        )
        total_subs = sum(len(v) for v in category_subcategories.values())
        logger.info("Generated %d subcategories total.", total_subs)

        # Step 3 — keywords (batched)
        logger.info("Step 3/3: Generating keywords for %d subcategories...", total_subs)
        category_subcategory_keywords = keyword_agent.generate_for_subcategories(
            categories, category_subcategories, country_or_culture,
            batch_size=batch_size, batch_delay=batch_delay,
        )
        total_kws = sum(
            len(kws) for sub_dict in category_subcategory_keywords.values()
            for kws in sub_dict.values()
        )
        logger.info("Generated %d keywords total.", total_kws)

        return {
            "categories": categories,
            "category_subcategories": category_subcategories,
            "category_subcategory_keywords": category_subcategory_keywords,
        }
