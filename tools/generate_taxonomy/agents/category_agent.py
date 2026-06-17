"""Agent for extracting cultural categories from free text."""

import json
import logging
from typing import Any, Dict, List

from core.llm import LLMClient, LLMRequest

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert cultural anthropologist and data analyst specializing in comprehensive cultural documentation and analysis.

Your task is to generate detailed, comprehensive categories for documenting and understanding a specific culture, country, or region. You should create categories that capture the full spectrum of cultural expression, social practices, traditions, and contemporary trends.

Output a JSON object with the following structure:
{
    "categories": [
        {
            "name": "category_name_in_english",
            "description": "detailed, comprehensive description of what this category represents and what types of data it encompasses"
        }
    ]
}

You must be thorough and comprehensive. Generate categories that cover:

1. **Traditional Customs & Practices**:
   - Traditional ceremonies, rituals, and celebrations
   - Folk traditions and cultural practices
   - Religious and spiritual customs
   - Traditional crafts and artisanal work
   - Historical cultural practices

2. **Social Structures & Daily Life**:
   - Family structures and relationships
   - Social hierarchies and community organization
   - Daily routines and lifestyle patterns
   - Social gatherings and community events
   - Interpersonal relationships and social norms

3. **Cultural Arts & Expression**:
   - Traditional music, dance, and performing arts
   - Visual arts, crafts, and traditional design
   - Literature, poetry, and oral traditions
   - Cultural symbols, motifs, and iconography
   - Contemporary artistic expressions

4. **Cuisine & Food Culture**:
   - Traditional dishes and recipes
   - Food preparation methods and techniques
   - Dining customs and etiquette
   - Food-related celebrations and traditions
   - Regional culinary variations

5. **Clothing & Fashion**:
   - Traditional attire and costumes
   - Cultural significance of clothing
   - Contemporary fashion trends
   - Textiles and fabric traditions
   - Accessories and adornments

6. **Architecture & Living Spaces**:
   - Traditional architecture and building styles
   - Interior design and home organization
   - Urban and rural living environments
   - Cultural significance of spaces
   - Modern architectural trends

7. **Language & Communication**:
   - Language use and dialects
   - Communication styles and etiquette
   - Written and oral traditions
   - Contemporary language trends
   - Multilingual aspects

8. **Recent Trends & Modern Culture**:
   - Contemporary social movements
   - Modern lifestyle trends
   - Technology integration in culture
   - Youth culture and generational differences
   - Globalization influences

9. **Geographical & Environmental Context**:
   - Landscape and natural environment
   - Urban vs rural distinctions
   - Regional variations within the culture
   - Environmental practices and traditions
   - Seasonal cultural variations

10. **Economic & Professional Life**:
    - Traditional occupations and trades
    - Modern professional environments
    - Economic activities and markets
    - Work-life balance and practices
    - Entrepreneurship and innovation

Generate 8-15 comprehensive categories that together provide a complete picture of the culture. Each category should be specific enough to be actionable for data collection, yet broad enough to encompass related sub-topics.

Return only valid JSON."""


class CategoryAgent:
    """Extracts comprehensive cultural categories from free text."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def extract_categories(self, country_or_culture: str) -> List[Dict[str, str]]:
        """
        Extract comprehensive categories for a specific country or culture.

        Args:
            country_or_culture: e.g. "Kazakhstan", "Japanese culture"

        Returns:
            List of ``{"name": ..., "description": ...}`` dicts.
        """
        user_message = (
            f"Generate comprehensive data categories for documenting and "
            f"understanding the culture, customs, traditions, and recent trends "
            f"of: {country_or_culture}\n\n"
            f"Create detailed categories that would enable comprehensive data "
            f"collection covering traditional customs, social structures, "
            f"cultural arts, cuisine, clothing, architecture, language, "
            f"recent trends, geography, and economic life.\n\n"
            f"Provide a thorough, well-organized set of categories that "
            f"capture the full spectrum of this culture."
        )

        request = LLMRequest(
            request_id="categories",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
        )
        resp = self.client.generate_json_sync(request)
        if not resp.success:
            logger.error("Category extraction failed: %s", resp.error)
            return []

        categories = resp.data.get("categories", [])
        logger.info("Extracted %d categories for '%s'.", len(categories), country_or_culture)
        return categories

    def refine_categories(
        self,
        country_or_culture: str,
        categories: List[Dict[str, str]],
        quality_report: Dict[str, Any],
        culture_profile: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        user_message = (
            "Refine the generated cultural category list before downstream data collection.\n\n"
            f"Country/Culture: {country_or_culture}\n"
            f"Culture profile:\n{json.dumps(culture_profile, ensure_ascii=False, indent=2)}\n\n"
            f"Current categories:\n{json.dumps(categories, ensure_ascii=False, indent=2)}\n\n"
            f"Quality report:\n{json.dumps(quality_report, ensure_ascii=False, indent=2)}\n\n"
            "Return a full revised list of categories. Preserve good categories, fix empty or generic "
            "descriptions, remove duplicates, and add missing cultural dimensions when needed. "
            "Return only the JSON object with key 'categories'."
        )
        request = LLMRequest(
            request_id="categories_refine",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
        )
        resp = self.client.generate_json_sync(request)
        if not resp.success:
            logger.error("Category refinement failed: %s", resp.error)
            return categories
        refined = resp.data.get("categories", [])
        logger.info("Refined categories from %d to %d.", len(categories), len(refined))
        return refined or categories
