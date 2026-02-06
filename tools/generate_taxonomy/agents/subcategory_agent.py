"""Agent for generating subcategories — supports batched execution."""

import logging
from typing import Dict, List

from core.llm import LLMClient, LLMRequest

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert cultural analyst specializing in breaking down broad cultural categories into specific, meaningful, and comprehensive subcategories.

Your task is to analyze a cultural category within the context of a specific country or culture and create a detailed, professional list of subcategories that would help organize data collections more granularly and comprehensively.

Output a JSON object with the following structure:
{
    "subcategories": [
        {
            "name": "subcategory_name_in_english",
            "description": "detailed, comprehensive description of what this subcategory represents, what types of data it encompasses, and its significance within the cultural context"
        }
    ]
}

When generating subcategories, you must:

1. **Understand the Cultural Context**:
   - Consider the specific country or culture being documented
   - Account for unique cultural characteristics, traditions, and practices
   - Reflect authentic cultural expressions and variations

2. **Be Comprehensive and Detailed**:
   - Cover all major aspects and variations of the category
   - Include both traditional and contemporary elements
   - Consider regional, social, and temporal variations
   - Think about different perspectives and use cases

3. **Ensure Professional Quality**:
   - Use precise, descriptive language
   - Make subcategories specific enough to be actionable
   - Ensure subcategories are distinct and non-overlapping
   - Consider practical data collection needs

4. **Consider the Category Description**:
   - Use the category description to understand the scope
   - Ensure subcategories align with the category's purpose
   - Create subcategories that logically fit within the category

Generate 4-10 comprehensive, well-described subcategories per category. Each subcategory should be specific, meaningful, and useful for organizing cultural data collection.

Return only valid JSON."""


class SubcategoryAgent:
    """Generates subcategories for cultural categories — supports batching."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @staticmethod
    def _build_user_message(
        category_name: str,
        category_description: str,
        country_or_culture: str,
    ) -> str:
        return (
            f"Generate comprehensive subcategories for the following cultural category:\n\n"
            f"Category: {category_name}\n"
            f"Category Description: {category_description}\n"
            f"Country/Culture: {country_or_culture}\n\n"
            f"Create detailed, professional subcategories that are specific to the "
            f"cultural context, cover both traditional and contemporary elements, "
            f"and are well-described and actionable for data collection."
        )

    def generate_for_categories(
        self,
        categories: List[Dict[str, str]],
        country_or_culture: str,
        *,
        batch_size: int = 5,
        batch_delay: float = 1.5,
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate subcategories for **all** categories in one batched call.

        Returns:
            ``{category_name: [{"name": ..., "description": ...}, ...]}``
        """
        requests = [
            LLMRequest(
                request_id=cat["name"],
                system_prompt=SYSTEM_PROMPT,
                user_message=self._build_user_message(
                    cat["name"], cat.get("description", ""), country_or_culture,
                ),
            )
            for cat in categories
        ]

        responses = self.client.generate_json_batch_sync(
            requests, batch_size=batch_size, batch_delay_seconds=batch_delay,
        )

        result: Dict[str, List[Dict[str, str]]] = {}
        for resp in responses:
            if resp.success:
                result[resp.request_id] = resp.data.get("subcategories", [])
                logger.info(
                    "Generated %d subcategories for '%s'.",
                    len(result[resp.request_id]), resp.request_id,
                )
            else:
                logger.error("Subcategory generation failed for '%s': %s", resp.request_id, resp.error)
                result[resp.request_id] = []
        return result
