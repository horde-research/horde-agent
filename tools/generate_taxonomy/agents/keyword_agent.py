"""Agent for generating search keywords — supports batched execution."""

import logging
from typing import Dict, List

from core.llm import LLMClient, LLMRequest

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert at creating effective search keywords for internet search engines (Google, Bing, etc.) and data collection.

Your task is to generate specific, searchable keywords that would return relevant data (images, text, etc.) for a given subcategory within a cultural context.

Output a JSON object with the following structure:
{
    "keywords": [
        "search keyword phrase 1",
        "search keyword phrase 2",
        "search keyword phrase 3"
    ]
}

Guidelines for creating effective keywords:

1. **Search Engine Optimization**:
   - Use 2-5 word phrases that people actually search for
   - Include the country/culture name when relevant
   - Use specific, descriptive terms
   - Think about what would return the best results on Google Images or web search

2. **Multilingual Keywords (REQUIRED)**:
   - **ALWAYS generate keywords in BOTH native language(s) AND English**
   - For each country/culture, include keywords in:
     * Native language(s) of the country (e.g., for Kazakhstan: Kazakh language using Cyrillic script like "Қазақстан дәстүрлі", "қазақ киімі")
     * English translations and descriptions (e.g., "Kazakhstan traditional", "Kazakh culture")
   - If the country has multiple official languages, include keywords in all major languages
   - Use native script (Cyrillic, Arabic, Latin, etc.) as appropriate for the country
   - Include transliterations when helpful (e.g., "qazaq kiyimi" for Kazakh)
   - Create mixed-language combinations (native term + English descriptor)
   - Balance: approximately 40-50% native language, 40-50% English, 10-20% mixed

3. **Cultural Context**:
   - Include cultural identifiers (country name, ethnic group, etc.)
   - Use culturally appropriate terminology
   - Consider regional variations
   - Include both traditional and modern terms

4. **Variety and Coverage**:
   - Generate diverse keywords covering different aspects
   - Include synonyms and alternative phrasings
   - Consider different search intents (informational, visual, etc.)
   - Mix broad and specific terms

Generate 8-12 effective keywords per subcategory. Return only a flat list of keyword strings, not objects.

Return only valid JSON."""


class KeywordAgent:
    """Generates search keywords per subcategory — supports batching."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @staticmethod
    def _build_user_message(
        category_name: str,
        subcategory_name: str,
        subcategory_description: str,
        country_or_culture: str,
    ) -> str:
        return (
            f"Generate effective search keywords for the following subcategory:\n\n"
            f"Category: {category_name}\n"
            f"Subcategory: {subcategory_name}\n"
            f"Subcategory Description: {subcategory_description}\n"
            f"Country/Culture: {country_or_culture}\n\n"
            f"REQUIREMENT: Generate 8-12 search keywords that MUST include BOTH "
            f"native language(s) of {country_or_culture} AND English keywords. "
            f"Use native script where appropriate. Balance: ~40-50% native, "
            f"~40-50% English, ~10-20% mixed."
        )

    def generate_for_subcategories(
        self,
        categories: List[Dict[str, str]],
        category_subcategories: Dict[str, List[Dict[str, str]]],
        country_or_culture: str,
        *,
        batch_size: int = 5,
        batch_delay: float = 1.5,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate keywords for **all** subcategories in one batched call.

        Returns:
            ``{category_name: {subcategory_name: [kw1, kw2, ...]}}``
        """
        requests: List[LLMRequest] = []
        # map request_id → (cat_name, sub_name)
        id_map: Dict[str, tuple] = {}

        for cat in categories:
            cat_name = cat["name"]
            for sub in category_subcategories.get(cat_name, []):
                rid = f"{cat_name}||{sub['name']}"
                id_map[rid] = (cat_name, sub["name"])
                requests.append(
                    LLMRequest(
                        request_id=rid,
                        system_prompt=SYSTEM_PROMPT,
                        user_message=self._build_user_message(
                            cat_name,
                            sub["name"],
                            sub.get("description", ""),
                            country_or_culture,
                        ),
                    )
                )

        responses = self.client.generate_json_batch_sync(
            requests, batch_size=batch_size, batch_delay_seconds=batch_delay,
        )

        result: Dict[str, Dict[str, List[str]]] = {}
        for resp in responses:
            cat_name, sub_name = id_map[resp.request_id]
            result.setdefault(cat_name, {})

            if resp.success:
                keywords = resp.data.get("keywords", [])
                # Handle list-of-dicts fallback
                if keywords and isinstance(keywords[0], dict):
                    keywords = [kw.get("keyword", "") for kw in keywords if kw.get("keyword")]
                keywords = [kw for kw in keywords if kw and isinstance(kw, str)]
                result[cat_name][sub_name] = keywords
                logger.info(
                    "Generated %d keywords for '%s / %s'.",
                    len(keywords), cat_name, sub_name,
                )
            else:
                logger.error("Keyword generation failed for '%s / %s': %s", cat_name, sub_name, resp.error)
                result[cat_name][sub_name] = []

        return result
