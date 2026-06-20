"""Agent for generating search queries — supports batched execution."""

import json
import logging
from typing import Any, Dict, List

from core.llm import LLMClient, LLMRequest

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert at creating effective Google search queries for web data collection.

Your task is to generate specific, ready-to-search Google queries that would return relevant pages (articles, encyclopedias, news) for a given subcategory within a cultural context.

Output a JSON object with the following structure:
{
    "search_queries": [
        "full search query phrase 1",
        "full search query phrase 2",
        "full search query phrase 3"
    ]
}

Guidelines for creating effective search queries:

1. **Google-Ready Queries**:
   - Write complete phrases you would type into Google (5-15 words)
   - Include the country/culture name in every query
   - Be specific: prefer "traditional Kazakh beshbarmak recipe preparation" over "beshbarmak"
   - Mix informational queries ("What is ...", "History of ...") with factual ones

2. **Multilingual Queries (REQUIRED)**:
   - Generate queries in BOTH native language(s) AND English
   - For Kazakhstan: include Kazakh (Cyrillic) and English queries
   - Use native script where appropriate (e.g., "Қазақстанның дәстүрлі тағамдары")
   - Balance: approximately 50% native language, 50% English

3. **Coverage and Variety**:
   - Cover different aspects: history, traditions, modern practices, notable examples
   - Include both broad overviews and specific deep-dives
   - Consider different source types: Wikipedia-style, news, academic, cultural guides

Generate 8-12 search queries per subcategory. Return only valid JSON."""


class QueryAgent:
    """Generates search queries per subcategory — supports batching."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @staticmethod
    def _build_user_message(
        category_name: str,
        subcategory_name: str,
        subcategory_description: str,
        country_or_culture: str,
        focus: str = "",
    ) -> str:
        focus_line = f"Scope/focus: {focus}\n" if focus else ""
        return (
            f"Generate Google search queries for the following subcategory:\n\n"
            f"Category: {category_name}\n"
            f"Subcategory: {subcategory_name}\n"
            f"Subcategory Description: {subcategory_description}\n"
            f"Country/Culture: {country_or_culture}\n\n"
            f"{focus_line}"
            f"REQUIREMENT: Generate 8-12 complete, ready-to-search Google queries "
            f"that MUST include BOTH native language(s) of {country_or_culture} AND "
            f"English queries. Use native script where appropriate."
        )

    def generate_for_subcategories(
        self,
        categories: List[Dict[str, str]],
        category_subcategories: Dict[str, List[Dict[str, str]]],
        country_or_culture: str,
        *,
        batch_size: int = 5,
        batch_delay: float = 1.5,
        focus: str = "",
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate search queries for all subcategories in one batched call.

        Returns:
            ``{category_name: {subcategory_name: [query1, query2, ...]}}``
        """
        requests: List[LLMRequest] = []
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
                            focus,
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
                queries = resp.data.get("search_queries", []) or resp.data.get("keywords", [])
                if queries and isinstance(queries[0], dict):
                    queries = [q.get("query", "") or q.get("keyword", "") for q in queries if isinstance(q, dict)]
                queries = [q for q in queries if q and isinstance(q, str)]
                result[cat_name][sub_name] = queries
                logger.info(
                    "Generated %d search queries for '%s / %s'.",
                    len(queries), cat_name, sub_name,
                )
            else:
                logger.error("Query generation failed for '%s / %s': %s", cat_name, sub_name, resp.error)
                result[cat_name][sub_name] = []

        return result

    def repair_for_subcategory(
        self,
        category: Dict[str, str],
        subcategory: Dict[str, str],
        existing_queries: List[str],
        country_or_culture: str,
        quality_report: Dict[str, Any],
        culture_profile: Dict[str, Any],
        *,
        focus: str = "",
    ) -> List[str]:
        category_name = category["name"]
        subcategory_name = subcategory["name"]
        focus_line = f"Scope/focus: {focus}\n" if focus else ""
        user_message = (
            "Repair Google search queries for one cultural subcategory before web collection.\n\n"
            f"Country/Culture: {country_or_culture}\n"
            f"{focus_line}"
            f"Culture profile:\n{json.dumps(culture_profile, ensure_ascii=False, indent=2)}\n\n"
            f"Category:\n{json.dumps(category, ensure_ascii=False, indent=2)}\n\n"
            f"Subcategory:\n{json.dumps(subcategory, ensure_ascii=False, indent=2)}\n\n"
            f"Existing queries:\n{json.dumps(existing_queries, ensure_ascii=False, indent=2)}\n\n"
            f"Query quality report:\n{json.dumps(quality_report, ensure_ascii=False, indent=2)}\n\n"
            "Return a complete replacement list of search queries for this subcategory only. "
            "Queries should be specific, searchable, include the target culture or aliases, and mix "
            "English with native-language/native-script queries where appropriate. Return only a JSON "
            "object with key 'search_queries'."
        )
        request = LLMRequest(
            request_id=f"repair_queries::{category_name}||{subcategory_name}",
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
        )
        resp = self.client.generate_json_sync(request)
        if not resp.success:
            logger.error("Query repair failed for '%s / %s': %s", category_name, subcategory_name, resp.error)
            return existing_queries
        queries = resp.data.get("search_queries", []) or resp.data.get("keywords", [])
        if queries and isinstance(queries[0], dict):
            queries = [q.get("query", "") or q.get("keyword", "") for q in queries if isinstance(q, dict)]
        repaired = [q for q in queries if q and isinstance(q, str)]
        logger.info("Repaired queries for '%s / %s': %d -> %d.", category_name, subcategory_name, len(existing_queries), len(repaired))
        return repaired or existing_queries


# Backward-compatible alias
KeywordAgent = QueryAgent
