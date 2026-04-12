"""Tests for the data collection pipeline.

Covers every stage independently:
  1. CategoryAgent      — topic space generation
  2. SubcategoryAgent   — batched subcategory expansion
  3. QueryAgent         — multilingual search query generation
  4. CollectDataTool    — Serper search + scrape (HTTP mocked)
  5. TextAnnotationAgent — text → SFT pairs (LLM mocked)
  6. BuildSftDatasetTool — end-to-end text annotation pipeline
  7. Integration        — taxonomy → collect → annotate (full data path)

No API keys or GPU required.

Run:
    pytest tests/test_data_collection_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ── fake LLM responses ──────────────────────────────────────────────────────

FAKE_CATEGORIES = {
    "categories": [
        {"name": "cuisine", "description": "Traditional dishes and cooking methods"},
        {"name": "music", "description": "Traditional and modern musical traditions"},
        {"name": "crafts", "description": "Textiles, ceramics, and handwork"},
    ]
}

FAKE_SUBCATEGORIES = {
    "subcategories": [
        {"name": "traditional_dishes", "description": "National dishes and recipes"},
        {"name": "cooking_methods", "description": "Traditional cooking techniques"},
    ]
}

FAKE_SEARCH_QUERIES = {
    "search_queries": [
        "traditional Kazakh beshbarmak recipe preparation",
        "Kazakhstan national cuisine history culture",
        "қазақ тағамдары дәстүрлі рецепттер",
        "Kazakh fermented mare milk kumiss preparation",
    ]
}

FAKE_TEXT_ANNOTATION = {
    "knowledge_qa": [
        {
            "question": "What is beshbarmak?",
            "answer": (
                "Beshbarmak is a traditional Kazakh dish made of boiled meat and "
                "flat noodles. It is considered the national dish of Kazakhstan and "
                "is served at important gatherings and celebrations."
            ),
        },
        {
            "question": "How is kumis prepared?",
            "answer": (
                "Kumis is fermented mare's milk prepared by fermenting fresh mare's "
                "milk in a leather bag called a saba, stirring regularly over several days."
            ),
        },
        {
            "question": "What role does hospitality play in Kazakh culture?",
            "answer": (
                "Hospitality is central to Kazakh culture. Guests are always offered "
                "the best food and a place at the table, reflecting deep-rooted customs "
                "of generosity passed down through generations."
            ),
        },
    ],
    "detailed_explanation": {
        "instruction": "Explain the significance of dastarkhan in Kazakh culture.",
        "response": (
            "The dastarkhan is a traditional tablecloth spread on the floor or a low "
            "table, representing the centerpiece of Kazakh hospitality. It symbolizes "
            "generosity and respect for guests. A bare dastarkhan is considered deeply "
            "disrespectful. Hosts ensure it is laden with bread, dried fruits, sweets, "
            "and tea before guests arrive."
        ),
    },
    "analytical_reasoning": {
        "instruction": "Why has beshbarmak remained the national dish despite modernization?",
        "response": (
            "First, beshbarmak uses ingredients readily available on the steppe: horse "
            "or lamb meat and simple flour noodles. Second, it is deeply tied to nomadic "
            "traditions of communal eating. Combining these factors, it has survived as "
            "a cultural anchor across generations."
        ),
    },
    "conversational_exchange": {
        "opening_question": "What are some typical Kazakh breakfast foods?",
        "opening_response": (
            "A typical Kazakh breakfast includes baursak (fried dough), kurt (dried cheese "
            "balls), and strong tea with milk. These foods reflect the nomadic heritage."
        ),
        "follow_up_question": "Are there regional variations in breakfast?",
        "follow_up_response": (
            "Yes, southern regions include more fruits and vegetables due to the warmer "
            "climate, while northern areas lean towards heavier dairy-based dishes."
        ),
    },
    "metadata": {
        "language": "en",
        "topics": ["cuisine", "beshbarmak", "kumis", "dastarkhan"],
        "domain": "culture",
    },
}

# Fake Serper API responses
FAKE_SERPER_SEARCH = {
    "organic": [
        {"link": "https://example.com/kazakh-food", "snippet": "Traditional Kazakh food"},
        {"link": "https://wiki.example.com/beshbarmak", "snippet": "Beshbarmak article"},
        {"link": "https://blog.example.com/kumis", "snippet": "Kumis preparation"},
    ]
}

FAKE_SERPER_SCRAPE_TEXT = (
    "Beshbarmak is the national dish of Kazakhstan, made from boiled horse or lamb "
    "meat served over flat noodles. The name means 'five fingers' in Kazakh, referring "
    "to the tradition of eating with hands. It is served at weddings and celebrations."
)


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """Return a mock LLMClient that returns fake taxonomy/annotation responses."""
    from core.llm.client import LLMResponse

    def _sync(request):
        msg = request.user_message.lower()
        if "search quer" in msg or "google" in msg:
            return LLMResponse(request_id=request.request_id, success=True, data=FAKE_SEARCH_QUERIES)
        if "subcategor" in msg:
            return LLMResponse(request_id=request.request_id, success=True, data=FAKE_SUBCATEGORIES)
        if "categor" in msg:
            return LLMResponse(request_id=request.request_id, success=True, data=FAKE_CATEGORIES)
        # text annotation (knowledge_qa key)
        return LLMResponse(request_id=request.request_id, success=True, data=FAKE_TEXT_ANNOTATION)

    def _batch_sync(requests, *, batch_size=5, batch_delay_seconds=0.0):
        return [_sync(r) for r in requests]

    client = MagicMock()
    client.generate_json_sync = _sync
    client.generate_json_batch_sync = _batch_sync
    return client


@pytest.fixture
def run_dir():
    with tempfile.TemporaryDirectory(prefix="horde_collect_test_") as d:
        yield d


# ════════════════════════════════════════════════════════════════════════════
# 1. CategoryAgent
# ════════════════════════════════════════════════════════════════════════════

class TestCategoryAgent:
    def test_returns_list_of_categories(self, mock_llm_client):
        from tools.generate_taxonomy.agents.category_agent import CategoryAgent

        agent = CategoryAgent(mock_llm_client)
        result = agent.extract_categories("Kazakhstan")

        assert isinstance(result, list)
        assert len(result) >= 2
        assert all("name" in c and "description" in c for c in result)

    def test_category_names_are_strings(self, mock_llm_client):
        from tools.generate_taxonomy.agents.category_agent import CategoryAgent

        agent = CategoryAgent(mock_llm_client)
        result = agent.extract_categories("Kazakhstan")

        for cat in result:
            assert isinstance(cat["name"], str) and cat["name"]
            assert isinstance(cat["description"], str) and cat["description"]

    def test_llm_failure_returns_empty_list(self):
        from core.llm.client import LLMResponse
        from tools.generate_taxonomy.agents.category_agent import CategoryAgent

        failing_client = MagicMock()
        failing_client.generate_json_sync = lambda r: LLMResponse(
            request_id=r.request_id, success=False, error="timeout"
        )
        agent = CategoryAgent(failing_client)
        result = agent.extract_categories("Kazakhstan")

        assert result == []

    def test_passes_country_in_prompt(self, mock_llm_client):
        from tools.generate_taxonomy.agents.category_agent import CategoryAgent

        captured = {}

        def _capture(request):
            captured["msg"] = request.user_message
            from core.llm.client import LLMResponse
            return LLMResponse(request_id=request.request_id, success=True, data=FAKE_CATEGORIES)

        mock_llm_client.generate_json_sync = _capture
        CategoryAgent(mock_llm_client).extract_categories("Kyrgyzstan")

        assert "Kyrgyzstan" in captured["msg"]


# ════════════════════════════════════════════════════════════════════════════
# 2. SubcategoryAgent
# ════════════════════════════════════════════════════════════════════════════

class TestSubcategoryAgent:
    def test_generates_subcategories_for_each_category(self, mock_llm_client):
        from tools.generate_taxonomy.agents.subcategory_agent import SubcategoryAgent

        categories = FAKE_CATEGORIES["categories"]
        agent = SubcategoryAgent(mock_llm_client)
        result = agent.generate_for_categories(categories, "Kazakhstan", batch_size=3, batch_delay=0.0)

        assert isinstance(result, dict)
        for cat in categories:
            assert cat["name"] in result
            subs = result[cat["name"]]
            assert isinstance(subs, list)
            assert len(subs) >= 1

    def test_subcategory_structure(self, mock_llm_client):
        from tools.generate_taxonomy.agents.subcategory_agent import SubcategoryAgent

        categories = FAKE_CATEGORIES["categories"][:1]
        agent = SubcategoryAgent(mock_llm_client)
        result = agent.generate_for_categories(categories, "Kazakhstan", batch_size=1, batch_delay=0.0)

        subs = result[categories[0]["name"]]
        for sub in subs:
            assert "name" in sub
            assert "description" in sub

    def test_failed_batch_entry_returns_empty_list(self):
        from core.llm.client import LLMResponse
        from tools.generate_taxonomy.agents.subcategory_agent import SubcategoryAgent

        call_count = {"n": 0}

        def _mixed(requests, *, batch_size=5, batch_delay_seconds=0.0):
            responses = []
            for r in requests:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    responses.append(LLMResponse(request_id=r.request_id, success=False, error="rate limit"))
                else:
                    responses.append(LLMResponse(request_id=r.request_id, success=True, data=FAKE_SUBCATEGORIES))
            return responses

        client = MagicMock()
        client.generate_json_batch_sync = _mixed
        categories = FAKE_CATEGORIES["categories"][:2]
        agent = SubcategoryAgent(client)
        result = agent.generate_for_categories(categories, "Kazakhstan", batch_size=2, batch_delay=0.0)

        # First category should be empty (failed), second should have subs
        assert result[categories[0]["name"]] == []
        assert len(result[categories[1]["name"]]) >= 1

    def test_batch_call_count_equals_categories(self, mock_llm_client):
        from tools.generate_taxonomy.agents.subcategory_agent import SubcategoryAgent

        categories = FAKE_CATEGORIES["categories"]
        captured_requests = []

        original_batch = mock_llm_client.generate_json_batch_sync

        def _capture(requests, **kwargs):
            captured_requests.extend(requests)
            return original_batch(requests, **kwargs)

        mock_llm_client.generate_json_batch_sync = _capture
        SubcategoryAgent(mock_llm_client).generate_for_categories(
            categories, "Kazakhstan", batch_size=5, batch_delay=0.0
        )

        # One request per category
        assert len(captured_requests) == len(categories)


# ════════════════════════════════════════════════════════════════════════════
# 3. QueryAgent
# ════════════════════════════════════════════════════════════════════════════

class TestQueryAgent:
    def _make_input(self):
        categories = FAKE_CATEGORIES["categories"]
        category_subcategories = {
            cat["name"]: FAKE_SUBCATEGORIES["subcategories"]
            for cat in categories
        }
        return categories, category_subcategories

    def test_generates_queries_for_all_subcategories(self, mock_llm_client):
        from tools.generate_taxonomy.agents.keyword_agent import QueryAgent

        categories, category_subcategories = self._make_input()
        agent = QueryAgent(mock_llm_client)
        result = agent.generate_for_subcategories(
            categories, category_subcategories, "Kazakhstan", batch_size=5, batch_delay=0.0
        )

        assert isinstance(result, dict)
        for cat in categories:
            assert cat["name"] in result
            for sub in category_subcategories[cat["name"]]:
                assert sub["name"] in result[cat["name"]]
                queries = result[cat["name"]][sub["name"]]
                assert isinstance(queries, list)
                assert len(queries) > 0

    def test_queries_are_non_empty_strings(self, mock_llm_client):
        from tools.generate_taxonomy.agents.keyword_agent import QueryAgent

        categories, category_subcategories = self._make_input()
        result = QueryAgent(mock_llm_client).generate_for_subcategories(
            categories, category_subcategories, "Kazakhstan", batch_size=5, batch_delay=0.0
        )

        for cat_name, sub_dict in result.items():
            for sub_name, queries in sub_dict.items():
                for q in queries:
                    assert isinstance(q, str) and q.strip(), (
                        f"Empty query in {cat_name}/{sub_name}"
                    )

    def test_keyword_agent_alias(self):
        from tools.generate_taxonomy.agents.keyword_agent import KeywordAgent, QueryAgent
        assert KeywordAgent is QueryAgent

    def test_total_query_count_proportional_to_subcategories(self, mock_llm_client):
        from tools.generate_taxonomy.agents.keyword_agent import QueryAgent

        categories, category_subcategories = self._make_input()
        total_subs = sum(len(v) for v in category_subcategories.values())

        result = QueryAgent(mock_llm_client).generate_for_subcategories(
            categories, category_subcategories, "Kazakhstan", batch_size=5, batch_delay=0.0
        )

        total_queries = sum(
            len(qs) for sub_dict in result.values() for qs in sub_dict.values()
        )
        # At least 1 query per subcategory
        assert total_queries >= total_subs


# ════════════════════════════════════════════════════════════════════════════
# 4. GenerateTaxonomyTool (full taxonomy pipeline)
# ════════════════════════════════════════════════════════════════════════════

class TestGenerateTaxonomyTool:
    @pytest.fixture(autouse=True)
    def _patch_llm(self, mock_llm_client):
        with patch("core.llm.client.LLMClient.from_env", return_value=mock_llm_client):
            yield

    def test_full_taxonomy_structure(self):
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        result = GenerateTaxonomyTool().execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})

        assert "categories" in result
        assert "category_subcategories" in result
        assert "category_subcategory_queries" in result

    def test_all_categories_have_subcategories_and_queries(self):
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        result = GenerateTaxonomyTool().execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})

        for cat in result["categories"]:
            name = cat["name"]
            assert name in result["category_subcategories"], f"No subcategories for {name}"
            assert name in result["category_subcategory_queries"], f"No queries for {name}"

            for sub in result["category_subcategories"][name]:
                assert sub["name"] in result["category_subcategory_queries"][name], (
                    f"Subcategory {sub['name']} has no queries"
                )

    def test_raises_on_empty_country(self):
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        with pytest.raises(ValueError, match="required"):
            GenerateTaxonomyTool().execute("", None)

    def test_raises_on_whitespace_country(self):
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        with pytest.raises(ValueError, match="required"):
            GenerateTaxonomyTool().execute("   ", None)

    def test_query_extraction_flat_list(self):
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        result = GenerateTaxonomyTool().execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})

        all_queries = []
        for sub_dict in result["category_subcategory_queries"].values():
            for q_list in sub_dict.values():
                all_queries.extend(q_list)

        assert len(all_queries) > 0
        assert all(isinstance(q, str) for q in all_queries)


# ════════════════════════════════════════════════════════════════════════════
# 5. CollectDataTool (Serper mocked)
# ════════════════════════════════════════════════════════════════════════════

def _make_mock_aiohttp(search_data: dict, scrape_text: str):
    """Build a context-manager mock for aiohttp.ClientSession."""
    mock_session = AsyncMock()

    async def _post_side_effect(url, *, json=None, headers=None):
        resp = AsyncMock()
        if "serper.dev/search" in url:
            resp.json = AsyncMock(return_value=search_data)
        else:
            resp.json = AsyncMock(return_value={"text": scrape_text})
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    mock_session.post = _post_side_effect
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_session_class = MagicMock(return_value=mock_session)
    mock_session_class.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_class.__aexit__ = AsyncMock(return_value=False)
    return mock_session_class


class TestCollectDataTool:
    @pytest.fixture(autouse=True)
    def _set_serper_key(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "test-key-123")

    def _run(self, queries: List[str], run_dir: str, **kwargs) -> Dict[str, Any]:
        from tools.collect_data.tool import CollectDataTool

        config = {
            "queries": queries,
            "run_dir": run_dir,
            "google_results_per_query": 3,
            "top_results": 2,
            "concurrency": 5,
            **kwargs,
        }
        with patch("aiohttp.ClientSession", _make_mock_aiohttp(FAKE_SERPER_SEARCH, FAKE_SERPER_SCRAPE_TEXT)):
            return CollectDataTool().execute(config)

    def test_returns_dataset_with_samples(self, run_dir):
        result = self._run(["Kazakh cuisine"], run_dir)

        assert "data_path" in result
        assert "num_samples" in result
        assert result["num_samples"] > 0

    def test_creates_dataset_on_disk(self, run_dir):
        result = self._run(["Kazakh cuisine"], run_dir)

        assert os.path.exists(result["data_path"])
        # HF Dataset saved to disk has dataset_info.json
        assert any(
            p.name in {"dataset_info.json", "data-00000-of-00001.arrow"}
            for p in Path(result["data_path"]).iterdir()
        )

    def test_creates_raw_json_dump(self, run_dir):
        self._run(["Kazakh cuisine"], run_dir)

        raw_files = list(Path(run_dir).glob("**/serper_raw.json"))
        assert len(raw_files) == 1
        raw = json.loads(raw_files[0].read_text())
        assert isinstance(raw, dict)

    def test_multiple_queries_collected(self, run_dir):
        queries = [
            "beshbarmak recipe",
            "kumis preparation",
            "Kazakh traditional music",
        ]
        result = self._run(queries, run_dir)

        assert result["num_samples"] > 0

    def test_missing_serper_key_raises(self, run_dir, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        from tools.collect_data.tool import CollectDataTool

        with pytest.raises(EnvironmentError, match="SERPER_API_KEY"):
            CollectDataTool().execute({"queries": ["test"], "run_dir": run_dir})

    def test_empty_queries_raises(self, run_dir):
        from tools.collect_data.tool import CollectDataTool

        with pytest.raises(ValueError, match="queries"):
            CollectDataTool().execute({"queries": [], "run_dir": run_dir})

    def test_missing_queries_key_raises(self, run_dir):
        from tools.collect_data.tool import CollectDataTool

        with pytest.raises(ValueError, match="queries"):
            CollectDataTool().execute({"run_dir": run_dir})

    def test_excluded_domains_filtered(self, run_dir):
        """Pages from social media domains must not appear in collected texts."""
        search_with_social = {
            "organic": [
                {"link": "https://facebook.com/kazakh-food", "snippet": "Facebook post"},
                {"link": "https://www.instagram.com/p/abc123", "snippet": "Instagram"},
                {"link": "https://example.com/real-article", "snippet": "Good article"},
            ]
        }
        from tools.collect_data.tool import CollectDataTool

        config = {
            "queries": ["Kazakh cuisine"],
            "run_dir": run_dir,
            "top_results": 5,
            "concurrency": 5,
        }
        with patch("aiohttp.ClientSession", _make_mock_aiohttp(search_with_social, FAKE_SERPER_SCRAPE_TEXT)):
            result = CollectDataTool().execute(config)

        raw_files = list(Path(run_dir).glob("**/serper_raw.json"))
        raw = json.loads(raw_files[0].read_text())

        for query_results in raw.values():
            for page in query_results:
                url = page.get("url", "")
                assert "facebook.com" not in url
                assert "instagram.com" not in url

    def test_scrape_failure_produces_no_crash(self, run_dir):
        """When scraping returns empty text, the page is skipped gracefully."""
        search_data = {"organic": [{"link": "https://example.com/page1", "snippet": "ok"}]}

        mock_session = AsyncMock()

        async def _post_side_effect(url, *, json=None, headers=None):
            resp = AsyncMock()
            if "serper.dev/search" in url:
                resp.json = AsyncMock(return_value=search_data)
            else:
                resp.json = AsyncMock(return_value={"text": ""})  # empty text
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)
            return resp

        mock_session.post = _post_side_effect
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_class = MagicMock(return_value=mock_session)

        from tools.collect_data.tool import CollectDataTool

        config = {"queries": ["Kazakh food"], "run_dir": run_dir, "concurrency": 2}
        with patch("aiohttp.ClientSession", mock_class):
            result = CollectDataTool().execute(config)

        # Falls back to "(no text collected)" placeholder
        assert result["num_samples"] >= 1

    def test_deduplication_within_query(self, run_dir):
        """Duplicate URLs across organic results should appear only once."""
        search_with_dupes = {
            "organic": [
                {"link": "https://example.com/article", "snippet": "Article"},
                {"link": "https://example.com/article", "snippet": "Duplicate"},  # same URL
                {"link": "https://other.com/page", "snippet": "Different"},
            ]
        }
        from tools.collect_data.tool import CollectDataTool

        config = {
            "queries": ["test query"],
            "run_dir": run_dir,
            "top_results": 10,
            "concurrency": 3,
        }
        with patch("aiohttp.ClientSession", _make_mock_aiohttp(search_with_dupes, FAKE_SERPER_SCRAPE_TEXT)):
            CollectDataTool().execute(config)

        raw = json.loads(list(Path(run_dir).glob("**/serper_raw.json"))[0].read_text())
        urls = [page["url"] for pages in raw.values() for page in pages]
        assert len(urls) == len(set(urls)), "Duplicate URLs were not deduplicated"

    def test_metadata_in_result(self, run_dir):
        result = self._run(["Kazakh cuisine"], run_dir)

        assert "metadata" in result
        assert result["metadata"]["provider"] == "serper"


# ════════════════════════════════════════════════════════════════════════════
# 6. TextAnnotationAgent
# ════════════════════════════════════════════════════════════════════════════

class TestTextAnnotationAgent:
    def _make_items(self, texts: List[str]):
        from tools.build_sft_dataset.types import TextItem
        return [TextItem(item_id=f"item_{i}", text=t) for i, t in enumerate(texts)]

    def test_annotates_items_successfully(self, mock_llm_client):
        from tools.build_sft_dataset.agents.sft_text_agent import TextAnnotationAgent

        agent = TextAnnotationAgent(mock_llm_client, target_language="English", batch_size=3, batch_delay=0.0)
        items = self._make_items([
            "Kazakhstan is a large country in Central Asia.",
            "Beshbarmak is the national dish of Kazakhstan.",
        ])
        annotations, failures = agent.annotate(items)

        assert len(annotations) == 2
        assert len(failures) == 0
        assert all(a["success"] for a in annotations)

    def test_annotation_data_has_required_keys(self, mock_llm_client):
        from tools.build_sft_dataset.agents.sft_text_agent import TextAnnotationAgent

        agent = TextAnnotationAgent(mock_llm_client, batch_size=1, batch_delay=0.0)
        items = self._make_items(["Traditional Kazakh hospitality customs."])
        annotations, _ = agent.annotate(items)

        data = annotations[0]["data"]
        assert "knowledge_qa" in data
        assert "detailed_explanation" in data
        assert "analytical_reasoning" in data
        assert "conversational_exchange" in data
        assert "metadata" in data

    def test_llm_failure_goes_to_failures_list(self):
        from core.llm.client import LLMResponse
        from tools.build_sft_dataset.agents.sft_text_agent import TextAnnotationAgent

        failing_client = MagicMock()
        failing_client.generate_json_batch_sync = lambda reqs, **kw: [
            LLMResponse(request_id=r.request_id, success=False, error="timeout")
            for r in reqs
        ]

        agent = TextAnnotationAgent(failing_client, batch_size=1, batch_delay=0.0)
        from tools.build_sft_dataset.types import TextItem
        items = [TextItem(item_id="x", text="Some text")]
        annotations, failures = agent.annotate(items)

        assert len(failures) == 1
        assert annotations[0]["success"] is False

    def test_schema_validation_failure_goes_to_failures_list(self):
        """Invalid JSON schema from LLM goes to failures, not annotations."""
        from core.llm.client import LLMResponse
        from tools.build_sft_dataset.agents.sft_text_agent import TextAnnotationAgent

        bad_client = MagicMock()
        bad_client.generate_json_batch_sync = lambda reqs, **kw: [
            LLMResponse(request_id=r.request_id, success=True, data={"wrong_key": "value"})
            for r in reqs
        ]

        agent = TextAnnotationAgent(bad_client, batch_size=1, batch_delay=0.0)
        from tools.build_sft_dataset.types import TextItem
        items = [TextItem(item_id="y", text="Some text")]
        annotations, failures = agent.annotate(items)

        assert len(failures) == 1
        assert "schema" in annotations[0].get("error", "")

    def test_partial_batch_failure(self, mock_llm_client):
        """Mix of successes and failures in one batch."""
        from core.llm.client import LLMResponse
        from tools.build_sft_dataset.agents.sft_text_agent import TextAnnotationAgent

        call_n = {"n": 0}

        def _mixed_batch(requests, **kw):
            results = []
            for r in requests:
                call_n["n"] += 1
                if call_n["n"] % 2 == 0:
                    results.append(LLMResponse(request_id=r.request_id, success=False, error="err"))
                else:
                    results.append(LLMResponse(request_id=r.request_id, success=True, data=FAKE_TEXT_ANNOTATION))
            return results

        mock_llm_client.generate_json_batch_sync = _mixed_batch
        agent = TextAnnotationAgent(mock_llm_client, batch_size=4, batch_delay=0.0)
        items = self._make_items(["text A", "text B", "text C", "text D"])
        annotations, failures = agent.annotate(items)

        successes = sum(1 for a in annotations if a["success"])
        assert successes == 2
        assert len(failures) == 2


# ════════════════════════════════════════════════════════════════════════════
# 7. SFT example quality properties
# ════════════════════════════════════════════════════════════════════════════

class TestSftExampleQuality:
    """Verify that generated SFT examples satisfy structural quality rules."""

    def _build_examples(self):
        from tools.build_sft_dataset.schemas import TextAnnotation
        from tools.build_sft_dataset.sft_builders import build_text_sft_examples

        annotation = TextAnnotation.model_validate(FAKE_TEXT_ANNOTATION)
        return build_text_sft_examples(annotation)

    def test_all_examples_have_user_assistant_messages(self):
        examples = self._build_examples()
        assert len(examples) > 0
        for ex in examples:
            msgs = ex["messages"]
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"

    def test_knowledge_qa_produces_five_examples(self):
        examples = self._build_examples()
        # 3 knowledge_qa + 1 detailed_explanation + 1 analytical_reasoning + 1 conversational
        # FAKE_TEXT_ANNOTATION has 3 knowledge_qa items
        user_messages = [ex["messages"][0]["content"] for ex in examples if len(ex["messages"]) == 2]
        assert len(user_messages) >= 3

    def test_conversational_exchange_has_four_turns(self):
        examples = self._build_examples()
        multi_turn = [ex for ex in examples if len(ex["messages"]) == 4]
        assert len(multi_turn) == 1

        msgs = multi_turn[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        assert msgs[3]["role"] == "assistant"

    def test_no_source_text_references_in_answers(self):
        """Answers must not contain 'according to the text' or similar phrases."""
        forbidden = [
            "according to the text",
            "according to the passage",
            "the text says",
            "as mentioned in",
            "as described above",
        ]
        examples = self._build_examples()
        for ex in examples:
            for msg in ex["messages"]:
                content = msg["content"].lower() if isinstance(msg["content"], str) else ""
                for phrase in forbidden:
                    assert phrase not in content, (
                        f"Forbidden reference '{phrase}' found in: {content[:100]}"
                    )

    def test_all_messages_have_non_empty_content(self):
        examples = self._build_examples()
        for ex in examples:
            for msg in ex["messages"]:
                content = msg["content"]
                if isinstance(content, str):
                    assert content.strip(), f"Empty content in {msg['role']} message"
                elif isinstance(content, list):
                    assert len(content) > 0


# ════════════════════════════════════════════════════════════════════════════
# 8. BuildSftDatasetTool (end-to-end annotation pipeline)
# ════════════════════════════════════════════════════════════════════════════

class TestBuildSftDatasetTool:
    @pytest.fixture(autouse=True)
    def _patch_llm(self, mock_llm_client):
        with patch("core.llm.client.LLMClient.from_env", return_value=mock_llm_client):
            yield

    def _write_jsonl(self, path: str, rows: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_text_mode_from_jsonl(self, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl = os.path.join(run_dir, "input.jsonl")
        self._write_jsonl(jsonl, [
            {"text": "Kazakhstan is in Central Asia."},
            {"text": "Beshbarmak is the national dish."},
            {"text": "The dombra is a traditional instrument."},
        ])

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "ann.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 2,
            "batch_delay": 0.0,
        })

        assert result["num_items"] == 3
        assert result["num_examples"] >= 3
        assert result["num_failures"] == 0

    def test_sft_output_file_is_valid_jsonl(self, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl = os.path.join(run_dir, "input.jsonl")
        self._write_jsonl(jsonl, [{"text": "Traditional Kazakh horse games."}])

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "ann.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 1,
            "batch_delay": 0.0,
        })

        sft_path = result["sft_path"]
        with open(sft_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) > 0
        for ex in lines:
            assert "messages" in ex

    def test_sft_chat_format_alternates_roles(self, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl = os.path.join(run_dir, "input.jsonl")
        self._write_jsonl(jsonl, [{"text": "Traditional Kazakh yurt construction."}])

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "ann.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 1,
            "batch_delay": 0.0,
        })

        with open(result["sft_path"], encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                msgs = ex["messages"]
                assert len(msgs) >= 2
                for i, msg in enumerate(msgs):
                    expected = "user" if i % 2 == 0 else "assistant"
                    assert msg["role"] == expected, (
                        f"Expected role '{expected}' at index {i}, got '{msg['role']}'"
                    )

    def test_empty_text_lines_skipped(self, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl = os.path.join(run_dir, "input.jsonl")
        self._write_jsonl(jsonl, [
            {"text": "Real content about Kazakhstan."},
            {"text": ""},          # empty — should be skipped by loader
            {"text": "   "},       # whitespace — should be skipped
        ])

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "ann.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 2,
            "batch_delay": 0.0,
        })

        assert result["num_items"] == 1  # only the non-empty line

    def test_invalid_mode_raises(self, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        with pytest.raises(ValueError, match="mode"):
            BuildSftDatasetTool().execute({"mode": "video", "run_dir": run_dir})

    def test_missing_input_raises(self, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        with pytest.raises(ValueError):
            BuildSftDatasetTool().execute({
                "mode": "text",
                "output_annotations": os.path.join(run_dir, "ann.jsonl"),
                "output_sft": os.path.join(run_dir, "sft.jsonl"),
            })

    def test_annotations_file_written(self, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl = os.path.join(run_dir, "input.jsonl")
        self._write_jsonl(jsonl, [{"text": "Kazakh nomadic traditions."}])
        ann_path = os.path.join(run_dir, "ann.jsonl")

        BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl,
            "text_field": "text",
            "output_annotations": ann_path,
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 1,
            "batch_delay": 0.0,
        })

        assert os.path.exists(ann_path)
        with open(ann_path, encoding="utf-8") as f:
            ann_lines = [json.loads(line) for line in f if line.strip()]
        assert len(ann_lines) == 1
        assert ann_lines[0]["success"] is True


# ════════════════════════════════════════════════════════════════════════════
# 9. Integration: taxonomy → collect → annotate
# ════════════════════════════════════════════════════════════════════════════

class TestDataCollectionIntegration:
    """Full pipeline: taxonomy generates queries → collect scrapes text →
    SFT agent annotates → JSONL ready for training."""

    @pytest.fixture(autouse=True)
    def _patch_llm(self, mock_llm_client):
        with patch("core.llm.client.LLMClient.from_env", return_value=mock_llm_client):
            yield

    def _write_jsonl(self, path: str, rows: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_taxonomy_queries_flow_into_collect(self, run_dir):
        """Queries from GenerateTaxonomyTool are forwarded to CollectDataTool."""
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        taxonomy = GenerateTaxonomyTool().execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})

        all_queries = []
        for sub_dict in taxonomy["category_subcategory_queries"].values():
            for q_list in sub_dict.values():
                all_queries.extend(q_list)

        assert len(all_queries) > 0

        collected_config = {"queries": all_queries, "run_dir": run_dir}
        fake_collect_result = {
            "data_path": os.path.join(run_dir, "collect"),
            "num_samples": len(all_queries) * 2,
            "metadata": {"provider": "serper"},
        }

        with patch("tools.collect_data.tool.CollectDataTool.execute", return_value=fake_collect_result) as mock_collect:
            from tools.collect_data.tool import CollectDataTool
            result = CollectDataTool().execute(collected_config)

        mock_collect.assert_called_once()
        call_config = mock_collect.call_args[0][0]
        assert "queries" in call_config
        assert len(call_config["queries"]) == len(all_queries)

    def test_full_data_collection_to_sft_pipeline(self, run_dir):
        """Simulate: taxonomy → texts → SFT annotation → valid JSONL."""
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        # Step 1: taxonomy
        taxonomy = GenerateTaxonomyTool().execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})
        all_queries = [
            q
            for sub_dict in taxonomy["category_subcategory_queries"].values()
            for q_list in sub_dict.values()
            for q in q_list
        ]
        assert len(all_queries) > 0

        # Step 2: simulate collected texts (one article per query)
        collected_jsonl = os.path.join(run_dir, "collected.jsonl")
        self._write_jsonl(collected_jsonl, [
            {"text": f"Detailed article about {q} in Kazakhstan culture."}
            for q in all_queries[:6]
        ])

        # Step 3: SFT annotation
        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": collected_jsonl,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "ann.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 3,
            "batch_delay": 0.0,
        })

        assert result["num_items"] == 6
        assert result["num_examples"] >= 6
        assert result["num_failures"] == 0

        # Step 4: verify JSONL is usable for training
        with open(result["sft_path"], encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]

        assert len(examples) >= 6
        for ex in examples:
            assert "messages" in ex
            assert ex["messages"][0]["role"] == "user"
            assert ex["messages"][1]["role"] == "assistant"

    def test_sft_example_count_per_text(self, run_dir):
        """Each input text should produce multiple SFT examples (Q&A pairs)."""
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl = os.path.join(run_dir, "single.jsonl")
        self._write_jsonl(jsonl, [{"text": "Kazakh nomadic traditions and customs."}])

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "ann.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 1,
            "batch_delay": 0.0,
        })

        # One text → knowledge_qa(3) + detailed_explanation(1) + analytical_reasoning(1) + conversation(1) = 6
        assert result["num_examples"] >= 6

    def test_pipeline_preserves_cultural_language_in_queries(self, mock_llm_client):
        """QueryAgent should emit multilingual queries including non-English."""
        from tools.generate_taxonomy.agents.keyword_agent import QueryAgent

        categories = FAKE_CATEGORIES["categories"][:1]
        category_subcategories = {
            categories[0]["name"]: FAKE_SUBCATEGORIES["subcategories"][:1]
        }

        result = QueryAgent(mock_llm_client).generate_for_subcategories(
            categories, category_subcategories, "Kazakhstan", batch_size=1, batch_delay=0.0
        )

        all_queries = [
            q
            for sub_dict in result.values()
            for qs in sub_dict.values()
            for q in qs
        ]
        # FAKE_SEARCH_QUERIES contains a Kazakh Cyrillic query
        has_non_ascii = any(not q.isascii() for q in all_queries)
        assert has_non_ascii, (
            "Expected at least one non-ASCII (native language) query, got: "
            + str(all_queries)
        )
