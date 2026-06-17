from __future__ import annotations

from unittest.mock import patch

from core.llm.client import LLMResponse
from tools.generate_taxonomy.quality import (
    validate_categories,
    validate_query_groups,
    validate_subcategories,
)
from tools.generate_taxonomy.tool import GenerateTaxonomyTool


def _response(request_id: str, data: dict) -> LLMResponse:
    return LLMResponse(request_id=request_id, success=True, data=data)


class ScriptedTaxonomyClient:
    def __init__(self) -> None:
        self.sync_request_ids: list[str] = []
        self.batch_request_ids: list[str] = []

    def generate_json_sync(self, request):
        self.sync_request_ids.append(request.request_id)
        if request.request_id == "categories":
            return _response(
                request.request_id,
                {"categories": [{"name": "culture", "description": ""}]},
            )
        if request.request_id == "categories_refine":
            return _response(
                request.request_id,
                {
                    "categories": [
                        {"name": "cuisine", "description": "Traditional and modern Kazakh food culture."},
                        {"name": "language", "description": "Kazakh, Russian, and multilingual communication."},
                    ]
                },
            )
        if request.request_id == "repair_subcategories::language":
            return _response(
                request.request_id,
                {
                    "subcategories": [
                        {"name": "kazakh_language", "description": "Kazakh usage, scripts, and everyday speech."},
                        {"name": "multilingual_media", "description": "Kazakh, Russian, and English media use."},
                    ]
                },
            )
        if request.request_id == "repair_queries::language||kazakh_language":
            return _response(
                request.request_id,
                {
                    "search_queries": [
                        "Kazakhstan Kazakh language everyday speech",
                        "қазақ тілі күнделікті сөйлеу Қазақстан",
                        "Kazakh Cyrillic language culture Kazakhstan",
                    ]
                },
            )
        raise AssertionError(f"Unexpected sync request: {request.request_id}")

    def generate_json_batch_sync(self, requests, *, batch_size=5, batch_delay_seconds=1.5):
        self.batch_request_ids.extend(request.request_id for request in requests)
        responses = []
        for request in requests:
            if request.request_id == "cuisine":
                responses.append(
                    _response(
                        request.request_id,
                        {
                            "subcategories": [
                                {"name": "traditional_dishes", "description": "Beshbarmak, baursak, and rituals."},
                                {"name": "modern_food", "description": "Contemporary restaurants and food trends."},
                            ]
                        },
                    )
                )
            elif request.request_id == "language":
                responses.append(_response(request.request_id, {"subcategories": []}))
            elif request.request_id == "cuisine||traditional_dishes":
                responses.append(
                    _response(
                        request.request_id,
                        {
                            "search_queries": [
                                "Kazakhstan traditional dishes beshbarmak history",
                                "қазақ ұлттық тағамдары бешбармақ",
                                "Kazakh baursak food culture Kazakhstan",
                            ]
                        },
                    )
                )
            elif request.request_id == "cuisine||modern_food":
                responses.append(
                    _response(
                        request.request_id,
                        {
                            "search_queries": [
                                "Kazakhstan modern food culture restaurants",
                                "қазақстан заманауи тағам мәдениеті",
                                "Kazakh contemporary cuisine trends",
                            ]
                        },
                    )
                )
            elif request.request_id == "language||kazakh_language":
                responses.append(_response(request.request_id, {"search_queries": ["Kazakh language"]}))
            elif request.request_id == "language||multilingual_media":
                responses.append(
                    _response(
                        request.request_id,
                        {
                            "search_queries": [
                                "Kazakhstan multilingual media Kazakh Russian",
                                "қазақстан көптілді медиа қазақ орыс",
                                "Kazakh Russian English media Kazakhstan",
                            ]
                        },
                    )
                )
            else:
                raise AssertionError(f"Unexpected batch request: {request.request_id}")
        return responses


def test_taxonomy_validators_identify_failing_units() -> None:
    category_report = validate_categories([{"name": "culture", "description": ""}], min_categories=2)
    assert not category_report["passed"]
    assert category_report["gate_status"] == "repair"
    assert category_report["decision"] == "repair"
    assert set(category_report["issue_categories"]) == {"missing_coverage", "schema"}
    assert "category_count_below_minimum" in category_report["blocking_issues"]
    assert "empty_category_description" in category_report["blocking_issues"]

    sub_report = validate_subcategories(
        [{"name": "language", "description": "Language use."}],
        {"language": []},
        min_subcategories_per_category=2,
    )
    assert not sub_report["passed"]
    assert sub_report["gate_status"] == "repair"
    assert sub_report["decision"] == "repair"
    assert sub_report["issue_categories"] == ["missing_coverage"]
    assert sub_report["failed_categories"] == ["language"]

    query_report = validate_query_groups(
        [{"name": "language", "description": "Language use."}],
        {"language": [{"name": "kazakh_language", "description": "Kazakh language."}]},
        {"language": {"kazakh_language": ["Kazakh"]}},
        min_queries_per_subcategory=3,
    )
    assert not query_report["passed"]
    assert query_report["gate_status"] == "repair"
    assert query_report["decision"] == "repair"
    assert query_report["issue_categories"] == ["missing_coverage"]
    assert query_report["failed_query_groups"] == [{"category": "language", "subcategory": "kazakh_language"}]


def test_taxonomy_validator_warns_without_blocking_on_missing_recommended_dimensions() -> None:
    report = validate_categories(
        [
            {"name": "cuisine", "description": "Food culture and meals."},
            {"name": "music", "description": "Songs and instruments."},
        ],
        min_categories=2,
        required_dimension_terms={"regional": ["landscape"]},
    )

    assert report["passed"] is True
    assert report["gate_status"] == "warn"
    assert report["decision"] == "continue"
    assert report["issue_categories"] == ["missing_coverage"]


def test_generate_taxonomy_mini_loop_repairs_categories_subcategories_and_queries() -> None:
    client = ScriptedTaxonomyClient()
    with patch("core.llm.client.LLMClient.from_env", return_value=client):
        result = GenerateTaxonomyTool().execute(
            "Kazakhstan",
            {
                "batch_size": 4,
                "batch_delay": 0.0,
                "taxonomy_min_categories": 2,
                "taxonomy_min_subcategories": 2,
                "taxonomy_min_queries": 3,
                "taxonomy_repair_attempts": 2,
            },
        )

    assert [category["name"] for category in result["categories"]] == ["cuisine", "language"]
    assert len(result["category_subcategories"]["language"]) == 2
    assert len(result["category_subcategory_queries"]["language"]["kazakh_language"]) == 3
    assert result["image_taxonomy"]["schema_version"] == "image_taxonomy_v1"
    assert result["image_taxonomy"]["slots"]

    quality = result["taxonomy_quality"]
    assert quality["passed"] is True
    assert quality["category_report"]["passed"] is True
    assert quality["subcategory_report"]["passed"] is True
    assert quality["query_report"]["passed"] is True
    assert [attempt["stage"] for attempt in quality["repair_attempts"]] == [
        "categories",
        "subcategories",
        "queries",
    ]

    assert "categories_refine" in client.sync_request_ids
    assert "repair_subcategories::language" in client.sync_request_ids
    assert "repair_queries::language||kazakh_language" in client.sync_request_ids
