from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.collect_data.image_search import (
    _image_extension_from_content_type,
    _image_extension_from_url,
    _normalize_image_record,
)
from tools.collect_data.tool import CollectDataTool


async def _fake_search_and_scrape(**kwargs: Any) -> dict[str, list[dict[str, str]]]:
    return {
        "kazakh food": [
            {
                "url": "https://example.com/article",
                "google_snippet": "snippet",
                "full_text": "Collected article text.",
            }
        ]
    }


def test_serper_image_record_normalization_prefers_full_image_url() -> None:
    record = _normalize_image_record(
        "kazakh food",
        {
            "title": "Beshbarmak",
            "source": "example",
            "link": "https://example.com/page",
            "imageUrl": "https://cdn.example.com/full.webp",
            "thumbnailUrl": "https://cdn.example.com/thumb.jpg",
        },
    )

    assert record == {
        "query": "kazakh food",
        "title": "Beshbarmak",
        "source": "example",
        "url": "https://example.com/page",
        "img_url": "https://cdn.example.com/full.webp",
        "thumbnail_url": "https://cdn.example.com/thumb.jpg",
    }


def test_serper_image_extension_detection() -> None:
    assert _image_extension_from_content_type("image/webp; charset=binary") == ".webp"
    assert _image_extension_from_url("https://example.com/path/photo.PNG?width=100") == ".png"
    assert _image_extension_from_url("https://example.com/path/photo.txt") == ""


def test_collect_data_uses_serper_image_search_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_collect_images_from_serper(
        queries: list[str],
        out_dir: Path,
        *,
        serper_key: str,
        results_per_query: int,
        concurrency: int,
        min_width: int,
        min_height: int,
    ) -> list[dict[str, str]]:
        captured.update(
            {
                "queries": queries,
                "out_dir": out_dir,
                "serper_key": serper_key,
                "results_per_query": results_per_query,
                "concurrency": concurrency,
                "min_width": min_width,
                "min_height": min_height,
            }
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / "image.jpg"
        image_path.write_bytes(b"fake-image")
        return [
            {
                "query": queries[0],
                "url": "https://example.com/source",
                "img_url": "https://example.com/image.jpg",
                "file_path": str(image_path),
            }
        ]

    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")
    monkeypatch.setattr("tools.collect_data.tool._search_and_scrape", _fake_search_and_scrape)
    monkeypatch.setattr("tools.collect_data.tool.collect_images_from_serper", fake_collect_images_from_serper)

    result = CollectDataTool().execute(
        {
            "queries": ["kazakh food"],
            "run_dir": str(tmp_path),
            "collect_images": True,
            "image_search_results_per_query": 7,
            "concurrency": 3,
            "image_min_width": 512,
            "image_min_height": 384,
        }
    )

    assert captured == {
        "queries": ["kazakh food"],
        "out_dir": tmp_path / "images",
        "serper_key": "test-serper-key",
        "results_per_query": 7,
        "concurrency": 3,
        "min_width": 512,
        "min_height": 384,
    }
    assert result["num_samples"] == 1
    assert result["metadata"]["image_collection_mode"] == "serper"
    assert result["metadata"]["image_search_results_per_query"] == 7
    assert result["metadata"]["num_images"] == 1
    assert Path(result["metadata"]["images_index"]).exists()


def test_collect_data_uses_image_taxonomy_query_specs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    image_taxonomy = {
        "schema_version": "image_taxonomy_v1",
        "slots": [
            {
                "slot_id": "food_drink_table_culture__traditional_food",
                "domain_id": "food_drink_table_culture",
                "subdomain_id": "traditional_food",
                "domain_label": "Food, Drink, and Table Culture",
                "subdomain_label": "Traditional Food",
                "visual_skills": ["object_recognition"],
                "queries": [
                    {"query": "Kazakhstan traditional food close-up", "query_intent": "object_closeup"}
                ],
            }
        ],
    }

    async def fake_collect_images_from_serper(
        queries: list[dict[str, str]],
        out_dir: Path,
        *,
        serper_key: str,
        results_per_query: int,
        concurrency: int,
        min_width: int,
        min_height: int,
    ) -> list[dict[str, str]]:
        captured["queries"] = queries
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / "food.jpg"
        image_path.write_bytes(b"fake-image")
        return [
            {
                **queries[0],
                "url": "https://example.com/source",
                "img_url": "https://example.com/food.jpg",
                "file_path": str(image_path),
            }
        ]

    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")
    monkeypatch.setattr("tools.collect_data.tool._search_and_scrape", _fake_search_and_scrape)
    monkeypatch.setattr("tools.collect_data.tool.collect_images_from_serper", fake_collect_images_from_serper)

    result = CollectDataTool().execute(
        {
            "queries": ["kazakh food"],
            "run_dir": str(tmp_path),
            "collect_images": True,
            "image_taxonomy": image_taxonomy,
        }
    )

    assert captured["queries"][0]["query"] == "Kazakhstan traditional food close-up"
    assert captured["queries"][0]["domain_id"] == "food_drink_table_culture"
    assert captured["queries"][0]["subdomain_id"] == "traditional_food"
    assert result["metadata"]["num_image_query_specs"] == 1
    assert result["metadata"]["image_taxonomy_schema_version"] == "image_taxonomy_v1"


def test_collect_data_runs_image_dedup_when_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def fake_collect_images_from_serper(
        queries: list[str],
        out_dir: Path,
        *,
        serper_key: str,
        results_per_query: int,
        concurrency: int,
        min_width: int,
        min_height: int,
    ) -> list[dict[str, str]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        first = out_dir / "first.jpg"
        second = out_dir / "second.jpg"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        return [
            {"query": queries[0], "img_url": "https://example.com/1.jpg", "file_path": str(first)},
            {"query": queries[0], "img_url": "https://example.com/2.jpg", "file_path": str(second)},
        ]

    def fake_deduplicate_image_records(
        records: list[dict[str, str]],
        *,
        output_dir: str | Path,
        threshold: float,
        model_path: str,
        model_url: str,
        batch_size: int,
        max_reported_pairs: int,
        device: str | None,
    ) -> dict[str, Any]:
        assert threshold == 0.91
        assert batch_size == 8
        return {
            "records": [records[0]],
            "report": {
                "schema_version": "image_dedup.v1",
                "method": "sscd",
                "threshold": threshold,
                "model_path": model_path,
                "model_url": model_url,
                "device": device or "cuda",
                "downloaded_model": True,
                "num_input_records": 2,
                "num_kept_records": 1,
                "num_removed_records": 1,
                "num_duplicate_clusters": 1,
            },
        }

    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")
    monkeypatch.setattr("tools.collect_data.tool._search_and_scrape", _fake_search_and_scrape)
    monkeypatch.setattr("tools.collect_data.tool.collect_images_from_serper", fake_collect_images_from_serper)
    monkeypatch.setattr("tools.collect_data.tool.deduplicate_image_records", fake_deduplicate_image_records)

    result = CollectDataTool().execute(
        {
            "queries": ["kazakh food"],
            "run_dir": str(tmp_path),
            "collect_images": True,
            "image_dedup_enable": True,
            "image_dedup_threshold": 0.91,
            "image_dedup_batch_size": 8,
        }
    )

    metadata = result["metadata"]
    assert metadata["num_images"] == 1
    assert metadata["num_images_before_dedup"] == 2
    assert metadata["num_images_removed_by_dedup"] == 1
    assert metadata["image_dedup_enabled"] is True
    assert metadata["image_dedup_downloaded_model"] is True
    assert Path(metadata["raw_images_index"]).exists()
    assert Path(metadata["image_dedup_report_path"]).exists()
    final_manifest = json.loads(Path(metadata["images_index"]).read_text(encoding="utf-8"))
    raw_manifest = json.loads(Path(metadata["raw_images_index"]).read_text(encoding="utf-8"))
    assert len(final_manifest) == 1
    assert len(raw_manifest) == 2


def test_collect_data_can_use_legacy_html_image_collection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_collect_images(
        links: list[str],
        out_dir: Path,
        *,
        concurrency: int,
        context_size: int,
        min_width: int,
        min_height: int,
    ) -> list[dict[str, str]]:
        captured.update(
            {
                "links": links,
                "out_dir": out_dir,
                "concurrency": concurrency,
                "context_size": context_size,
                "min_width": min_width,
                "min_height": min_height,
            }
        )
        return []

    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")
    monkeypatch.setattr("tools.collect_data.tool._search_and_scrape", _fake_search_and_scrape)
    monkeypatch.setattr("tools.collect_data.tool.collect_images", fake_collect_images)

    result = CollectDataTool().execute(
        {
            "queries": ["kazakh food"],
            "run_dir": str(tmp_path),
            "collect_images": True,
            "image_collection_mode": "html",
            "image_context_size": 250,
        }
    )

    assert captured["links"] == ["https://example.com/article"]
    assert captured["out_dir"] == tmp_path / "images"
    assert captured["context_size"] == 250
    assert result["metadata"]["image_collection_mode"] == "html"
    assert result["metadata"]["num_images"] == 0
