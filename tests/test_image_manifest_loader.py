from __future__ import annotations

import json
from pathlib import Path

from tools.build_sft_dataset.loaders import load_images_from_manifest
from tools.build_sft_dataset.tool import BuildSftDatasetTool


def test_load_images_from_manifest_uses_taxonomy_metadata_for_topic_hint(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "food.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"fake-image")
    manifest_path = tmp_path / "images.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "file_path": str(image_path),
                    "domain_id": "food_drink_table_culture",
                    "subdomain_id": "traditional_food",
                    "domain_label": "Food, Drink, and Table Culture",
                    "subdomain_label": "Traditional Food",
                    "query": "Kazakhstan traditional food close-up",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    items = load_images_from_manifest(str(manifest_path))

    assert len(items) == 1
    assert items[0].image_path == str(image_path)
    assert "Food, Drink, and Table Culture" in (items[0].topic_hint or "")
    assert "Traditional Food" in (items[0].topic_hint or "")
    assert "Kazakhstan traditional food close-up" in (items[0].topic_hint or "")


def test_build_sft_image_mode_prefers_manifest_when_provided(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "food.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"fake-image")
    manifest_path = tmp_path / "images.json"
    manifest_path.write_text(
        json.dumps([{"file_path": str(image_path), "domain_id": "food", "subdomain_id": "traditional_food"}]),
        encoding="utf-8",
    )

    items = BuildSftDatasetTool()._load_items(
        "image",
        {
            "input_dir": str(tmp_path / "images"),
            "image_manifest": str(manifest_path),
        },
    )

    assert len(items) == 1
    assert items[0].item_id == str(image_path)
    assert items[0].topic_hint == "food / traditional_food"
