from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, load_from_disk

from config import PipelineConfig
from core.agentic.action_space import ActionType
from core.agentic.models import ActionRequest, PipelineState
from core.agentic.tool_adapters import AgenticToolAdapter
from core.agentic.validators import validate_collection_output, validate_sft_output
from tools.build_sft_dataset.tool import _annotation_cache_signature
from tools.train.tool import TrainTool


class FakeTool:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls = []

    def execute(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.output


class ExplodingTool:
    def execute(self, *args, **kwargs):
        raise AssertionError("real tool should not be called in debug stub mode")


class RecordingSftTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(config)
        input_path = Path(config["input_jsonl"])
        rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        output_sft = Path(config["output_sft"])
        output_annotations = Path(config["output_annotations"])
        output_sft.parent.mkdir(parents=True, exist_ok=True)
        output_annotations.parent.mkdir(parents=True, exist_ok=True)
        with output_sft.open("w", encoding="utf-8") as handle:
            for idx, row in enumerate(rows):
                handle.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": f"Question {idx}"},
                                {"role": "assistant", "content": "Answer"},
                            ],
                            "group_key": row.get("group_key"),
                            "source_url": row.get("source_url"),
                            "source_excerpt": row.get("source_excerpt"),
                            "collection_iteration": row.get("collection_iteration"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        output_annotations.write_text('{"success": true}\n', encoding="utf-8")
        return {
            "mode": config["mode"],
            "num_items": len(rows),
            "num_annotations": len(rows),
            "num_examples": len(rows),
            "num_failures": 0,
            "annotations_path": str(output_annotations),
            "sft_path": str(output_sft),
            "prompt_preset": config.get("prompt_preset", "default"),
        }


def test_text_annotation_cache_signature_includes_focus() -> None:
    focused = _annotation_cache_signature(
        None,
        target_language="English",
        prompt_preset="default",
        focus="traditional culture",
        provider="fake-provider",
        model="fake-model",
    )
    broad = _annotation_cache_signature(
        None,
        target_language="English",
        prompt_preset="default",
        focus="",
        provider="fake-provider",
        model="fake-model",
    )

    assert focused["focus"] == "traditional culture"
    assert focused != broad


def test_generate_taxonomy_adapter_flattens_queries(tmp_path: Path) -> None:
    taxonomy_tool = FakeTool(
        {
            "categories": ["food", "music"],
            "category_subcategories": {"food": ["dishes"], "music": ["instruments"]},
            "category_subcategory_queries": {
                "food": {"dishes": ["kazakh food", "beshbarmak"]},
                "music": {"instruments": ["dombra music"]},
            },
        }
    )
    adapter = AgenticToolAdapter({"generate_taxonomy": taxonomy_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "country": "Kazakhstan",
            "focus": "traditional culture",
            "llm_batch_size": 2,
            "llm_batch_delay": 0.0,
        },
    )

    result = adapter.execute_generate_taxonomy(
        state,
        ActionRequest(ActionType.GENERATE_TAXONOMY),
    )

    assert result.status == "success"
    assert result.quality_report and result.quality_report.passed
    assert result.artifacts["search_queries"] == ["kazakh food", "beshbarmak", "dombra music"]
    assert taxonomy_tool.calls[0]["args"][0] == "Kazakhstan"
    assert taxonomy_tool.calls[0]["args"][1]["batch_size"] == 2
    assert taxonomy_tool.calls[0]["args"][1]["focus"] == "traditional culture"


def test_generate_taxonomy_adapter_fails_when_taxonomy_quality_fails(tmp_path: Path) -> None:
    taxonomy_tool = FakeTool(
        {
            "categories": [{"name": "culture", "description": "Too broad."}],
            "category_subcategories": {"culture": [{"name": "general", "description": "Generic."}]},
            "category_subcategory_queries": {"culture": {"general": ["Kazakhstan culture"]}},
            "taxonomy_quality": {"passed": False, "score": 0.3},
        }
    )
    adapter = AgenticToolAdapter({"generate_taxonomy": taxonomy_tool})
    state = PipelineState(run_dir=str(tmp_path), config={"country": "Kazakhstan"})

    result = adapter.execute_generate_taxonomy(state, ActionRequest(ActionType.GENERATE_TAXONOMY))

    assert result.status == "failed"
    assert result.quality_report
    assert "taxonomy_quality_failed" in result.quality_report.blocking_issues


def test_generate_taxonomy_adapter_can_limit_queries_per_category(tmp_path: Path) -> None:
    taxonomy_tool = FakeTool(
        {
            "categories": ["food", "music"],
            "category_subcategories": {"food": ["dishes", "drinks"], "music": ["instruments"]},
            "category_subcategory_queries": {
                "food": {"dishes": ["food q1", "food q2"], "drinks": ["food q3"]},
                "music": {"instruments": ["music q1", "music q2"]},
            },
        }
    )
    adapter = AgenticToolAdapter({"generate_taxonomy": taxonomy_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "country": "Kazakhstan",
            "max_queries_per_category": 1,
            "max_queries": 2,
        },
    )

    result = adapter.execute_generate_taxonomy(state, ActionRequest(ActionType.GENERATE_TAXONOMY))

    assert result.status == "success"
    assert result.artifacts["search_queries"] == ["food q1", "music q1"]


def test_collect_data_adapter_forwards_image_collection_config(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "collect" / "dataset"
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    dataset_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    images_index.write_text("[]", encoding="utf-8")

    collect_tool = FakeTool(
        {
            "data_path": str(dataset_dir),
            "num_samples": 4,
            "metadata": {
                "provider": "serper",
                "images_dir": str(images_dir),
                "images_index": str(images_index),
                "num_images": 3,
            },
        }
    )
    adapter = AgenticToolAdapter({"collect_data": collect_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "serper_results_per_query": 7,
            "serper_top_results": 2,
            "serper_concurrency": 3,
            "collect_images": True,
            "image_min_width": 512,
            "image_min_height": 384,
            "image_context_size": 250,
            "image_collection_mode": "serper",
            "image_search_results_per_query": 11,
            "coverage_added_queries": ["q3", "q1"],
        },
        artifacts={"search_queries": ["q1", "q2"]},
    )

    result = adapter.execute_collect_data(state, ActionRequest(ActionType.COLLECT_DATA))

    called_config = collect_tool.calls[0]["args"][0]
    assert called_config["queries"] == ["q1", "q2", "q3"]
    assert called_config["collect_images"] is True
    assert called_config["image_min_width"] == 512
    assert called_config["image_min_height"] == 384
    assert called_config["image_context_size"] == 250
    assert called_config["image_collection_mode"] == "serper"
    assert called_config["image_search_results_per_query"] == 11
    assert result.status == "success"
    assert result.quality_report and result.quality_report.passed
    assert result.artifacts["images_dir"] == str(images_dir)
    assert result.artifacts["num_images"] == 3


def test_collect_data_adapter_writes_text_quality_when_raw_path_present(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "collect" / "dataset"
    dataset_dir.mkdir(parents=True)
    raw_path = tmp_path / "collect" / "serper_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "q1": [
                    {"url": "https://example.com/a?utm=1", "full_text": "Same collected text."},
                    {"url": "https://www.example.com/a", "full_text": "Same collected text."},
                ]
            }
        ),
        encoding="utf-8",
    )
    collect_tool = FakeTool(
        {
            "data_path": str(dataset_dir),
            "num_samples": 2,
            "metadata": {"provider": "serper", "raw_result_path": str(raw_path)},
        }
    )
    adapter = AgenticToolAdapter({"collect_data": collect_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"text_quality_enable_embeddings": False},
        artifacts={"search_queries": ["q1"]},
    )

    result = adapter.execute_collect_data(state, ActionRequest(ActionType.COLLECT_DATA))

    assert result.status == "success"
    quality_path = Path(result.artifacts["collection_text_quality_path"])
    assert quality_path.exists()
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["exact_duplicate_count"] == 1
    assert quality["url_duplicate_count"] == 1
    assert result.artifacts["collection_text_quality_summary"]["embedding_enabled"] is False


def test_assess_coverage_refines_weak_collection_queries(tmp_path: Path) -> None:
    raw_path = tmp_path / "collect" / "serper_raw.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"kazakh food": [], "dombra music": [{"url": "u"}]}), encoding="utf-8")
    adapter = AgenticToolAdapter({})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"country": "Kazakhstan", "coverage_min_text_samples": 3},
        artifacts={
            "search_queries": ["kazakh food", "dombra music"],
            "num_samples": 1,
            "collection_metadata": {"raw_result_path": str(raw_path)},
        },
    )

    result = adapter.execute_assess_coverage_and_refine_queries(
        state,
        ActionRequest(ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES),
    )

    assert result.status == "failed"
    assert result.quality_report
    assert "coverage_text_samples_below_minimum" in result.quality_report.blocking_issues
    assert result.artifacts["coverage_added_queries"]
    assert "kazakh food" in result.artifacts["coverage_review"]["weak_text_queries"]


def test_assess_coverage_does_not_emit_repair_queries_when_passing(tmp_path: Path) -> None:
    raw_path = tmp_path / "collect" / "serper_raw.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps({"kazakh food": [{"url": "u1"}], "dombra music": [{"url": "u2"}]}),
        encoding="utf-8",
    )
    adapter = AgenticToolAdapter({})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"country": "Kazakhstan", "coverage_min_text_samples": 1},
        artifacts={
            "search_queries": ["kazakh food", "dombra music"],
            "num_samples": 2,
            "collection_metadata": {"raw_result_path": str(raw_path)},
        },
    )

    result = adapter.execute_assess_coverage_and_refine_queries(
        state,
        ActionRequest(ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES),
    )

    assert result.status == "success"
    assert result.artifacts["coverage_added_queries"] == []
    assert result.artifacts["coverage_review"]["added_queries"] == []
    assert result.artifacts["coverage_review"]["candidate_text_queries"] == []


def test_assess_source_quality_filters_text_dataset_and_updates_data_path(tmp_path: Path) -> None:
    raw_dataset_dir = tmp_path / "collect" / "dataset"
    Dataset.from_list(
        [
            {
                "text": "Kazakh yurt construction uses a shanyrak crown, kerege lattice walls, and felt cover.",
                "source_url": "https://good.example/yurt",
                "source_query": "Kazakh yurt shanyrak",
                "group_key": "good",
            },
            {
                "text": "Menu Login Privacy Search Tags Categories Subscribe Copyright",
                "source_url": "https://bad.example/search?q=yurt",
                "source_query": "Kazakh yurt shanyrak",
                "group_key": "bad",
            },
        ]
    ).save_to_disk(str(raw_dataset_dir))
    adapter = AgenticToolAdapter({})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "training_modality": "text",
            "source_quality_oracle_enable": False,
            "source_quality_min_kept_rows": 1,
            "source_quality_min_source_groups": 1,
            "source_quality_max_domain_share": 1.0,
            "source_quality_min_avg_score": 0.0,
            "source_quality_min_quality_score": 0.10,
            "source_quality_accumulate_kept_sources": False,
        },
        artifacts={
            "raw_data_path": str(raw_dataset_dir),
            "taxonomy": {"categories": ["Kazakh yurt"]},
            "search_queries": ["Kazakh yurt shanyrak"],
        },
    )

    result = adapter.execute_assess_source_quality(state, ActionRequest(ActionType.ASSESS_SOURCE_QUALITY))

    assert result.status == "success"
    assert result.quality_report and result.quality_report.passed
    assert result.artifacts["data_path"] == result.artifacts["source_quality_filtered_data_path"]
    filtered = load_from_disk(result.artifacts["data_path"])
    assert len(filtered) == 1
    assert filtered[0]["group_key"] == "good"
    assert Path(result.artifacts["source_quality_report_path"]).exists()


def test_build_sft_adapter_uses_collected_images_for_image_mode(tmp_path: Path) -> None:
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    images_dir.mkdir(parents=True)
    images_index.write_text("[]", encoding="utf-8")
    image_path = images_dir / "image.jpg"
    image_path.write_bytes(b"fake")
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "Describe this image."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "A caption."}]},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_path.write_text('{"success": true}\n', encoding="utf-8")

    sft_tool = FakeTool(
        {
            "mode": "image",
            "num_items": 1,
            "num_annotations": 1,
            "num_examples": 1,
            "num_failures": 0,
            "annotations_path": str(annotations_path),
            "sft_path": str(sft_path),
        }
    )
    adapter = AgenticToolAdapter({"build_sft_dataset": sft_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "sft_mode": "image",
            "sft_target_language": "English",
            "sft_prompt_preset": "schema_strict",
            "focus": "traditional culture",
        },
        artifacts={"images_dir": str(images_dir), "images_index": str(images_index)},
    )

    result = adapter.execute_build_sft_dataset(
        state,
        ActionRequest(ActionType.BUILD_SFT_DATASET),
    )

    called_config = sft_tool.calls[0]["args"][0]
    assert called_config["mode"] == "image"
    assert called_config["input_dir"] == str(images_dir)
    assert called_config["image_manifest"] == str(images_index)
    assert called_config["prompt_preset"] == "schema_strict"
    assert called_config["focus"] == "traditional culture"
    assert called_config["image_tasks"] == ["caption"]
    assert "input_jsonl" not in called_config
    assert result.status == "success"
    assert result.quality_report and result.quality_report.passed
    assert result.artifacts["sft_path"] == str(sft_path)
    assert result.artifacts["training_modality"] == "image"


def test_build_sft_adapter_writes_text_quality(tmp_path: Path) -> None:
    collected_path = tmp_path / "sft" / "collected_texts.jsonl"
    collected_path.parent.mkdir(parents=True)
    collected_path.write_text('{"text": "source"}\n', encoding="utf-8")
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    sft_path.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}),
                json.dumps({"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_path.write_text('{"success": true}\n', encoding="utf-8")
    sft_tool = FakeTool(
        {
            "mode": "text",
            "num_items": 2,
            "num_annotations": 2,
            "num_examples": 2,
            "num_failures": 0,
            "annotations_path": str(annotations_path),
            "sft_path": str(sft_path),
        }
    )
    adapter = AgenticToolAdapter({"build_sft_dataset": sft_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"sft_mode": "text", "sft_target_language": "English", "text_quality_enable_embeddings": False},
        artifacts={"collected_texts_jsonl": str(collected_path)},
    )

    result = adapter.execute_build_sft_dataset(state, ActionRequest(ActionType.BUILD_SFT_DATASET))

    assert result.status == "success"
    quality_path = Path(result.artifacts["sft_text_quality_path"])
    assert quality_path.exists()
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["exact_duplicate_count"] == 1
    assert result.artifacts["sft_text_quality_summary"]["num_records"] == 2


def test_build_sft_adapter_creates_heldout_source_eval_set(tmp_path: Path) -> None:
    collected_path = tmp_path / "sft" / "collected_texts.jsonl"
    collected_path.parent.mkdir(parents=True)
    collected_rows = [
        {"text": "Source A text", "group_key": "source-a", "source_excerpt": "A"},
        {"text": "Source B text", "group_key": "source-b", "source_excerpt": "B"},
        {"text": "Source C text", "group_key": "source-c", "source_excerpt": "C"},
        {"text": "Source D text", "group_key": "source-d", "source_excerpt": "D"},
    ]
    collected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in collected_rows),
        encoding="utf-8",
    )
    sft_tool = RecordingSftTool()
    adapter = AgenticToolAdapter({"build_sft_dataset": sft_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "sft_mode": "text",
            "sft_target_language": "English",
            "source_eval_ratio": 0.5,
            "source_eval_max_items": 2,
            "seed": 7,
            "focus": "traditional culture",
        },
        artifacts={"collected_texts_jsonl": str(collected_path)},
    )

    result = adapter.execute_build_sft_dataset(state, ActionRequest(ActionType.BUILD_SFT_DATASET))

    assert result.status == "success"
    assert len(sft_tool.calls) == 2
    assert sft_tool.calls[0]["focus"] == "traditional culture"
    assert sft_tool.calls[1]["focus"] == "traditional culture"
    assert result.artifacts["heldout_eval_sft_path"].endswith("heldout_eval_sft.jsonl")
    assert Path(result.artifacts["heldout_eval_sft_path"]).exists()
    summary = result.artifacts["source_split_summary"]
    assert summary["split_strategy"] == "source_group"
    assert summary["num_train_source_rows"] == 2
    assert summary["num_eval_source_rows"] == 2
    train_groups = {
        json.loads(line)["group_key"]
        for line in Path(sft_tool.calls[0]["input_jsonl"]).read_text(encoding="utf-8").splitlines()
    }
    eval_groups = {
        json.loads(line)["group_key"]
        for line in Path(sft_tool.calls[1]["input_jsonl"]).read_text(encoding="utf-8").splitlines()
    }
    assert train_groups.isdisjoint(eval_groups)


def test_build_sft_adapter_merges_text_sources_across_collection_iterations(tmp_path: Path) -> None:
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir(parents=True)
    previous_rows = [
        {
            "text": "Old Source A text",
            "source_url": "https://example.com/a",
            "group_key": "source-a",
            "collection_iteration": "iteration_0",
        },
        {
            "text": "Shared Source B text",
            "source_url": "https://example.com/b",
            "group_key": "source-b",
            "collection_iteration": "iteration_0",
        },
    ]
    (sft_dir / "collected_texts_merged.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in previous_rows),
        encoding="utf-8",
    )
    current_path = sft_dir / "current_collection.jsonl"
    current_rows = [
        {"text": "Shared Source B text", "source_url": "https://example.com/b", "group_key": "source-b"},
        {"text": "New Source C text", "source_url": "https://example.com/c", "group_key": "source-c"},
    ]
    current_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in current_rows),
        encoding="utf-8",
    )
    sft_tool = RecordingSftTool()
    adapter = AgenticToolAdapter({"build_sft_dataset": sft_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "sft_mode": "text",
            "sft_target_language": "English",
            "source_eval_ratio": 0.34,
            "source_eval_max_items": 1,
            "seed": 3,
        },
        artifacts={"collected_texts_jsonl": str(current_path)},
        retry_counts={ActionType.COLLECT_DATA.value: 1},
    )

    result = adapter.execute_build_sft_dataset(state, ActionRequest(ActionType.BUILD_SFT_DATASET))

    assert result.status == "success"
    registry_rows = [
        json.loads(line)
        for line in Path(result.artifacts["source_registry_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(registry_rows) == 3
    by_url = {row["source_url"]: row for row in registry_rows}
    assert by_url["https://example.com/a"]["collection_iteration"] == "iteration_0"
    assert by_url["https://example.com/c"]["collection_iteration"] == "iteration_1"
    assert by_url["https://example.com/b"]["seen_collection_iterations"] == ["iteration_0", "iteration_1"]
    summary = result.artifacts["source_registry_summary"]
    assert summary["num_previous_source_rows"] == 2
    assert summary["num_current_source_rows"] == 2
    assert summary["num_merged_source_rows"] == 3
    assert summary["num_new_source_rows"] == 1
    assert summary["num_existing_source_rows_seen"] == 1


def test_build_sft_adapter_does_not_push_dataset_before_split(monkeypatch, tmp_path: Path) -> None:
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    images_dir.mkdir(parents=True)
    images_index.write_text("[]", encoding="utf-8")
    image_path = images_dir / "image.jpg"
    image_path.write_bytes(b"fake")
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "Describe this image."},
                        ],
                    },
                    {"role": "assistant", "content": "A caption."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_path.write_text('{"success": true}\n', encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def _push_dataset(local_path, repo_name, *, username=None, private=True, card_readme=None):
        calls.append(
            {
                "local_path": local_path,
                "repo_name": repo_name,
                "username": username,
                "private": private,
                "card_readme": card_readme,
            }
        )
        return f"{username}/{repo_name}"

    monkeypatch.setattr("core.hf_hub.push_dataset", _push_dataset)
    sft_tool = FakeTool(
        {
            "mode": "image",
            "num_items": 1,
            "num_annotations": 1,
            "num_examples": 1,
            "num_failures": 0,
            "annotations_path": str(annotations_path),
            "sft_path": str(sft_path),
        }
    )
    adapter = AgenticToolAdapter({"build_sft_dataset": sft_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "sft_mode": "image",
            "sft_target_language": "English",
            "hf_dataset_repo": "test-owner/test-dataset",
        },
        artifacts={"images_dir": str(images_dir), "images_index": str(images_index)},
    )

    result = adapter.execute_build_sft_dataset(state, ActionRequest(ActionType.BUILD_SFT_DATASET))

    assert result.status == "success"
    assert "dataset_repo_id" not in result.artifacts
    assert calls == []


def test_build_dataset_adapter_pushes_split_dataset_to_hf(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    calls: list[dict[str, Any]] = []

    def _push_dataset(local_path, repo_name, *, username=None, private=True, card_readme=None):
        calls.append(
            {
                "local_path": local_path,
                "repo_name": repo_name,
                "username": username,
                "private": private,
                "card_readme": card_readme,
            }
        )
        return f"{username}/{repo_name}"

    monkeypatch.setattr("core.hf_hub.push_dataset", _push_dataset)
    build_dataset_tool = FakeTool(
        {
            "dataset_ref": {
                "kind": "hf",
                "data_path": str(dataset_dir),
                "split": "train",
                "eval_split": "validation",
                "split_counts": {"train": 8, "validation": 2},
            },
            "dataset_summary": {
                "data_path": str(dataset_dir),
                "columns": ["messages", "group_key"],
                "sample_count": 10,
                "split_counts": {"train": 8, "validation": 2},
                "split_strategy": "group",
                "group_key_column": "group_key",
            },
            "dataset_manifest_path": str(manifest_path),
        }
    )
    adapter = AgenticToolAdapter({"build_dataset": build_dataset_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"sft_mode": "text", "sft_target_language": "English", "hf_dataset_repo": "test-owner/test-dataset"},
        artifacts={
            "sft_path": str(tmp_path / "sft.jsonl"),
            "num_sft_examples": 10,
        },
    )

    result = adapter.execute_build_dataset(state, ActionRequest(ActionType.BUILD_DATASET))

    assert result.status == "success"
    assert result.artifacts["dataset_repo_id"] == "test-owner/test-dataset"
    assert result.artifacts["hf_dataset_card_updated"] is True
    assert calls[0]["local_path"] == str(dataset_dir)
    assert calls[0]["repo_name"] == "test-dataset"
    assert calls[0]["username"] == "test-owner"
    assert "Split strategy" in calls[0]["card_readme"]
    assert "group" in calls[0]["card_readme"]


def test_pipeline_config_syncs_training_modality_and_legacy_sft_mode(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        mode="full_agentic",
        country="Kazakhstan",
        run_dir=str(tmp_path),
        training_modality="image",
        sft_target_language="English",
        hf_model_id="test-model",
    )

    assert cfg.training_modality == "image"
    assert cfg.sft_mode == "image"

    legacy_cfg = PipelineConfig(
        mode="full_agentic",
        country="Kazakhstan",
        run_dir=str(tmp_path),
        sft_mode="image",
        sft_target_language="English",
        hf_model_id="test-model",
    )

    assert legacy_cfg.training_modality == "image"
    assert legacy_cfg.sft_mode == "image"


def test_pipeline_config_normalizes_image_sft_tasks() -> None:
    cfg = PipelineConfig(
        country="Kazakhstan",
        sft_target_language="English",
        hf_model_id="test-model",
        image_sft_tasks="caption,vqa,ocr",
    )

    assert cfg.image_sft_tasks == ["caption", "vqa", "ocr"]

    all_cfg = PipelineConfig(
        country="Kazakhstan",
        sft_target_language="English",
        hf_model_id="test-model",
        image_sft_tasks="all",
    )

    assert all_cfg.image_sft_tasks == ["caption", "vqa", "ocr", "reason", "instruct_follow"]


def test_real_train_tool_fails_fast_for_unsupported_modality(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="Only text and single-image vision-language SFT"):
        TrainTool().execute(
            {"kind": "hf", "data_path": str(tmp_path / "sft.jsonl"), "split": "train"},
            {
                "method": "sft",
                "run_dir": str(tmp_path),
                "hf_model_id": "test-model",
                "training_modality": "multimodal",
            },
        )


def test_collection_validator_fails_image_gate_when_images_requested_without_images(tmp_path: Path) -> None:
    report = validate_collection_output(
        {
            "data_path": str(tmp_path / "dataset"),
            "num_samples": 5,
            "metadata": {"num_images": 0},
        },
        collect_images=True,
    )

    assert not report.passed
    assert "num_images_below_minimum" in report.blocking_issues


def test_collection_validator_fails_when_text_filter_removes_all_rows(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    report = validate_collection_output(
        {
            "data_path": str(dataset_dir),
            "num_samples": 1,
            "metadata": {
                "text_filter_summary": {
                    "enabled": True,
                    "num_input": 4,
                    "num_kept": 0,
                    "num_removed": 4,
                    "removal_rate": 1.0,
                }
            },
        }
    )

    assert not report.passed
    assert "text_filter_removed_all_samples" in report.blocking_issues
    assert report.metrics["text_filter_removed"] == 4


def test_sft_validator_requires_examples_and_sft_path(tmp_path: Path) -> None:
    missing_report = validate_sft_output({"num_examples": 0, "sft_path": str(tmp_path / "missing.jsonl")})
    assert not missing_report.passed
    assert "num_examples_below_minimum" in missing_report.blocking_issues

    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]})
        + "\n",
        encoding="utf-8",
    )
    passing_report = validate_sft_output({"mode": "text", "num_examples": 1, "sft_path": str(sft_path)})
    assert passing_report.passed

    invalid_path = tmp_path / "invalid_sft.jsonl"
    invalid_path.write_text('{"messages": []}\n', encoding="utf-8")
    invalid_report = validate_sft_output({"mode": "text", "num_examples": 1, "sft_path": str(invalid_path)})
    assert not invalid_report.passed
    assert "sft_messages_missing" in invalid_report.blocking_issues


def test_sft_validator_rejects_invalid_image_rows(tmp_path: Path) -> None:
    image_sft_path = tmp_path / "image_sft.jsonl"
    image_sft_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Describe"}, {"role": "assistant", "content": "A"}]})
        + "\n",
        encoding="utf-8",
    )

    report = validate_sft_output({"mode": "image", "num_examples": 1, "sft_path": str(image_sft_path)})

    assert not report.passed
    assert "sft_image_content_missing" in report.blocking_issues


def test_debug_stub_train_returns_valid_training_contract(tmp_path: Path) -> None:
    adapter = AgenticToolAdapter({"train": ExplodingTool()})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "debug_stub_train": True,
            "max_steps": 7,
            "train_batch_size": 2,
            "train_grad_accum": 1,
            "train_lr": 0.001,
            "hf_adapter_repo": "should-not-upload",
        },
        artifacts={"dataset_ref": {"kind": "hf", "data_path": str(tmp_path / "sft.jsonl"), "split": "train"}},
    )

    result = adapter.execute_train_model(state, ActionRequest(ActionType.TRAIN_MODEL))

    assert result.status == "success"
    assert result.quality_report and result.quality_report.passed
    assert Path(result.artifacts["adapter_path"]).exists()
    assert result.artifacts["train_metrics"]["steps"] == 7
    assert result.artifacts["iterations"][0]["metrics"]["steps"] == 7
    assert "hf_adapter_upload_skipped" not in result.artifacts
    assert result.raw_output["training_modality"] == "text"
    assert result.raw_output["debug_stub"] is True


def test_train_adapter_does_not_push_adapter_before_eval(monkeypatch, tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def _push_adapter(local_path, repo_name, *, username=None, private=True, card_readme=None):
        calls.append(
            {
                "local_path": local_path,
                "repo_name": repo_name,
                "username": username,
                "private": private,
                "card_readme": card_readme,
            }
        )
        return f"{username}/{repo_name}"

    monkeypatch.setattr("core.hf_hub.push_adapter", _push_adapter)
    train_tool = FakeTool(
        {
            "adapter_path": str(adapter_dir),
            "log_paths": {},
            "metrics": {"steps": 1, "last_train_loss": 1.0},
            "iteration_record": {"iter_idx": 0, "metrics": {"steps": 1}},
        }
    )
    adapter = AgenticToolAdapter({"train": train_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "hf_model_id": "test-model",
            "hf_adapter_repo": "test-owner/test-adapter",
        },
        artifacts={"dataset_ref": {"kind": "hf", "data_path": str(tmp_path / "sft.jsonl"), "split": "train"}},
    )

    result = adapter.execute_train_model(state, ActionRequest(ActionType.TRAIN_MODEL))

    assert result.status == "success"
    assert "adapter_repo_id" not in result.artifacts
    assert calls == []


def test_eval_adapter_pushes_hf_adapter_after_eval_passes(monkeypatch, tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    predictions_path.write_text('{"id": 1}\n', encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def _push_adapter(local_path, repo_name, *, username=None, private=True, card_readme=None):
        calls.append(
            {
                "local_path": local_path,
                "repo_name": repo_name,
                "username": username,
                "private": private,
                "card_readme": card_readme,
            }
        )
        return f"{username}/{repo_name}"

    monkeypatch.setattr("core.hf_hub.push_adapter", _push_adapter)
    eval_tool = FakeTool(
        {
            "predictions_path": str(predictions_path),
            "failures_path": str(failures_path),
            "cluster_preview": {"clusters": []},
            "metrics": {
                "failure_rate": 0.0,
                "num_predictions": 1,
                "training_health": {"gate_status": "pass"},
                "judge": {
                    "enabled": True,
                    "gate_status": "pass",
                    "quality_score": 1.0,
                    "major_failure_rate": 0.0,
                    "unsupported_grounding_rate": 0.0,
                    "failure_category_counts": {},
                },
            },
            "training_health": {"gate_status": "pass"},
            "judge_summary": {
                "enabled": True,
                "gate_status": "pass",
                "quality_score": 1.0,
                "major_failure_rate": 0.0,
                "unsupported_grounding_rate": 0.0,
            },
        }
    )
    adapter = AgenticToolAdapter({"eval_model": eval_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "hf_model_id": "test-model",
            "eval_enable_llm_judge": True,
            "hf_adapter_repo": "test-owner/test-adapter",
            "focus": "traditional culture",
        },
        artifacts={
            "adapter_path": str(tmp_path / "adapter"),
            "dataset_ref": {"kind": "jsonl", "data_path": str(tmp_path / "sft.jsonl"), "split": "validation"},
            "train_metrics": {"last_train_loss": 1.0},
            "dataset_repo_id": "test-owner/test-dataset",
        },
    )

    result = adapter.execute_evaluate_model(state, ActionRequest(ActionType.EVALUATE_MODEL))

    assert result.status == "success"
    assert eval_tool.calls[0]["args"][2]["focus"] == "traditional culture"
    assert result.artifacts["adapter_repo_id"] == "test-owner/test-adapter"
    assert result.artifacts["hf_adapter_card_updated"] is True
    assert calls[0]["local_path"] == str(tmp_path / "adapter")
    assert calls[0]["repo_name"] == "test-adapter"
    assert calls[0]["username"] == "test-owner"
    assert "Evaluation Summary" in calls[0]["card_readme"]
    assert "Unsupported grounding rate" in calls[0]["card_readme"]


def test_eval_adapter_does_not_push_hf_adapter_when_eval_fails(monkeypatch, tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    predictions_path.write_text('{"id": 1}\n', encoding="utf-8")
    failures_path.write_text('{"id": 1, "label": "knowledge_missing"}\n', encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def _push_adapter(local_path, repo_name, *, username=None, private=True, card_readme=None):
        calls.append({"local_path": local_path, "repo_name": repo_name, "username": username})
        return f"{username}/{repo_name}"

    monkeypatch.setattr("core.hf_hub.push_adapter", _push_adapter)
    eval_tool = FakeTool(
        {
            "predictions_path": str(predictions_path),
            "failures_path": str(failures_path),
            "cluster_preview": {"clusters": [{"label": "knowledge_missing", "count": 1}]},
            "metrics": {
                "failure_rate": 1.0,
                "num_predictions": 1,
                "training_health": {"gate_status": "pass"},
                "judge": {"enabled": False},
            },
            "training_health": {"gate_status": "pass"},
        }
    )
    adapter = AgenticToolAdapter({"eval_model": eval_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"hf_model_id": "test-model", "hf_adapter_repo": "test-owner/test-adapter"},
        artifacts={
            "adapter_path": str(tmp_path / "adapter"),
            "dataset_ref": {"kind": "jsonl", "data_path": str(tmp_path / "sft.jsonl"), "split": "validation"},
        },
    )

    result = adapter.execute_evaluate_model(state, ActionRequest(ActionType.EVALUATE_MODEL))

    assert result.status == "failed"
    assert "adapter_repo_id" not in result.artifacts
    assert calls == []


def test_debug_stub_eval_returns_valid_happy_eval_contract(tmp_path: Path) -> None:
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Question"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter = AgenticToolAdapter({"eval_model": ExplodingTool()})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"debug_stub_eval": True, "eval_max_samples": 2},
        artifacts={
            "adapter_path": str(adapter_dir),
            "dataset_ref": {"kind": "hf", "data_path": str(data_path), "split": "train"},
        },
    )

    result = adapter.execute_evaluate_model(state, ActionRequest(ActionType.EVALUATE_MODEL))

    assert result.status == "success"
    assert result.quality_report and result.quality_report.passed
    assert Path(result.artifacts["predictions_path"]).exists()
    assert Path(result.artifacts["failures_path"]).exists()
    assert "eval/attempt_0" in result.artifacts["predictions_path"]
    assert result.artifacts["eval_attempt"] == 0
    assert result.artifacts["cluster_preview"] == {"clusters": []}
    assert result.raw_output["debug_stub"] is True
