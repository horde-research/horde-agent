from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from config import PipelineConfig
from core.agentic.action_space import ActionType
from core.agentic.models import ActionRequest, PipelineState
from core.agentic.tool_adapters import AgenticToolAdapter
from core.agentic.validators import validate_collection_output, validate_sft_output
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
        config={"country": "Kazakhstan", "llm_batch_size": 2, "llm_batch_delay": 0.0},
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


def test_build_sft_adapter_uses_collected_images_for_image_mode(tmp_path: Path) -> None:
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    images_dir.mkdir(parents=True)
    images_index.write_text("[]", encoding="utf-8")
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text('{"messages": []}\n', encoding="utf-8")
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
        config={"sft_mode": "image", "sft_target_language": "English", "sft_prompt_preset": "schema_strict"},
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
    assert "input_jsonl" not in called_config
    assert result.status == "success"
    assert result.quality_report and result.quality_report.passed
    assert result.artifacts["sft_path"] == str(sft_path)
    assert result.artifacts["training_modality"] == "image"


def test_build_sft_adapter_pushes_dataset_to_hf_when_configured(monkeypatch, tmp_path: Path) -> None:
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    images_dir.mkdir(parents=True)
    images_index.write_text("[]", encoding="utf-8")
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text('{"messages": []}\n', encoding="utf-8")
    annotations_path.write_text('{"success": true}\n', encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def _push_dataset(local_path, repo_name, *, username=None, private=True):
        calls.append({"local_path": local_path, "repo_name": repo_name, "username": username, "private": private})
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
    assert result.artifacts["dataset_repo_id"] == "test-owner/test-dataset"
    assert result.raw_output["dataset_repo_id"] == "test-owner/test-dataset"
    assert calls == [
        {
            "local_path": str(sft_path),
            "repo_name": "test-dataset",
            "username": "test-owner",
            "private": True,
        }
    ]


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


def test_sft_validator_requires_examples_and_sft_path(tmp_path: Path) -> None:
    missing_report = validate_sft_output({"num_examples": 0, "sft_path": str(tmp_path / "missing.jsonl")})
    assert not missing_report.passed
    assert "num_examples_below_minimum" in missing_report.blocking_issues

    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text('{"messages": []}\n', encoding="utf-8")
    passing_report = validate_sft_output({"num_examples": 1, "sft_path": str(sft_path)})
    assert passing_report.passed


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
    assert result.artifacts["hf_adapter_upload_skipped"] == "debug_stub_train"
    assert result.raw_output["training_modality"] == "text"
    assert result.raw_output["debug_stub"] is True


def test_train_adapter_pushes_adapter_to_hf_when_configured(monkeypatch, tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def _push_adapter(local_path, repo_name, *, username=None, private=True):
        calls.append({"local_path": local_path, "repo_name": repo_name, "username": username, "private": private})
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
    assert result.artifacts["adapter_repo_id"] == "test-owner/test-adapter"
    assert result.raw_output["adapter_repo_id"] == "test-owner/test-adapter"
    assert calls == [
        {
            "local_path": str(adapter_dir),
            "repo_name": "test-adapter",
            "username": "test-owner",
            "private": True,
        }
    ]


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
    assert result.artifacts["cluster_preview"] == {"clusters": []}
    assert result.raw_output["debug_stub"] is True
