from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.workflow import WorkflowRunner
from config import PipelineConfig
from core.agentic.action_space import ActionType
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.agentic.state_store import PipelineStateStore


class FakeObserver:
    def start_run(self, state: PipelineState) -> None:
        pass

    def before_action(self, state: PipelineState, request: ActionRequest) -> None:
        pass

    def after_action(self, state: PipelineState, result: ActionResult) -> None:
        pass

    def finish_run(self, state: PipelineState) -> None:
        pass


class FakeTool:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls = []

    def execute(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.output


class FakeReportingTool:
    def __init__(self, report_path: str) -> None:
        self.report_path = report_path
        self.calls = []

    def finalize(self, payload):
        self.calls.append(payload)
        Path(self.report_path).write_text("<html>report</html>", encoding="utf-8")
        return self.report_path


class ExplodingTool:
    def execute(self, *args, **kwargs):
        raise AssertionError("real train/eval tool should not run in debug stub mode")


def test_full_agentic_workflow_runs_known_graph_with_image_collection(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "collect" / "dataset"
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    adapter_dir = tmp_path / "adapter"
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    report_path = tmp_path / "report.html"

    for path in (dataset_dir, images_dir, adapter_dir):
        path.mkdir(parents=True)
    for path in (images_index, sft_path, annotations_path, manifest_path, predictions_path, failures_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    tools = {
        "generate_taxonomy": FakeTool(
            {
                "categories": ["food"],
                "category_subcategories": {"food": ["dishes"]},
                "category_subcategory_queries": {"food": {"dishes": ["kazakh food"]}},
            }
        ),
        "collect_data": FakeTool(
            {
                "data_path": str(dataset_dir),
                "num_samples": 3,
                "metadata": {
                    "provider": "test",
                    "images_dir": str(images_dir),
                    "images_index": str(images_index),
                    "num_images": 1,
                },
            }
        ),
        "build_sft_dataset": FakeTool(
            {
                "mode": "image",
                "num_items": 1,
                "num_annotations": 1,
                "num_examples": 1,
                "num_failures": 0,
                "annotations_path": str(annotations_path),
                "sft_path": str(sft_path),
            }
        ),
        "build_dataset": FakeTool(
            {
                "dataset_ref": {"kind": "hf", "data_path": str(sft_path), "split": "train"},
                "dataset_summary": {
                    "sample_count": 1,
                    "columns": ["messages"],
                    "validation_warnings": [],
                },
                "dataset_manifest_path": str(manifest_path),
            }
        ),
        "train": FakeTool(
            {
                "adapter_path": str(adapter_dir),
                "log_paths": {"train_log": str(tmp_path / "train.log")},
                "metrics": {"last_train_loss": 0.1},
                "iteration_record": {
                    "iter_idx": 0,
                    "config": {
                        "lr": 0.0002,
                        "batch_size": 4,
                        "grad_accum": 4,
                        "max_steps": 1,
                        "warmup_ratio": 0.03,
                        "weight_decay": 0.0,
                        "max_seq_len": 512,
                        "eval_steps": 50,
                        "seed": 42,
                    },
                    "metrics": {"last_train_loss": 0.1},
                    "adapter_path": str(adapter_dir),
                    "log_paths": {"train_log": str(tmp_path / "train.log")},
                },
            }
        ),
        "eval_model": FakeTool(
            {
                "predictions_path": str(predictions_path),
                "failures_path": str(failures_path),
                "cluster_preview": {"clusters": []},
            }
        ),
        "reporting": FakeReportingTool(str(report_path)),
    }
    cfg = PipelineConfig(
        mode="full_agentic",
        country="Kazakhstan",
        run_dir=str(tmp_path),
        collect_images=True,
        sft_mode="image",
        sft_target_language="English",
        hf_model_id="test-model",
        max_steps=1,
    )

    result = WorkflowRunner(tools, cfg, agent_observer=FakeObserver()).run()

    assert result["termination_reason"] == "full_graph_complete"
    assert result["completed_stages"] == [
        "generate_taxonomy",
        "collect_data",
        "assess_coverage_and_refine_queries",
        "build_sft_dataset",
        "build_dataset",
        "train_model",
        "evaluate_model",
        "generate_report",
    ]
    assert result["artifacts"]["images_dir"] == str(images_dir)
    assert result["artifacts"]["report_path"] == str(report_path)
    sft_config = tools["build_sft_dataset"].calls[0]["args"][0]
    assert sft_config["mode"] == "image"
    assert sft_config["input_dir"] == str(images_dir)
    report_payload = tools["reporting"].calls[0]
    taxonomy_summary = report_payload["pipeline_summary"]["taxonomy_summary"]
    assert taxonomy_summary["num_categories"] == 1
    assert taxonomy_summary["num_subcategories"] == 1
    assert taxonomy_summary["categories"] == [{"name": "food", "subcategories": ["dishes"]}]
    assert report_payload["error_analysis"] == {"status": "No evaluation failures detected."}
    assert "taxonomy" not in report_payload["error_analysis"]


def test_full_agentic_debug_flow_stubs_train_and_eval_but_generates_report(tmp_path: Path) -> None:
    from tools.reporting.tool import ReportingTool

    dataset_dir = tmp_path / "collect" / "dataset"
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"

    for path in (dataset_dir, images_dir):
        path.mkdir(parents=True)
    images_index.write_text("[]", encoding="utf-8")
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text('{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}\n', encoding="utf-8")
    annotations_path.write_text('{"success": true}\n', encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")

    tools = {
        "generate_taxonomy": FakeTool(
            {
                "categories": ["food"],
                "category_subcategories": {"food": ["dishes"]},
                "category_subcategory_queries": {"food": {"dishes": ["kazakh food"]}},
            }
        ),
        "collect_data": FakeTool(
            {
                "data_path": str(dataset_dir),
                "num_samples": 3,
                "metadata": {
                    "provider": "test",
                    "images_dir": str(images_dir),
                    "images_index": str(images_index),
                    "num_images": 1,
                },
            }
        ),
        "build_sft_dataset": FakeTool(
            {
                "mode": "image",
                "num_items": 1,
                "num_annotations": 1,
                "num_examples": 1,
                "num_failures": 0,
                "annotations_path": str(annotations_path),
                "sft_path": str(sft_path),
            }
        ),
        "build_dataset": FakeTool(
            {
                "dataset_ref": {"kind": "hf", "data_path": str(sft_path), "split": "train"},
                "dataset_summary": {
                    "sample_count": 1,
                    "columns": ["messages"],
                    "validation_warnings": [],
                },
                "dataset_manifest_path": str(manifest_path),
            }
        ),
        "train": ExplodingTool(),
        "eval_model": ExplodingTool(),
        "reporting": ReportingTool({"run_dir": str(tmp_path)}),
    }
    cfg = PipelineConfig(
        mode="full_agentic",
        country="Kazakhstan",
        run_dir=str(tmp_path),
        collect_images=True,
        sft_mode="image",
        sft_target_language="English",
        hf_model_id="test-model",
        max_steps=3,
        debug_stub_train=True,
        debug_stub_eval=True,
    )

    result = WorkflowRunner(tools, cfg, agent_observer=FakeObserver()).run()

    assert result["termination_reason"] == "full_graph_complete"
    report_path = Path(result["artifacts"]["report_path"])
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Training Iterations" in report_text
    assert "## Pipeline Summary" in report_text
    assert "- food (1 subcategories)" in report_text
    assert '"status": "No evaluation failures detected."' in report_text
    assert Path(result["artifacts"]["adapter_path"]).exists()
    assert Path(result["artifacts"]["predictions_path"]).exists()


def test_full_agentic_restart_from_stage_rebuilds_stale_taxonomy_state(tmp_path: Path) -> None:
    from tools.reporting.tool import ReportingTool

    stale_state = PipelineState(
        run_dir=str(tmp_path),
        mode="full",
        config={
            "mode": "full_agentic",
            "country": "Kazakhstan",
            "max_queries": 500,
            "sft_mode": "image",
            "sft_target_language": "English",
            "hf_model_id": "old-model",
        },
        artifacts={
            "taxonomy": {"categories": ["old"]},
            "search_queries": ["old q1", "old q2"],
        },
        completed_stages=["generate_taxonomy"],
        quality_reports={
            "generate_taxonomy": QualityReport(
                stage=ActionType.GENERATE_TAXONOMY,
                passed=True,
                recoverable=True,
            )
        },
        blockers=["resume_confirmation_required:generate_taxonomy"],
        termination_reason="resume_confirmation_required",
    )
    PipelineStateStore(tmp_path).save(stale_state)

    dataset_dir = tmp_path / "collect" / "dataset"
    images_dir = tmp_path / "collect" / "images"
    images_index = tmp_path / "collect" / "images.json"
    sft_path = tmp_path / "sft" / "sft.jsonl"
    annotations_path = tmp_path / "sft" / "annotations.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"

    for path in (dataset_dir, images_dir):
        path.mkdir(parents=True)
    images_index.write_text("[]", encoding="utf-8")
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text('{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}\n', encoding="utf-8")
    annotations_path.write_text('{"success": true}\n', encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")

    taxonomy_tool = FakeTool(
        {
            "categories": ["new"],
            "category_subcategories": {"new": ["slot"]},
            "category_subcategory_queries": {"new": {"slot": ["new q1", "new q2", "new q3"]}},
        }
    )
    collect_tool = FakeTool(
        {
            "data_path": str(dataset_dir),
            "num_samples": 3,
            "metadata": {
                "provider": "test",
                "images_dir": str(images_dir),
                "images_index": str(images_index),
                "num_images": 1,
            },
        }
    )
    tools = {
        "generate_taxonomy": taxonomy_tool,
        "collect_data": collect_tool,
        "build_sft_dataset": FakeTool(
            {
                "mode": "image",
                "num_items": 1,
                "num_annotations": 1,
                "num_examples": 1,
                "num_failures": 0,
                "annotations_path": str(annotations_path),
                "sft_path": str(sft_path),
            }
        ),
        "build_dataset": FakeTool(
            {
                "dataset_ref": {"kind": "hf", "data_path": str(sft_path), "split": "train"},
                "dataset_summary": {
                    "sample_count": 1,
                    "columns": ["messages"],
                    "validation_warnings": [],
                },
                "dataset_manifest_path": str(manifest_path),
            }
        ),
        "train": ExplodingTool(),
        "eval_model": ExplodingTool(),
        "reporting": ReportingTool({"run_dir": str(tmp_path)}),
    }
    cfg = PipelineConfig(
        mode="full_agentic",
        country="Kazakhstan",
        run_dir=str(tmp_path),
        collect_images=True,
        sft_mode="image",
        sft_target_language="English",
        hf_model_id="new-model",
        max_queries=2,
        restart_from_stage="generate_taxonomy",
        debug_stub_train=True,
        debug_stub_eval=True,
    )

    result = WorkflowRunner(tools, cfg, agent_observer=FakeObserver()).run()

    assert result["termination_reason"] == "full_graph_complete"
    assert taxonomy_tool.calls
    collect_config = collect_tool.calls[0]["args"][0]
    assert collect_config["queries"] == ["new q1", "new q2"]
    assert result["config"]["max_queries"] == 2
    assert not result["blockers"]
    assert Path(result["artifacts"]["report_path"]).exists()
