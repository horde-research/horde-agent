from __future__ import annotations

from pathlib import Path

from core.agentic.action_space import ActionType
from core.agentic.models import ActionRequest, PipelineState
from core.agentic.tool_adapters import AgenticToolAdapter
from tools.reporting.report import write_report


class FakeReportingTool:
    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path
        self.calls: list[dict] = []

    def finalize(self, eval_results):
        self.calls.append(eval_results)
        self.report_path.write_text("# report\n", encoding="utf-8")
        return str(self.report_path)


def test_write_report_renders_dataset_example_readably(tmp_path: Path) -> None:
    report_path = write_report(
        out_dir=str(tmp_path),
        dataset_summary={
            "data_path": "dataset",
            "resolved_data_id": "source.jsonl",
            "columns": ["messages"],
            "sample_count": 1,
            "split_counts": {"train": 1},
            "split_strategy": "group",
            "group_key_column": "group_key",
            "split_group_counts": {"train": 1},
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "/tmp/image.jpg"},
                            {"type": "text", "text": "Describe this image in detail."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "A red square."}],
                    },
                ],
                "group_key": "group-1",
                "source_url": "https://example.com/source",
            },
            "modality_candidates": ["image"],
            "validation_warnings": [],
        },
        component_selection={
            "dataset_loader_key": "hf_image_default",
            "model_loader_key": "hf_image_text_default",
            "lora_preset_key": "lora_attn_small",
            "trainer_key": "vision_language_sft",
            "hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "primary_metric": "eval_loss",
            "rationale": "agentic full mode selection",
        },
        iterations=[],
        failures_path="",
        cluster_preview={},
        error_analysis={"status": "No evaluation failures detected."},
        pipeline_summary={},
    )

    report_text = Path(report_path).read_text(encoding="utf-8")
    assert "## Dataset Example" in report_text
    assert "[image: /tmp/image.jpg]" in report_text
    assert "A red square." in report_text
    assert '"image": null' not in report_text


def test_generate_report_defaults_to_image_model_loader_for_image_runs(tmp_path: Path) -> None:
    report_tool = FakeReportingTool(tmp_path / "report.md")
    adapter = AgenticToolAdapter({"reporting": report_tool})
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "training_modality": "image",
            "hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "dataset_loader_key": "hf_image_default",
        },
        artifacts={},
    )

    result = adapter.execute_generate_report(state, ActionRequest(ActionType.GENERATE_REPORT))

    assert result.status == "success"
    assert report_tool.calls
    component_selection = report_tool.calls[0]["component_selection"]
    assert component_selection["model_loader_key"] == "hf_image_text_default"
    assert component_selection["trainer_key"] == "vision_language_sft"
