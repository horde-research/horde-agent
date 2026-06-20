from __future__ import annotations

import logging

from agent.main import _install_tool_log_record_factory, _setup_logging, _tool_name_from_logger, build_parser, main


def test_tool_name_from_logger_maps_tool_modules() -> None:
    assert _tool_name_from_logger("tools.collect_data.tool") == "CollectDataTool"
    assert _tool_name_from_logger("tools.generate_taxonomy.agents.category_agent") == "CategoryAgent"
    assert _tool_name_from_logger("tools.build_dataset.tool") == "BuildDatasetTool"
    assert _tool_name_from_logger("core.agentic.langgraph_runtime") == "AgenticRuntime"
    assert _tool_name_from_logger("__main__") == "PipelineCLI"


def test_log_record_factory_adds_tool_name() -> None:
    _install_tool_log_record_factory()

    record = logging.getLogger("tools.collect_data.tool").makeRecord(
        name="tools.collect_data.tool",
        level=logging.INFO,
        fn=__file__,
        lno=1,
        msg="message",
        args=(),
        exc_info=None,
    )

    assert record.tool_name == "CollectDataTool"


def test_setup_logging_suppresses_http_request_noise() -> None:
    _setup_logging("INFO")

    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING


def test_parser_accepts_lora_and_grad_accum_training_flags() -> None:
    args = build_parser().parse_args(
        [
            "--focus",
            "traditional culture",
            "--lora_preset_key",
            "lora_attn_medium",
            "--train-grad-accum",
            "8",
        ]
    )

    assert args.focus == "traditional culture"
    assert args.lora_preset_key == "lora_attn_medium"
    assert args.train_grad_accum == 8


def test_main_forwards_lora_and_grad_accum_overrides(monkeypatch) -> None:
    captured = {}

    class FakeOrchestrator:
        def __init__(self, tools, **overrides):
            captured["tools"] = tools
            captured["overrides"] = overrides

        def run(self):
            return {"termination_reason": "test_complete"}

    monkeypatch.setattr("agent.main.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent.main",
            "--mode",
            "full_agentic",
            "--hf_model_id",
            "test-model",
            "--focus",
            "traditional culture",
            "--lora-preset-key",
            "lora_attn_medium",
            "--train_grad_accum",
            "8",
        ],
    )

    main()

    assert captured["tools"] is None
    assert captured["overrides"]["mode"] == "full_agentic"
    assert captured["overrides"]["hf_model_id"] == "test-model"
    assert captured["overrides"]["focus"] == "traditional culture"
    assert captured["overrides"]["lora_preset_key"] == "lora_attn_medium"
    assert captured["overrides"]["train_grad_accum"] == 8
