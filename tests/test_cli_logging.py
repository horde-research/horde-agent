from __future__ import annotations

import logging

from agent.main import _install_tool_log_record_factory, _setup_logging, _tool_name_from_logger


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
