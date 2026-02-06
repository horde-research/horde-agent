"""Main agent orchestrator.

This is the new orchestration entrypoint that runs the migrated tools.
"""

from __future__ import annotations

from typing import Any, Dict

from agent.workflow import WorkflowRunner
from tools.build_dataset.tool import BuildDatasetTool
from tools.eval_model.tool import EvalModelTool
from tools.reporting.tool import ReportingTool
from tools.train.tool import TrainTool


class Orchestrator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        run_dir = config.get("run_dir") or config.get("out_dir")
        self.tools = {
            "build_dataset": BuildDatasetTool(),
            "train": TrainTool(),
            "eval_model": EvalModelTool(),
            "reporting": ReportingTool({"run_dir": run_dir}),
        }

    def run(self) -> Dict[str, Any]:
        runner = WorkflowRunner(self.tools, self.config)
        return runner.run()
