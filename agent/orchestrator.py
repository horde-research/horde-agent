"""Main agent orchestrator.

Wires all pipeline tools and delegates to WorkflowRunner.
"""

from __future__ import annotations

from typing import Any, Dict

from agent.workflow import WorkflowRunner
from tools.build_dataset.tool import BuildDatasetTool
from tools.build_sft_dataset.tool import BuildSftDatasetTool
from tools.collect_data.tool import CollectDataTool
from tools.eval_model.tool import EvalModelTool
from tools.generate_taxonomy.tool import GenerateTaxonomyTool
from tools.reporting.tool import ReportingTool
from tools.train.tool import TrainTool


class Orchestrator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        run_dir = config.get("run_dir") or config.get("out_dir")
        self.tools = {
            "generate_taxonomy": GenerateTaxonomyTool(),
            "collect_data": CollectDataTool(),
            "build_sft_dataset": BuildSftDatasetTool(),
            "build_dataset": BuildDatasetTool(),
            "train": TrainTool(),
            "eval_model": EvalModelTool(),
            "reporting": ReportingTool({"run_dir": run_dir}),
        }

    def run(self) -> Dict[str, Any]:
        runner = WorkflowRunner(self.tools, self.config)
        return runner.run()
