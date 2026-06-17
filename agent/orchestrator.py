"""Main agent orchestrator.

Wires all pipeline tools and delegates to WorkflowRunner.
Config is loaded once from .env via PipelineConfig.
"""

from __future__ import annotations

from typing import Any, Dict

from config import PipelineConfig
from agent.workflow import WorkflowRunner


class Orchestrator:
    def __init__(
        self,
        config: PipelineConfig | Dict[str, Any] | None = None,
        *,
        agent_observer: Any = None,
        agent_resume_provider: Any = None,
        **overrides,
    ) -> None:
        if isinstance(config, PipelineConfig):
            self.cfg = config
        elif isinstance(config, dict):
            self.cfg = PipelineConfig(**(config | overrides))
        else:
            self.cfg = PipelineConfig.from_env(**overrides)

        from tools.build_dataset.tool import BuildDatasetTool
        from tools.build_sft_dataset.tool import BuildSftDatasetTool
        from tools.collect_data.tool import CollectDataTool
        from tools.eval_model.tool import EvalModelTool
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool
        from tools.reporting.tool import ReportingTool
        from tools.train.tool import TrainTool

        self.tools = {
            "generate_taxonomy": GenerateTaxonomyTool(),
            "collect_data": CollectDataTool(),
            "build_sft_dataset": BuildSftDatasetTool(),
            "build_dataset": BuildDatasetTool(),
            "train": TrainTool(),
            "eval_model": EvalModelTool(),
            "reporting": ReportingTool({"run_dir": self.cfg.run_dir}),
        }
        self.agent_observer = agent_observer
        self.agent_resume_provider = agent_resume_provider

    def run(self) -> Dict[str, Any]:
        runner = WorkflowRunner(
            self.tools,
            self.cfg,
            agent_observer=self.agent_observer,
            agent_resume_provider=self.agent_resume_provider,
        )
        return runner.run()
