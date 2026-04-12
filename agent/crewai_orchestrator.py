"""CrewAI-based orchestration backend.

This module adds CrewAI as an orchestration layer on top of the existing
WorkflowRunner execution logic.

Design:
    1) CrewAI runs a lightweight 3-agent orchestration pass (plan/execute/review)
       to capture an execution intent and checkpoints.
    2) Existing WorkflowRunner performs the actual pipeline execution.

This keeps behavior compatible with current tools while allowing teams to use
CrewAI workflow semantics from CLI.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from config import PipelineConfig
from agent.workflow import WorkflowRunner
from tools.build_dataset.tool import BuildDatasetTool
from tools.build_sft_dataset.tool import BuildSftDatasetTool
from tools.collect_data.tool import CollectDataTool
from tools.eval_model.tool import EvalModelTool
from tools.generate_taxonomy.tool import GenerateTaxonomyTool
from tools.reporting.tool import ReportingTool
from tools.train.tool import TrainTool

logger = logging.getLogger(__name__)


class CrewAIOrchestrator:
    """Alternative orchestrator that uses CrewAI as orchestration layer."""

    def __init__(self, config: PipelineConfig | Dict[str, Any] | None = None, **overrides) -> None:
        if isinstance(config, PipelineConfig):
            self.cfg = config
        elif isinstance(config, dict):
            self.cfg = PipelineConfig(**(config | overrides))
        else:
            self.cfg = PipelineConfig.from_env(**overrides)

        self.tools = {
            "generate_taxonomy": GenerateTaxonomyTool(),
            "collect_data": CollectDataTool(),
            "build_sft_dataset": BuildSftDatasetTool(),
            "build_dataset": BuildDatasetTool(),
            "train": TrainTool(),
            "eval_model": EvalModelTool(),
            "reporting": ReportingTool({"run_dir": self.cfg.run_dir}),
        }

    def _run_crewai_pass(self) -> str:
        """Run CrewAI planning/orchestration pass and return summary text."""
        try:
            from crewai import Agent, Crew, Process, Task
        except Exception as exc:
            raise RuntimeError(
                "CrewAI orchestration requested, but 'crewai' is not installed. "
                "Install dependencies and retry: pip install crewai"
            ) from exc

        planner = Agent(
            role="Pipeline Planner",
            goal="Create a concise execution plan for the selected horde pipeline mode.",
            backstory="Expert in ML pipeline decomposition and risk-aware planning.",
            verbose=False,
            allow_delegation=False,
        )
        executor = Agent(
            role="Pipeline Executor",
            goal="Define the concrete execution checkpoints and artifact boundaries.",
            backstory="Executes data/LLM/training workflows with strict step ordering.",
            verbose=False,
            allow_delegation=False,
        )
        reviewer = Agent(
            role="Pipeline Reviewer",
            goal="Validate execution assumptions and produce a pre-flight checklist.",
            backstory="Identifies configuration risks before long-running jobs start.",
            verbose=False,
            allow_delegation=False,
        )

        t1 = Task(
            description=(
                "Given the runtime inputs, produce a short execution intent and step list. "
                "Focus on mode-specific behavior (full/workflow/minimal_agentic)."
            ),
            expected_output="A concise plan with step order and expected artifacts.",
            agent=planner,
        )
        t2 = Task(
            description=(
                "Based on the plan, provide concrete checkpoints and stop conditions for each step."
            ),
            expected_output="Checkpoint list with pass/fail signals per step.",
            agent=executor,
        )
        t3 = Task(
            description=(
                "Review the plan and checkpoints; list top configuration risks and mitigations."
            ),
            expected_output="A short pre-flight checklist with highest-risk items first.",
            agent=reviewer,
        )

        crew = Crew(
            agents=[planner, executor, reviewer],
            tasks=[t1, t2, t3],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff(
            inputs={
                "mode": self.cfg.mode,
                "country": self.cfg.country,
                "run_dir": self.cfg.run_dir,
                "data_path": self.cfg.data_path or "",
                "max_iters": self.cfg.max_iters,
                "max_queries": self.cfg.max_queries,
            }
        )
        return str(result)

    def run(self) -> Dict[str, Any]:
        logger.info("Running CrewAI orchestration pass...")
        crew_summary = self._run_crewai_pass()
        logger.info("CrewAI pass complete. Executing WorkflowRunner...")

        runner = WorkflowRunner(self.tools, self.cfg)
        result = runner.run()
        result["orchestrator_backend"] = "crewai"
        result["crewai_summary"] = crew_summary
        return result

