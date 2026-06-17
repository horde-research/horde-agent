"""LangSmith observability hooks for agentic runs."""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Protocol

from langsmith.run_trees import RunTree

from core.agentic.models import ActionRequest, ActionResult, PipelineState

logger = logging.getLogger(__name__)


class RunObserver(Protocol):
    def start_run(self, state: PipelineState) -> None:
        ...

    def before_action(self, state: PipelineState, request: ActionRequest) -> None:
        ...

    def after_action(self, state: PipelineState, result: ActionResult) -> None:
        ...

    def finish_run(self, state: PipelineState) -> None:
        ...


class LangSmithObserver:
    """Explicit LangSmith RunTree observer for the full-pipeline controller."""

    def __init__(
        self,
        *,
        project_name: Optional[str] = None,
    ) -> None:
        self.project_name = project_name or os.getenv("LANGSMITH_PROJECT") or "horde-agent"
        self._root = None
        self._active_children: List[Any] = []

    def start_run(self, state: PipelineState) -> None:
        self._root = RunTree(
            name="Horde Agent Full Run",
            run_type="chain",
            inputs={
                "run_dir": state.run_dir,
                "mode": state.mode,
                "config": state.config,
            },
            project_name=self.project_name,
        )
        self._root.post()
        state.artifacts["langsmith_project"] = self.project_name
        state.artifacts["langsmith_run_id"] = str(getattr(self._root, "id", ""))
        logger.info(
            "LangSmith trace started: project=%s run_id=%s",
            self.project_name,
            state.artifacts["langsmith_run_id"],
        )

    def before_action(self, state: PipelineState, request: ActionRequest) -> None:
        if self._root is None:
            raise RuntimeError("LangSmith root run has not been started.")
        child = self._root.create_child(
            name=request.action_type.value,
            run_type="tool",
            inputs={
                "request": request.to_dict(),
                "completed_stages": list(state.completed_stages),
                "retry_counts": dict(state.retry_counts),
            },
        )
        child.post()
        logger.info(
            "LangSmith stage span started: stage=%s run_id=%s",
            request.action_type.value,
            getattr(child, "id", ""),
        )
        self._active_children.append(child)

    def after_action(self, state: PipelineState, result: ActionResult) -> None:
        if not self._active_children:
            raise RuntimeError("No active LangSmith child run to finish.")
        child = self._active_children.pop()
        child.end(outputs={"result": result.to_dict()})
        child.patch()
        logger.info(
            "LangSmith stage span finished: stage=%s status=%s",
            result.action_type.value,
            result.status,
        )

    def finish_run(self, state: PipelineState) -> None:
        if self._root is None:
            raise RuntimeError("LangSmith root run has not been started.")
        self._root.end(
            outputs={
                "termination_reason": state.termination_reason,
                "completed_stages": list(state.completed_stages),
                "retry_counts": dict(state.retry_counts),
                "blockers": list(state.blockers),
            }
        )
        self._root.patch()
        logger.info(
            "LangSmith trace finished: project=%s run_id=%s",
            self.project_name,
            getattr(self._root, "id", ""),
        )


def build_observer_from_env() -> RunObserver:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required for agent observability.")
    return LangSmithObserver(project_name=os.getenv("LANGSMITH_PROJECT"))
