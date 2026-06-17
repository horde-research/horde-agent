"""Executable v1 controller for the constrained full-pipeline agent."""

from __future__ import annotations

from typing import Callable, Dict, Mapping

from core.agentic.action_space import ActionType, FULL_GRAPH_ACTIONS, TERMINAL_ACTIONS
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.agentic.observability import RunObserver, build_observer_from_env
from core.agentic.policy import choose_next_action

StageExecutor = Callable[[PipelineState, ActionRequest], ActionResult]


class AgenticFullController:
    """Run the v1 full graph using injected stage executors.

    The controller is deliberately independent from the existing tools. Tool
    adapters can be attached later by supplying one executor per full-graph
    action. This keeps the controller testable without importing the current
    heavier workflow modules.
    """

    def __init__(
        self,
        *,
        executors: Mapping[ActionType, StageExecutor],
        observer: RunObserver | None = None,
    ) -> None:
        self.executors: Dict[ActionType, StageExecutor] = dict(executors)
        self.observer = observer or build_observer_from_env()

    def run(self, state: PipelineState) -> PipelineState:
        if state.mode != "full":
            state.termination_reason = "unsupported_mode"
            return state

        self.observer.start_run(state)
        while True:
            request = choose_next_action(state)
            state.record_decision(request)

            if request.action_type in TERMINAL_ACTIONS:
                state.termination_reason = request.reason
                break

            if request.retry_attempt:
                state.clear_stage_and_downstream(request.action_type)
                state.apply_config_delta(request.config_delta, reason=request.reason)
                state.retry_counts[request.action_type.value] = request.retry_attempt
            elif request.config_delta:
                state.apply_config_delta(request.config_delta, reason=request.reason)

            self.observer.before_action(state, request)
            result = self._execute(state, request)
            state.record_action_result(result)
            self.observer.after_action(state, result)

            if len(state.decision_history) >= state.max_graph_steps:
                state.termination_reason = "graph_step_limit_exhausted"
                break

        self.observer.finish_run(state)
        return state

    def _execute(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        executor = self.executors.get(request.action_type)
        if executor is None:
            return ActionResult(
                action_type=request.action_type,
                status="failed",
                quality_report=QualityReport(
                    stage=request.action_type,
                    passed=False,
                    recoverable=False,
                    blocking_issues=[f"missing_executor:{request.action_type.value}"],
                ),
                error=f"No executor registered for {request.action_type.value}",
            )
        return executor(state, request)


def missing_executors(executors: Mapping[ActionType, StageExecutor]) -> list[ActionType]:
    return [stage for stage in FULL_GRAPH_ACTIONS if stage not in executors]
