"""Deterministic v1 policy for the constrained full-pipeline graph."""

from __future__ import annotations

from core.agentic.action_space import ActionType, FULL_GRAPH_ACTIONS
from core.agentic.models import ActionRequest, PipelineState
from core.agentic.recovery import build_recovery_plan


def _first_incomplete_stage(state: PipelineState) -> ActionType | None:
    completed = set(state.completed_stages)
    for stage in FULL_GRAPH_ACTIONS:
        if stage.value not in completed:
            return stage
    return None


def choose_next_action(state: PipelineState) -> ActionRequest:
    """Choose the next legal action for the v1 full graph.

    This policy is intentionally conservative: it only follows the known full
    stage sequence, but recoverable failures now produce bounded recovery plans
    instead of blind same-stage retries.
    """
    if state.mode != "full":
        return ActionRequest(
            action_type=ActionType.STOP_FAILURE,
            reason="unsupported_mode",
        )

    if state.blockers:
        return ActionRequest(
            action_type=ActionType.STOP_FAILURE,
            reason="unresolved_blockers",
        )

    last_result = state.last_action_result
    if last_result and last_result.quality_report and not last_result.quality_report.passed:
        plan = build_recovery_plan(state, last_result)
        stage = plan.target_stage
        if not last_result.quality_report.recoverable:
            return ActionRequest(
                action_type=ActionType.STOP_FAILURE,
                stage=last_result.action_type,
                reason="unrecoverable_quality_failure",
            )
        current_retries = int(state.retry_counts.get(stage.value, 0))
        if current_retries >= state.max_stage_retries:
            return ActionRequest(
                action_type=ActionType.STOP_FAILURE,
                stage=stage,
                reason="retry_limit_exhausted",
            )
        return ActionRequest(
            action_type=stage,
            stage=stage,
            reason=plan.reason,
            config_delta=plan.config_delta,
            retry_attempt=current_retries + 1,
        )

    if len(state.decision_history) >= state.max_graph_steps:
        return ActionRequest(
            action_type=ActionType.STOP_FAILURE,
            reason="graph_step_limit_exhausted",
        )

    next_stage = _first_incomplete_stage(state)
    if next_stage is None:
        return ActionRequest(
            action_type=ActionType.STOP_SUCCESS,
            reason="full_graph_complete",
        )

    return ActionRequest(
        action_type=next_stage,
        stage=next_stage,
        reason="next_required_stage",
    )
