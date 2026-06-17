"""Resume confirmation policy for reused agentic stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.agentic.models import PipelineState


class ResumeDecisionProvider(Protocol):
    def confirm_reuse(self, *, stage: str, state: PipelineState) -> bool:
        ...


@dataclass
class StaticResumeDecisionProvider:
    """Deterministic provider used by tests and non-interactive callers."""

    confirm_all: bool = True
    decisions: dict[str, bool] = field(default_factory=dict)

    def confirm_reuse(self, *, stage: str, state: PipelineState) -> bool:
        return bool(self.decisions.get(stage, self.confirm_all))


def apply_resume_confirmations(
    state: PipelineState,
    provider: ResumeDecisionProvider | None,
) -> PipelineState:
    """Confirm whether completed stages from a loaded state may be reused.

    If no provider is supplied, the runtime pauses as a failure-like terminal
    state instead of silently skipping already completed work.
    """
    if not state.completed_stages:
        return state

    for stage in list(state.completed_stages):
        if stage in state.resume_confirmations:
            continue
        if provider is None:
            state.termination_reason = "resume_confirmation_required"
            blocker = f"resume_confirmation_required:{stage}"
            if blocker not in state.blockers:
                state.blockers.append(blocker)
            return state

        confirmed = provider.confirm_reuse(stage=stage, state=state)
        state.resume_confirmations[stage] = confirmed
        if not confirmed:
            _clear_stage_and_downstream(state, stage)
            break

    return state


def _clear_stage_and_downstream(state: PipelineState, stage: str) -> None:
    state.clear_stage_and_downstream(stage)
