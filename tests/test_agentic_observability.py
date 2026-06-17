from __future__ import annotations

from core.agentic.action_space import ActionType
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.agentic.observability import LangSmithObserver


class FakeRunTree:
    def __init__(self, *args, **kwargs) -> None:
        self.id = kwargs.get("id", "root-run-id")
        self.children = []
        self.ended = False

    def post(self) -> None:
        pass

    def create_child(self, *args, **kwargs):
        child = FakeRunTree(id="child-run-id")
        self.children.append(child)
        return child

    def end(self, outputs=None) -> None:
        self.outputs = outputs
        self.ended = True

    def patch(self) -> None:
        pass


def test_langsmith_observer_persists_trace_identifiers(monkeypatch) -> None:
    monkeypatch.setattr("core.agentic.observability.RunTree", FakeRunTree)
    state = PipelineState(run_dir="/tmp/run")
    observer = LangSmithObserver(project_name="test-project")

    observer.start_run(state)
    observer.before_action(state, ActionRequest(ActionType.GENERATE_TAXONOMY))
    observer.after_action(
        state,
        ActionResult(
            action_type=ActionType.GENERATE_TAXONOMY,
            status="success",
            quality_report=QualityReport(stage=ActionType.GENERATE_TAXONOMY, passed=True),
        ),
    )
    observer.finish_run(state)

    assert state.artifacts["langsmith_project"] == "test-project"
    assert state.artifacts["langsmith_run_id"] == "root-run-id"
