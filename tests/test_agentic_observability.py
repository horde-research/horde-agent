from __future__ import annotations

from core.agentic.action_space import ActionType
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.agentic.observability import LangSmithObserver


class FakeRunTree:
    def __init__(self, *args, **kwargs) -> None:
        self.id = kwargs.get("id", "root-run-id")
        self.inputs = kwargs.get("inputs")
        self.name = kwargs.get("name")
        self.children = []
        self.ended = False
        self.outputs = None

    def post(self) -> None:
        pass

    def create_child(self, *args, **kwargs):
        child = FakeRunTree(id="child-run-id", **kwargs)
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


def test_langsmith_observer_redacts_trace_payloads(monkeypatch) -> None:
    monkeypatch.setattr("core.agentic.observability.RunTree", FakeRunTree)
    state = PipelineState(
        run_dir="/tmp/run",
        config={
            "llm_api_key": "llm-secret",
            "serper_api_key": "serper-secret",
            "hf_token": "hf-secret",
            "safe": "visible",
        },
    )
    observer = LangSmithObserver(project_name="test-project")

    observer.start_run(state)
    observer.before_action(state, ActionRequest(ActionType.GENERATE_TAXONOMY))
    observer.after_action(
        state,
        ActionResult(
            action_type=ActionType.GENERATE_TAXONOMY,
            status="failed",
            error="HTTP 403 for https://example.test?key=llm-secret",
            raw_output={"nested": {"token": "hf-secret"}},
            quality_report=QualityReport(
                stage=ActionType.GENERATE_TAXONOMY,
                passed=False,
                blocking_issues=["Authorization: Bearer serper-secret"],
            ),
        ),
    )

    root = observer._root
    assert root.inputs["config"]["llm_api_key"] == "[REDACTED]"
    assert root.inputs["config"]["serper_api_key"] == "[REDACTED]"
    assert root.inputs["config"]["hf_token"] == "[REDACTED]"
    assert root.inputs["config"]["safe"] == "visible"
    payload = json_dump(root.children[0].outputs)
    assert "llm-secret" not in payload
    assert "serper-secret" not in payload
    assert "hf-secret" not in payload
    assert "?key=[REDACTED]" in payload


def json_dump(value) -> str:
    import json

    return json.dumps(value, sort_keys=True)
