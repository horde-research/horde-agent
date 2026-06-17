from __future__ import annotations

import os
from pathlib import Path

from core.agentic.action_space import ActionType, FULL_GRAPH_ACTIONS
from core.agentic.controller import AgenticFullController
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.agentic.policy import choose_next_action


class FakeObserver:
    def __init__(self) -> None:
        self.events = []

    def start_run(self, state: PipelineState) -> None:
        self.events.append({"event": "start_run", "run_dir": state.run_dir})

    def before_action(self, state: PipelineState, request: ActionRequest) -> None:
        self.events.append({"event": "before_action", "request": request.to_dict()})

    def after_action(self, state: PipelineState, result: ActionResult) -> None:
        self.events.append({"event": "after_action", "result": result.to_dict()})

    def finish_run(self, state: PipelineState) -> None:
        self.events.append({"event": "finish_run", "termination_reason": state.termination_reason})


def _passing_report(stage: ActionType) -> QualityReport:
    return QualityReport(
        stage=stage,
        passed=True,
        score=0.92,
        metrics={"example_metric": 1.0},
    )


def test_policy_starts_at_first_full_graph_stage(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path))

    action = choose_next_action(state)

    assert action.action_type == ActionType.GENERATE_TAXONOMY
    assert action.stage == ActionType.GENERATE_TAXONOMY
    assert action.reason == "next_required_stage"


def test_quality_report_infers_categorical_gate_status() -> None:
    repair_report = QualityReport(
        stage=ActionType.COLLECT_DATA,
        passed=False,
        recoverable=True,
        blocking_issues=["num_samples_below_minimum"],
    )
    fail_report = QualityReport(
        stage=ActionType.GENERATE_REPORT,
        passed=False,
        recoverable=False,
        blocking_issues=["report_path_missing"],
    )
    warn_report = QualityReport(
        stage=ActionType.BUILD_DATASET,
        passed=True,
        warnings=["small_eval_sample"],
    )

    assert repair_report.gate_status == "repair"
    assert repair_report.decision == "repair"
    assert fail_report.gate_status == "fail"
    assert fail_report.decision == "stop"
    assert warn_report.gate_status == "warn"
    assert warn_report.decision == "continue"


def test_policy_advances_through_full_graph_after_quality_passes(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path))
    state.mark_stage_complete(
        ActionType.GENERATE_TAXONOMY,
        quality_report=_passing_report(ActionType.GENERATE_TAXONOMY),
        artifacts={"taxonomy": "taxonomy.json"},
    )

    action = choose_next_action(state)

    assert action.action_type == ActionType.COLLECT_DATA
    assert action.stage == ActionType.COLLECT_DATA


def test_policy_runs_coverage_assessment_after_collection_passes(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path))
    state.mark_stage_complete(
        ActionType.GENERATE_TAXONOMY,
        quality_report=_passing_report(ActionType.GENERATE_TAXONOMY),
        artifacts={"search_queries": ["q1"]},
    )
    state.mark_stage_complete(
        ActionType.COLLECT_DATA,
        quality_report=_passing_report(ActionType.COLLECT_DATA),
        artifacts={"num_samples": 3},
    )

    action = choose_next_action(state)

    assert action.action_type == ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES
    assert action.stage == ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES


def test_policy_retries_recoverable_failed_stage_before_advancing(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path), max_stage_retries=2)
    state.last_action_result = ActionResult(
        action_type=ActionType.COLLECT_DATA,
        status="failed",
        quality_report=QualityReport(
            stage=ActionType.COLLECT_DATA,
            passed=False,
            score=0.2,
            recoverable=True,
            blocking_issues=["sample_count_below_minimum"],
        ),
    )

    action = choose_next_action(state)

    assert action.action_type == ActionType.COLLECT_DATA
    assert action.stage == ActionType.COLLECT_DATA
    assert action.reason == "retry_recoverable_quality_failure"
    assert action.retry_attempt == 1


def test_policy_adds_collection_recovery_config_delta(tmp_path: Path) -> None:
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "collect_images": True,
            "serper_results_per_query": 10,
            "serper_top_results": 5,
            "image_search_results_per_query": 10,
            "image_taxonomy_queries_per_slot": 4,
            "image_min_width": 512,
            "image_min_height": 384,
        },
    )
    state.last_action_result = ActionResult(
        action_type=ActionType.COLLECT_DATA,
        status="failed",
        quality_report=QualityReport(
            stage=ActionType.COLLECT_DATA,
            passed=False,
            score=0.2,
            recoverable=True,
            blocking_issues=["num_samples_below_minimum", "num_images_below_minimum"],
        ),
    )

    action = choose_next_action(state)

    assert action.action_type == ActionType.COLLECT_DATA
    assert action.reason == "recovery_expand_collection_coverage"
    assert action.config_delta["serper_results_per_query"] == 15
    assert action.config_delta["serper_top_results"] == 7
    assert action.config_delta["image_search_results_per_query"] == 15
    assert action.config_delta["image_taxonomy_queries_per_slot"] == 5
    assert action.config_delta["image_min_width"] == 409
    assert action.config_delta["image_min_height"] == 307


def test_policy_routes_failed_coverage_assessment_back_to_collection(tmp_path: Path) -> None:
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"serper_results_per_query": 10, "serper_top_results": 5},
    )
    state.last_action_result = ActionResult(
        action_type=ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES,
        status="failed",
        raw_output={"added_queries": ["Kazakhstan kazakh food culture source"]},
        quality_report=QualityReport(
            stage=ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES,
            passed=False,
            recoverable=True,
            blocking_issues=["coverage_text_samples_below_minimum"],
        ),
    )

    action = choose_next_action(state)

    assert action.action_type == ActionType.COLLECT_DATA
    assert action.reason == "recovery_refine_collection_queries"
    assert action.config_delta["coverage_added_queries"] == ["Kazakhstan kazakh food culture source"]
    assert action.config_delta["serper_results_per_query"] == 13


def test_policy_routes_eval_knowledge_failure_back_to_collection(tmp_path: Path) -> None:
    state = PipelineState(
        run_dir=str(tmp_path),
        config={"serper_results_per_query": 10, "serper_top_results": 5},
    )
    state.last_action_result = ActionResult(
        action_type=ActionType.EVALUATE_MODEL,
        status="failed",
        metrics={"failure_clusters": ["knowledge_missing"]},
        quality_report=QualityReport(
            stage=ActionType.EVALUATE_MODEL,
            passed=False,
            score=0.1,
            recoverable=True,
            blocking_issues=["eval_knowledge_missing"],
        ),
    )

    action = choose_next_action(state)

    assert action.action_type == ActionType.COLLECT_DATA
    assert action.reason == "recovery_eval_requests_more_source_coverage"
    assert action.config_delta["serper_results_per_query"] == 15


def test_policy_stops_after_retry_limit(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path), max_stage_retries=1)
    state.retry_counts[ActionType.COLLECT_DATA.value] = 1
    state.last_action_result = ActionResult(
        action_type=ActionType.COLLECT_DATA,
        status="failed",
        quality_report=QualityReport(
            stage=ActionType.COLLECT_DATA,
            passed=False,
            recoverable=True,
            blocking_issues=["sample_count_below_minimum"],
        ),
    )

    action = choose_next_action(state)

    assert action.action_type == ActionType.STOP_FAILURE
    assert action.reason == "retry_limit_exhausted"


def test_policy_stops_success_when_full_graph_complete(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path))
    for stage in FULL_GRAPH_ACTIONS:
        state.mark_stage_complete(
            stage,
            quality_report=_passing_report(stage),
            artifacts={stage.value: f"{stage.value}.artifact"},
        )

    action = choose_next_action(state)

    assert action.action_type == ActionType.STOP_SUCCESS
    assert action.reason == "full_graph_complete"


def test_controller_runs_full_graph_and_records_observer_events(tmp_path: Path) -> None:
    def _executor(stage: ActionType):
        def _run(state: PipelineState, request):
            return ActionResult(
                action_type=stage,
                status="success",
                quality_report=_passing_report(stage),
                artifacts={stage.value: f"{stage.value}.artifact"},
            )

        return _run

    observer = FakeObserver()
    controller = AgenticFullController(
        executors={stage: _executor(stage) for stage in FULL_GRAPH_ACTIONS},
        observer=observer,
    )

    state = controller.run(PipelineState(run_dir=str(tmp_path)))

    assert state.termination_reason == "full_graph_complete"
    assert state.completed_stages == [stage.value for stage in FULL_GRAPH_ACTIONS]
    assert len(state.result_history) == len(FULL_GRAPH_ACTIONS)
    assert observer.events[0]["event"] == "start_run"
    assert observer.events[-1]["event"] == "finish_run"
    assert len([event for event in observer.events if event["event"] == "before_action"]) == len(FULL_GRAPH_ACTIONS)
    assert len([event for event in observer.events if event["event"] == "after_action"]) == len(FULL_GRAPH_ACTIONS)


def test_controller_requires_langsmith_api_key_without_injected_observer(tmp_path: Path) -> None:
    previous_key = os.environ.pop("LANGSMITH_API_KEY", None)
    try:
        try:
            AgenticFullController(executors={})
        except RuntimeError as exc:
            assert "LANGSMITH_API_KEY is required" in str(exc)
        else:
            raise AssertionError("Expected controller construction to require LANGSMITH_API_KEY")
    finally:
        if previous_key is not None:
            os.environ["LANGSMITH_API_KEY"] = previous_key


def test_controller_retries_recoverable_stage_then_continues(tmp_path: Path) -> None:
    attempts = {"collect": 0}

    def _success_executor(stage: ActionType):
        def _run(state: PipelineState, request):
            return ActionResult(
                action_type=stage,
                status="success",
                quality_report=_passing_report(stage),
                artifacts={stage.value: f"{stage.value}.artifact"},
            )

        return _run

    def _collect_executor(state: PipelineState, request):
        attempts["collect"] += 1
        if attempts["collect"] == 1:
            return ActionResult(
                action_type=ActionType.COLLECT_DATA,
                status="failed",
                quality_report=QualityReport(
                    stage=ActionType.COLLECT_DATA,
                    passed=False,
                    score=0.25,
                    recoverable=True,
                    blocking_issues=["sample_count_below_minimum"],
                ),
            )
        return ActionResult(
            action_type=ActionType.COLLECT_DATA,
            status="success",
            quality_report=_passing_report(ActionType.COLLECT_DATA),
            artifacts={"collect_data": "collect_data.artifact"},
        )

    executors = {stage: _success_executor(stage) for stage in FULL_GRAPH_ACTIONS}
    executors[ActionType.COLLECT_DATA] = _collect_executor

    controller = AgenticFullController(executors=executors, observer=FakeObserver())

    state = controller.run(PipelineState(run_dir=str(tmp_path), max_stage_retries=2))

    assert state.termination_reason == "full_graph_complete"
    assert attempts["collect"] == 2
    assert state.retry_counts[ActionType.COLLECT_DATA.value] == 1
    assert len([r for r in state.result_history if r["action_type"] == ActionType.COLLECT_DATA.value]) == 2


def test_controller_applies_recovery_config_delta_before_execution(tmp_path: Path) -> None:
    observed_config = {}

    def _collect_executor(state: PipelineState, request: ActionRequest):
        observed_config.update(state.config)
        return ActionResult(
            action_type=ActionType.COLLECT_DATA,
            status="success",
            quality_report=_passing_report(ActionType.COLLECT_DATA),
            artifacts={"collect_data": "collect_data.artifact"},
        )

    state = PipelineState(
        run_dir=str(tmp_path),
        config={"serper_results_per_query": 10, "serper_top_results": 5},
        max_stage_retries=2,
    )
    state.last_action_result = ActionResult(
        action_type=ActionType.COLLECT_DATA,
        status="failed",
        quality_report=QualityReport(
            stage=ActionType.COLLECT_DATA,
            passed=False,
            recoverable=True,
            blocking_issues=["num_samples_below_minimum"],
        ),
    )
    controller = AgenticFullController(
        executors={ActionType.COLLECT_DATA: _collect_executor},
        observer=FakeObserver(),
    )

    final_state = controller.run(state)

    assert observed_config["serper_results_per_query"] == 15
    assert observed_config["serper_top_results"] == 7
    assert final_state.config_history[0]["reason"] == "recovery_expand_collection_coverage"
