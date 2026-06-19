from __future__ import annotations

import json
import logging
from pathlib import Path

from core.agentic.action_space import ActionType, FULL_GRAPH_ACTIONS
from core.agentic.langgraph_runtime import LangGraphAgentRuntime, _log_eval_failure_details
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.agentic.resume import StaticResumeDecisionProvider
from core.agentic.state_store import PipelineStateStore


class FakeObserver:
    def __init__(self) -> None:
        self.events = []

    def start_run(self, state: PipelineState) -> None:
        self.events.append(("start", list(state.completed_stages)))

    def before_action(self, state: PipelineState, request: ActionRequest) -> None:
        self.events.append(("before", request.action_type.value))

    def after_action(self, state: PipelineState, result: ActionResult) -> None:
        self.events.append(("after", result.action_type.value))

    def finish_run(self, state: PipelineState) -> None:
        self.events.append(("finish", state.termination_reason))


def _passing_report(stage: ActionType) -> QualityReport:
    return QualityReport(stage=stage, passed=True, score=1.0)


def _executors(call_counts: dict[str, int] | None = None):
    counts = call_counts if call_counts is not None else {}

    def _executor(stage: ActionType):
        def _run(state: PipelineState, request: ActionRequest) -> ActionResult:
            counts[stage.value] = counts.get(stage.value, 0) + 1
            return ActionResult(
                action_type=stage,
                status="success",
                quality_report=_passing_report(stage),
                artifacts={stage.value: f"{stage.value}.artifact"},
            )

        return _run

    return {stage: _executor(stage) for stage in FULL_GRAPH_ACTIONS}


def test_langgraph_runtime_runs_full_graph_and_persists_state(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="core.agentic.langgraph_runtime")
    observer = FakeObserver()
    runtime = LangGraphAgentRuntime(
        executors=_executors(),
        observer=observer,
        thread_id="test-full-run",
    )

    final_state = runtime.run(PipelineState(run_dir=str(tmp_path)))

    assert final_state.termination_reason == "full_graph_complete"
    assert final_state.completed_stages == [stage.value for stage in FULL_GRAPH_ACTIONS]
    assert (tmp_path / "agent_state.json").exists()
    assert (tmp_path / "decision_history.jsonl").exists()
    assert (tmp_path / "quality_history.jsonl").exists()
    assert (tmp_path / "result_history.jsonl").exists()
    assert (tmp_path / "config_history.jsonl").exists()
    assert (tmp_path / "agent_trace.jsonl").exists()
    assert (tmp_path / "run_summary.json").exists()

    loaded = PipelineStateStore(tmp_path).load()
    assert loaded.termination_reason == "full_graph_complete"
    assert loaded.completed_stages == final_state.completed_stages
    assert observer.events[0][0] == "start"
    assert observer.events[-1] == ("finish", "full_graph_complete")
    assert "Agent decision: stage=generate_taxonomy" in caplog.text
    assert "Agent stage start: stage=generate_taxonomy" in caplog.text
    assert "Agent stage result: stage=generate_taxonomy" in caplog.text
    assert "Agent: next I will run 'generate_taxonomy'." in caplog.text
    assert "Agent: doing 'generate_taxonomy' now." in caplog.text
    assert "Agent: 'generate_taxonomy' is done." in caplog.text


def test_state_store_redacts_secrets_and_writes_artifact_manifest(tmp_path: Path) -> None:
    sft_path = tmp_path / "sft" / "sft.jsonl"
    sft_path.parent.mkdir(parents=True)
    sft_path.write_text('{"messages": []}\n', encoding="utf-8")
    state = PipelineState(
        run_dir=str(tmp_path),
        config={
            "llm_api_key": "secret-llm",
            "hf_token": "secret-hf",
            "eval_max_new_tokens": 128,
        },
    )
    state.record_action_result(
        ActionResult(
            action_type=ActionType.BUILD_SFT_DATASET,
            status="success",
            quality_report=_passing_report(ActionType.BUILD_SFT_DATASET),
            artifacts={"sft_path": str(sft_path)},
            raw_output={"large": "passed output"},
        )
    )

    PipelineStateStore(tmp_path).save(state)

    payload = json.loads((tmp_path / "agent_state.json").read_text(encoding="utf-8"))
    assert payload["config"]["llm_api_key"] == "[REDACTED]"
    assert payload["config"]["hf_token"] == "[REDACTED]"
    assert payload["config"]["eval_max_new_tokens"] == 128
    assert payload["result_history"][0]["raw_output"]["omitted"] == "passed_stage_raw_output"
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text(encoding="utf-8"))
    sft_rows = [row for row in manifest["artifacts"] if row["key"] == "sft_path"]
    assert sft_rows
    assert sft_rows[0]["stage"] == "build_sft_dataset"
    assert sft_rows[0]["exists"] is True


def test_langgraph_runtime_resume_requires_confirmation_without_provider(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path))
    state.mark_stage_complete(
        ActionType.GENERATE_TAXONOMY,
        quality_report=_passing_report(ActionType.GENERATE_TAXONOMY),
        artifacts={"search_queries": ["q1"]},
    )
    PipelineStateStore(tmp_path).save(state)
    call_counts: dict[str, int] = {}
    runtime = LangGraphAgentRuntime(
        executors=_executors(call_counts),
        observer=FakeObserver(),
        thread_id="test-resume-required",
    )

    final_state = runtime.resume(tmp_path)

    assert final_state.termination_reason == "resume_confirmation_required"
    assert final_state.blockers == ["resume_confirmation_required:generate_taxonomy"]
    assert call_counts == {}


def test_langgraph_runtime_confirmed_resume_skips_completed_stage(tmp_path: Path) -> None:
    state = PipelineState(run_dir=str(tmp_path))
    state.mark_stage_complete(
        ActionType.GENERATE_TAXONOMY,
        quality_report=_passing_report(ActionType.GENERATE_TAXONOMY),
        artifacts={"search_queries": ["q1"]},
    )
    PipelineStateStore(tmp_path).save(state)
    call_counts: dict[str, int] = {}
    runtime = LangGraphAgentRuntime(
        executors=_executors(call_counts),
        observer=FakeObserver(),
        resume_decision_provider=StaticResumeDecisionProvider(confirm_all=True),
        thread_id="test-confirmed-resume",
    )

    final_state = runtime.resume(tmp_path)

    assert final_state.termination_reason == "full_graph_complete"
    assert final_state.resume_confirmations["generate_taxonomy"] is True
    assert call_counts.get("generate_taxonomy") is None
    assert call_counts["collect_data"] == 1
    assert final_state.completed_stages == [stage.value for stage in FULL_GRAPH_ACTIONS]


def test_langgraph_runtime_logs_collect_data_recovery_iteration(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="core.agentic.langgraph_runtime")
    call_counts: dict[str, int] = {}

    def _executor(stage: ActionType):
        def _run(state: PipelineState, request: ActionRequest) -> ActionResult:
            call_counts[stage.value] = call_counts.get(stage.value, 0) + 1
            if stage == ActionType.COLLECT_DATA and call_counts[stage.value] == 1:
                return ActionResult(
                    action_type=stage,
                    status="failed",
                    quality_report=QualityReport(
                        stage=stage,
                        passed=False,
                        recoverable=True,
                        blocking_issues=["num_samples_below_minimum"],
                    ),
                )
            return ActionResult(
                action_type=stage,
                status="success",
                quality_report=_passing_report(stage),
                artifacts={stage.value: f"{stage.value}.artifact"},
            )

        return _run

    runtime = LangGraphAgentRuntime(
        executors={stage: _executor(stage) for stage in FULL_GRAPH_ACTIONS},
        observer=FakeObserver(),
        thread_id="test-collect-recovery-logs",
    )

    final_state = runtime.run(
        PipelineState(
            run_dir=str(tmp_path),
            config={"serper_results_per_query": 1, "serper_top_results": 1, "max_queries": 1},
        )
    )

    assert final_state.termination_reason == "full_graph_complete"
    assert call_counts["collect_data"] == 2
    assert final_state.config["serper_results_per_query"] == 6
    assert "Agent recovery iteration: stage=collect_data attempt=1" in caplog.text
    assert "recovery_expand_collection_coverage" in caplog.text


def test_langgraph_runtime_logs_eval_failure_details(caplog) -> None:
    caplog.set_level(logging.INFO, logger="core.agentic.langgraph_runtime")
    report = QualityReport(
        stage=ActionType.EVALUATE_MODEL,
        passed=False,
        gate_status="repair",
        decision="repair",
        blocking_issues=["eval_failure_rate_too_high", "eval_training_failure"],
        metrics={"failure_rate": 1.0, "training_health_gate": "repair"},
    )
    result = ActionResult(
        action_type=ActionType.EVALUATE_MODEL,
        status="failed",
        quality_report=report,
        artifacts={
            "predictions_path": "predictions.jsonl",
            "failures_path": "failures.jsonl",
            "eval_metrics_path": "eval_metrics.json",
            "cluster_preview": {
                "clusters": [
                    {"label": "semantic_mismatch", "count": 8, "examples": [{"input": "hidden"}]},
                ]
            },
            "eval_metrics": {
                "failure_rate": 1.0,
                "num_failures": 8,
                "num_predictions": 8,
                "avg_similarity": 0.04,
                "failure_reason_counts": {"low": 8, "repetition": 1},
            },
            "training_health": {
                "gate_status": "repair",
                "blocking_issues": ["training_steps_incomplete"],
                "metrics": {
                    "last_step": 5,
                    "expected_steps": 200,
                    "step_completion_ratio": 0.025,
                    "first_train_loss": 2.7,
                    "last_train_loss": 2.7,
                    "best_eval_loss": None,
                    "loss_trend": "insufficient",
                },
            },
            "judge_summary": {
                "enabled": False,
                "gate_status": "pass",
                "major_failure_rate": 0.0,
            },
        },
    )

    _log_eval_failure_details(result, report)

    assert "Agent eval failure details:" in caplog.text
    assert '"failure_rate": 1.0' in caplog.text
    assert '"failure_reason_counts": {"low": 8, "repetition": 1}' in caplog.text
    assert '"last_step": 5' in caplog.text
    assert '"expected_steps": 200' in caplog.text
    assert '"last_train_loss": 2.7' in caplog.text
    assert "hidden" not in caplog.text
