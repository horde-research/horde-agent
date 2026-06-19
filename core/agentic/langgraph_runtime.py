"""LangGraph runtime for the constrained full-pipeline agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.agentic.action_space import ActionType, TERMINAL_ACTIONS
from core.agentic.controller import StageExecutor
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.agentic.observability import RunObserver, build_observer_from_env
from core.agentic.policy import choose_next_action
from core.agentic.resume import ResumeDecisionProvider, apply_resume_confirmations
from core.agentic.state_store import PipelineStateStore

logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict, total=False):
    pipeline_state: dict
    pending_request: dict


RouteFn = Callable[[AgentGraphState], str]


class LangGraphAgentRuntime:
    """Run the v1 constrained agent with LangGraph loop control."""

    def __init__(
        self,
        *,
        executors: Mapping[ActionType, StageExecutor],
        observer: RunObserver | None = None,
        resume_decision_provider: ResumeDecisionProvider | None = None,
        checkpointer=None,
        thread_id: str | None = None,
    ) -> None:
        self.executors: Dict[ActionType, StageExecutor] = dict(executors)
        self.observer = observer or build_observer_from_env()
        self.resume_decision_provider = resume_decision_provider
        self.checkpointer = checkpointer or MemorySaver()
        self.thread_id = thread_id or "full-agentic"
        self._store: PipelineStateStore | None = None
        self._started = False
        self._graph = self._build_graph()

    def run(self, state: PipelineState) -> PipelineState:
        self._store = PipelineStateStore(state.run_dir)
        self._started = False
        input_state: AgentGraphState = {"pipeline_state": state.to_dict()}
        output = self._graph.invoke(input_state, config=self._graph_config())
        return PipelineState.from_dict(output["pipeline_state"])

    def resume(self, run_dir: str | Path) -> PipelineState:
        store = PipelineStateStore(run_dir)
        state = store.load()
        state.termination_reason = None
        state.blockers = [blocker for blocker in state.blockers if not blocker.startswith("resume_confirmation_required:")]
        return self.run(state)

    def _build_graph(self):
        graph = StateGraph(AgentGraphState)
        graph.add_node("initialize", self._initialize)
        graph.add_node("confirm_resume", self._confirm_resume)
        graph.add_node("choose_action", self._choose_action)
        graph.add_node("execute_action", self._execute_action)
        graph.add_node("finish", self._finish)

        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "confirm_resume")
        graph.add_conditional_edges(
            "confirm_resume",
            self._route_after_confirm_resume,
            {"choose_action": "choose_action", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "choose_action",
            self._route_after_choose_action,
            {"execute_action": "execute_action", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "execute_action",
            self._route_after_execute_action,
            {"choose_action": "choose_action", "finish": "finish"},
        )
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=self.checkpointer, name="horde-full-agentic")

    def _initialize(self, graph_state: AgentGraphState) -> AgentGraphState:
        state = self._pipeline_state(graph_state)
        if state.mode != "full":
            state.termination_reason = "unsupported_mode"
        logger.info(
            "Agent run start: run_dir=%s thread_id=%s completed_stages=%s",
            state.run_dir,
            self.thread_id,
            state.completed_stages,
        )
        logger.info(
            "Agent: checking current pipeline state. Completed stages: %s.",
            state.completed_stages or "none",
        )
        self.observer.start_run(state)
        self._started = True
        self._save(state)
        return {"pipeline_state": state.to_dict()}

    def _confirm_resume(self, graph_state: AgentGraphState) -> AgentGraphState:
        state = self._pipeline_state(graph_state)
        apply_resume_confirmations(state, self.resume_decision_provider)
        if state.termination_reason == "resume_confirmation_required":
            logger.info(
                "Agent resume paused: blockers=%s completed_stages=%s",
                state.blockers,
                state.completed_stages,
            )
        elif state.resume_confirmations:
            logger.info("Agent resume confirmations: %s", state.resume_confirmations)
        self._save(state)
        return {"pipeline_state": state.to_dict()}

    def _choose_action(self, graph_state: AgentGraphState) -> AgentGraphState:
        state = self._pipeline_state(graph_state)
        request = choose_next_action(state)
        state.record_decision(request)
        if request.action_type in TERMINAL_ACTIONS:
            state.termination_reason = request.reason
            logger.info(
                "Agent decision: terminal=%s reason=%s completed_stages=%s blockers=%s",
                request.action_type.value,
                request.reason,
                state.completed_stages,
                state.blockers,
            )
            logger.info(
                "Agent: no more pipeline actions to run. I am finishing with reason '%s'.",
                request.reason,
            )
        else:
            logger.info(
                "Agent decision: stage=%s reason=%s retry_attempt=%d config_delta=%s",
                request.action_type.value,
                request.reason,
                request.retry_attempt,
                _compact_json(request.config_delta),
            )
            if request.retry_attempt:
                logger.info(
                    "Agent: quality checks requested repair. I will redo '%s' with updated settings: %s.",
                    request.action_type.value,
                    _compact_json(request.config_delta),
                )
            else:
                logger.info(
                    "Agent: next I will run '%s'. Reason: %s.",
                    request.action_type.value,
                    request.reason,
                )
        self._save(state)
        return {
            "pipeline_state": state.to_dict(),
            "pending_request": request.to_dict(),
        }

    def _execute_action(self, graph_state: AgentGraphState) -> AgentGraphState:
        state = self._pipeline_state(graph_state)
        request_payload = graph_state.get("pending_request")
        if not request_payload:
            state.termination_reason = "missing_pending_request"
            self._save(state)
            return {"pipeline_state": state.to_dict()}

        request = ActionRequest.from_dict(request_payload)
        if request.retry_attempt:
            logger.info(
                "Agent recovery iteration: stage=%s attempt=%d reason=%s config_delta=%s",
                request.action_type.value,
                request.retry_attempt,
                request.reason,
                _compact_json(request.config_delta),
            )
            state.clear_stage_and_downstream(request.action_type)
            state.record_recovery_fingerprint(request.recovery_fingerprint)
            state.apply_config_delta(request.config_delta, reason=request.reason)
            state.retry_counts[request.action_type.value] = request.retry_attempt
        elif request.config_delta:
            state.apply_config_delta(request.config_delta, reason=request.reason)

        logger.info(
            "Agent stage start: stage=%s retry_attempt=%d completed_before=%s",
            request.action_type.value,
            request.retry_attempt,
            state.completed_stages,
        )
        logger.info("Agent: doing '%s' now.", request.action_type.value)
        self.observer.before_action(state, request)
        result = self._execute(state, request)
        state.record_action_result(result)
        self.observer.after_action(state, result)
        report = result.quality_report
        logger.info(
            "Agent stage result: stage=%s status=%s passed=%s gate=%s decision=%s issues=%s metrics=%s artifacts=%s",
            result.action_type.value,
            result.status,
            report.passed if report else None,
            report.gate_status if report else None,
            report.decision if report else None,
            report.blocking_issues if report else [],
            _compact_json(report.metrics if report else result.metrics),
            sorted(result.artifacts.keys()),
        )
        _log_eval_failure_details(result, report)
        if report:
            logger.info(
                "Agent: '%s' is done. Gate=%s, decision=%s, issues=%s. I will inspect the state and choose the next step.",
                result.action_type.value,
                report.gate_status,
                report.decision,
                report.blocking_issues or report.warnings or "none",
            )
        else:
            logger.info(
                "Agent: '%s' is done without a quality report. I will inspect the state and choose the next step.",
                result.action_type.value,
            )

        if len(state.decision_history) >= state.max_graph_steps:
            state.termination_reason = "graph_step_limit_exhausted"

        self._save(state)
        return {"pipeline_state": state.to_dict(), "pending_request": {}}

    def _finish(self, graph_state: AgentGraphState) -> AgentGraphState:
        state = self._pipeline_state(graph_state)
        logger.info(
            "Agent run finish: termination_reason=%s completed_stages=%s retry_counts=%s blockers=%s",
            state.termination_reason,
            state.completed_stages,
            state.retry_counts,
            state.blockers,
        )
        logger.info(
            "Agent: run finished. Completed stages=%s. Retry counts=%s. Blockers=%s.",
            state.completed_stages,
            state.retry_counts,
            state.blockers or "none",
        )
        if self._started:
            self.observer.finish_run(state)
        self._save(state)
        return {"pipeline_state": state.to_dict()}

    def _route_after_confirm_resume(self, graph_state: AgentGraphState) -> str:
        state = self._pipeline_state(graph_state)
        return "finish" if state.termination_reason else "choose_action"

    def _route_after_choose_action(self, graph_state: AgentGraphState) -> str:
        state = self._pipeline_state(graph_state)
        return "finish" if state.termination_reason else "execute_action"

    def _route_after_execute_action(self, graph_state: AgentGraphState) -> str:
        state = self._pipeline_state(graph_state)
        return "finish" if state.termination_reason else "choose_action"

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

    def _pipeline_state(self, graph_state: AgentGraphState) -> PipelineState:
        return PipelineState.from_dict(graph_state["pipeline_state"])

    def _save(self, state: PipelineState) -> None:
        if self._store is None:
            self._store = PipelineStateStore(state.run_dir)
        self._store.save(state)

    def _graph_config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}


def _compact_json(value: Any) -> str:
    if value in (None, {}, []):
        return "{}" if isinstance(value, dict) or value is None else "[]"
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _log_eval_failure_details(result: ActionResult, report: QualityReport | None) -> None:
    if result.action_type != ActionType.EVALUATE_MODEL or not report:
        return
    if report.passed and report.gate_status == "pass":
        return

    artifacts = result.artifacts or {}
    eval_metrics = _as_mapping(artifacts.get("eval_metrics"))
    report_metrics = _as_mapping(report.metrics)
    training_health = _as_mapping(artifacts.get("training_health")) or _as_mapping(
        eval_metrics.get("training_health")
    )
    training_metrics = _as_mapping(training_health.get("metrics"))
    judge = _as_mapping(artifacts.get("judge_summary")) or _as_mapping(eval_metrics.get("judge"))

    details = _drop_empty(
        {
            "failure_rate": eval_metrics.get("failure_rate", report_metrics.get("failure_rate")),
            "num_failures": eval_metrics.get("num_failures"),
            "num_predictions": eval_metrics.get("num_predictions"),
            "avg_similarity": eval_metrics.get("avg_similarity"),
            "failure_reason_counts": eval_metrics.get("failure_reason_counts"),
            "clusters": _cluster_summary(artifacts.get("cluster_preview")),
            "training_health": _drop_empty(
                {
                    "gate": training_health.get("gate_status"),
                    "blocking_issues": training_health.get("blocking_issues"),
                    "warnings": training_health.get("warnings"),
                    "last_step": training_metrics.get("last_step"),
                    "expected_steps": training_metrics.get("expected_steps"),
                    "step_completion_ratio": training_metrics.get("step_completion_ratio"),
                    "first_train_loss": training_metrics.get("first_train_loss"),
                    "last_train_loss": training_metrics.get("last_train_loss"),
                    "best_eval_loss": training_metrics.get("best_eval_loss"),
                    "max_grad_norm": training_metrics.get("max_grad_norm"),
                    "loss_trend": training_metrics.get("loss_trend"),
                }
            ),
            "judge": _drop_empty(
                {
                    "enabled": judge.get("enabled"),
                    "gate": judge.get("gate_status"),
                    "major_failure_rate": judge.get("major_failure_rate"),
                    "major_failure_count": judge.get("major_failure_count"),
                    "warning_count": judge.get("warning_count"),
                    "failure_category_counts": judge.get("failure_category_counts"),
                }
            ),
            "paths": _drop_empty(
                {
                    "eval_metrics": artifacts.get("eval_metrics_path"),
                    "failures": artifacts.get("failures_path"),
                    "predictions": artifacts.get("predictions_path"),
                }
            ),
        }
    )
    if details:
        logger.info("Agent eval failure details: %s", _compact_json(details))


def _as_mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _cluster_summary(cluster_preview: Any) -> list[dict[str, Any]]:
    preview = _as_mapping(cluster_preview)
    clusters = preview.get("clusters")
    if not isinstance(clusters, list):
        return []
    summary: list[dict[str, Any]] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        item = _drop_empty(
            {
                "label": cluster.get("label"),
                "count": cluster.get("count"),
            }
        )
        if item:
            summary.append(item)
    return summary


def _drop_empty(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}
