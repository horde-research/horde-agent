"""Serializable state and result models for the v1 agentic controller.

These models intentionally use the Python standard library instead of Pydantic
so the agent core can be imported before the heavier runtime dependencies are
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.agentic.action_space import ActionType, FULL_GRAPH_ACTIONS, coerce_action_type


_STAGE_ARTIFACT_KEYS: dict[str, set[str]] = {
    ActionType.GENERATE_TAXONOMY.value: {"taxonomy", "search_queries", "image_taxonomy"},
    ActionType.COLLECT_DATA.value: {
        "raw_data_path",
        "data_path",
        "num_samples",
        "collection_metadata",
        "collection_text_quality_path",
        "collection_text_quality_summary",
        "images_dir",
        "images_index",
        "num_images",
        "raw_images_index",
        "image_dedup_report_path",
        "image_dedup_enabled",
        "image_dedup_method",
        "image_dedup_threshold",
        "image_dedup_model_path",
        "image_dedup_model_url",
        "image_dedup_device",
        "image_dedup_downloaded_model",
        "num_images_before_dedup",
        "num_images_removed_by_dedup",
        "num_image_dedup_clusters",
    },
    ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES.value: {
        "coverage_review",
        "coverage_added_queries",
        "image_query_specs",
    },
    ActionType.BUILD_SFT_DATASET.value: {
        "sft_mode",
        "training_modality",
        "sft_path",
        "annotations_path",
        "num_sft_examples",
        "sft_text_quality_path",
        "sft_text_quality_summary",
        "dataset_repo_id",
        "hf_dataset_upload_error",
    },
    ActionType.BUILD_DATASET.value: {
        "dataset_ref",
        "dataset_summary",
        "dataset_manifest_path",
    },
    ActionType.TRAIN_MODEL.value: {
        "adapter_path",
        "train_log_paths",
        "train_metrics",
        "iterations",
        "adapter_repo_id",
        "hf_adapter_upload_error",
        "hf_adapter_upload_skipped",
    },
    ActionType.EVALUATE_MODEL.value: {
        "eval_attempt",
        "eval_attempt_dir",
        "predictions_path",
        "failures_path",
        "cluster_preview",
        "eval_metrics_path",
        "eval_metrics",
        "training_health",
        "judge_summary",
    },
    ActionType.GENERATE_REPORT.value: {"report_path"},
}


def _as_plain(value: Any) -> Any:
    if isinstance(value, ActionType):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_plain(v) for v in value]
    return value


def _infer_gate_status(*, passed: bool, recoverable: bool, warnings: List[str]) -> str:
    if passed:
        return "warn" if warnings else "pass"
    return "repair" if recoverable else "fail"


def _decision_from_gate_status(gate_status: str) -> str:
    normalized = (gate_status or "").strip().lower()
    if normalized in {"pass", "warn"}:
        return "continue"
    if normalized == "repair":
        return "repair"
    return "stop"


@dataclass
class QualityReport:
    stage: ActionType | str
    passed: bool
    gate_status: str = ""
    decision: str = ""
    score: float = 0.0
    recoverable: bool = False
    issue_categories: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    suggested_adjustments: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stage = coerce_action_type(self.stage)
        if not self.gate_status:
            self.gate_status = _infer_gate_status(
                passed=self.passed,
                recoverable=self.recoverable,
                warnings=self.warnings,
            )
        if not self.decision:
            self.decision = _decision_from_gate_status(self.gate_status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "passed": self.passed,
            "gate_status": self.gate_status,
            "decision": self.decision,
            "score": self.score,
            "recoverable": self.recoverable,
            "issue_categories": list(self.issue_categories),
            "blocking_issues": list(self.blocking_issues),
            "warnings": list(self.warnings),
            "metrics": _as_plain(self.metrics),
            "recommended_actions": list(self.recommended_actions),
            "suggested_adjustments": _as_plain(self.suggested_adjustments),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QualityReport":
        return cls(
            stage=payload["stage"],
            passed=bool(payload["passed"]),
            gate_status=str(payload.get("gate_status") or ""),
            decision=str(payload.get("decision") or ""),
            score=float(payload.get("score", 0.0)),
            recoverable=bool(payload.get("recoverable", False)),
            issue_categories=list(payload.get("issue_categories") or []),
            blocking_issues=list(payload.get("blocking_issues") or []),
            warnings=list(payload.get("warnings") or []),
            metrics=dict(payload.get("metrics") or {}),
            recommended_actions=list(payload.get("recommended_actions") or []),
            suggested_adjustments=dict(payload.get("suggested_adjustments") or {}),
        )


@dataclass
class ActionRequest:
    action_type: ActionType | str
    stage: ActionType | str | None = None
    reason: str = ""
    config_delta: Dict[str, Any] = field(default_factory=dict)
    retry_attempt: int = 0

    def __post_init__(self) -> None:
        self.action_type = coerce_action_type(self.action_type)
        if self.stage is None:
            self.stage = self.action_type
        else:
            self.stage = coerce_action_type(self.stage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "stage": self.stage.value if isinstance(self.stage, ActionType) else self.stage,
            "reason": self.reason,
            "config_delta": _as_plain(self.config_delta),
            "retry_attempt": self.retry_attempt,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionRequest":
        return cls(
            action_type=payload["action_type"],
            stage=payload.get("stage"),
            reason=payload.get("reason", ""),
            config_delta=dict(payload.get("config_delta") or {}),
            retry_attempt=int(payload.get("retry_attempt") or 0),
        )


@dataclass
class ActionResult:
    action_type: ActionType | str
    status: str
    stage: ActionType | str | None = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    raw_output: Any = None
    quality_report: Optional[QualityReport] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.action_type = coerce_action_type(self.action_type)
        if self.stage is None:
            self.stage = self.action_type
        else:
            self.stage = coerce_action_type(self.stage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "stage": self.stage.value if isinstance(self.stage, ActionType) else self.stage,
            "status": self.status,
            "artifacts": _as_plain(self.artifacts),
            "metrics": _as_plain(self.metrics),
            "warnings": list(self.warnings),
            "raw_output": _as_plain(self.raw_output),
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionResult":
        quality_payload = payload.get("quality_report")
        return cls(
            action_type=payload["action_type"],
            status=payload["status"],
            stage=payload.get("stage"),
            artifacts=dict(payload.get("artifacts") or {}),
            metrics=dict(payload.get("metrics") or {}),
            warnings=list(payload.get("warnings") or []),
            raw_output=payload.get("raw_output"),
            quality_report=QualityReport.from_dict(quality_payload) if quality_payload else None,
            error=payload.get("error"),
        )


@dataclass
class PipelineState:
    run_dir: str
    mode: str = "full"
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    completed_stages: List[str] = field(default_factory=list)
    quality_reports: Dict[str, QualityReport] = field(default_factory=dict)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    config_history: List[Dict[str, Any]] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    result_history: List[Dict[str, Any]] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    resume_confirmations: Dict[str, bool] = field(default_factory=dict)
    last_action_result: Optional[ActionResult] = None
    max_stage_retries: int = 2
    max_graph_steps: int = 20
    termination_reason: Optional[str] = None

    def mark_stage_complete(
        self,
        stage: ActionType | str,
        *,
        quality_report: QualityReport,
        artifacts: Dict[str, Any] | None = None,
    ) -> None:
        stage_type = coerce_action_type(stage)
        stage_key = stage_type.value
        if stage_key not in self.completed_stages:
            self.completed_stages.append(stage_key)
        self.quality_reports[stage_key] = quality_report
        if artifacts:
            self.artifacts.update(artifacts)

    def record_decision(self, request: ActionRequest) -> None:
        self.decision_history.append(request.to_dict())

    def apply_config_delta(self, delta: Dict[str, Any], *, reason: str = "") -> None:
        if not delta:
            return
        before = {key: self.config.get(key) for key in delta}
        self.config.update(delta)
        self.config_history.append(
            {
                "reason": reason,
                "before": _as_plain(before),
                "delta": _as_plain(delta),
                "after": _as_plain({key: self.config.get(key) for key in delta}),
            }
        )

    def clear_stage_and_downstream(self, stage: ActionType | str) -> None:
        stage_type = coerce_action_type(stage)
        ordered = [action.value for action in FULL_GRAPH_ACTIONS]
        if stage_type.value not in ordered:
            return
        blocked = set(ordered[ordered.index(stage_type.value) :])
        self.completed_stages = [completed for completed in self.completed_stages if completed not in blocked]
        for blocked_stage in blocked:
            self.quality_reports.pop(blocked_stage, None)
            self.retry_counts.pop(blocked_stage, None)
            for artifact_key in _STAGE_ARTIFACT_KEYS.get(blocked_stage, set()):
                self.artifacts.pop(artifact_key, None)
        self.last_action_result = None
        self.termination_reason = None

    def record_action_result(self, result: ActionResult) -> None:
        self.last_action_result = result
        stage_key = result.action_type.value
        report = result.quality_report
        if report and report.passed:
            self.mark_stage_complete(
                result.action_type,
                quality_report=report,
                artifacts=result.artifacts,
            )
        self.result_history.append(result.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "mode": self.mode,
            "config": _as_plain(self.config),
            "artifacts": _as_plain(self.artifacts),
            "completed_stages": list(self.completed_stages),
            "quality_reports": {k: v.to_dict() for k, v in self.quality_reports.items()},
            "retry_counts": dict(self.retry_counts),
            "config_history": _as_plain(self.config_history),
            "decision_history": _as_plain(self.decision_history),
            "result_history": _as_plain(self.result_history),
            "blockers": list(self.blockers),
            "resume_confirmations": dict(self.resume_confirmations),
            "last_action_result": self.last_action_result.to_dict() if self.last_action_result else None,
            "max_stage_retries": self.max_stage_retries,
            "max_graph_steps": self.max_graph_steps,
            "termination_reason": self.termination_reason,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipelineState":
        quality_reports = {
            str(stage): QualityReport.from_dict(report)
            for stage, report in dict(payload.get("quality_reports") or {}).items()
        }
        last_action_payload = payload.get("last_action_result")
        return cls(
            run_dir=str(payload["run_dir"]),
            mode=str(payload.get("mode", "full")),
            config=dict(payload.get("config") or {}),
            artifacts=dict(payload.get("artifacts") or {}),
            completed_stages=list(payload.get("completed_stages") or []),
            quality_reports=quality_reports,
            retry_counts=dict(payload.get("retry_counts") or {}),
            config_history=list(payload.get("config_history") or []),
            decision_history=list(payload.get("decision_history") or []),
            result_history=list(payload.get("result_history") or []),
            blockers=list(payload.get("blockers") or []),
            resume_confirmations=dict(payload.get("resume_confirmations") or {}),
            last_action_result=ActionResult.from_dict(last_action_payload) if last_action_payload else None,
            max_stage_retries=int(payload.get("max_stage_retries", 2)),
            max_graph_steps=int(payload.get("max_graph_steps", 20)),
            termination_reason=payload.get("termination_reason"),
        )
