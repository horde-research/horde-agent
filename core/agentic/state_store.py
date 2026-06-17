"""Durable state snapshots for agentic runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.agentic.models import PipelineState


class PipelineStateStore:
    """Persist the latest PipelineState and compact inspection artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "agent_state.json"
        self.decision_history_path = self.run_dir / "decision_history.jsonl"
        self.quality_history_path = self.run_dir / "quality_history.jsonl"
        self.result_history_path = self.run_dir / "result_history.jsonl"
        self.config_history_path = self.run_dir / "config_history.jsonl"
        self.agent_trace_path = self.run_dir / "agent_trace.jsonl"
        self.run_summary_path = self.run_dir / "run_summary.json"

    def exists(self) -> bool:
        return self.state_path.exists()

    def save(self, state: PipelineState) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = state.to_dict()
        self.state_path.write_text(_to_json(payload), encoding="utf-8")
        self._write_jsonl(self.decision_history_path, state.decision_history)
        self._write_jsonl(self.quality_history_path, [report.to_dict() for report in state.quality_reports.values()])
        self._write_jsonl(self.result_history_path, state.result_history)
        self._write_jsonl(self.config_history_path, state.config_history)
        self._write_jsonl(self.agent_trace_path, _build_agent_trace(state))
        self.run_summary_path.write_text(
            _to_json(
                {
                    "run_dir": state.run_dir,
                    "mode": state.mode,
                    "termination_reason": state.termination_reason,
                    "completed_stages": list(state.completed_stages),
                    "retry_counts": dict(state.retry_counts),
                    "blockers": list(state.blockers),
                }
            ),
            encoding="utf-8",
        )

    def load(self) -> PipelineState:
        if not self.state_path.exists():
            raise FileNotFoundError(f"No agent state snapshot found: {self.state_path}")
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return PipelineState.from_dict(payload)

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _build_agent_trace(state: PipelineState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, decision in enumerate(state.decision_history):
        rows.append({"event": "decision", "idx": idx, **decision})
        if idx < len(state.result_history):
            result = state.result_history[idx]
            report = result.get("quality_report") or {}
            rows.append(
                {
                    "event": "result",
                    "idx": idx,
                    "action_type": result.get("action_type"),
                    "status": result.get("status"),
                    "gate_status": report.get("gate_status"),
                    "decision": report.get("decision"),
                    "passed": report.get("passed"),
                    "blocking_issues": report.get("blocking_issues") or [],
                    "metrics": report.get("metrics") or result.get("metrics") or {},
                    "error": result.get("error"),
                }
            )
    for idx, change in enumerate(state.config_history):
        rows.append({"event": "config_update", "idx": idx, **change})
    return rows
