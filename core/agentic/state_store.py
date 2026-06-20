"""Durable state snapshots for agentic runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from core.agentic.models import PipelineState, _STAGE_ARTIFACT_KEYS
from core.redaction import is_secret_key, redact_secrets


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
        self.artifact_manifest_path = self.run_dir / "artifact_manifest.json"

    def exists(self) -> bool:
        return self.state_path.exists()

    def save(self, state: PipelineState) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = _compact_state_payload(state.to_dict())
        self.state_path.write_text(_to_json(_redact_secrets(payload)), encoding="utf-8")
        self._write_jsonl(self.decision_history_path, state.decision_history)
        self._write_jsonl(self.quality_history_path, [report.to_dict() for report in state.quality_reports.values()])
        self._write_jsonl(self.result_history_path, state.result_history)
        self._write_jsonl(self.config_history_path, state.config_history)
        self._write_jsonl(self.agent_trace_path, _build_agent_trace(state))
        self.artifact_manifest_path.write_text(
            _to_json(_build_artifact_manifest(state, self.run_dir)),
            encoding="utf-8",
        )
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
                handle.write(json.dumps(_redact_secrets(row), ensure_ascii=False, default=str) + "\n")


def _to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _redact_secrets(value: Any, key: str = "") -> Any:
    return redact_secrets(value, key)


def _compact_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for result in payload.get("result_history") or []:
        if isinstance(result, dict):
            _compact_result_payload(result)
    last_result = payload.get("last_action_result")
    if isinstance(last_result, dict):
        _compact_result_payload(last_result)
    return payload


def _compact_result_payload(result: dict[str, Any]) -> None:
    report = result.get("quality_report") if isinstance(result.get("quality_report"), dict) else {}
    if not report.get("passed"):
        return
    if result.get("raw_output") in (None, {}, []):
        return
    result["raw_output"] = {
        "omitted": "passed_stage_raw_output",
        "artifact_keys": sorted((result.get("artifacts") or {}).keys()),
    }


def _is_secret_key(key: str) -> bool:
    return is_secret_key(key)


def _build_artifact_manifest(
    state: PipelineState,
    run_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stage_by_key = _stage_by_artifact_key()
    for key, value in state.artifacts.items():
        stage = stage_by_key.get(key)
        rows.extend(_artifact_rows(key, value, run_dir, stage=stage))
    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda row: (str(row.get("stage") or ""), str(row.get("key") or "")))
    return {
        "schema_version": "artifact_manifest.v1",
        "run_dir": str(run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": rows,
    }


def _stage_by_artifact_key() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for stage, keys in _STAGE_ARTIFACT_KEYS.items():
        for key in keys:
            mapping[key] = stage
    return mapping


def _artifact_rows(
    key: str,
    value: Any,
    run_dir: Path,
    *,
    stage: str | None,
) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for child_key, child_value in value.items():
            rows.extend(_artifact_rows(f"{key}.{child_key}", child_value, run_dir, stage=stage))
        return rows
    if isinstance(value, list):
        rows = []
        for idx, item in enumerate(value):
            rows.extend(_artifact_rows(f"{key}.{idx}", item, run_dir, stage=stage))
        return rows
    row = _artifact_row(key, value, run_dir, stage=stage)
    return [row] if row else []


def _artifact_row(
    key: str,
    value: Any,
    run_dir: Path,
    *,
    stage: str | None,
) -> dict[str, Any] | None:
    if not isinstance(value, (str, Path)):
        return None
    raw = str(value).strip()
    if not raw or not _looks_like_artifact_path(key, raw):
        return None
    path = Path(raw)
    if not path.is_absolute():
        candidate = run_dir / path
        path = candidate if candidate.exists() else path
    exists = path.exists()
    if not exists and not _looks_like_artifact_key(key):
        return None
    kind = "directory" if exists and path.is_dir() else "file"
    stat = path.stat() if exists else None
    return {
        "key": key,
        "stage": stage or "unknown",
        "path": str(path),
        "exists": exists,
        "kind": kind,
        "size_bytes": stat.st_size if stat and path.is_file() else None,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
        "attempt": _path_marker(path, "attempt_"),
        "iteration": _path_marker(path, "iter_"),
        "active": True,
    }


def _looks_like_artifact_path(key: str, value: str) -> bool:
    if _looks_like_artifact_key(key):
        return True
    path = Path(value)
    return path.is_absolute() or value.startswith(("./", "../"))


def _looks_like_artifact_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in ("path", "dir", "index", "manifest", "log"))


def _path_marker(path: Path, prefix: str) -> int | None:
    for part in path.parts:
        if part.startswith(prefix):
            try:
                return int(part[len(prefix) :])
            except ValueError:
                return None
    return None


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
