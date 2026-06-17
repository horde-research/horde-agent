"""Training-health checks derived from trainer metrics logs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


LOSS_KEYS = ("loss", "train_loss")
EVAL_LOSS_KEYS = ("eval_loss",)
GRAD_KEYS = ("grad_norm", "gradient_norm")


def evaluate_training_health(
    train_log_paths: Dict[str, Any] | None,
    *,
    expected_steps: int | None = None,
) -> Dict[str, Any]:
    """Return categorical training-health report from metrics JSONL."""
    train_log_paths = train_log_paths or {}
    metrics_path = train_log_paths.get("metrics")
    records = _read_metrics_jsonl(metrics_path)
    train_losses = _values(records, LOSS_KEYS)
    eval_losses = _values(records, EVAL_LOSS_KEYS)
    grad_norms = _values(records, GRAD_KEYS)
    steps = [int(record.get("step", 0) or 0) for record in records if _is_intish(record.get("step"))]
    last_step = max(steps) if steps else 0

    blocking_issues: list[str] = []
    warnings: list[str] = []
    issue_categories: list[str] = []

    if not metrics_path or not Path(str(metrics_path)).exists():
        warnings.append("training_metrics_missing")
        issue_categories.append("training_health")
    elif not records:
        warnings.append("training_metrics_empty")
        issue_categories.append("training_health")

    if any(not math.isfinite(value) for value in [*train_losses, *eval_losses, *grad_norms]):
        blocking_issues.append("training_nonfinite_metric")
        issue_categories.append("training_health")

    if train_losses:
        if _near_zero(train_losses):
            blocking_issues.append("training_loss_degenerate_zero")
            issue_categories.append("training_health")
        trend = _loss_trend(train_losses)
        if trend == "exploding":
            blocking_issues.append("training_loss_exploding")
            issue_categories.append("training_health")
        elif trend == "not_improving":
            warnings.append("training_loss_not_improving")
            issue_categories.append("training_health")
    else:
        warnings.append("training_loss_missing")
        issue_categories.append("training_health")

    if grad_norms and max(grad_norms) > 100.0:
        blocking_issues.append("training_gradient_norm_exploding")
        issue_categories.append("training_health")

    if expected_steps and expected_steps > 0 and last_step:
        completion_ratio = last_step / expected_steps
        if completion_ratio < 0.8:
            blocking_issues.append("training_steps_incomplete")
            issue_categories.append("training_health")
    else:
        completion_ratio = None

    gate_status = "repair" if blocking_issues else ("warn" if warnings else "pass")
    return {
        "passed": not blocking_issues,
        "gate_status": gate_status,
        "decision": "repair" if gate_status == "repair" else "continue",
        "issue_categories": sorted(set(issue_categories)),
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "metrics": {
            "num_metric_records": len(records),
            "last_step": last_step,
            "expected_steps": expected_steps,
            "step_completion_ratio": completion_ratio,
            "first_train_loss": train_losses[0] if train_losses else None,
            "last_train_loss": train_losses[-1] if train_losses else None,
            "best_eval_loss": min(eval_losses) if eval_losses else None,
            "max_grad_norm": max(grad_norms) if grad_norms else None,
            "loss_trend": _loss_trend(train_losses),
        },
    }


def _read_metrics_jsonl(path_value: Any) -> List[Dict[str, Any]]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _values(records: Iterable[Dict[str, Any]], keys: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for record in records:
        for key in keys:
            if key not in record:
                continue
            try:
                values.append(float(record[key]))
            except (TypeError, ValueError):
                values.append(float("nan"))
            break
    return values


def _loss_trend(losses: list[float]) -> str:
    finite = [loss for loss in losses if math.isfinite(loss)]
    if len(finite) < 3:
        return "insufficient"
    window = max(1, len(finite) // 5)
    start = mean(finite[:window])
    end = mean(finite[-window:])
    if end > start * 1.5 + 0.25:
        return "exploding"
    if end >= start * 0.98:
        return "not_improving"
    return "improving"


def _near_zero(values: list[float]) -> bool:
    finite = [abs(value) for value in values if math.isfinite(value)]
    return bool(finite) and max(finite) < 1e-8


def _is_intish(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
