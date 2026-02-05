"""Parse training logs into a metrics summary."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_train_pipeline.types import MetricsSummary


def parse_metrics(metrics_path: str) -> MetricsSummary:
    best_eval = None
    last_train = None
    last_eval = None
    last_step = 0

    path = Path(metrics_path)
    if not path.exists():
        return MetricsSummary(steps=0, best_eval_loss=None, last_train_loss=None, last_eval_loss=None)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = int(record.get("step", 0))
            last_step = max(last_step, step)
            # Handle both "loss" and "train_loss" keys
            if "loss" in record:
                last_train = float(record["loss"])
            elif "train_loss" in record:
                last_train = float(record["train_loss"])
            # Handle eval loss
            if "eval_loss" in record:
                last_eval = float(record["eval_loss"])
                if best_eval is None or last_eval < best_eval:
                    best_eval = last_eval

    return MetricsSummary(
        steps=last_step,
        best_eval_loss=best_eval,
        last_train_loss=last_train,
        last_eval_loss=last_eval,
    )


def read_log_tail(log_path: str, max_lines: int = 25) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = lines[-max_lines:]
    return "\n".join(tail)
