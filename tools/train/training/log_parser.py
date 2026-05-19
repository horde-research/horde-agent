"""Parse training logs into a metrics summary.

Copied from `agentic_train_pipeline/training/log_parser.py` and adjusted for new package layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.types.pipeline_types import MetricsSummary


def parse_metrics(metrics_path: str) -> MetricsSummary:
    best_eval = None
    last_train = None
    last_eval = None
    last_reward = None
    reward_sum = 0.0
    reward_count = 0
    last_kl = None
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
            reward_value = record.get("reward") or record.get("rewards/mean") or record.get("mean_reward")
            if reward_value is not None:
                last_reward = float(reward_value)
                reward_sum += last_reward
                reward_count += 1
            kl_value = record.get("kl") or record.get("objective/kl")
            if kl_value is not None:
                last_kl = float(kl_value)

    return MetricsSummary(
        steps=last_step,
        best_eval_loss=best_eval,
        last_train_loss=last_train,
        last_eval_loss=last_eval,
        last_reward=last_reward,
        mean_reward=(reward_sum / reward_count) if reward_count else None,
        last_kl=last_kl,
    )


def read_log_tail(log_path: str, max_lines: int = 25) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = lines[-max_lines:]
    return "\n".join(tail)

