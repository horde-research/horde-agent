"""Apply bounded training adjustments."""

from dataclasses import dataclass

from agentic_train_pipeline.agent.schemas import TrainingAdjustmentDecision
from agentic_train_pipeline.types import TrainConfig


@dataclass
class TuningBounds:
    min_lr: float = 1e-5
    max_lr: float = 5e-4
    min_batch_size: int = 1
    max_batch_size: int = 16
    min_grad_accum: int = 1
    max_grad_accum: int = 16
    min_steps: int = 50
    max_steps: int = 1000


def generate_random_candidates(
    base: TrainConfig, bounds: TuningBounds, num_trials: int, seed: int
) -> list[TrainConfig]:
    import random

    rng = random.Random(seed)
    candidates = [base]
    for _ in range(max(0, num_trials - 1)):
        candidates.append(
            base.model_copy(
                update={
                    "lr": rng.uniform(bounds.min_lr, bounds.max_lr),
                    "batch_size": rng.randint(bounds.min_batch_size, bounds.max_batch_size),
                    "grad_accum": rng.randint(bounds.min_grad_accum, bounds.max_grad_accum),
                    "max_steps": rng.randint(bounds.min_steps, bounds.max_steps),
                }
            )
        )
    return candidates


def apply_adjustments(
    config: TrainConfig, decision: TrainingAdjustmentDecision, bounds: TuningBounds
) -> TrainConfig:
    lr = min(max(config.lr * decision.lr_multiplier, bounds.min_lr), bounds.max_lr)
    batch_size = min(max(config.batch_size + decision.batch_size_delta, bounds.min_batch_size), bounds.max_batch_size)
    grad_accum = min(max(config.grad_accum + decision.grad_accum_delta, bounds.min_grad_accum), bounds.max_grad_accum)
    max_steps = min(max(config.max_steps + decision.max_steps_delta, bounds.min_steps), bounds.max_steps)

    return config.model_copy(
        update={
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "max_steps": max_steps,
        }
    )
