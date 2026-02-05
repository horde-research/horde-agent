"""High-level agent wrapper for pipeline decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from agentic_train_pipeline.agent.client import OpenAIJsonClient
from agentic_train_pipeline.agent.schemas import (
    ComponentSelectionDecision,
    ErrorAnalysisDecision,
    ModalityDecision,
    TrainingAdjustmentDecision,
)
from agentic_train_pipeline.prompts.loader import load_prompt
from agentic_train_pipeline.registry.registry import RegistrySnapshot


class Agent:
    def __init__(self, out_dir: str, registry_snapshot: RegistrySnapshot) -> None:
        self.client = OpenAIJsonClient()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.registry_snapshot = registry_snapshot
        self.system_prompt = load_prompt("system.txt")
        self.decisions_path = self.out_dir / "agent_decisions.jsonl"

    def _append_decision(self, stage: str, payload: Dict[str, Any]) -> None:
        record = {"stage": stage, "payload": payload}
        with self.decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _validate_registry_keys(self, decision: ComponentSelectionDecision) -> bool:
        snapshot = self.registry_snapshot
        return (
            decision.dataset_loader_key in snapshot.dataset_loader_keys
            and decision.model_loader_key in snapshot.model_loader_keys
            and decision.lora_preset_key in snapshot.lora_preset_keys
            and decision.trainer_key in snapshot.trainer_keys
        )

    def _validate_adjustment_keys(self, decision: TrainingAdjustmentDecision) -> bool:
        if decision.switch_lora_preset_key is None:
            return True
        return decision.switch_lora_preset_key in self.registry_snapshot.lora_preset_keys

    def decide_modality(self, dataset_summary: Dict[str, Any]) -> ModalityDecision:
        user_prompt = load_prompt("stage1_modality.txt", dataset_summary=json.dumps(dataset_summary, indent=2))
        payload = self.client.request_json(self.system_prompt, user_prompt)
        decision = ModalityDecision.model_validate(payload)
        self._append_decision("decide_modality", decision.model_dump())
        return decision

    def select_components(
        self, dataset_summary: Dict[str, Any], modality: str
    ) -> ComponentSelectionDecision:
        user_prompt = load_prompt(
            "stage2_components.txt",
            dataset_summary=json.dumps(dataset_summary, indent=2),
            modality=modality,
            registry_snapshot=json.dumps(self.registry_snapshot.model_dump(), indent=2),
        )
        payload = self.client.request_json(self.system_prompt, user_prompt)
        decision = ComponentSelectionDecision.model_validate(payload)
        if not self._validate_registry_keys(decision):
            correction = (
                "One or more keys are not in the registry. "
                "Choose only from the registry keys provided."
            )
            user_prompt = user_prompt + "\n\n" + correction
            payload = self.client.request_json(self.system_prompt, user_prompt)
            decision = ComponentSelectionDecision.model_validate(payload)
        self._append_decision("select_components", decision.model_dump())
        return decision

    def suggest_training_adjustments(
        self, metrics_summary: Dict[str, Any], log_tail: str, bounds: Dict[str, Any]
    ) -> TrainingAdjustmentDecision:
        user_prompt = load_prompt(
            "stage6_train_adjust.txt",
            metrics_summary=json.dumps(metrics_summary, indent=2),
            log_tail=log_tail,
            bounds=json.dumps(bounds, indent=2),
            registry_snapshot=json.dumps(self.registry_snapshot.model_dump(), indent=2),
        )
        payload = self.client.request_json(self.system_prompt, user_prompt)
        decision = TrainingAdjustmentDecision.model_validate(payload)
        if not self._validate_adjustment_keys(decision):
            correction = (
                "The lora preset key is invalid. "
                "Choose only from the registry keys provided."
            )
            user_prompt = user_prompt + "\n\n" + correction
            payload = self.client.request_json(self.system_prompt, user_prompt)
            decision = TrainingAdjustmentDecision.model_validate(payload)
        self._append_decision("suggest_training_adjustments", decision.model_dump())
        return decision

    def analyze_errors(self, failure_overview: Dict[str, Any], cluster_preview: Dict[str, Any]) -> ErrorAnalysisDecision:
        user_prompt = load_prompt(
            "stage8_error_analysis.txt",
            failure_overview=json.dumps(failure_overview, indent=2),
            cluster_preview=json.dumps(cluster_preview, indent=2),
        )
        payload = self.client.request_json(self.system_prompt, user_prompt)
        decision = ErrorAnalysisDecision.model_validate(payload)
        self._append_decision("analyze_errors", decision.model_dump())
        return decision
