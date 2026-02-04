"""LangGraph-based orchestration for agentic LoRA SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agentic_train_pipeline.agent.agent import Agent
from agentic_train_pipeline.agent.schemas import TrainingAdjustmentDecision
from agentic_train_pipeline.config import DEFAULT_SEED
from agentic_train_pipeline.data.hf_dataset import load_dataset_from_path
from agentic_train_pipeline.data.manifest import write_manifest
from agentic_train_pipeline.data.modality import build_example_preview, infer_modality
from agentic_train_pipeline.data.validation import validate_text_columns
from agentic_train_pipeline.eval.error_analysis import cluster_failures
from agentic_train_pipeline.eval.failures import collect_failures
from agentic_train_pipeline.eval.inference import run_inference
from agentic_train_pipeline.models.lora import attach_lora, preset_from_dict
from agentic_train_pipeline.registry.builtins import build_registry
from agentic_train_pipeline.registry.registry import RegistrySnapshot
from agentic_train_pipeline.reporting.report import write_report
from agentic_train_pipeline.training.log_parser import parse_metrics, read_log_tail
from agentic_train_pipeline.training.sft_runner import run_sft_iteration
from agentic_train_pipeline.training.tuning import (
    TuningBounds,
    apply_adjustments,
    generate_random_candidates,
)
from agentic_train_pipeline.types import DatasetSummary, IterationRecord, TrainConfig


class PipelineState(TypedDict, total=False):
    data_path: str
    task: str
    out_dir: str
    max_iters: int
    hf_model_id_override: Optional[str]
    search_trials: int
    registry: Any
    registry_snapshot: RegistrySnapshot
    out_dir_path: Path
    dataset: Any
    dataset_summary: DatasetSummary
    agent: Agent
    component_selection: Any
    dataset_loader: Any
    model_loader: Any
    lora_preset_dict: Dict[str, Any]
    trainer_cls: Any
    train_config: TrainConfig
    bounds: TuningBounds
    iterations: List[IterationRecord]
    iter_idx: int
    model: Any
    tokenizer: Any
    last_adjustment: TrainingAdjustmentDecision
    report_path: str


def _build_dataset_summary(data_path: str, dataset, resolved_id: str) -> DatasetSummary:
    columns = list(dataset.column_names)
    features = dataset.features
    modality_candidates = infer_modality(columns, features)

    example = build_example_preview(dataset[0]) if len(dataset) > 0 else {}
    text_columns = [c for c in columns if c in {"text", "prompt", "response", "instruction", "output"}]
    warnings = validate_text_columns(dataset, text_columns)

    return DatasetSummary(
        data_path=data_path,
        resolved_data_id=resolved_id,
        columns=columns,
        sample_count=len(dataset),
        example=example,
        modality_candidates=modality_candidates,
        validation_warnings=warnings,
    )


def _initialize(state: PipelineState) -> PipelineState:
    registry = build_registry()
    snapshot = registry.snapshot()

    out_dir_path = Path(state["out_dir"])
    out_dir_path.mkdir(parents=True, exist_ok=True)

    dataset, resolved_id = load_dataset_from_path(state["data_path"])
    dataset_summary = _build_dataset_summary(state["data_path"], dataset, resolved_id)
    write_manifest(state["out_dir"], dataset_summary.model_dump())

    agent = Agent(state["out_dir"], snapshot)
    modality_decision = agent.decide_modality(dataset_summary.model_dump())
    component_selection = agent.select_components(
        dataset_summary.model_dump(), modality_decision.modality
    )
    if state.get("hf_model_id_override"):
        component_selection = component_selection.model_copy(
            update={"hf_model_id": state["hf_model_id_override"]}
        )
    if not component_selection.hf_model_id:
        component_selection = component_selection.model_copy(
            update={"hf_model_id": snapshot.default_hf_model_id}
        )

    dataset_loader = registry.get_dataset_loader(component_selection.dataset_loader_key)
    model_loader = registry.get_model_loader(component_selection.model_loader_key)
    lora_preset_dict = registry.get_lora_preset(component_selection.lora_preset_key)
    trainer_cls = registry.get_trainer(component_selection.trainer_key)

    dataset, _ = dataset_loader(state["data_path"])

    return {
        "registry": registry,
        "registry_snapshot": snapshot,
        "out_dir_path": out_dir_path,
        "dataset": dataset,
        "dataset_summary": dataset_summary,
        "agent": agent,
        "component_selection": component_selection,
        "dataset_loader": dataset_loader,
        "model_loader": model_loader,
        "lora_preset_dict": lora_preset_dict,
        "trainer_cls": trainer_cls,
        "train_config": TrainConfig(seed=DEFAULT_SEED),
        "bounds": TuningBounds(),
        "iterations": [],
        "iter_idx": 0,
    }


def _maybe_search(state: PipelineState) -> PipelineState:
    search_trials = state.get("search_trials", 0)
    if search_trials <= 0:
        return {}

    search_dir = state["out_dir_path"] / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    candidates = generate_random_candidates(state["train_config"], state["bounds"], search_trials, DEFAULT_SEED)

    best_metric = None
    best_config = state["train_config"]
    for idx, candidate in enumerate(candidates):
        model, tokenizer = state["model_loader"](state["component_selection"].hf_model_id)
        model = attach_lora(model, preset_from_dict(state["lora_preset_dict"]))
        trial_dir = search_dir / f"trial_{idx}"
        adapter_path, log_paths = run_sft_iteration(
            model=model,
            tokenizer=tokenizer,
            dataset=state["dataset"],
            trainer_cls=state["trainer_cls"],
            train_config=candidate,
            out_dir=str(trial_dir),
        )
        _ = adapter_path
        metrics = parse_metrics(log_paths["metrics"])
        score = metrics.best_eval_loss if metrics.best_eval_loss is not None else metrics.last_train_loss
        if score is not None and (best_metric is None or score < best_metric):
            best_metric = score
            best_config = candidate

    return {"train_config": best_config}


def _init_model(state: PipelineState) -> PipelineState:
    model, tokenizer = state["model_loader"](state["component_selection"].hf_model_id)
    model = attach_lora(model, preset_from_dict(state["lora_preset_dict"]))
    return {"model": model, "tokenizer": tokenizer}


def _train_iteration(state: PipelineState) -> PipelineState:
    iter_dir = state["out_dir_path"] / "iterations" / f"iter_{state['iter_idx']}"
    adapter_path, log_paths = run_sft_iteration(
        model=state["model"],
        tokenizer=state["tokenizer"],
        dataset=state["dataset"],
        trainer_cls=state["trainer_cls"],
        train_config=state["train_config"],
        out_dir=str(iter_dir),
    )

    metrics = parse_metrics(log_paths["metrics"])
    log_tail = read_log_tail(log_paths["train_log"])

    record = IterationRecord(
        iter_idx=state["iter_idx"],
        config=state["train_config"],
        metrics=metrics,
        adapter_path=adapter_path,
        log_paths=log_paths,
    )
    iterations = list(state["iterations"])
    iterations.append(record)

    decision = state["agent"].suggest_training_adjustments(
        metrics_summary=metrics.model_dump(),
        log_tail=log_tail,
        bounds=state["bounds"].__dict__,
    )

    return {"iterations": iterations, "last_adjustment": decision}


def _should_continue(state: PipelineState) -> str:
    decision = state["last_adjustment"]
    if not decision.should_retry:
        return "post_training"
    if state["iter_idx"] + 1 >= state["max_iters"]:
        return "post_training"
    return "apply_adjustments"


def _apply_adjustments(state: PipelineState) -> PipelineState:
    decision = state["last_adjustment"]
    component_selection = state["component_selection"]
    lora_preset_dict = state["lora_preset_dict"]
    model = state["model"]
    tokenizer = state["tokenizer"]

    if decision.switch_lora_preset_key and decision.switch_lora_preset_key != component_selection.lora_preset_key:
        component_selection = component_selection.model_copy(
            update={"lora_preset_key": decision.switch_lora_preset_key}
        )
        lora_preset_dict = state["registry"].get_lora_preset(component_selection.lora_preset_key)
        model, tokenizer = state["model_loader"](component_selection.hf_model_id)
        model = attach_lora(model, preset_from_dict(lora_preset_dict))

    train_config = apply_adjustments(state["train_config"], decision, state["bounds"])

    return {
        "component_selection": component_selection,
        "lora_preset_dict": lora_preset_dict,
        "model": model,
        "tokenizer": tokenizer,
        "train_config": train_config,
        "iter_idx": state["iter_idx"] + 1,
    }


def _post_training(state: PipelineState) -> PipelineState:
    iterations_path = state["out_dir_path"] / "iterations.json"
    iterations_path.write_text(
        json.dumps([record.model_dump() for record in state["iterations"]], indent=2),
        encoding="utf-8",
    )

    predictions_path = run_inference(state["model"], state["tokenizer"], state["dataset"], state["out_dir"])
    failures_path = collect_failures(predictions_path, state["out_dir"])
    cluster_preview = cluster_failures(failures_path, state["out_dir"])

    failure_overview = {
        "total_failures": sum(1 for _ in open(failures_path, "r", encoding="utf-8")),
        "failures_path": failures_path,
    }
    error_analysis = state["agent"].analyze_errors(failure_overview, cluster_preview).model_dump()

    report_path = write_report(
        out_dir=state["out_dir"],
        dataset_summary=state["dataset_summary"].model_dump(),
        component_selection=state["component_selection"].model_dump(),
        iterations=state["iterations"],
        failures_path=failures_path,
        cluster_preview=cluster_preview,
        error_analysis=error_analysis,
    )

    return {"report_path": report_path}


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("initialize", _initialize)
    graph.add_node("maybe_search", _maybe_search)
    graph.add_node("init_model", _init_model)
    graph.add_node("train_iteration", _train_iteration)
    graph.add_node("apply_adjustments", _apply_adjustments)
    graph.add_node("post_training", _post_training)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "maybe_search")
    graph.add_edge("maybe_search", "init_model")
    graph.add_edge("init_model", "train_iteration")
    graph.add_conditional_edges(
        "train_iteration",
        _should_continue,
        {"apply_adjustments": "apply_adjustments", "post_training": "post_training"},
    )
    graph.add_edge("apply_adjustments", "train_iteration")
    graph.add_edge("post_training", END)
    return graph


def run_langgraph_pipeline(
    data_path: str,
    task: str,
    out_dir: str,
    max_iters: int,
    hf_model_id_override: Optional[str] = None,
    search_trials: int = 0,
) -> str:
    graph = build_graph().compile()
    result = graph.invoke(
        {
            "data_path": data_path,
            "task": task,
            "out_dir": out_dir,
            "max_iters": max_iters,
            "hf_model_id_override": hf_model_id_override,
            "search_trials": search_trials,
        }
    )
    return result["report_path"]
