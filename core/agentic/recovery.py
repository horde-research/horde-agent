"""Bounded recovery planning for the constrained full-pipeline agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from core.agentic.action_space import ActionType
from core.agentic.models import ActionResult, PipelineState


@dataclass
class RecoveryPlan:
    """A legal, bounded intervention after a recoverable quality failure."""

    target_stage: ActionType
    reason: str
    config_delta: Dict[str, Any] = field(default_factory=dict)


def build_recovery_plan(state: PipelineState, result: ActionResult) -> RecoveryPlan:
    report = result.quality_report
    issues = set(report.blocking_issues if report else [])

    if result.action_type == ActionType.GENERATE_TAXONOMY:
        return _taxonomy_plan(state, issues)
    if result.action_type == ActionType.COLLECT_DATA:
        return _collection_plan(state, issues)
    if result.action_type == ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES:
        return _coverage_plan(state, issues, result)
    if result.action_type == ActionType.BUILD_SFT_DATASET:
        return _sft_plan(state, issues)
    if result.action_type == ActionType.BUILD_DATASET:
        return _dataset_plan(state, issues)
    if result.action_type == ActionType.TRAIN_MODEL:
        return _training_plan(state, issues)
    if result.action_type == ActionType.EVALUATE_MODEL:
        return _eval_plan(state, issues, result)
    return RecoveryPlan(
        target_stage=result.action_type,
        reason=f"recovery_retry:{result.action_type.value}",
    )


def _taxonomy_plan(state: PipelineState, issues: set[str]) -> RecoveryPlan:
    delta: Dict[str, Any] = {
        "enable_taxonomy_mini_loop": True,
        "taxonomy_repair_attempts": min(_int_cfg(state, "taxonomy_repair_attempts", 2) + 1, 4),
    }
    if "queries_missing" in issues:
        delta["taxonomy_min_queries"] = max(1, _int_cfg(state, "taxonomy_min_queries", 1))
    return RecoveryPlan(
        target_stage=ActionType.GENERATE_TAXONOMY,
        reason="recovery_expand_taxonomy_generation",
        config_delta=delta,
    )


def _collection_plan(state: PipelineState, issues: set[str]) -> RecoveryPlan:
    delta: Dict[str, Any] = {}
    if issues & {"num_samples_below_minimum", "data_path_missing"}:
        delta["serper_results_per_query"] = min(_int_cfg(state, "serper_results_per_query", 10) + 5, 30)
        delta["serper_top_results"] = min(_int_cfg(state, "serper_top_results", 5) + 2, 12)
        if state.config.get("max_queries") is not None:
            delta["max_queries"] = min(_int_cfg(state, "max_queries", 10) + 10, 200)

    if issues & {"num_images_below_minimum", "images_dir_missing", "images_index_missing", "image_slot_coverage_missing"}:
        delta["image_search_results_per_query"] = min(
            _int_cfg(state, "image_search_results_per_query", _int_cfg(state, "serper_results_per_query", 10)) + 5,
            40,
        )
        delta["image_taxonomy_queries_per_slot"] = min(_int_cfg(state, "image_taxonomy_queries_per_slot", 4) + 1, 8)
        delta["image_min_width"] = max(224, int(_int_cfg(state, "image_min_width", 300) * 0.8))
        delta["image_min_height"] = max(224, int(_int_cfg(state, "image_min_height", 300) * 0.8))

    if not delta:
        return RecoveryPlan(
            target_stage=ActionType.COLLECT_DATA,
            reason="retry_recoverable_quality_failure",
        )

    return RecoveryPlan(
        target_stage=ActionType.COLLECT_DATA,
        reason="recovery_expand_collection_coverage",
        config_delta=delta,
    )


def _coverage_plan(state: PipelineState, issues: set[str], result: ActionResult) -> RecoveryPlan:
    review = result.raw_output if isinstance(result.raw_output, dict) else {}
    added_queries = _merge_list_cfg(state.config.get("coverage_added_queries"), review.get("added_queries"))
    image_specs = review.get("image_query_specs") if isinstance(review.get("image_query_specs"), list) else []
    delta: Dict[str, Any] = {}
    if added_queries:
        delta["coverage_added_queries"] = added_queries
        delta["serper_results_per_query"] = min(_int_cfg(state, "serper_results_per_query", 10) + 3, 30)
        delta["serper_top_results"] = min(_int_cfg(state, "serper_top_results", 5) + 1, 12)
    if image_specs:
        delta["image_query_specs"] = image_specs
        delta["image_search_results_per_query"] = min(
            _int_cfg(state, "image_search_results_per_query", _int_cfg(state, "serper_results_per_query", 10)) + 5,
            40,
        )
    if not delta:
        return _collection_plan(state, issues)
    return RecoveryPlan(
        target_stage=ActionType.COLLECT_DATA,
        reason="recovery_refine_collection_queries",
        config_delta=delta,
    )


def _sft_plan(state: PipelineState, issues: set[str]) -> RecoveryPlan:
    delta: Dict[str, Any] = {
        "llm_batch_size": max(1, _int_cfg(state, "llm_batch_size", 5) // 2),
        "llm_batch_delay": max(float(state.config.get("llm_batch_delay", 1.0)), 1.0) + 0.5,
    }
    if issues & {"all_annotations_failed", "num_examples_below_minimum"}:
        delta["sft_prompt_preset"] = "schema_strict"
    return RecoveryPlan(
        target_stage=ActionType.BUILD_SFT_DATASET,
        reason="recovery_adjust_sft_annotation",
        config_delta=delta,
    )


def _dataset_plan(state: PipelineState, issues: set[str]) -> RecoveryPlan:
    if issues & {"sample_count_below_minimum", "columns_missing"}:
        return RecoveryPlan(
            target_stage=ActionType.BUILD_SFT_DATASET,
            reason="recovery_rebuild_sft_for_dataset_quality",
            config_delta={"sft_prompt_preset": "schema_strict"},
        )
    return RecoveryPlan(
        target_stage=ActionType.BUILD_DATASET,
        reason="recovery_rebuild_dataset",
    )


def _training_plan(state: PipelineState, issues: set[str]) -> RecoveryPlan:
    lr = _float_cfg(state, "train_lr", 2e-4)
    delta = {
        "train_lr": max(lr * 0.5, 1e-6),
        "max_steps": min(_int_cfg(state, "max_steps", 200) + 50, 1000),
        "train_grad_accum": min(_int_cfg(state, "train_grad_accum", 4) + 1, 16),
    }
    return RecoveryPlan(
        target_stage=ActionType.TRAIN_MODEL,
        reason="recovery_stabilize_training",
        config_delta=delta,
    )


def _eval_plan(state: PipelineState, issues: set[str], result: ActionResult) -> RecoveryPlan:
    labels = _failure_labels(result)
    if issues & {"eval_knowledge_missing", "eval_grounding_failure"} or _contains_any(
        labels,
        ("knowledge", "coverage", "missing", "grounding", "hallucination"),
    ):
        return RecoveryPlan(
            target_stage=ActionType.COLLECT_DATA,
            reason="recovery_eval_requests_more_source_coverage",
            config_delta={
                "serper_results_per_query": min(_int_cfg(state, "serper_results_per_query", 10) + 5, 30),
                "serper_top_results": min(_int_cfg(state, "serper_top_results", 5) + 2, 12),
                "image_search_results_per_query": min(
                    _int_cfg(state, "image_search_results_per_query", _int_cfg(state, "serper_results_per_query", 10)) + 5,
                    40,
                ),
            },
        )
    if issues & {"eval_formatting_failure", "eval_schema_failure"} or _contains_any(labels, ("format", "schema", "json")):
        return RecoveryPlan(
            target_stage=ActionType.BUILD_SFT_DATASET,
            reason="recovery_eval_requests_sft_prompt_repair",
            config_delta={"sft_prompt_preset": "schema_strict"},
        )
    if issues & {"eval_training_failure"} or _contains_any(labels, ("unstable", "loss", "training")):
        return _training_plan(state, issues)
    return RecoveryPlan(
        target_stage=ActionType.EVALUATE_MODEL,
        reason="recovery_rerun_evaluation",
        config_delta={"eval_max_samples": min(_int_cfg(state, "eval_max_samples", 64) + 32, 256)},
    )


def _failure_labels(result: ActionResult) -> list[str]:
    labels: list[str] = []
    for source in (result.metrics, result.artifacts, result.raw_output if isinstance(result.raw_output, dict) else {}):
        cluster_preview = source.get("cluster_preview") if isinstance(source, dict) else None
        if isinstance(cluster_preview, dict):
            clusters = cluster_preview.get("clusters") or []
            for cluster in clusters:
                if isinstance(cluster, dict) and cluster.get("label"):
                    labels.append(str(cluster["label"]).lower())
        explicit = source.get("failure_clusters") if isinstance(source, dict) else None
        if isinstance(explicit, Iterable) and not isinstance(explicit, (str, bytes)):
            labels.extend(str(item).lower() for item in explicit)
    return labels


def _contains_any(values: list[str], needles: tuple[str, ...]) -> bool:
    return any(any(needle in value for needle in needles) for value in values)


def _merge_list_cfg(existing: Any, additions: Any) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in [*_iter_list(existing), *_iter_list(additions)]:
        key = str(value).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def _iter_list(value: Any) -> list[Any]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return []


def _int_cfg(state: PipelineState, key: str, default: int) -> int:
    try:
        return int(state.config.get(key, default))
    except (TypeError, ValueError):
        return default


def _float_cfg(state: PipelineState, key: str, default: float) -> float:
    try:
        return float(state.config.get(key, default))
    except (TypeError, ValueError):
        return default
