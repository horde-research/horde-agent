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


def build_recovery_fingerprint(result: ActionResult, plan: RecoveryPlan) -> Dict[str, Any]:
    report = result.quality_report
    return _normalize_value(
        {
            "failed_stage": result.action_type.value,
            "target_stage": plan.target_stage.value,
            "reason": plan.reason,
            "config_delta": plan.config_delta,
            "blocking_issues": sorted(report.blocking_issues if report else []),
            "issue_categories": sorted(report.issue_categories if report else []),
            "key_metrics": _recovery_key_metrics(result),
        }
    )


def recovery_fingerprint_seen(state: PipelineState, fingerprint: Dict[str, Any]) -> bool:
    normalized = _normalize_value(fingerprint)
    return any(_normalize_value(existing) == normalized for existing in state.recovery_fingerprints)


def build_recovery_plan(state: PipelineState, result: ActionResult) -> RecoveryPlan:
    report = result.quality_report
    issues = set(report.blocking_issues if report else [])

    if result.action_type == ActionType.GENERATE_TAXONOMY:
        return _taxonomy_plan(state, issues)
    if result.action_type == ActionType.COLLECT_DATA:
        return _collection_plan(state, issues, result)
    if result.action_type == ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES:
        return _coverage_plan(state, issues, result)
    if result.action_type == ActionType.ASSESS_SOURCE_QUALITY:
        return _source_quality_plan(state, issues, result)
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


def _collection_plan(state: PipelineState, issues: set[str], result: ActionResult | None = None) -> RecoveryPlan:
    delta: Dict[str, Any] = {}
    text_filter = _text_filter_summary(result)
    if "text_filter_removed_all_samples" in issues or _text_filter_removed_all(text_filter):
        delta.update(_expand_text_collection_delta(state))
        delta.update(_relax_text_filter_delta(state, text_filter))
        return RecoveryPlan(
            target_stage=ActionType.COLLECT_DATA,
            reason="recovery_expand_collection_after_text_filter",
            config_delta=delta,
        )

    if issues & {"num_samples_below_minimum", "data_path_missing"}:
        delta.update(_expand_text_collection_delta(state))
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
        return _collection_plan(state, issues, result)
    return RecoveryPlan(
        target_stage=ActionType.COLLECT_DATA,
        reason="recovery_refine_collection_queries",
        config_delta=delta,
    )


def _source_quality_plan(state: PipelineState, issues: set[str], result: ActionResult) -> RecoveryPlan:
    raw = result.raw_output if isinstance(result.raw_output, dict) else {}
    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
    query_refinements = _merge_list_cfg(
        state.config.get("coverage_added_queries"),
        raw.get("query_refinements") or result.artifacts.get("source_quality_query_refinements"),
    )
    delta = _expand_text_collection_delta(state)
    if query_refinements:
        delta["coverage_added_queries"] = query_refinements
    if issues & {
        "source_quality_removed_all_rows",
        "source_quality_average_score_too_low",
    }:
        delta["source_quality_min_quality_score"] = max(
            0.05,
            _float_cfg(state, "source_quality_min_quality_score", 0.20) - 0.05,
        )
        delta["source_quality_keep_borderline"] = True
    if issues & {"source_quality_domain_concentration_too_high", "source_quality_source_groups_below_minimum"}:
        delta["serper_results_per_query"] = min(_int_cfg(state, "serper_results_per_query", 10) + 8, 40)
        delta["serper_top_results"] = min(_int_cfg(state, "serper_top_results", 5) + 3, 15)
    if _float_value(summary.get("removal_rate"), 0.0) > 0.80:
        delta["source_quality_keep_borderline"] = True
    return RecoveryPlan(
        target_stage=ActionType.COLLECT_DATA,
        reason="recovery_source_quality_collect_more",
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
    judge = _judge_summary(result)
    if (
        "eval_grounding_failure" in issues
        or _float_value(judge.get("unsupported_grounding_rate"), 0.0) > 0.2
        or _contains_any(labels, ("grounding", "unsupported", "hallucination", "wrong_fact"))
    ):
        delta = _expand_text_collection_delta(state)
        delta["coverage_min_text_samples"] = min(_int_cfg(state, "coverage_min_text_samples", 3) + 2, 25)
        return RecoveryPlan(
            target_stage=ActionType.COLLECT_DATA,
            reason="recovery_eval_requests_grounded_sources",
            config_delta=delta,
        )
    if "eval_failure_rate_too_high" in issues and not bool(state.config.get("eval_enable_llm_judge")):
        return RecoveryPlan(
            target_stage=ActionType.EVALUATE_MODEL,
            reason="recovery_enable_llm_judge_for_eval",
            config_delta={"eval_enable_llm_judge": True},
        )
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


def _expand_text_collection_delta(state: PipelineState) -> Dict[str, Any]:
    return {
        "serper_results_per_query": min(_int_cfg(state, "serper_results_per_query", 10) + 5, 30),
        "serper_top_results": min(_int_cfg(state, "serper_top_results", 5) + 2, 12),
    }


def _relax_text_filter_delta(state: PipelineState, summary: dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {
        "text_filter_min_chars": max(120, int(_int_cfg(state, "text_filter_min_chars", 300) * 0.8)),
        "text_filter_min_words": max(20, int(_int_cfg(state, "text_filter_min_words", 40) * 0.8)),
    }
    reason_counts = summary.get("removed_reason_counts") if isinstance(summary.get("removed_reason_counts"), dict) else {}
    near_removed = int(reason_counts.get("near_duplicate_text") or 0)
    exact_removed = int(reason_counts.get("exact_duplicate_text") or 0)
    if near_removed > exact_removed:
        delta["text_filter_shingle_threshold"] = min(
            _float_cfg(state, "text_filter_shingle_threshold", 0.90) + 0.03,
            0.98,
        )
    return delta


def _text_filter_summary(result: ActionResult | None) -> dict[str, Any]:
    if not result:
        return {}
    summary = result.artifacts.get("text_filter_summary") if isinstance(result.artifacts, dict) else None
    if isinstance(summary, dict):
        return summary
    metadata = result.artifacts.get("collection_metadata") if isinstance(result.artifacts, dict) else None
    if isinstance(metadata, dict) and isinstance(metadata.get("text_filter_summary"), dict):
        return metadata["text_filter_summary"]
    raw = result.raw_output if isinstance(result.raw_output, dict) else {}
    raw_metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    return raw_metadata.get("text_filter_summary") if isinstance(raw_metadata.get("text_filter_summary"), dict) else {}


def _text_filter_removed_all(summary: dict[str, Any]) -> bool:
    return bool(summary.get("enabled")) and _int_value(summary.get("num_input"), 0) > 0 and _int_value(summary.get("num_kept"), 0) == 0


def _judge_summary(result: ActionResult) -> dict[str, Any]:
    for source in (result.artifacts, result.metrics, result.raw_output if isinstance(result.raw_output, dict) else {}):
        if not isinstance(source, dict):
            continue
        direct = source.get("judge_summary") or source.get("judge")
        if isinstance(direct, dict):
            return direct
        metrics = source.get("metrics")
        if isinstance(metrics, dict):
            nested = metrics.get("judge") or metrics.get("judge_summary")
            if isinstance(nested, dict):
                return nested
    return {}


def _recovery_key_metrics(result: ActionResult) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for source in (result.metrics, result.artifacts, result.raw_output if isinstance(result.raw_output, dict) else {}):
        if not isinstance(source, dict):
            continue
        _copy_metric(metrics, source, "failure_rate")
        _copy_metric(metrics, source, "judge_unsupported_grounding_rate")
        _copy_metric(metrics, source, "judge_major_failure_rate")
        _copy_metric(metrics, source, "training_health_gate")
        _copy_metric(metrics, source, "num_samples")
        _copy_metric(metrics, source, "text_filter_kept")
        _copy_metric(metrics, source, "text_filter_removed")
        _copy_metric(metrics, source, "text_filter_removal_rate")
        _copy_metric(metrics, source, "num_kept_rows", target_key="source_quality_kept")
        _copy_metric(metrics, source, "num_removed_rows", target_key="source_quality_removed")
        _copy_metric(metrics, source, "top_domain_share", target_key="source_quality_top_domain_share")
        _copy_metric(metrics, source, "avg_kept_quality_score", target_key="source_quality_avg_score")
        judge = source.get("judge_summary") or source.get("judge")
        if isinstance(judge, dict):
            _copy_metric(metrics, judge, "quality_score", target_key="judge_quality_score")
            _copy_metric(metrics, judge, "major_failure_rate", target_key="judge_major_failure_rate")
            _copy_metric(metrics, judge, "unsupported_grounding_rate", target_key="judge_unsupported_grounding_rate")
        text_filter = source.get("text_filter_summary")
        if isinstance(text_filter, dict):
            _copy_metric(metrics, text_filter, "num_kept", target_key="text_filter_kept")
            _copy_metric(metrics, text_filter, "num_removed", target_key="text_filter_removed")
            _copy_metric(metrics, text_filter, "removal_rate", target_key="text_filter_removal_rate")
        source_quality = source.get("source_quality_summary") or source.get("summary")
        if isinstance(source_quality, dict):
            _copy_metric(metrics, source_quality, "num_kept_rows", target_key="source_quality_kept")
            _copy_metric(metrics, source_quality, "num_removed_rows", target_key="source_quality_removed")
            _copy_metric(metrics, source_quality, "top_domain_share", target_key="source_quality_top_domain_share")
            _copy_metric(metrics, source_quality, "avg_kept_quality_score", target_key="source_quality_avg_score")
    return _normalize_value(metrics)


def _copy_metric(metrics: Dict[str, Any], source: Dict[str, Any], key: str, *, target_key: str | None = None) -> None:
    if source.get(key) is not None:
        metrics[target_key or key] = source[key]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, ActionType):
        return value.value
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


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


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
