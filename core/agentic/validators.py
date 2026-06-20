"""Deterministic quality gates for v1 agentic stage outputs."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any, Dict, List

from core.agentic.action_space import ActionType
from core.agentic.models import QualityReport


def _path_exists(value: Any) -> bool:
    return bool(value) and Path(str(value)).exists()


def _collection_is_placeholder_only(data_path: Any) -> bool:
    if not _path_exists(data_path):
        return False
    try:
        from datasets import load_from_disk

        dataset = load_from_disk(str(data_path))
        if len(dataset) != 1:
            return False
        row = dataset[0]
    except Exception:
        return False
    if not isinstance(row, dict):
        return False
    placeholder_values = {"(no text collected)", "empty"}
    text = str(row.get("text") or row.get("source_text") or "").strip()
    source_id = str(row.get("source_id") or "").strip()
    group_key = str(row.get("group_key") or "").strip()
    return text in placeholder_values or (source_id == "empty" and group_key == "empty")


def _flatten_queries(output: Dict[str, Any]) -> List[str]:
    nested = output.get("category_subcategory_queries") or {}
    queries: List[str] = []
    for subcategories in nested.values():
        if not isinstance(subcategories, dict):
            continue
        for query_list in subcategories.values():
            if isinstance(query_list, Iterable) and not isinstance(query_list, (str, bytes)):
                queries.extend(str(query).strip() for query in query_list if str(query).strip())
    return queries


def validate_taxonomy_output(output: Dict[str, Any]) -> QualityReport:
    categories = output.get("categories") or []
    queries = _flatten_queries(output)
    taxonomy_quality = output.get("taxonomy_quality") or {}
    duplicate_count = len(queries) - len(set(queries))
    duplicate_rate = duplicate_count / len(queries) if queries else 0.0

    blocking_issues: List[str] = []
    if not categories:
        blocking_issues.append("categories_missing")
    if not queries:
        blocking_issues.append("queries_missing")
    if duplicate_rate > 0.5:
        blocking_issues.append("query_duplicate_rate_too_high")
    if taxonomy_quality and not taxonomy_quality.get("passed", False):
        blocking_issues.append("taxonomy_quality_failed")
    warnings = list(taxonomy_quality.get("warnings") or [])
    issue_categories = sorted(
        set(taxonomy_quality.get("issue_categories") or [])
        | set(_issue_categories(blocking_issues, warnings))
    )
    gate_status = str(taxonomy_quality.get("gate_status") or _gate_status(blocking_issues, warnings))

    return QualityReport(
        stage=ActionType.GENERATE_TAXONOMY,
        passed=not blocking_issues,
        gate_status=gate_status,
        decision=_decision_from_gate_status(gate_status),
        score=float(taxonomy_quality.get("score", 1.0 if not blocking_issues else 0.0)),
        recoverable=True,
        issue_categories=issue_categories,
        blocking_issues=blocking_issues,
        warnings=warnings,
        metrics={
            "num_categories": len(categories),
            "num_queries": len(queries),
            "query_duplicate_rate": duplicate_rate,
            "taxonomy_quality_gate_status": taxonomy_quality.get("gate_status"),
            "taxonomy_quality_score": taxonomy_quality.get("score"),
        },
    )


def validate_collection_output(
    output: Dict[str, Any],
    *,
    collect_images: bool = False,
    min_samples: int = 1,
    min_images: int = 1,
) -> QualityReport:
    metadata = output.get("metadata") or {}
    num_samples = int(output.get("num_samples") or 0)
    num_images = int(metadata.get("num_images") or 0)
    text_filter = metadata.get("text_filter_summary") if isinstance(metadata.get("text_filter_summary"), dict) else {}
    text_filter_input = _safe_int(text_filter.get("num_input"))
    text_filter_kept = _safe_int(text_filter.get("num_kept"))
    text_filter_removed = _safe_int(text_filter.get("num_removed"))
    text_filter_removal_rate = _safe_float(text_filter.get("removal_rate"))

    blocking_issues: List[str] = []
    warnings: List[str] = []
    recommended_actions: List[str] = []
    suggested_adjustments: Dict[str, Any] = {}
    if num_samples < min_samples:
        blocking_issues.append("num_samples_below_minimum")
        recommended_actions.append("increase_collection_coverage")
        suggested_adjustments.update({"serper_results_per_query": "increase", "serper_top_results": "increase"})
    if _collection_is_placeholder_only(output.get("data_path")):
        blocking_issues.append("placeholder_collection_row")
        recommended_actions.append("increase_collection_coverage")
        suggested_adjustments.update({"serper_results_per_query": "increase", "serper_top_results": "increase"})
    if text_filter.get("enabled") and text_filter_input > 0 and text_filter_kept == 0:
        blocking_issues.append("text_filter_removed_all_samples")
        recommended_actions.append("expand_collection_or_relax_text_filter")
        suggested_adjustments.update(
            {
                "serper_results_per_query": "increase",
                "serper_top_results": "increase",
                "text_filter_min_chars": "decrease",
                "text_filter_min_words": "decrease",
            }
        )
    elif text_filter.get("enabled") and text_filter_removal_rate > 0.6:
        warnings.append("text_filter_removed_many_samples")
    if not _path_exists(output.get("data_path")):
        blocking_issues.append("data_path_missing")
    if collect_images:
        if num_images < min_images:
            blocking_issues.append("num_images_below_minimum")
            recommended_actions.append("increase_image_search_coverage")
            suggested_adjustments.update(
                {
                    "image_search_results_per_query": "increase",
                    "image_taxonomy_queries_per_slot": "increase",
                    "image_min_width": "decrease",
                    "image_min_height": "decrease",
                }
            )
        if not _path_exists(metadata.get("images_dir")):
            blocking_issues.append("images_dir_missing")
        if not _path_exists(metadata.get("images_index")):
            blocking_issues.append("images_index_missing")
        slot_count = _image_slot_count(metadata.get("images_index"))
        expected_slots = int(metadata.get("num_image_taxonomy_slots") or 0)
        if expected_slots > 0 and slot_count <= 0:
            blocking_issues.append("image_slot_coverage_missing")
            recommended_actions.append("repair_image_taxonomy_queries")
    else:
        slot_count = 0

    return QualityReport(
        stage=ActionType.COLLECT_DATA,
        passed=not blocking_issues,
        gate_status=_gate_status(blocking_issues, warnings),
        decision=_decision_from_gate_status(_gate_status(blocking_issues, warnings)),
        score=1.0 if not blocking_issues else 0.0,
        recoverable=True,
        issue_categories=_issue_categories(blocking_issues, warnings),
        blocking_issues=blocking_issues,
        warnings=warnings,
        metrics={
            "num_samples": num_samples,
            "num_images": num_images,
            "image_slot_count": slot_count,
            "num_image_taxonomy_slots": int(metadata.get("num_image_taxonomy_slots") or 0),
            "text_filter_input": text_filter_input,
            "text_filter_kept": text_filter_kept,
            "text_filter_removed": text_filter_removed,
            "text_filter_removal_rate": text_filter_removal_rate,
        },
        recommended_actions=recommended_actions,
        suggested_adjustments=suggested_adjustments,
    )


def validate_sft_output(output: Dict[str, Any]) -> QualityReport:
    num_examples = int(output.get("num_examples") or 0)
    num_failures = int(output.get("num_failures") or 0)
    num_items = int(output.get("num_items") or 0)
    failure_rate = num_failures / num_items if num_items else 0.0

    blocking_issues: List[str] = []
    recommended_actions: List[str] = []
    suggested_adjustments: Dict[str, Any] = {}
    if num_examples <= 0:
        blocking_issues.append("num_examples_below_minimum")
        recommended_actions.append("retry_sft_with_stricter_prompt")
        suggested_adjustments["sft_prompt_preset"] = "schema_strict"
    if not _path_exists(output.get("sft_path")):
        blocking_issues.append("sft_path_missing")
    if failure_rate >= 1.0 and num_items:
        blocking_issues.append("all_annotations_failed")
        recommended_actions.append("reduce_sft_batch_size")
        suggested_adjustments["llm_batch_size"] = "decrease"

    return QualityReport(
        stage=ActionType.BUILD_SFT_DATASET,
        passed=not blocking_issues,
        gate_status=_gate_status(blocking_issues, []),
        decision=_decision_from_gate_status(_gate_status(blocking_issues, [])),
        score=1.0 if not blocking_issues else 0.0,
        recoverable=True,
        issue_categories=_issue_categories(blocking_issues, []),
        blocking_issues=blocking_issues,
        metrics={
            "num_items": num_items,
            "num_examples": num_examples,
            "num_failures": num_failures,
            "failure_rate": failure_rate,
        },
        recommended_actions=recommended_actions,
        suggested_adjustments=suggested_adjustments,
    )


def validate_dataset_output(output: Dict[str, Any]) -> QualityReport:
    summary = output.get("dataset_summary") or {}
    dataset_ref = output.get("dataset_ref") or {}
    sample_count = int(summary.get("sample_count") or 0)
    columns = summary.get("columns") or []
    warnings = summary.get("validation_warnings") or []
    split_counts = dataset_ref.get("split_counts") or summary.get("split_counts") or {}

    blocking_issues: List[str] = []
    if not dataset_ref:
        blocking_issues.append("dataset_ref_missing")
    if sample_count <= 0:
        blocking_issues.append("sample_count_below_minimum")
    if not columns:
        blocking_issues.append("columns_missing")
    if not _path_exists(output.get("dataset_manifest_path")):
        blocking_issues.append("dataset_manifest_missing")
    if split_counts and isinstance(split_counts, dict) and sample_count > 1 and int(split_counts.get("validation") or 0) <= 0:
        blocking_issues.append("validation_split_empty")

    return QualityReport(
        stage=ActionType.BUILD_DATASET,
        passed=not blocking_issues,
        gate_status=_gate_status(blocking_issues, warnings),
        decision=_decision_from_gate_status(_gate_status(blocking_issues, warnings)),
        score=1.0 if not blocking_issues else 0.0,
        recoverable=True,
        issue_categories=_issue_categories(blocking_issues, [str(warning) for warning in warnings]),
        blocking_issues=blocking_issues,
        warnings=[str(warning) for warning in warnings],
        metrics={"sample_count": sample_count, "num_columns": len(columns), "split_counts": split_counts},
    )


def validate_training_output(output: Dict[str, Any]) -> QualityReport:
    blocking_issues: List[str] = []
    recommended_actions: List[str] = []
    suggested_adjustments: Dict[str, Any] = {}
    if not _path_exists(output.get("adapter_path")):
        blocking_issues.append("adapter_path_missing")
    if not output.get("metrics"):
        blocking_issues.append("metrics_missing")
    if not output.get("iteration_record"):
        blocking_issues.append("iteration_record_missing")
    if blocking_issues:
        recommended_actions.append("stabilize_training_config")
        suggested_adjustments.update({"train_lr": "decrease", "train_grad_accum": "increase"})

    return QualityReport(
        stage=ActionType.TRAIN_MODEL,
        passed=not blocking_issues,
        gate_status=_gate_status(blocking_issues, []),
        decision=_decision_from_gate_status(_gate_status(blocking_issues, [])),
        score=1.0 if not blocking_issues else 0.0,
        recoverable=True,
        issue_categories=_issue_categories(blocking_issues, []),
        blocking_issues=blocking_issues,
        metrics=output.get("metrics") or {},
        recommended_actions=recommended_actions,
        suggested_adjustments=suggested_adjustments,
    )


def validate_eval_output(output: Dict[str, Any]) -> QualityReport:
    blocking_issues: List[str] = []
    warnings: List[str] = []
    recommended_actions: List[str] = []
    suggested_adjustments: Dict[str, Any] = {}
    if not _path_exists(output.get("predictions_path")):
        blocking_issues.append("predictions_path_missing")
    if not _path_exists(output.get("failures_path")):
        blocking_issues.append("failures_path_missing")
    if output.get("cluster_preview") is None:
        blocking_issues.append("cluster_preview_missing")
    metrics_payload = output.get("metrics") or {}
    training_health = metrics_payload.get("training_health") if isinstance(metrics_payload, dict) else {}
    judge = metrics_payload.get("judge") if isinstance(metrics_payload, dict) else {}
    judge_enabled = isinstance(judge, dict) and bool(judge.get("enabled"))
    failure_rate = _failure_rate(output)
    if not judge_enabled and failure_rate is not None and failure_rate > 0.5:
        blocking_issues.append("eval_failure_rate_too_high")
        recommended_actions.append("route_to_upstream_recovery")
    labels = [] if judge_enabled else _cluster_labels(output.get("cluster_preview"))
    if isinstance(training_health, dict) and training_health.get("gate_status") == "repair":
        blocking_issues.append("eval_training_failure")
        suggested_adjustments.update({"train_lr": "decrease", "train_grad_accum": "increase"})
    judge_labels = _judge_labels(judge)
    labels.extend(judge_labels)
    if judge_enabled and judge.get("gate_status") == "repair":
        blocking_issues.append("eval_judge_quality_failure")
    elif judge_enabled and judge.get("gate_status") == "warn":
        warnings.append("eval_judge_quality_warn")
    unsupported_grounding_rate = _safe_float(judge.get("unsupported_grounding_rate")) if isinstance(judge, dict) else 0.0
    if judge_enabled and unsupported_grounding_rate > 0.2:
        blocking_issues.append("eval_grounding_failure")
        labels.append("grounding_unsupported")
        suggested_adjustments.update({"serper_results_per_query": "increase", "serper_top_results": "increase"})
    elif judge_enabled and unsupported_grounding_rate > 0.1:
        warnings.append("eval_grounding_warn")
    if any(_has_any(label, ("knowledge", "missing", "coverage", "hallucination", "grounding")) for label in labels):
        blocking_issues.append("eval_knowledge_missing")
        suggested_adjustments.update({"serper_results_per_query": "increase", "serper_top_results": "increase"})
    elif any(_has_any(label, ("format", "schema", "json")) for label in labels):
        blocking_issues.append("eval_formatting_failure")
        suggested_adjustments["sft_prompt_preset"] = "schema_strict"
    elif labels:
        warnings.append("eval_failure_clusters_present")

    return QualityReport(
        stage=ActionType.EVALUATE_MODEL,
        passed=not blocking_issues,
        gate_status=_gate_status(blocking_issues, warnings),
        decision=_decision_from_gate_status(_gate_status(blocking_issues, warnings)),
        score=1.0 if not blocking_issues else 0.0,
        recoverable=True,
        issue_categories=_issue_categories(blocking_issues, warnings),
        blocking_issues=blocking_issues,
        warnings=warnings,
        metrics={
            "failure_rate": failure_rate,
            "failure_clusters": labels,
            "training_health_gate": training_health.get("gate_status") if isinstance(training_health, dict) else None,
            "judge_gate": judge.get("gate_status") if isinstance(judge, dict) else None,
            "judge_major_failure_rate": judge.get("major_failure_rate") if isinstance(judge, dict) else None,
            "judge_unsupported_grounding_rate": unsupported_grounding_rate,
        },
        recommended_actions=recommended_actions,
        suggested_adjustments=suggested_adjustments,
    )


def validate_report_output(output: Dict[str, Any]) -> QualityReport:
    blocking_issues: List[str] = []
    if not _path_exists(output.get("report_path")):
        blocking_issues.append("report_path_missing")

    return QualityReport(
        stage=ActionType.GENERATE_REPORT,
        passed=not blocking_issues,
        gate_status=_gate_status(blocking_issues, [], recoverable=False),
        decision=_decision_from_gate_status(_gate_status(blocking_issues, [], recoverable=False)),
        score=1.0 if not blocking_issues else 0.0,
        recoverable=False,
        issue_categories=_issue_categories(blocking_issues, []),
        blocking_issues=blocking_issues,
        metrics={},
    )


def _image_slot_count(images_index: Any) -> int:
    if not _path_exists(images_index):
        return 0
    try:
        payload = json.loads(Path(str(images_index)).read_text(encoding="utf-8"))
    except Exception:
        return 0
    records = payload.get("images", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return 0
    return len({str(record.get("slot_id") or "") for record in records if isinstance(record, dict) and record.get("slot_id")})


def _failure_rate(output: Dict[str, Any]) -> float | None:
    for source in (output, output.get("metrics") or {}):
        if isinstance(source, dict) and source.get("failure_rate") is not None:
            try:
                return float(source["failure_rate"])
            except (TypeError, ValueError):
                return None
    return None


def _cluster_labels(cluster_preview: Any) -> List[str]:
    if not isinstance(cluster_preview, dict):
        return []
    clusters = cluster_preview.get("clusters") or []
    labels: List[str] = []
    for cluster in clusters:
        if isinstance(cluster, dict) and cluster.get("label"):
            labels.append(str(cluster["label"]).lower())
    return labels


def _judge_labels(judge: Any) -> List[str]:
    if not isinstance(judge, dict):
        return []
    counts = judge.get("failure_category_counts") or {}
    if not isinstance(counts, dict):
        return []
    return [str(label).lower() for label, count in counts.items() if count]


def _has_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gate_status(blocking_issues: List[str], warnings: List[str], *, recoverable: bool = True) -> str:
    if blocking_issues:
        return "repair" if recoverable else "fail"
    if warnings:
        return "warn"
    return "pass"


def _decision_from_gate_status(gate_status: str) -> str:
    normalized = (gate_status or "").strip().lower()
    if normalized in {"pass", "warn"}:
        return "continue"
    if normalized == "repair":
        return "repair"
    return "stop"


def _issue_categories(blocking_issues: List[str], warnings: List[str]) -> List[str]:
    categories: set[str] = set()
    for issue in [*blocking_issues, *warnings]:
        if "alias" in issue or "language" in issue:
            categories.add("culture_specificity")
        elif "duplicate" in issue:
            categories.add("duplicates")
        elif "schema" in issue or "format" in issue or "empty" in issue or "columns" in issue:
            categories.add("schema")
        elif "missing" in issue or "below_minimum" in issue or "count" in issue or "coverage" in issue:
            categories.add("missing_coverage")
        elif "training" in issue or "metrics" in issue or "adapter" in issue:
            categories.add("training_health")
        elif "failure" in issue or "cluster" in issue:
            categories.add("evaluation_quality")
        else:
            categories.add("other")
    return sorted(categories)
