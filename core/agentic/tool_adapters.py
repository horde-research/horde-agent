"""Adapters that normalize existing pipeline tools into agentic ActionResult objects."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Callable, Dict

from core.agentic.action_space import ActionType, FULL_GRAPH_ACTIONS
from core.agentic.coverage import assess_coverage_and_refine_queries
from core.agentic.models import ActionRequest, ActionResult, PipelineState, QualityReport
from core.data.source_quality import assess_text_source_quality
from core.data.text_quality import (
    DEFAULT_EMBEDDING_MODEL,
    records_from_serper_raw,
    records_from_sft_jsonl,
    summarize_text_quality,
    write_text_quality_report,
)
from core.data.image_sft_tasks import normalize_image_sft_tasks
from core.redaction import sanitize_secret_text
from core.agentic.validators import (
    validate_collection_output,
    validate_dataset_output,
    validate_eval_output,
    validate_report_output,
    validate_source_quality_output,
    validate_sft_output,
    validate_taxonomy_output,
    validate_training_output,
)

logger = logging.getLogger(__name__)

StageExecutor = Callable[[PipelineState, ActionRequest], ActionResult]


class AgenticToolAdapter:
    """Wrap repo tools behind stable agentic stage contracts."""

    def __init__(self, tools: Mapping[str, Any]) -> None:
        self.tools = dict(tools)

    def executors(self) -> Dict[ActionType, StageExecutor]:
        return {
            ActionType.GENERATE_TAXONOMY: self.execute_generate_taxonomy,
            ActionType.COLLECT_DATA: self.execute_collect_data,
            ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES: self.execute_assess_coverage_and_refine_queries,
            ActionType.ASSESS_SOURCE_QUALITY: self.execute_assess_source_quality,
            ActionType.BUILD_SFT_DATASET: self.execute_build_sft_dataset,
            ActionType.BUILD_DATASET: self.execute_build_dataset,
            ActionType.TRAIN_MODEL: self.execute_train_model,
            ActionType.EVALUATE_MODEL: self.execute_evaluate_model,
            ActionType.GENERATE_REPORT: self.execute_generate_report,
        }

    def missing_tools(self) -> list[ActionType]:
        required = {
            ActionType.GENERATE_TAXONOMY: "generate_taxonomy",
            ActionType.COLLECT_DATA: "collect_data",
            ActionType.BUILD_SFT_DATASET: "build_sft_dataset",
            ActionType.BUILD_DATASET: "build_dataset",
            ActionType.TRAIN_MODEL: "train",
            ActionType.EVALUATE_MODEL: "eval_model",
            ActionType.GENERATE_REPORT: "reporting",
        }
        return [stage for stage, key in required.items() if key not in self.tools]

    def execute_generate_taxonomy(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            country = str(cfg.get("country") or "").strip()
            if not country:
                raise ValueError("state.config['country'] is required.")
            output = self.tools["generate_taxonomy"].execute(
                country,
                {
                    "batch_size": cfg.get("llm_batch_size", 5),
                    "batch_delay": cfg.get("llm_batch_delay", 1.5),
                    "provider": cfg.get("llm_provider"),
                    "model": cfg.get("llm_model"),
                    "api_key": cfg.get("llm_api_key"),
                    "temperature": cfg.get("llm_temperature", 0.2),
                    "focus": cfg.get("focus", ""),
                    "enable_image_taxonomy": cfg.get("enable_image_taxonomy", True),
                    "image_taxonomy_queries_per_slot": cfg.get("image_taxonomy_queries_per_slot", 4),
                    "image_taxonomy_max_slots": cfg.get("image_taxonomy_max_slots"),
                },
            )
            selected_queries = _select_queries(
                output,
                max_queries_per_category=cfg.get("max_queries_per_category"),
                max_queries=cfg.get("max_queries"),
            )
            report = validate_taxonomy_output(output)
            return ActionResult(
                action_type=ActionType.GENERATE_TAXONOMY,
                status=_status_from_report(report),
                artifacts={
                    "taxonomy": output,
                    "search_queries": selected_queries,
                    "image_taxonomy": output.get("image_taxonomy"),
                },
                metrics=report.metrics,
                quality_report=report,
                raw_output=output,
            )
        except Exception as exc:
            return _failed_result(ActionType.GENERATE_TAXONOMY, exc)

    def execute_collect_data(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            queries = state.artifacts.get("search_queries") or cfg.get("queries")
            if not queries:
                raise ValueError("search_queries artifact is required before collection.")
            query_list = _merge_queries(list(queries), cfg.get("coverage_added_queries"))
            collect_images = bool(cfg.get("collect_images", False))
            output = self.tools["collect_data"].execute(
                {
                    "queries": query_list,
                    "run_dir": str(Path(state.run_dir) / "collect"),
                    "google_results_per_query": cfg.get("serper_results_per_query", 10),
                    "serper_api_key": cfg.get("serper_api_key"),
                    "top_results": cfg.get("serper_top_results", 5),
                    "concurrency": cfg.get("serper_concurrency", 50),
                    "collect_images": collect_images,
                    "image_min_width": cfg.get("image_min_width", 300),
                    "image_min_height": cfg.get("image_min_height", 300),
                    "image_context_size": cfg.get("image_context_size", 500),
                    "image_collection_mode": cfg.get("image_collection_mode", "serper"),
                    "image_search_results_per_query": cfg.get(
                        "image_search_results_per_query",
                        cfg.get("serper_results_per_query", 10),
                    ),
                    "image_taxonomy": state.artifacts.get("image_taxonomy") or cfg.get("image_taxonomy"),
                    "image_query_specs": cfg.get("image_query_specs"),
                    "image_search_queries": cfg.get("image_search_queries"),
                    "image_dedup_enable": cfg.get("image_dedup_enable", False),
                    "image_dedup_threshold": cfg.get("image_dedup_threshold", 0.90),
                    "image_dedup_model_path": cfg.get("image_dedup_model_path"),
                    "image_dedup_model_url": cfg.get("image_dedup_model_url"),
                    "image_dedup_batch_size": cfg.get("image_dedup_batch_size", 32),
                    "image_dedup_max_reported_pairs": cfg.get("image_dedup_max_reported_pairs", 100),
                    "image_dedup_device": cfg.get("image_dedup_device"),
                    "text_filter_enable": cfg.get("text_filter_enable", True),
                    "text_filter_min_chars": cfg.get("text_filter_min_chars", 300),
                    "text_filter_min_words": cfg.get("text_filter_min_words", 40),
                    "text_filter_min_unique_word_ratio": cfg.get("text_filter_min_unique_word_ratio", 0.15),
                    "text_filter_shingle_threshold": cfg.get("text_filter_shingle_threshold", 0.90),
                    "text_filter_max_near_duplicate_items": cfg.get(
                        "text_filter_max_near_duplicate_items",
                        1000,
                    ),
                    "text_filter_max_reported_rows": cfg.get("text_filter_max_reported_rows", 50),
                }
            )
            report = validate_collection_output(output, collect_images=collect_images)
            metadata = output.get("metadata") or {}
            text_quality_artifacts = _write_collection_text_quality_artifacts(state, metadata)
            artifacts = {
                "raw_data_path": output.get("data_path"),
                "data_path": output.get("data_path"),
                "num_samples": output.get("num_samples"),
                "collection_metadata": metadata,
                **text_quality_artifacts,
            }
            for key in (
                "text_filter_report_path",
                "text_filter_summary",
                "text_filter_enabled",
                "num_text_rows_before_filter",
                "num_text_rows_after_filter",
                "num_text_rows_removed_by_filter",
                "images_dir",
                "images_index",
                "num_images",
                "raw_images_index",
                "image_dedup_report_path",
                "image_dedup_enabled",
                "image_dedup_method",
                "image_dedup_threshold",
                "image_dedup_model_path",
                "image_dedup_model_url",
                "image_dedup_device",
                "image_dedup_downloaded_model",
                "num_images_before_dedup",
                "num_images_removed_by_dedup",
                "num_image_dedup_clusters",
            ):
                if key in metadata:
                    artifacts[key] = metadata[key]
            return ActionResult(
                action_type=ActionType.COLLECT_DATA,
                status=_status_from_report(report),
                artifacts=artifacts,
                metrics=report.metrics,
                quality_report=report,
                raw_output=output,
            )
        except Exception as exc:
            return _failed_result(ActionType.COLLECT_DATA, exc)

    def execute_assess_coverage_and_refine_queries(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            output = assess_coverage_and_refine_queries(state)
            report = output["report"]
            return ActionResult(
                action_type=ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES,
                status=_status_from_report(report),
                artifacts={
                    "coverage_review": output["coverage_review"],
                    "coverage_added_queries": output["coverage_added_queries"],
                    "image_query_specs": output["image_query_specs"],
                },
                metrics=report.metrics,
                quality_report=report,
                raw_output=output["coverage_review"],
            )
        except Exception as exc:
            return _failed_result(ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES, exc)

    def execute_assess_source_quality(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            if _training_modality(cfg) != "text":
                output = {
                    "skipped": True,
                    "enabled": False,
                    "reason": "source_quality_text_only",
                    "data_path": state.artifacts.get("data_path") or cfg.get("data_path"),
                }
                report = validate_source_quality_output(output)
                return ActionResult(
                    action_type=ActionType.ASSESS_SOURCE_QUALITY,
                    status=_status_from_report(report),
                    artifacts={
                        "source_quality_enabled": False,
                        "data_path": output.get("data_path"),
                        "source_quality_summary": {"skipped": True, "reason": output["reason"]},
                    },
                    metrics=report.metrics,
                    quality_report=report,
                    raw_output=output,
                )
            if not _as_bool(cfg.get("source_quality_enable", True)):
                output = {
                    "skipped": True,
                    "enabled": False,
                    "reason": "source_quality_disabled",
                    "data_path": state.artifacts.get("data_path") or cfg.get("data_path"),
                }
                report = validate_source_quality_output(output)
                return ActionResult(
                    action_type=ActionType.ASSESS_SOURCE_QUALITY,
                    status=_status_from_report(report),
                    artifacts={
                        "source_quality_enabled": False,
                        "data_path": output.get("data_path"),
                        "source_quality_summary": {"skipped": True, "reason": output["reason"]},
                    },
                    metrics=report.metrics,
                    quality_report=report,
                    raw_output=output,
                )

            data_path = state.artifacts.get("raw_data_path") or state.artifacts.get("data_path") or cfg.get("data_path")
            if not data_path:
                raise ValueError("data_path artifact is required for source quality assessment.")
            output = assess_text_source_quality(
                data_path=str(data_path),
                output_dir=Path(state.run_dir) / "source_quality",
                taxonomy=state.artifacts.get("taxonomy") or cfg.get("taxonomy") or {},
                queries=state.artifacts.get("search_queries") or cfg.get("queries") or [],
                config=cfg,
            )
            report = validate_source_quality_output(
                output,
                min_kept_rows=_int_value(cfg.get("source_quality_min_kept_rows"), 20),
                min_source_groups=_int_value(cfg.get("source_quality_min_source_groups"), 5),
                max_domain_share=_float_value(cfg.get("source_quality_max_domain_share"), 0.75),
                min_avg_quality_score=_float_value(cfg.get("source_quality_min_avg_score"), 0.20),
            )
            artifacts = {
                "source_quality_enabled": True,
                "source_quality_input_data_path": output.get("input_data_path"),
                "source_quality_filtered_data_path": output.get("filtered_data_path"),
                "source_quality_profile_path": output.get("profile_path"),
                "source_quality_clusters_path": output.get("clusters_path"),
                "source_quality_oracle_payload_path": output.get("oracle_payload_path"),
                "source_quality_policy_path": output.get("policy_path"),
                "source_quality_report_path": output.get("report_path"),
                "source_quality_decisions_path": output.get("decisions_path"),
                "source_quality_accepted_sources_path": output.get("accepted_sources_path"),
                "source_quality_summary": output.get("summary"),
                "source_quality_oracle": output.get("oracle"),
                "source_quality_query_refinements": output.get("query_refinements"),
            }
            if report.passed and output.get("filtered_data_path"):
                artifacts["data_path"] = output.get("filtered_data_path")
            return ActionResult(
                action_type=ActionType.ASSESS_SOURCE_QUALITY,
                status=_status_from_report(report),
                artifacts=artifacts,
                metrics=report.metrics,
                quality_report=report,
                raw_output=output,
            )
        except Exception as exc:
            return _failed_result(ActionType.ASSESS_SOURCE_QUALITY, exc)

    def execute_build_sft_dataset(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            mode = _training_modality(cfg)
            sft_dir = Path(state.run_dir) / "sft"
            sft_dir.mkdir(parents=True, exist_ok=True)
            tool_config = {
                "mode": mode,
                "output_annotations": str(sft_dir / "annotations.jsonl"),
                "output_sft": str(sft_dir / "sft.jsonl"),
                "target_language": cfg.get("sft_target_language", "English"),
                "batch_size": cfg.get("llm_batch_size", 5),
                "batch_delay": cfg.get("llm_batch_delay", 1.0),
                "provider": cfg.get("llm_provider"),
                "model": cfg.get("llm_model"),
                "api_key": cfg.get("llm_api_key"),
                "prompt_preset": cfg.get("sft_prompt_preset", "default"),
                "focus": cfg.get("focus", ""),
            }
            if mode == "image":
                images_dir = state.artifacts.get("images_dir") or cfg.get("input_dir")
                if not images_dir:
                    raise ValueError("images_dir artifact is required for image SFT mode.")
                tool_config["input_dir"] = str(images_dir)
                image_manifest = state.artifacts.get("images_index") or cfg.get("image_manifest")
                if image_manifest:
                    tool_config["image_manifest"] = str(image_manifest)
                tool_config["image_exts"] = cfg.get("image_exts", [".jpg", ".jpeg", ".png", ".webp"])
                tool_config["image_tasks"] = normalize_image_sft_tasks(cfg.get("image_sft_tasks"))
                source_split: Dict[str, Any] = {}
                source_registry: Dict[str, Any] = {}
            else:
                input_jsonl = state.artifacts.get("collected_texts_jsonl")
                if not input_jsonl:
                    data_path = state.artifacts.get("data_path") or cfg.get("data_path")
                    if not data_path:
                        raise ValueError("data_path artifact is required for text SFT mode.")
                    input_jsonl = str(sft_dir / f"collected_texts_{_collection_iteration_label(state)}.jsonl")
                    _export_hf_dataset_to_jsonl(str(data_path), input_jsonl)
                source_registry = _prepare_incremental_text_source_registry(
                    str(input_jsonl),
                    sft_dir,
                    collection_iteration=_collection_iteration_label(state),
                )
                input_jsonl = source_registry["merged_input_jsonl"]
                source_split = _prepare_text_source_split(
                    str(input_jsonl),
                    sft_dir,
                    enabled=_as_bool(cfg.get("source_eval_enable", True)),
                    ratio=_float_value(cfg.get("source_eval_ratio"), _float_value(cfg.get("dataset_val_ratio"), 0.1)),
                    seed=_int_value(cfg.get("seed"), 42),
                    max_eval_items=_int_value(cfg.get("source_eval_max_items"), 8),
                )
                input_jsonl = source_split.get("train_input_jsonl") or input_jsonl
                tool_config["input_jsonl"] = str(input_jsonl)
                tool_config["text_field"] = cfg.get("sft_text_field", "text")
                tool_config["reuse_annotations"] = _as_bool(cfg.get("sft_reuse_annotations", True))
                tool_config["annotation_cache_path"] = str(sft_dir / "text_annotation_cache.jsonl")
                tool_config["annotation_cache_metadata"] = _text_annotation_cache_signature(cfg)

            output = self.tools["build_sft_dataset"].execute(tool_config)
            report = validate_sft_output(output)
            text_quality_artifacts = _write_sft_text_quality_artifacts(state, output)
            heldout_eval_artifacts: Dict[str, Any] = {}
            if mode == "text" and source_split.get("eval_input_jsonl") and report.passed:
                eval_tool_config = {
                    **tool_config,
                    "input_jsonl": source_split["eval_input_jsonl"],
                    "output_annotations": str(sft_dir / "heldout_eval_annotations.jsonl"),
                    "output_sft": str(sft_dir / "heldout_eval_sft.jsonl"),
                }
                try:
                    eval_output = self.tools["build_sft_dataset"].execute(eval_tool_config)
                    heldout_eval_artifacts = {
                        "heldout_eval_source_path": source_split.get("eval_input_jsonl"),
                        "heldout_eval_sft_path": eval_output.get("sft_path"),
                        "heldout_eval_annotations_path": eval_output.get("annotations_path"),
                        "heldout_eval_num_examples": eval_output.get("num_examples"),
                        "heldout_eval_annotation_reuse_summary": eval_output.get("annotation_reuse"),
                    }
                except Exception as exc:
                    logger.warning("Failed to build held-out source eval set: %s", exc)
                    heldout_eval_artifacts = {
                        "heldout_eval_source_path": source_split.get("eval_input_jsonl"),
                        "heldout_eval_generation_error": f"{type(exc).__name__}: {exc}",
                    }
            sft_artifacts = {
                "sft_mode": output.get("mode"),
                "training_modality": output.get("mode"),
                "sft_path": output.get("sft_path"),
                "annotations_path": output.get("annotations_path"),
                "num_sft_examples": output.get("num_examples"),
                "sft_prompt_preset": output.get("prompt_preset"),
                "image_sft_tasks": output.get("image_tasks") if mode == "image" else None,
                "source_registry_path": source_registry.get("source_registry_path") if mode == "text" else None,
                "source_registry_summary": source_registry.get("summary") if mode == "text" else None,
                "annotation_cache_path": output.get("annotation_cache_path"),
                "annotation_reuse_summary": output.get("annotation_reuse"),
                "source_split_summary": source_split.get("summary") if mode == "text" else None,
                **heldout_eval_artifacts,
                **text_quality_artifacts,
            }
            return ActionResult(
                action_type=ActionType.BUILD_SFT_DATASET,
                status=_status_from_report(report),
                artifacts=sft_artifacts,
                metrics=report.metrics,
                quality_report=report,
                raw_output=output,
            )
        except Exception as exc:
            return _failed_result(ActionType.BUILD_SFT_DATASET, exc)

    def execute_build_dataset(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            data_path = state.artifacts.get("sft_path") or cfg.get("data_path")
            if not data_path:
                raise ValueError("sft_path artifact or config data_path is required.")
            output = self.tools["build_dataset"].execute(
                str(data_path),
                {
                    "run_dir": state.run_dir,
                    "validation_ratio": cfg.get("dataset_val_ratio", 0.1),
                    "eval_split": cfg.get("eval_split", "validation"),
                    "seed": cfg.get("seed", 42),
                },
            )
            report = validate_dataset_output(output)
            hub_info = {}
            if report.passed:
                dataset_ref = output.get("dataset_ref") if isinstance(output.get("dataset_ref"), dict) else {}
                dataset_path = dataset_ref.get("data_path") or (output.get("dataset_summary") or {}).get("data_path")
                hub_info = _push_hf_outputs_if_configured(
                    cfg,
                    dataset_path=dataset_path,
                    dataset_card_readme=_build_dataset_card(
                        state,
                        dataset_summary=output.get("dataset_summary"),
                    ),
                )
            return ActionResult(
                action_type=ActionType.BUILD_DATASET,
                status=_status_from_report(report),
                artifacts={
                    "dataset_ref": output.get("dataset_ref"),
                    "dataset_summary": output.get("dataset_summary"),
                    "dataset_manifest_path": output.get("dataset_manifest_path"),
                    **hub_info,
                },
                metrics=report.metrics,
                quality_report=report,
                raw_output={**output, **hub_info},
            )
        except Exception as exc:
            return _failed_result(ActionType.BUILD_DATASET, exc)

    def execute_train_model(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            dataset_ref = state.artifacts.get("dataset_ref")
            if not dataset_ref:
                raise ValueError("dataset_ref artifact is required before training.")
            iter_idx = int(state.retry_counts.get(ActionType.TRAIN_MODEL.value, 0))
            train_config = _train_config_dict(cfg)
            training_modality = _training_modality(cfg)
            if _as_bool(cfg.get("debug_stub_train", False)):
                output = _debug_train_output(
                    run_dir=state.run_dir,
                    iter_idx=iter_idx,
                    train_config=train_config,
                    dataset_ref=dataset_ref,
                    training_modality=training_modality,
                )
            else:
                output = self.tools["train"].execute(
                    dataset_ref,
                    {
                        "method": "sft",
                        "run_dir": state.run_dir,
                        "iter_idx": iter_idx,
                        "hf_model_id": cfg.get("hf_model_id"),
                        "trainer_key": cfg.get("trainer_key", "static_sft_default"),
                        "lora_preset_key": cfg.get("lora_preset_key", "lora_attn_small"),
                        "model_loader_key": cfg.get("model_loader_key", "hf_causal_lm_default"),
                        "train_config": train_config,
                        "max_samples": cfg.get("max_samples"),
                        "training_modality": training_modality,
                    },
                )
            report = validate_training_output(output)
            train_artifacts = {
                "adapter_path": output.get("adapter_path"),
                "train_log_paths": output.get("log_paths"),
                "train_metrics": output.get("metrics"),
                "iterations": [output.get("iteration_record")] if output.get("iteration_record") else [],
            }
            return ActionResult(
                action_type=ActionType.TRAIN_MODEL,
                status=_status_from_report(report),
                artifacts=train_artifacts,
                metrics=report.metrics,
                quality_report=report,
                raw_output=output,
            )
        except Exception as exc:
            return _failed_result(ActionType.TRAIN_MODEL, exc)

    def execute_evaluate_model(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            adapter_path = state.artifacts.get("adapter_path")
            dataset_ref = state.artifacts.get("dataset_ref") or {}
            heldout_eval_path = state.artifacts.get("heldout_eval_sft_path")
            data_path = heldout_eval_path or dataset_ref.get("data_path") or state.artifacts.get("sft_path") or cfg.get("data_path")
            eval_split = "train" if heldout_eval_path else dataset_ref.get("eval_split") or cfg.get("eval_split", "validation")
            if not adapter_path:
                raise ValueError("adapter_path artifact is required before evaluation.")
            if not data_path:
                raise ValueError("dataset data_path is required before evaluation.")
            eval_attempt = _stage_attempt(state, ActionType.EVALUATE_MODEL)
            eval_run_dir = Path(state.run_dir) / "eval" / f"attempt_{eval_attempt}"
            eval_config = {
                "run_dir": str(eval_run_dir),
                "hf_model_id": cfg.get("hf_model_id"),
                "split": eval_split,
                "max_samples": cfg.get("eval_max_samples", 64),
                "max_new_tokens": cfg.get("eval_max_new_tokens", 128),
                "max_input_tokens": cfg.get("eval_max_input_tokens") or cfg.get("train_max_seq_len"),
                "training_modality": _training_modality(cfg),
                "train_log_paths": state.artifacts.get("train_log_paths"),
                "max_steps": cfg.get("max_steps"),
                "eval_enable_llm_judge": cfg.get("eval_enable_llm_judge", False),
                "eval_compare_base_model": cfg.get("eval_compare_base_model", True),
                "eval_judge_max_samples": cfg.get("eval_judge_max_samples", 32),
                "eval_judge_batch_size": cfg.get("eval_judge_batch_size", 3),
                "eval_judge_batch_delay": cfg.get("eval_judge_batch_delay", 1.0),
                "target_language": cfg.get("sft_target_language"),
                "focus": cfg.get("focus", ""),
                "llm_provider": cfg.get("llm_provider"),
                "llm_model": cfg.get("llm_model"),
                "llm_api_key": cfg.get("llm_api_key"),
                "llm_batch_size": cfg.get("llm_batch_size"),
                "llm_batch_delay": cfg.get("llm_batch_delay"),
            }
            if _as_bool(cfg.get("debug_stub_eval", False)):
                output = _debug_eval_output(
                    run_dir=str(eval_run_dir),
                    data_path=str(data_path),
                    split=str(eval_split),
                    max_samples=int(eval_config["max_samples"] or 64),
                    failure_rate=float(cfg.get("debug_eval_failure_rate") or 0.0),
                )
            else:
                output = self.tools["eval_model"].execute(
                    str(adapter_path),
                    str(data_path),
                    eval_config,
                )
            report = validate_eval_output(output)
            eval_artifacts = {
                "eval_attempt": eval_attempt,
                "eval_attempt_dir": str(eval_run_dir),
                "predictions_path": output.get("predictions_path"),
                "failures_path": output.get("failures_path"),
                "cluster_preview": output.get("cluster_preview"),
                "eval_metrics_path": output.get("eval_metrics_path"),
                "eval_metrics": output.get("metrics"),
                "training_health": output.get("training_health"),
                "judge_summary": output.get("judge_summary"),
                "base_predictions_path": output.get("base_predictions_path"),
                "base_failures_path": output.get("base_failures_path"),
                "base_judge_summary": output.get("base_judge_summary"),
                "base_eval_metrics_path": output.get("base_eval_metrics_path"),
                "eval_lift_summary": output.get("lift_summary"),
                "eval_lift_summary_path": output.get("lift_summary_path"),
            }
            hub_info = {}
            if report.passed:
                if _as_bool(cfg.get("debug_stub_train", False)) or _as_bool(cfg.get("debug_stub_eval", False)):
                    if cfg.get("hf_adapter_repo"):
                        skipped_by = "debug_stub_train" if _as_bool(cfg.get("debug_stub_train", False)) else "debug_stub_eval"
                        hub_info["hf_adapter_upload_skipped"] = skipped_by
                else:
                    hub_info = _push_hf_outputs_if_configured(
                        cfg,
                        adapter_path=adapter_path,
                        adapter_card_readme=_build_adapter_card(
                            state,
                            current_artifacts=eval_artifacts,
                            eval_report=report,
                        ),
                    )
            return ActionResult(
                action_type=ActionType.EVALUATE_MODEL,
                status=_status_from_report(report),
                artifacts={
                    **eval_artifacts,
                    **hub_info,
                },
                metrics=report.metrics,
                quality_report=report,
                raw_output={**output, **hub_info},
            )
        except Exception as exc:
            return _failed_result(ActionType.EVALUATE_MODEL, exc)

    def execute_generate_report(self, state: PipelineState, request: ActionRequest) -> ActionResult:
        try:
            cfg = state.config
            training_modality = _training_modality(cfg)
            model_loader_key = cfg.get("model_loader_key")
            if not model_loader_key or (training_modality == "image" and model_loader_key == "hf_causal_lm_default"):
                model_loader_key = "hf_image_text_default" if training_modality == "image" else "hf_causal_lm_default"
            trainer_key = cfg.get("trainer_key")
            if not trainer_key or (training_modality == "image" and trainer_key == "static_sft_default"):
                trainer_key = "vision_language_sft" if training_modality == "image" else "static_sft_default"
            component_selection = {
                "dataset_loader_key": cfg.get("dataset_loader_key", f"hf_{training_modality}_default"),
                "model_loader_key": model_loader_key,
                "lora_preset_key": cfg.get("lora_preset_key", "lora_attn_small"),
                "trainer_key": trainer_key,
                "hf_model_id": cfg.get("hf_model_id"),
                "primary_metric": cfg.get("primary_metric", "eval_loss"),
                "rationale": "agentic full mode selection",
            }
            report_path = self.tools["reporting"].finalize(
                {
                    "dataset_summary": state.artifacts.get("dataset_summary") or {},
                    "component_selection": component_selection,
                    "iterations": state.artifacts.get("iterations") or [],
                    "failures_path": state.artifacts.get("failures_path") or "",
                    "cluster_preview": state.artifacts.get("cluster_preview") or {},
                    "pipeline_summary": _build_pipeline_summary(state),
                    "error_analysis": _build_eval_error_analysis(state),
                }
            )
            output = {"report_path": report_path}
            report = validate_report_output(output)
            return ActionResult(
                action_type=ActionType.GENERATE_REPORT,
                status=_status_from_report(report),
                artifacts=output,
                metrics=report.metrics,
                quality_report=report,
                raw_output=output,
            )
        except Exception as exc:
            return _failed_result(ActionType.GENERATE_REPORT, exc)


def _failed_result(stage: ActionType, exc: Exception) -> ActionResult:
    safe_error = str(sanitize_secret_text(f"{type(exc).__name__}:{exc}"))
    return ActionResult(
        action_type=stage,
        status="failed",
        quality_report=QualityReport(
            stage=stage,
            passed=False,
            recoverable=_is_recoverable_exception(exc),
            blocking_issues=[safe_error],
        ),
        error=str(sanitize_secret_text(str(exc))),
    )


def _is_recoverable_exception(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError))


def _status_from_report(report: QualityReport) -> str:
    return "success" if report.passed else "failed"


def _flatten_queries(output: Dict[str, Any]) -> list[str]:
    nested = output.get("category_subcategory_queries") or {}
    queries: list[str] = []
    for subcategories in nested.values():
        if not isinstance(subcategories, dict):
            continue
        for query_list in subcategories.values():
            if isinstance(query_list, Iterable) and not isinstance(query_list, (str, bytes)):
                queries.extend(str(query).strip() for query in query_list if str(query).strip())
    return queries


def _limit_queries(queries: list[str], max_queries: Any) -> list[str]:
    if max_queries is None:
        return queries
    max_queries_int = int(max_queries)
    if max_queries_int <= 0:
        return queries
    return queries[:max_queries_int]


def _merge_queries(base_queries: list[str], added_queries: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for query in [*base_queries, *_strings_from_any(added_queries)]:
        normalized = str(query).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def _strings_from_any(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _select_queries(
    output: Dict[str, Any],
    *,
    max_queries_per_category: Any,
    max_queries: Any,
) -> list[str]:
    per_category = _optional_positive_int(max_queries_per_category)
    if per_category is None:
        return _limit_queries(_flatten_queries(output), max_queries)

    nested = output.get("category_subcategory_queries") or {}
    queries: list[str] = []
    for subcategories in nested.values():
        if not isinstance(subcategories, dict):
            continue
        category_queries: list[str] = []
        for query_list in subcategories.values():
            if isinstance(query_list, Iterable) and not isinstance(query_list, (str, bytes)):
                category_queries.extend(str(query).strip() for query in query_list if str(query).strip())
        queries.extend(category_queries[:per_category])
    return _limit_queries(queries, max_queries)


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _export_hf_dataset_to_jsonl(hf_dataset_path: str, jsonl_path: str) -> None:
    from datasets import load_from_disk

    dataset = load_from_disk(hf_dataset_path)
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _collection_iteration_label(state: PipelineState) -> str:
    return f"iteration_{int(state.retry_counts.get(ActionType.COLLECT_DATA.value, 0) or 0)}"


def _prepare_incremental_text_source_registry(
    current_input_jsonl: str,
    sft_dir: Path,
    *,
    collection_iteration: str,
) -> Dict[str, Any]:
    registry_path = sft_dir / "collected_texts_merged.jsonl"
    alias_path = sft_dir / "collected_texts.jsonl"
    previous_rows = _read_existing_source_registry(registry_path, alias_path, current_input_jsonl)
    current_rows = _read_jsonl_rows(current_input_jsonl)

    merged_rows: list[Dict[str, Any]] = []
    merged_by_key: dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(previous_rows):
        normalized = _normalize_source_registry_row(row, idx, fallback_iteration="iteration_0")
        key = str(normalized["source_registry_key"])
        if key not in merged_by_key:
            merged_by_key[key] = normalized
            merged_rows.append(normalized)

    num_new_rows = 0
    num_existing_rows_seen = 0
    for idx, row in enumerate(current_rows):
        normalized = _normalize_source_registry_row(row, idx, fallback_iteration=collection_iteration)
        key = str(normalized["source_registry_key"])
        if key in merged_by_key:
            num_existing_rows_seen += 1
            _mark_source_row_seen(merged_by_key[key], collection_iteration)
            continue
        num_new_rows += 1
        merged_by_key[key] = normalized
        merged_rows.append(normalized)

    _write_jsonl_rows(registry_path, merged_rows)
    _write_jsonl_rows(alias_path, merged_rows)
    summary = {
        "schema_version": "text_source_registry.v1",
        "collection_iteration": collection_iteration,
        "current_source_path": current_input_jsonl,
        "source_registry_path": str(registry_path),
        "merged_source_path": str(alias_path),
        "num_previous_source_rows": len(previous_rows),
        "num_current_source_rows": len(current_rows),
        "num_merged_source_rows": len(merged_rows),
        "num_new_source_rows": num_new_rows,
        "num_existing_source_rows_seen": num_existing_rows_seen,
    }
    return {
        "merged_input_jsonl": str(registry_path),
        "source_registry_path": str(registry_path),
        "summary": summary,
    }


def _read_existing_source_registry(registry_path: Path, alias_path: Path, current_input_jsonl: str) -> list[Dict[str, Any]]:
    current_path = Path(current_input_jsonl).resolve()
    for path in (registry_path, alias_path):
        if path.exists() and path.resolve() != current_path:
            return _read_jsonl_rows(str(path))
    return []


def _normalize_source_registry_row(row: Mapping[str, Any], idx: int, *, fallback_iteration: str) -> Dict[str, Any]:
    normalized = dict(row)
    text = str(normalized.get("text") or normalized.get("source_text") or "").strip()
    source_identity = _source_identity(normalized, idx)
    text_hash = hashlib.sha256(_normalize_source_text(text).encode("utf-8")).hexdigest()
    registry_key = _source_registry_key(source_identity, text_hash)
    collection_iteration = str(normalized.get("collection_iteration") or fallback_iteration)
    normalized["id"] = registry_key
    normalized["source_registry_key"] = registry_key
    normalized["source_text_hash"] = text_hash
    normalized["source_identity"] = source_identity
    if text and not normalized.get("text"):
        normalized["text"] = text
    normalized["collection_iteration"] = collection_iteration
    normalized.setdefault("first_seen_collection_iteration", collection_iteration)
    normalized["last_seen_collection_iteration"] = str(normalized.get("last_seen_collection_iteration") or collection_iteration)
    seen = normalized.get("seen_collection_iterations")
    if not isinstance(seen, list):
        seen = [collection_iteration]
    if collection_iteration not in seen:
        seen.append(collection_iteration)
    normalized["seen_collection_iterations"] = seen
    normalized.setdefault("group_key", _source_group_key(normalized, idx))
    if text and not normalized.get("source_excerpt"):
        normalized["source_excerpt"] = text[:2000]
    return normalized


def _source_identity(row: Mapping[str, Any], idx: int) -> str:
    for key in ("source_identity", "source_url", "url", "group_key", "source_id", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"row:{idx}"


def _source_registry_key(source_identity: str, text_hash: str) -> str:
    payload = {"source_identity": source_identity, "text_hash": text_hash}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_source_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _mark_source_row_seen(row: Dict[str, Any], collection_iteration: str) -> None:
    row["last_seen_collection_iteration"] = collection_iteration
    seen = row.get("seen_collection_iterations")
    if not isinstance(seen, list):
        seen = []
    if collection_iteration not in seen:
        seen.append(collection_iteration)
    row["seen_collection_iterations"] = seen


def _text_annotation_cache_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "text_annotation.v1",
        "target_language": config.get("sft_target_language", "English"),
        "prompt_preset": config.get("sft_prompt_preset", "default"),
        "focus": config.get("focus", ""),
        "provider": config.get("llm_provider"),
        "model": config.get("llm_model"),
    }


def _prepare_text_source_split(
    input_jsonl: str,
    sft_dir: Path,
    *,
    enabled: bool,
    ratio: float,
    seed: int,
    max_eval_items: int,
) -> Dict[str, Any]:
    rows = _read_jsonl_rows(input_jsonl)
    summary: Dict[str, Any] = {
        "enabled": enabled,
        "split_strategy": "none",
        "source_eval_ratio": ratio,
        "num_source_rows": len(rows),
        "num_train_source_rows": len(rows),
        "num_eval_source_rows": 0,
        "num_source_groups": 0,
        "num_train_source_groups": 0,
        "num_eval_source_groups": 0,
    }
    if not enabled or ratio <= 0.0 or len(rows) <= 1:
        summary["reason"] = "disabled_or_insufficient_rows"
        return {"train_input_jsonl": input_jsonl, "summary": summary}

    group_by_row = [_source_group_key(row, idx) for idx, row in enumerate(rows)]
    groups = sorted(set(group_by_row))
    summary["num_source_groups"] = len(groups)
    if len(groups) <= 1:
        summary["reason"] = "insufficient_source_groups"
        return {"train_input_jsonl": input_jsonl, "summary": summary}

    shuffled_groups = list(groups)
    random.Random(seed).shuffle(shuffled_groups)
    eval_group_count = max(1, int(round(len(groups) * min(max(ratio, 0.0), 0.5))))
    eval_group_count = min(eval_group_count, len(groups) - 1)
    eval_groups = set(shuffled_groups[:eval_group_count])

    train_rows: list[Dict[str, Any]] = []
    eval_rows: list[Dict[str, Any]] = []
    train_groups: set[str] = set()
    actual_eval_groups: set[str] = set()
    for row, group_key in zip(rows, group_by_row):
        normalized_row = dict(row)
        normalized_row.setdefault("group_key", group_key)
        if group_key in eval_groups:
            eval_rows.append(normalized_row)
            actual_eval_groups.add(group_key)
        else:
            train_rows.append(normalized_row)
            train_groups.add(group_key)

    if max_eval_items > 0 and len(eval_rows) > max_eval_items:
        eval_rows = eval_rows[:max_eval_items]
        actual_eval_groups = {_source_group_key(row, idx) for idx, row in enumerate(eval_rows)}

    if not train_rows or not eval_rows:
        summary["reason"] = "empty_train_or_eval_split"
        return {"train_input_jsonl": input_jsonl, "summary": summary}

    train_path = sft_dir / "collected_texts_train_sources.jsonl"
    eval_path = sft_dir / "collected_texts_eval_sources.jsonl"
    _write_jsonl_rows(train_path, train_rows)
    _write_jsonl_rows(eval_path, eval_rows)

    summary.update(
        {
            "split_strategy": "source_group",
            "num_train_source_rows": len(train_rows),
            "num_eval_source_rows": len(eval_rows),
            "num_train_source_groups": len(train_groups),
            "num_eval_source_groups": len(actual_eval_groups),
            "train_source_path": str(train_path),
            "eval_source_path": str(eval_path),
            "max_eval_items": max_eval_items,
        }
    )
    return {
        "train_input_jsonl": str(train_path),
        "eval_input_jsonl": str(eval_path),
        "summary": summary,
    }


def _read_jsonl_rows(path: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _write_jsonl_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _source_group_key(row: Mapping[str, Any], idx: int) -> str:
    for key in ("group_key", "source_url", "source_id", "url", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"row:{idx}"


def _train_config_dict(config: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = config.get("train_config")
    if isinstance(explicit, dict):
        return explicit
    return {
        "lr": config.get("train_lr", 2e-4),
        "batch_size": config.get("train_batch_size", 4),
        "grad_accum": config.get("train_grad_accum", 4),
        "max_steps": config.get("max_steps", 200),
        "warmup_ratio": config.get("train_warmup_ratio", 0.03),
        "weight_decay": config.get("train_weight_decay", 0.0),
        "max_seq_len": config.get("train_max_seq_len", 512),
        "eval_steps": config.get("train_eval_steps", 50),
        "seed": config.get("seed", 42),
    }


def _write_collection_text_quality_artifacts(
    state: PipelineState,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_result_path = metadata.get("raw_result_path")
    if not raw_result_path:
        return {}
    out_path = Path(state.run_dir) / "collect" / "text_quality.json"
    try:
        report = write_text_quality_report(
            records_from_serper_raw(str(raw_result_path)),
            out_path,
            source="collection",
            **_text_quality_config(state.config),
        )
    except Exception as exc:
        logger.warning("Failed to write collection text quality report: %s", exc)
        return {}
    return {
        "collection_text_quality_path": str(out_path),
        "collection_text_quality_summary": summarize_text_quality(report),
    }


def _write_sft_text_quality_artifacts(
    state: PipelineState,
    output: Mapping[str, Any],
) -> Dict[str, Any]:
    sft_path = output.get("sft_path")
    if not sft_path:
        return {}
    out_path = Path(state.run_dir) / "sft" / "text_quality.json"
    try:
        report = write_text_quality_report(
            records_from_sft_jsonl(str(sft_path)),
            out_path,
            source="sft",
            **_text_quality_config(state.config),
        )
    except Exception as exc:
        logger.warning("Failed to write SFT text quality report: %s", exc)
        return {}
    return {
        "sft_text_quality_path": str(out_path),
        "sft_text_quality_summary": summarize_text_quality(report),
    }


def _text_quality_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "enable_embeddings": _as_bool(config.get("text_quality_enable_embeddings", False)),
        "embedding_model": _stripped(config.get("text_quality_embedding_model")) or DEFAULT_EMBEDDING_MODEL,
        "embedding_threshold": _float_value(config.get("text_quality_embedding_threshold"), 0.93),
        "max_embedding_items": _int_value(config.get("text_quality_max_embedding_items"), 256),
        "shingle_threshold": _float_value(config.get("text_quality_shingle_threshold"), 0.85),
        "max_shingle_items": _int_value(config.get("text_quality_max_shingle_items"), 1000),
        "max_reported_pairs": _int_value(config.get("text_quality_max_reported_pairs"), 50),
    }


def _stage_attempt(state: PipelineState, stage: ActionType) -> int:
    return sum(1 for result in state.result_history if result.get("action_type") == stage.value)


def _int_value(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _training_modality(config: Mapping[str, Any]) -> str:
    return str(config.get("training_modality") or config.get("sft_mode") or "text").strip().lower()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _push_hf_outputs_if_configured(
    config: Mapping[str, Any],
    *,
    dataset_path: Any = None,
    adapter_path: Any = None,
    dataset_card_readme: str | None = None,
    adapter_card_readme: str | None = None,
) -> Dict[str, Any]:
    token = _stripped(config.get("hf_token"))
    if token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token

    username = _stripped(config.get("hf_username")) or None
    pushed: Dict[str, Any] = {}

    dataset_repo = _stripped(config.get("hf_dataset_repo"))
    if dataset_path and dataset_repo:
        try:
            from core.hf_hub import push_dataset

            repo_name, repo_username = _hf_repo_name_and_username(dataset_repo, username)
            pushed["dataset_repo_id"] = push_dataset(
                str(dataset_path),
                repo_name,
                username=repo_username,
                card_readme=dataset_card_readme,
            )
            if dataset_card_readme:
                pushed["hf_dataset_card_updated"] = True
            logger.info("SFT dataset pushed to HF Hub: %s", pushed["dataset_repo_id"])
        except Exception as exc:
            safe_error = sanitize_secret_text(f"{type(exc).__name__}: {exc}")
            pushed["hf_dataset_upload_error"] = safe_error
            logger.error("Failed to push SFT dataset to HF Hub: %s", safe_error)

    adapter_repo = _stripped(config.get("hf_adapter_repo"))
    if adapter_path and adapter_repo:
        try:
            from core.hf_hub import push_adapter

            repo_name, repo_username = _hf_repo_name_and_username(adapter_repo, username)
            pushed["adapter_repo_id"] = push_adapter(
                str(adapter_path),
                repo_name,
                username=repo_username,
                card_readme=adapter_card_readme,
            )
            if adapter_card_readme:
                pushed["hf_adapter_card_updated"] = True
            logger.info("LoRA adapter pushed to HF Hub: %s", pushed["adapter_repo_id"])
        except Exception as exc:
            safe_error = sanitize_secret_text(f"{type(exc).__name__}: {exc}")
            pushed["hf_adapter_upload_error"] = safe_error
            logger.error("Failed to push LoRA adapter to HF Hub: %s", safe_error)

    return pushed


def _update_hf_dataset_card_if_configured(
    state: PipelineState,
    config: Mapping[str, Any],
    dataset_output: Mapping[str, Any],
) -> Dict[str, Any]:
    repo_id = _stripped(state.artifacts.get("dataset_repo_id"))
    if not repo_id:
        return {}
    token = _stripped(config.get("hf_token"))
    if token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
    try:
        from core.hf_hub import update_repo_readme

        readme = _build_dataset_card(
            state,
            dataset_summary=dataset_output.get("dataset_summary"),
        )
        update_repo_readme(repo_id, readme, repo_type="dataset")
        return {"hf_dataset_card_updated": True}
    except Exception as exc:
        safe_error = sanitize_secret_text(f"{type(exc).__name__}: {exc}")
        logger.error("Failed to update HF dataset card: %s", safe_error)
        return {"hf_dataset_card_update_error": safe_error}


def _update_hf_adapter_card_if_configured(
    state: PipelineState,
    config: Mapping[str, Any],
    eval_artifacts: Mapping[str, Any],
    eval_report: QualityReport,
) -> Dict[str, Any]:
    repo_id = _stripped(state.artifacts.get("adapter_repo_id"))
    if not repo_id:
        return {}
    token = _stripped(config.get("hf_token"))
    if token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
    try:
        from core.hf_hub import update_repo_readme

        readme = _build_adapter_card(state, current_artifacts=dict(eval_artifacts), eval_report=eval_report)
        update_repo_readme(repo_id, readme, repo_type="model")
        return {"hf_adapter_card_updated": True}
    except Exception as exc:
        safe_error = sanitize_secret_text(f"{type(exc).__name__}: {exc}")
        logger.error("Failed to update HF adapter card: %s", safe_error)
        return {"hf_adapter_card_update_error": safe_error}


def _build_dataset_card(
    state: PipelineState,
    *,
    current_artifacts: Mapping[str, Any] | None = None,
    dataset_summary: Mapping[str, Any] | None = None,
) -> str:
    artifacts = {**state.artifacts, **dict(current_artifacts or {})}
    collection = artifacts.get("collection_metadata") if isinstance(artifacts.get("collection_metadata"), dict) else {}
    text_filter = artifacts.get("text_filter_summary") or collection.get("text_filter_summary") or {}
    collection_quality = artifacts.get("collection_text_quality_summary") or {}
    sft_quality = artifacts.get("sft_text_quality_summary") or {}
    summary = dict(dataset_summary or artifacts.get("dataset_summary") or {})
    source_split = artifacts.get("source_split_summary") if isinstance(artifacts.get("source_split_summary"), dict) else {}
    source_registry = artifacts.get("source_registry_summary") if isinstance(artifacts.get("source_registry_summary"), dict) else {}
    annotation_reuse = artifacts.get("annotation_reuse_summary") if isinstance(artifacts.get("annotation_reuse_summary"), dict) else {}
    lines = [
        "---",
        "tags:",
        "- horde-agent",
        "- supervised-fine-tuning",
        "- lora-training",
        "license: other",
        "---",
        "",
        "# Horde Agent SFT Dataset",
        "",
        "This dataset was generated by the horde-agent full agentic workflow for supervised fine-tuning.",
        "Source metadata columns are retained to support source-grounded validation and split leakage checks.",
        "",
        "## Dataset Summary",
        _markdown_table(
            [
                ("Country", state.config.get("country")),
                ("Focus", state.config.get("focus")),
                (
                    "Training modality",
                    artifacts.get("training_modality") or artifacts.get("sft_mode") or _training_modality(state.config),
                ),
                ("Image SFT tasks", ", ".join(artifacts.get("image_sft_tasks") or []) or None),
                ("Target language", state.config.get("sft_target_language")),
                ("SFT examples", artifacts.get("num_sft_examples")),
                ("Held-out eval examples", artifacts.get("heldout_eval_num_examples")),
                ("Split strategy", summary.get("split_strategy")),
                ("Train rows", _nested(summary, "split_counts", "train")),
                ("Validation rows", _nested(summary, "split_counts", "validation")),
                ("Group key column", summary.get("group_key_column")),
            ]
        ),
        "",
        "## Source And Filtering",
        _markdown_table(
            [
                ("Collected samples", artifacts.get("num_samples")),
                ("Text filter enabled", text_filter.get("enabled")),
                ("Text rows before filter", text_filter.get("num_input")),
                ("Text rows kept", text_filter.get("num_kept")),
                ("Text rows removed", text_filter.get("num_removed")),
                ("Merged source rows", source_registry.get("num_merged_source_rows")),
                ("New source rows this iteration", source_registry.get("num_new_source_rows")),
                ("Existing source rows seen again", source_registry.get("num_existing_source_rows_seen")),
                ("Reused annotations", annotation_reuse.get("num_reused_annotations")),
                ("New annotation requests", annotation_reuse.get("num_llm_annotation_requests")),
                ("Source split strategy", source_split.get("split_strategy")),
                ("Train source groups", source_split.get("num_train_source_groups")),
                ("Eval source groups", source_split.get("num_eval_source_groups")),
                ("Collection exact duplicate rate", collection_quality.get("exact_duplicate_rate")),
                ("Collection URL duplicate rate", collection_quality.get("url_duplicate_rate")),
                ("SFT exact duplicate rate", sft_quality.get("exact_duplicate_rate")),
                ("SFT shingle near-duplicate pairs", sft_quality.get("shingle_pair_count")),
            ]
        ),
    ]
    if artifacts.get("image_dedup_enabled") is not None:
        lines.extend(
            [
                "",
                "## Image Deduplication",
                _markdown_table(
                    [
                        ("Enabled", artifacts.get("image_dedup_enabled")),
                        ("Method", artifacts.get("image_dedup_method")),
                        ("Threshold", artifacts.get("image_dedup_threshold")),
                        ("Images before dedup", artifacts.get("num_images_before_dedup")),
                        ("Images after dedup", artifacts.get("num_images")),
                        ("Removed images", artifacts.get("num_images_removed_by_dedup")),
                        ("Duplicate clusters", artifacts.get("num_image_dedup_clusters")),
                    ]
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Intended Use",
            (
                "Use this dataset for LoRA/SFT experiments in the same domain and modality. "
                "Review retained source metadata before public release or downstream redistribution."
            ),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_adapter_card(
    state: PipelineState,
    *,
    current_artifacts: Mapping[str, Any] | None = None,
    eval_report: QualityReport | None = None,
) -> str:
    artifacts = {**state.artifacts, **dict(current_artifacts or {})}
    train_metrics = artifacts.get("train_metrics") if isinstance(artifacts.get("train_metrics"), dict) else {}
    eval_metrics = artifacts.get("eval_metrics") if isinstance(artifacts.get("eval_metrics"), dict) else {}
    judge = artifacts.get("judge_summary") if isinstance(artifacts.get("judge_summary"), dict) else {}
    if not judge and isinstance(eval_metrics.get("judge"), dict):
        judge = eval_metrics["judge"]
    training_health = artifacts.get("training_health") if isinstance(artifacts.get("training_health"), dict) else {}
    if not training_health and isinstance(eval_metrics.get("training_health"), dict):
        training_health = eval_metrics["training_health"]
    lift = artifacts.get("eval_lift_summary") if isinstance(artifacts.get("eval_lift_summary"), dict) else {}
    if not lift and isinstance(eval_metrics.get("lift"), dict):
        lift = eval_metrics["lift"]
    lines = [
        "---",
        "tags:",
        "- horde-agent",
        "- lora",
        "- peft",
        "- supervised-fine-tuning",
        "license: other",
        "---",
        "",
        "# Horde Agent LoRA Adapter",
        "",
        "This LoRA adapter was trained by the horde-agent full agentic workflow.",
        "",
        "## Training Summary",
        _markdown_table(
            [
                ("Base model", state.config.get("hf_model_id")),
                ("Country", state.config.get("country")),
                ("Focus", state.config.get("focus")),
                ("Training modality", artifacts.get("training_modality") or _training_modality(state.config)),
                ("Dataset repo", artifacts.get("dataset_repo_id")),
                ("Max steps", state.config.get("max_steps")),
                ("Learning rate", state.config.get("train_lr")),
                ("Batch size", state.config.get("train_batch_size")),
                ("Gradient accumulation", state.config.get("train_grad_accum")),
                ("Last train loss", train_metrics.get("last_train_loss")),
                ("Best eval loss", train_metrics.get("best_eval_loss")),
            ]
        ),
        "",
        "## Evaluation Summary",
        _markdown_table(
            [
                ("Gate", eval_report.gate_status if eval_report else None),
                ("Failure rate", eval_metrics.get("failure_rate")),
                ("Predictions", eval_metrics.get("num_predictions")),
                ("Training health gate", training_health.get("gate_status")),
                ("Judge enabled", judge.get("enabled")),
                ("Judge gate", judge.get("gate_status")),
                ("Judge quality score", judge.get("quality_score")),
                ("Judge major failure rate", judge.get("major_failure_rate")),
                ("Unsupported grounding rate", judge.get("unsupported_grounding_rate")),
                ("Quality score delta vs base", lift.get("quality_score_delta")),
                ("Failure rate delta vs base", lift.get("failure_rate_delta")),
                ("Unsupported grounding delta vs base", lift.get("unsupported_grounding_rate_delta")),
            ]
        ),
        "",
        "## Intended Use",
        (
            "Load this repository as a PEFT/LoRA adapter on top of the base model listed above. "
            "Check the evaluation summary before using it outside the training domain."
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _hf_repo_name_and_username(repo_value: str, username: str | None) -> tuple[str, str | None]:
    if "/" not in repo_value:
        return repo_value, username
    repo_username, repo_name = repo_value.split("/", 1)
    return repo_name, username or repo_username


def _stripped(value: Any) -> str:
    return str(value or "").strip()


def _markdown_table(rows: Iterable[tuple[str, Any]]) -> str:
    table = ["| Field | Value |", "| --- | --- |"]
    for label, value in rows:
        if value in (None, "", [], {}):
            continue
        table.append(f"| {_escape_md(label)} | {_escape_md(_format_md_value(value))} |")
    if len(table) == 2:
        table.append("| None | Not available |")
    return "\n".join(table)


def _format_md_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _nested(mapping: Mapping[str, Any], key: str, nested_key: str) -> Any:
    value = mapping.get(key)
    if isinstance(value, Mapping):
        return value.get(nested_key)
    return None


def _build_pipeline_summary(state: PipelineState) -> Dict[str, Any]:
    metadata = state.artifacts.get("collection_metadata") or {}
    eval_metrics = state.artifacts.get("eval_metrics") or {}
    return {
        "run_summary": {
            "country": state.config.get("country"),
            "focus": state.config.get("focus"),
            "training_modality": _training_modality(state.config),
            "target_language": state.config.get("sft_target_language"),
            "hf_model_id": state.config.get("hf_model_id"),
        },
        "taxonomy_summary": _summarize_taxonomy(
            state.artifacts.get("taxonomy"),
            selected_queries=state.artifacts.get("search_queries"),
        ),
        "collection_summary": {
            "provider": metadata.get("provider"),
            "num_samples": state.artifacts.get("num_samples"),
            "collected_at": metadata.get("collected_at"),
            "raw_result_path": metadata.get("raw_result_path"),
            "text_filter_path": state.artifacts.get("text_filter_report_path"),
            "text_filter": state.artifacts.get("text_filter_summary"),
            "text_quality_path": state.artifacts.get("collection_text_quality_path"),
            "text_quality": state.artifacts.get("collection_text_quality_summary"),
            "image_dedup": {
                "enabled": state.artifacts.get("image_dedup_enabled"),
                "method": state.artifacts.get("image_dedup_method"),
                "threshold": state.artifacts.get("image_dedup_threshold"),
                "num_before": state.artifacts.get("num_images_before_dedup"),
                "num_after": state.artifacts.get("num_images"),
                "num_removed": state.artifacts.get("num_images_removed_by_dedup"),
                "num_clusters": state.artifacts.get("num_image_dedup_clusters"),
                "report_path": state.artifacts.get("image_dedup_report_path"),
                "raw_images_index": state.artifacts.get("raw_images_index"),
                "model_path": state.artifacts.get("image_dedup_model_path"),
                "downloaded_model": state.artifacts.get("image_dedup_downloaded_model"),
                "device": state.artifacts.get("image_dedup_device"),
            },
        },
        "sft_summary": {
            "mode": state.artifacts.get("sft_mode") or state.artifacts.get("training_modality"),
            "sft_path": state.artifacts.get("sft_path"),
            "num_examples": state.artifacts.get("num_sft_examples"),
            "source_split": state.artifacts.get("source_split_summary"),
            "heldout_eval_sft_path": state.artifacts.get("heldout_eval_sft_path"),
            "heldout_eval_num_examples": state.artifacts.get("heldout_eval_num_examples"),
            "text_quality_path": state.artifacts.get("sft_text_quality_path"),
            "text_quality": state.artifacts.get("sft_text_quality_summary"),
            "dataset_repo_id": state.artifacts.get("dataset_repo_id"),
            "hf_dataset_card_updated": state.artifacts.get("hf_dataset_card_updated"),
            "hf_dataset_upload_error": state.artifacts.get("hf_dataset_upload_error"),
            "hf_dataset_card_update_error": state.artifacts.get("hf_dataset_card_update_error"),
        },
        "training_summary": {
            "adapter_path": state.artifacts.get("adapter_path"),
            "adapter_repo_id": state.artifacts.get("adapter_repo_id"),
            "hf_adapter_card_updated": state.artifacts.get("hf_adapter_card_updated"),
            "hf_adapter_upload_error": state.artifacts.get("hf_adapter_upload_error"),
            "hf_adapter_card_update_error": state.artifacts.get("hf_adapter_card_update_error"),
            "hf_adapter_upload_skipped": state.artifacts.get("hf_adapter_upload_skipped"),
        },
        "eval_summary": {
            "attempt": state.artifacts.get("eval_attempt"),
            "attempt_dir": state.artifacts.get("eval_attempt_dir"),
            "failure_rate": eval_metrics.get("failure_rate") if isinstance(eval_metrics, dict) else None,
            "num_predictions": eval_metrics.get("num_predictions") if isinstance(eval_metrics, dict) else None,
            "predictions_path": state.artifacts.get("predictions_path"),
            "failures_path": state.artifacts.get("failures_path"),
            "judge": state.artifacts.get("judge_summary"),
            "training_health": state.artifacts.get("training_health"),
            "base_predictions_path": state.artifacts.get("base_predictions_path"),
            "base_failures_path": state.artifacts.get("base_failures_path"),
            "lift": state.artifacts.get("eval_lift_summary"),
            "lift_path": state.artifacts.get("eval_lift_summary_path"),
        },
    }


def _summarize_taxonomy(taxonomy: Any, *, selected_queries: Any = None) -> Dict[str, Any]:
    if not isinstance(taxonomy, dict):
        return {
            "num_categories": 0,
            "num_subcategories": 0,
            "num_generated_queries": 0,
            "num_selected_queries": _list_len(selected_queries),
            "categories": [],
        }

    category_subcategories = taxonomy.get("category_subcategories") or {}
    if not isinstance(category_subcategories, Mapping):
        category_subcategories = {}
    categories = taxonomy.get("categories") or []
    category_rows: list[dict[str, Any]] = []

    for category in categories:
        category_name = _name_from_item(category)
        if not category_name:
            continue
        subcategory_names = [
            name
            for name in (_name_from_item(item) for item in category_subcategories.get(category_name, []))
            if name
        ]
        category_rows.append({"name": category_name, "subcategories": subcategory_names})

    if not category_rows and isinstance(category_subcategories, dict):
        for category_name, subcategories in category_subcategories.items():
            subcategory_names = [
                name
                for name in (_name_from_item(item) for item in _iterable_items(subcategories))
                if name
            ]
            category_rows.append({"name": str(category_name), "subcategories": subcategory_names})

    return {
        "num_categories": len(category_rows),
        "num_subcategories": sum(len(row["subcategories"]) for row in category_rows),
        "num_generated_queries": _count_generated_queries(taxonomy),
        "num_selected_queries": _list_len(selected_queries),
        "categories": category_rows,
    }


def _build_eval_error_analysis(state: PipelineState) -> Dict[str, Any]:
    cluster_preview = state.artifacts.get("cluster_preview") or {}
    clusters = cluster_preview.get("clusters") if isinstance(cluster_preview, dict) else []
    eval_metrics = state.artifacts.get("eval_metrics") or {}
    training_health = state.artifacts.get("training_health")
    judge_summary = state.artifacts.get("judge_summary")
    training_health_has_issue = _gate_has_issue(training_health)
    judge_has_issue = _gate_has_issue(judge_summary)

    if not clusters and not training_health_has_issue and not judge_has_issue:
        return {"status": "No evaluation failures detected."}

    analysis = {
        "failure_clusters": clusters or [],
        "failure_rate": eval_metrics.get("failure_rate") if isinstance(eval_metrics, dict) else None,
    }
    if training_health_has_issue:
        analysis["training_health"] = training_health
    if judge_has_issue:
        analysis["judge_summary"] = judge_summary
    return analysis


def _gate_has_issue(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    gate_status = str(payload.get("gate_status") or "").strip().lower()
    return gate_status in {"repair", "fail"} or bool(payload.get("blocking_issues"))


def _count_generated_queries(taxonomy: Mapping[str, Any]) -> int:
    nested = taxonomy.get("category_subcategory_queries") or {}
    if not isinstance(nested, Mapping):
        return 0
    total = 0
    for subcategories in nested.values():
        if not isinstance(subcategories, Mapping):
            continue
        for queries in subcategories.values():
            total += _list_len(queries)
    return total


def _name_from_item(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("name") or "").strip()
    return str(item or "").strip()


def _iterable_items(value: Any) -> list[Any]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return list(value)
    return []


def _list_len(value: Any) -> int:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return len(list(value))
    return 0


def _debug_train_output(
    *,
    run_dir: str,
    iter_idx: int,
    train_config: Mapping[str, Any],
    dataset_ref: Mapping[str, Any],
    training_modality: str,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    adapter_dir = run_path / "debug_stub" / "adapter" / f"iter_{iter_idx}"
    log_dir = run_path / "debug_stub" / "logs"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "steps": int(train_config.get("max_steps") or 0),
        "best_eval_loss": 0.0,
        "last_train_loss": 0.0,
        "last_eval_loss": 0.0,
    }
    train_log = log_dir / f"train_iter_{iter_idx}.log"
    metrics_path = log_dir / f"metrics_iter_{iter_idx}.jsonl"
    adapter_config = adapter_dir / "adapter_config.json"
    train_log.write_text(
        (
            f"debug_stub_train=true iter_idx={iter_idx} "
            f"training_modality={training_modality} dataset={dataset_ref.get('data_path', '')}\n"
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False) + "\n", encoding="utf-8")
    adapter_config.write_text(
        json.dumps(
            {
                "debug_stub": True,
                "created_at": int(time.time()),
                "training_modality": training_modality,
                "dataset_ref": dict(dataset_ref),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    config = dict(train_config)
    iteration_record = {
        "iter_idx": iter_idx,
        "config": config,
        "metrics": metrics,
        "adapter_path": str(adapter_dir),
        "log_paths": {"train_log": str(train_log), "metrics": str(metrics_path)},
    }
    return {
        "adapter_path": str(adapter_dir),
        "log_paths": iteration_record["log_paths"],
        "metrics": metrics,
        "iteration_record": iteration_record,
        "training_modality": training_modality,
        "debug_stub": True,
    }


def _debug_eval_output(
    *,
    run_dir: str,
    data_path: str,
    split: str,
    max_samples: int,
    failure_rate: float,
) -> Dict[str, Any]:
    eval_dir = Path(run_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = eval_dir / "predictions.jsonl"
    failures_path = eval_dir / "failures.jsonl"

    rows = _read_debug_rows(data_path, split=split, max_samples=max_samples)
    with predictions_path.open("w", encoding="utf-8") as handle:
        if rows:
            for idx, row in enumerate(rows):
                handle.write(
                    json.dumps(
                        {
                            "id": idx,
                            "input": _preview_text(row),
                            "prediction": "debug_stub_prediction",
                            "reference": "debug_stub_prediction",
                            "passed": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        else:
            handle.write(
                json.dumps(
                    {
                        "id": 0,
                        "input": "",
                        "prediction": "debug_stub_prediction",
                        "reference": "debug_stub_prediction",
                        "passed": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    failure_count = 0
    if failure_rate > 0:
        failure_count = max(1, int(round(max(1, len(rows)) * failure_rate)))
    with failures_path.open("w", encoding="utf-8") as handle:
        for idx in range(failure_count):
            handle.write(
                json.dumps(
                    {
                        "id": idx,
                        "reason": "debug_stub_failure",
                        "label": "debug_stub",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    clusters = []
    if failure_count:
        clusters.append({"label": "debug_stub", "count": failure_count, "examples": []})
    return {
        "predictions_path": str(predictions_path),
        "failures_path": str(failures_path),
        "cluster_preview": {"clusters": clusters},
        "metrics": {"failure_rate": failure_rate, "num_predictions": max(1, len(rows))},
        "debug_stub": True,
    }


def _read_debug_rows(data_path: str, *, split: str, max_samples: int) -> list[Dict[str, Any]]:
    path = Path(data_path)
    rows: list[Dict[str, Any]] = []
    limit = max(1, max_samples)
    if path.exists() and path.is_dir():
        try:
            from core.data.hf_dataset import load_dataset_from_path

            dataset, _ = load_dataset_from_path(str(path), split=split)
            for row in dataset.select(range(min(len(dataset), limit))):
                rows.append(dict(row))
            return rows
        except Exception:
            return []
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = {"text": line}
            rows.append(parsed if isinstance(parsed, dict) else {"value": parsed})
    return rows


def _preview_text(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, Mapping):
            return str(first.get("content") or "")[:500]
    for key in ("text", "prompt", "instruction", "input", "query"):
        if row.get(key):
            return str(row[key])[:500]
    return json.dumps(dict(row), ensure_ascii=False)[:500]


def full_graph_executors(tools: Mapping[str, Any]) -> Dict[ActionType, StageExecutor]:
    adapter = AgenticToolAdapter(tools)
    return {stage: adapter.executors()[stage] for stage in FULL_GRAPH_ACTIONS}
