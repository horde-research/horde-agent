"""Markdown reporting for pipeline results.

Copied from `agentic_train_pipeline/reporting/report.py` and adjusted for new package layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types.pipeline_types import IterationRecord


def _parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        start = line.find("{")
        end = line.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(line[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_json_line(line)
            if parsed is not None:
                records.append(parsed)
    return records


def write_report(
    out_dir: str,
    dataset_summary: Dict[str, Any],
    component_selection: Dict[str, Any],
    iterations: List[IterationRecord],
    failures_path: str,
    cluster_preview: Dict[str, Any],
    error_analysis: Dict[str, Any],
    pipeline_summary: Dict[str, Any] | None = None,
) -> str:
    out_path = Path(out_dir) / "report.md"
    decisions_path = Path(out_dir) / "agent_decisions.jsonl"
    decisions = _read_jsonl(str(decisions_path))
    dataset_example = None
    dataset_summary_for_report = dict(dataset_summary)
    if isinstance(dataset_summary_for_report.get("example"), dict):
        dataset_example = dataset_summary_for_report.pop("example")

    lines: List[str] = []
    lines.append("# Agentic LoRA SFT Report\n")
    lines.append("## Dataset Summary\n")
    lines.append("```\n" + json.dumps(dataset_summary_for_report, indent=2, ensure_ascii=False) + "\n```\n")
    if dataset_example:
        lines.append("## Dataset Example\n")
        lines.extend(_format_dataset_example(dataset_example))
        lines.append("")

    lines.append("## Agent Decisions\n")
    lines.append("```\n" + json.dumps(decisions, indent=2, ensure_ascii=False) + "\n```\n")

    lines.append("## Selected Components\n")
    lines.append("```\n" + json.dumps(component_selection, indent=2, ensure_ascii=False) + "\n```\n")

    lines.append("## Training Iterations\n")
    for record in iterations:
        lines.append(f"### Iteration {record.iter_idx}\n")
        lines.append("```\n" + json.dumps(record.model_dump(), indent=2, ensure_ascii=False) + "\n```\n")
        if record.metrics.best_eval_loss is None and record.metrics.last_eval_loss is None:
            lines.append("- Eval loss not recorded for this iteration; `eval_steps` may exceed `max_steps`.\n")

    lines.append("## Failure Clusters\n")
    lines.append("```\n" + json.dumps(cluster_preview, indent=2, ensure_ascii=False) + "\n```\n")

    if pipeline_summary:
        lines.append("## Pipeline Summary\n")
        lines.extend(_format_pipeline_summary(pipeline_summary))

    lines.append("## Error Analysis\n")
    lines.append("```\n" + json.dumps(error_analysis, indent=2, ensure_ascii=False) + "\n```\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def _format_pipeline_summary(summary: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    run_summary = summary.get("run_summary") or {}
    if run_summary:
        lines.append("### Run\n")
        if run_summary.get("country"):
            lines.append(f"- Country/culture: {run_summary['country']}")
        if run_summary.get("focus"):
            lines.append(f"- Scope/focus: {run_summary['focus']}")
        if run_summary.get("training_modality"):
            lines.append(f"- Training modality: {run_summary['training_modality']}")
        if run_summary.get("target_language"):
            lines.append(f"- Target language: {run_summary['target_language']}")
        if run_summary.get("hf_model_id"):
            lines.append(f"- Base model: `{run_summary['hf_model_id']}`")
        lines.append("")

    taxonomy = summary.get("taxonomy_summary") or {}
    if taxonomy:
        lines.append("### Taxonomy\n")
        lines.append(f"- Categories: {taxonomy.get('num_categories', 0)}")
        lines.append(f"- Subcategories: {taxonomy.get('num_subcategories', 0)}")
        lines.append(f"- Generated search queries: {taxonomy.get('num_generated_queries', 0)}")
        lines.append(f"- Selected search queries: {taxonomy.get('num_selected_queries', 0)}")
        lines.append("")
        for category in taxonomy.get("categories") or []:
            if not isinstance(category, dict):
                continue
            name = category.get("name") or "Unnamed category"
            subcategories = category.get("subcategories") or []
            lines.append(f"- {name} ({len(subcategories)} subcategories)")
            for subcategory in subcategories:
                lines.append(f"  - {subcategory}")
        lines.append("")

    collection = summary.get("collection_summary") or {}
    if collection:
        lines.append("### Collection\n")
        lines.append(f"- Provider: {collection.get('provider') or 'unknown'}")
        lines.append(f"- Text samples: {collection.get('num_samples', 0)}")
        if collection.get("collected_at"):
            lines.append(f"- Collected at: {collection['collected_at']}")
        if collection.get("raw_result_path"):
            lines.append(f"- Raw result path: `{collection['raw_result_path']}`")
        lines.extend(_format_text_filter(collection.get("text_filter"), path=collection.get("text_filter_path")))
        lines.extend(_format_text_quality(collection.get("text_quality"), path=collection.get("text_quality_path")))
        lines.extend(_format_image_dedup(collection.get("image_dedup")))
        lines.append("")

    sft = summary.get("sft_summary") or {}
    if sft:
        lines.append("### SFT Dataset\n")
        lines.append(f"- Mode: {sft.get('mode') or 'unknown'}")
        lines.append(f"- Examples: {sft.get('num_examples', 0)}")
        source_split = sft.get("source_split") if isinstance(sft.get("source_split"), dict) else {}
        if source_split:
            lines.append(f"- Source split: `{source_split.get('split_strategy') or 'none'}`")
            lines.append(
                "- Source groups train/eval: "
                f"{source_split.get('num_train_source_groups', 0)}/{source_split.get('num_eval_source_groups', 0)}"
            )
        if sft.get("heldout_eval_sft_path"):
            lines.append(f"- Held-out eval SFT path: `{sft['heldout_eval_sft_path']}`")
        if sft.get("heldout_eval_num_examples") is not None:
            lines.append(f"- Held-out eval examples: {sft['heldout_eval_num_examples']}")
        if sft.get("sft_path"):
            lines.append(f"- SFT path: `{sft['sft_path']}`")
        lines.extend(_format_text_quality(sft.get("text_quality"), path=sft.get("text_quality_path")))
        if sft.get("dataset_repo_id"):
            lines.append(f"- HF dataset repo: `{sft['dataset_repo_id']}`")
        if sft.get("hf_dataset_card_updated"):
            lines.append("- HF dataset card updated: yes")
        if sft.get("hf_dataset_upload_error"):
            lines.append(f"- HF dataset upload error: `{sft['hf_dataset_upload_error']}`")
        if sft.get("hf_dataset_card_update_error"):
            lines.append(f"- HF dataset card update error: `{sft['hf_dataset_card_update_error']}`")
        lines.append("")

    training = summary.get("training_summary") or {}
    if training:
        lines.append("### Training\n")
        if training.get("adapter_path"):
            lines.append(f"- Adapter path: `{training['adapter_path']}`")
        if training.get("adapter_repo_id"):
            lines.append(f"- HF adapter repo: `{training['adapter_repo_id']}`")
        if training.get("hf_adapter_card_updated"):
            lines.append("- HF adapter card updated: yes")
        if training.get("hf_adapter_upload_skipped"):
            lines.append(f"- HF adapter upload skipped: `{training['hf_adapter_upload_skipped']}`")
        if training.get("hf_adapter_upload_error"):
            lines.append(f"- HF adapter upload error: `{training['hf_adapter_upload_error']}`")
        if training.get("hf_adapter_card_update_error"):
            lines.append(f"- HF adapter card update error: `{training['hf_adapter_card_update_error']}`")
        lines.append("")

    eval_summary = summary.get("eval_summary") or {}
    if eval_summary:
        lines.append("### Evaluation\n")
        if eval_summary.get("attempt") is not None:
            lines.append(f"- Attempt: {eval_summary['attempt']}")
        if eval_summary.get("attempt_dir"):
            lines.append(f"- Attempt directory: `{eval_summary['attempt_dir']}`")
        if eval_summary.get("failure_rate") is not None:
            lines.append(f"- Failure rate: {eval_summary['failure_rate']}")
        if eval_summary.get("num_predictions") is not None:
            lines.append(f"- Predictions: {eval_summary['num_predictions']}")
        if eval_summary.get("predictions_path"):
            lines.append(f"- Predictions path: `{eval_summary['predictions_path']}`")
        if eval_summary.get("failures_path"):
            lines.append(f"- Failures path: `{eval_summary['failures_path']}`")
        if eval_summary.get("base_predictions_path"):
            lines.append(f"- Base predictions path: `{eval_summary['base_predictions_path']}`")
        judge = eval_summary.get("judge") if isinstance(eval_summary.get("judge"), dict) else {}
        if judge:
            lines.append(f"- Judge gate: {judge.get('gate_status') or 'unknown'}")
            if judge.get("quality_score") is not None:
                lines.append(f"- Judge quality score: {judge['quality_score']}")
            if judge.get("unsupported_grounding_rate") is not None:
                lines.append(f"- Unsupported grounding rate: {judge['unsupported_grounding_rate']}")
        training_health = (
            eval_summary.get("training_health") if isinstance(eval_summary.get("training_health"), dict) else {}
        )
        if training_health.get("gate_status"):
            lines.append(f"- Training health gate: {training_health['gate_status']}")
        lift = eval_summary.get("lift") if isinstance(eval_summary.get("lift"), dict) else {}
        if lift.get("enabled"):
            if lift.get("quality_score_delta") is not None:
                lines.append(f"- Quality score delta vs base: {lift['quality_score_delta']}")
            if lift.get("failure_rate_delta") is not None:
                lines.append(f"- Failure rate delta vs base: {lift['failure_rate_delta']}")
            if lift.get("unsupported_grounding_rate_delta") is not None:
                lines.append(
                    "- Unsupported grounding delta vs base: "
                    f"{lift['unsupported_grounding_rate_delta']}"
                )
        if eval_summary.get("lift_path"):
            lines.append(f"- Lift summary path: `{eval_summary['lift_path']}`")
        lines.append("")

    if not lines:
        lines.append("No pipeline summary was provided.\n")
    return lines


def _format_dataset_example(example: Any) -> List[str]:
    if not isinstance(example, dict):
        return []
    lines: List[str] = []
    messages = example.get("messages") if isinstance(example.get("messages"), list) else []
    if messages:
        lines.append("```")
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or "unknown"
            rendered = _render_message_content(message.get("content"))
            lines.append(f"{role}: {rendered}")
        lines.append("```")
    metadata_lines = []
    for key in ("group_key", "source_url", "source_image_url", "source_query"):
        value = example.get(key)
        if value not in (None, "", [], {}):
            metadata_lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")
    lines.extend(metadata_lines)
    return lines


def _render_message_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image":
                image_ref = item.get("image") or item.get("path") or item.get("url")
                if image_ref:
                    parts.append(f"[image: {image_ref}]")
            elif item_type == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return " ".join(parts).strip()
    return str(content or "").strip()


def _format_text_filter(summary: Any, *, path: Any = None) -> List[str]:
    if not isinstance(summary, dict) and not path:
        return []
    lines: List[str] = []
    if isinstance(summary, dict):
        if summary.get("enabled") is not None:
            lines.append(f"- Text filter enabled: {summary['enabled']}")
        if summary.get("num_removed") is not None:
            lines.append(
                "- Text filter removed: "
                f"{summary.get('num_removed', 0)}/{summary.get('num_input', 0)} rows"
            )
        if summary.get("removed_reason_counts"):
            lines.append(f"- Text filter removal reasons: `{summary['removed_reason_counts']}`")
    if path:
        lines.append(f"- Text filter report: `{path}`")
    return lines


def _format_text_quality(quality: Any, *, path: Any = None) -> List[str]:
    if not isinstance(quality, dict) and not path:
        return []
    lines: List[str] = []
    if isinstance(quality, dict):
        if quality.get("exact_duplicate_rate") is not None:
            lines.append(
                "- Exact duplicate rate: "
                f"{quality['exact_duplicate_rate']:.3f} ({quality.get('exact_duplicate_count', 0)} duplicates)"
            )
        if quality.get("url_duplicate_rate") is not None:
            lines.append(
                "- URL duplicate rate: "
                f"{quality['url_duplicate_rate']:.3f} ({quality.get('url_duplicate_count', 0)} duplicates)"
            )
        if quality.get("shingle_pair_count") is not None:
            lines.append(
                "- Shingle near-duplicate pairs: "
                f"{quality['shingle_pair_count']} across {quality.get('shingle_num_compared', 0)} checked rows"
            )
        if quality.get("embedding_enabled"):
            model = quality.get("embedding_model") or "unknown"
            count = quality.get("embedding_pair_count", 0)
            embedded = quality.get("embedding_num_embedded", 0)
            lines.append(f"- Embedding near-duplicate pairs: {count} across {embedded} embedded rows (`{model}`)")
        elif quality.get("embedding_enabled") is False:
            model = quality.get("embedding_model") or "unknown"
            lines.append(f"- Embedding near-duplicate check: disabled (`{model}`)")
        if quality.get("embedding_error"):
            lines.append(f"- Embedding near-duplicate error: `{quality['embedding_error']}`")
    if path:
        lines.append(f"- Text quality path: `{path}`")
    return lines


def _format_image_dedup(summary: Any) -> List[str]:
    if not isinstance(summary, dict) or not summary.get("enabled"):
        return []
    before = summary.get("num_before")
    after = summary.get("num_after")
    removed = summary.get("num_removed")
    method = summary.get("method") or "unknown"
    threshold = summary.get("threshold")
    lines = [f"- Image dedup: `{method}` threshold={threshold} kept={after}/{before} removed={removed}"]
    if summary.get("num_clusters") is not None:
        lines.append(f"- Image dedup clusters: {summary['num_clusters']}")
    if summary.get("downloaded_model"):
        lines.append(f"- Image dedup downloaded model: `{summary.get('model_path')}`")
    elif summary.get("model_path"):
        lines.append(f"- Image dedup model: `{summary['model_path']}`")
    if summary.get("device"):
        lines.append(f"- Image dedup device: `{summary['device']}`")
    if summary.get("report_path"):
        lines.append(f"- Image dedup report: `{summary['report_path']}`")
    if summary.get("raw_images_index"):
        lines.append(f"- Raw image manifest: `{summary['raw_images_index']}`")
    return lines
