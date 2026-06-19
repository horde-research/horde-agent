"""Structured LLM judge for validation predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from core.llm import LLMClient, LLMRequest


PASS_VALUES = {"pass", "ok", "good", "none"}
MINOR_VALUES = {"minor_issue", "minor", "warn", "warning", "partial"}
MAJOR_VALUES = {"major_failure", "major", "fail", "bad"}
GROUNDING_VALUES = {"supported", "unsupported", "insufficient_source"}


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for supervised fine-tuning validation outputs.
Return only valid JSON. Judge semantic usefulness, not string similarity."""


def run_llm_judge(
    predictions_path: str,
    out_dir: str,
    *,
    modality: str,
    target_language: str = "",
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_samples: int = 32,
    batch_size: int = 3,
    batch_delay: float = 1.0,
) -> Dict[str, Any]:
    rows = _read_predictions(predictions_path, max_samples=max_samples)
    out_path = Path(out_dir) / "judge_results.jsonl"
    summary_path = Path(out_dir) / "judge_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        summary = _aggregate([], num_requested=0, judge_results_path=str(out_path))
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    client = LLMClient.from_env(provider=provider, model=model, api_key=api_key, temperature=0.0)
    requests = [
        LLMRequest(
            request_id=str(row.get("id", idx)),
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=_judge_prompt(row, modality=modality, target_language=target_language),
            images=[str(row["image_path"])] if modality == "image" and row.get("image_path") else None,
        )
        for idx, row in enumerate(rows)
    ]
    responses = client.generate_json_batch_sync(
        requests,
        batch_size=batch_size,
        batch_delay_seconds=batch_delay,
    )

    judged_rows: list[dict[str, Any]] = []
    by_id = {str(row.get("id", idx)): row for idx, row in enumerate(rows)}
    with out_path.open("w", encoding="utf-8") as handle:
        for response in responses:
            source = by_id.get(str(response.request_id), {})
            if response.success and isinstance(response.data, dict):
                judged = _normalize_judgement(response.data)
                judged["id"] = response.request_id
            else:
                judged = {
                    "id": response.request_id,
                    "verdict": "major_failure",
                    "major_failure": True,
                    "warning": False,
                    "categories": ["judge_error"],
                    "failure_categories": ["judge_error"],
                    "error": response.error or "judge_error",
                    "raw": response.data,
                }
            judged["input"] = source.get("input")
            judged["prediction"] = source.get("prediction")
            judged["source_url"] = source.get("source_url")
            judged["group_key"] = source.get("group_key")
            judged_rows.append(judged)
            handle.write(json.dumps(judged, ensure_ascii=False) + "\n")

    summary = _aggregate(judged_rows, num_requested=len(rows), judge_results_path=str(out_path))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _read_predictions(predictions_path: str, *, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(predictions_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _judge_prompt(row: dict[str, Any], *, modality: str, target_language: str) -> str:
    image_line = (
        "The image is attached to this request. Judge visual grounding against the image.\n"
        if modality == "image"
        else ""
    )
    source_excerpt = str(row.get("source_excerpt") or "").strip()
    source_url = str(row.get("source_url") or "").strip()
    source_block = (
        f"Source URL:\n{source_url or 'unspecified'}\n\nSource excerpt:\n{source_excerpt}\n"
        if source_excerpt
        else "Source excerpt:\nNot provided. Set grounding to insufficient_source.\n"
    )
    return f"""
Evaluate this validation answer.

Modality: {modality}
Target language: {target_language or "unspecified"}
{image_line}
Judge semantic usefulness, not string similarity.
When a source excerpt is provided, treat it as the primary evidence for factual support.
The reference answer is the expected answer, but correct paraphrases are acceptable.
Penalize answers that are factually wrong, contradict the reference, miss the core answer,
hallucinate unsupported details, or fail the requested format.
Do not penalize harmless wording differences or concise correct answers.

Rubric:
- pass: The answer satisfies the user request and is materially consistent with the reference.
- minor_issue: The answer is mostly correct but incomplete, mildly verbose, or has a small non-critical issue.
- major_failure: The answer is wrong, irrelevant, hallucinated, contradictory, empty, or misses the main point.

Allowed categories:
wrong_fact, missing_key_point, hallucination, irrelevant, format, language, unsafe, other

Grounding labels:
- supported: the model answer is supported by the source excerpt or attached image.
- unsupported: the model answer makes factual claims contradicted by or absent from provided evidence.
- insufficient_source: no usable source evidence was provided.

{source_block}

User prompt:
{row.get("input", "")}

Model answer:
{row.get("prediction", "")}

Reference answer:
{row.get("reference", "")}

Return JSON with exactly these keys:
{{
  "verdict": "pass|minor_issue|major_failure",
  "grounding": "supported|unsupported|insufficient_source",
  "categories": ["wrong_fact|missing_key_point|hallucination|irrelevant|format|language|unsafe|other"]
}}
""".strip()


def _normalize_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    verdict = _norm_verdict(payload.get("verdict"))
    grounding = _norm_grounding(payload.get("grounding"))
    categories = _categories(payload.get("categories") or payload.get("failure_categories"))
    if grounding == "unsupported":
        verdict = "major_failure"
        if not categories:
            categories = ["hallucination"]
    major_failure = verdict == "major_failure"
    warning = verdict == "minor_issue"
    return {
        "verdict": verdict,
        "grounding": grounding,
        "major_failure": major_failure,
        "warning": warning,
        "categories": categories,
        "failure_categories": categories,
        "raw": payload,
    }


def _aggregate(rows: list[dict[str, Any]], *, num_requested: int, judge_results_path: str) -> Dict[str, Any]:
    num_judged = len(rows)
    passes = sum(1 for row in rows if row.get("verdict") == "pass")
    minor = sum(1 for row in rows if row.get("verdict") == "minor_issue")
    major = sum(1 for row in rows if row.get("major_failure"))
    category_counts: dict[str, int] = {}
    grounding_counts: dict[str, int] = {}
    for row in rows:
        grounding = str(row.get("grounding") or "insufficient_source")
        grounding_counts[grounding] = grounding_counts.get(grounding, 0) + 1
        for category in row.get("categories") or row.get("failure_categories") or []:
            category_counts[str(category)] = category_counts.get(str(category), 0) + 1
    pass_rate = passes / num_judged if num_judged else 0.0
    minor_rate = minor / num_judged if num_judged else 0.0
    major_rate = major / num_judged if num_judged else 0.0
    quality_score = (passes + (0.5 * minor)) / num_judged if num_judged else 0.0
    unsupported_grounding_count = grounding_counts.get("unsupported", 0)
    unsupported_grounding_rate = unsupported_grounding_count / num_judged if num_judged else 0.0
    if not num_judged:
        gate_status = "pass"
    elif major_rate > 0.2 or quality_score < 0.70 or unsupported_grounding_rate > 0.2:
        gate_status = "repair"
    elif major_rate > 0.1 or quality_score < 0.85 or unsupported_grounding_rate > 0.1:
        gate_status = "warn"
    else:
        gate_status = "pass"
    return {
        "enabled": True,
        "passed": gate_status != "repair",
        "gate_status": gate_status,
        "decision": "repair" if gate_status == "repair" else "continue",
        "judge_results_path": judge_results_path,
        "num_requested": num_requested,
        "num_judged": num_judged,
        "pass_count": passes,
        "minor_issue_count": minor,
        "major_failure_count": major,
        "warning_count": minor,
        "quality_score": quality_score,
        "pass_rate": pass_rate,
        "minor_issue_rate": minor_rate,
        "major_failure_rate": major_rate,
        "warning_rate": minor_rate,
        "failure_category_counts": category_counts,
        "grounding_counts": grounding_counts,
        "unsupported_grounding_count": unsupported_grounding_count,
        "unsupported_grounding_rate": unsupported_grounding_rate,
    }


def disabled_judge_summary(out_dir: str) -> Dict[str, Any]:
    summary = {
        "enabled": False,
        "passed": True,
        "gate_status": "pass",
        "decision": "continue",
        "judge_results_path": "",
        "num_requested": 0,
        "num_judged": 0,
        "pass_count": 0,
        "minor_issue_count": 0,
        "major_failure_count": 0,
        "warning_count": 0,
        "quality_score": 0.0,
        "pass_rate": 0.0,
        "minor_issue_rate": 0.0,
        "major_failure_rate": 0.0,
        "warning_rate": 0.0,
        "failure_category_counts": {},
        "grounding_counts": {},
        "unsupported_grounding_count": 0,
        "unsupported_grounding_rate": 0.0,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "judge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _norm_verdict(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in PASS_VALUES:
        return "pass"
    if normalized in MINOR_VALUES:
        return "minor_issue"
    if normalized in MAJOR_VALUES:
        return "major_failure"
    return "minor_issue"


def _norm_grounding(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in GROUNDING_VALUES else "insufficient_source"


def _categories(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []
