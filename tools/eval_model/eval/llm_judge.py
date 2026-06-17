"""Structured LLM judge for validation predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from core.llm import LLMClient, LLMRequest


PASS_VALUES = {"pass", "ok", "good", "none"}
WARN_VALUES = {"warn", "minor", "partial"}
FAIL_VALUES = {"fail", "major", "bad"}


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for supervised fine-tuning validation outputs.
Return only valid JSON. Do not reward verbosity. Judge whether the model answer satisfies
the prompt and is semantically consistent with the reference answer."""


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
                    "major_failure": True,
                    "failure_categories": ["judge_error"],
                    "rationale": response.error or "judge_error",
                    "raw": response.data,
                }
            judged["input"] = source.get("input")
            judged["prediction"] = source.get("prediction")
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
    return f"""
Evaluate this validation example.

Modality: {modality}
Target language: {target_language or "unspecified"}
{image_line}
User prompt:
{row.get("input", "")}

Model answer:
{row.get("prediction", "")}

Reference answer:
{row.get("reference", "")}

Return JSON with exactly these keys:
{{
  "instruction_following": "pass|warn|fail",
  "semantic_correctness": "pass|warn|fail",
  "groundedness": "pass|warn|fail",
  "completeness": "pass|warn|fail",
  "language_quality": "pass|warn|fail",
  "hallucination_risk": "none|minor|major",
  "failure_categories": ["formatting|language|missing_knowledge|grounding|hallucination|incomplete|irrelevant|other"],
  "rationale": "one concise sentence"
}}
""".strip()


def _normalize_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    dimensions = {
        key: _norm_label(payload.get(key))
        for key in (
            "instruction_following",
            "semantic_correctness",
            "groundedness",
            "completeness",
            "language_quality",
        )
    }
    hallucination = _norm_hallucination(payload.get("hallucination_risk"))
    categories = [str(item) for item in payload.get("failure_categories") or []]
    major_failure = hallucination == "major" or any(value == "fail" for value in dimensions.values())
    warning = hallucination == "minor" or any(value == "warn" for value in dimensions.values())
    return {
        **dimensions,
        "hallucination_risk": hallucination,
        "major_failure": major_failure,
        "warning": warning and not major_failure,
        "failure_categories": categories,
        "rationale": str(payload.get("rationale") or ""),
        "raw": payload,
    }


def _aggregate(rows: list[dict[str, Any]], *, num_requested: int, judge_results_path: str) -> Dict[str, Any]:
    num_judged = len(rows)
    major = sum(1 for row in rows if row.get("major_failure"))
    warnings = sum(1 for row in rows if row.get("warning"))
    category_counts: dict[str, int] = {}
    for row in rows:
        for category in row.get("failure_categories") or []:
            category_counts[str(category)] = category_counts.get(str(category), 0) + 1
    major_rate = major / num_judged if num_judged else 0.0
    warning_rate = warnings / num_judged if num_judged else 0.0
    if major_rate > 0.2:
        gate_status = "repair"
    elif major_rate > 0.1 or warning_rate > 0.3:
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
        "major_failure_count": major,
        "warning_count": warnings,
        "major_failure_rate": major_rate,
        "warning_rate": warning_rate,
        "failure_category_counts": category_counts,
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
        "major_failure_count": 0,
        "warning_count": 0,
        "major_failure_rate": 0.0,
        "warning_rate": 0.0,
        "failure_category_counts": {},
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "judge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _norm_label(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in PASS_VALUES:
        return "pass"
    if normalized in WARN_VALUES:
        return "warn"
    if normalized in FAIL_VALUES:
        return "fail"
    return "warn"


def _norm_hallucination(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"none", "no", "pass"}:
        return "none"
    if normalized in {"minor", "warn"}:
        return "minor"
    if normalized in {"major", "fail"}:
        return "major"
    return "minor"
