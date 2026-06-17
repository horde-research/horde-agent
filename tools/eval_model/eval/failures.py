"""Failure heuristics for predictions.

Copied from `agentic_train_pipeline/eval/failures.py` and adjusted for new package layout.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _has_repetition(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return True
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    return unique_ratio < 0.3


def collect_failures(predictions_path: str, out_dir: str) -> str:
    failures_path, _ = collect_failures_with_metrics(predictions_path, out_dir)
    return failures_path


def collect_failures_with_metrics(predictions_path: str, out_dir: str) -> tuple[str, Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    total = 0
    similarities: list[float] = []
    prediction_lengths: list[int] = []
    reason_counts: dict[str, int] = {}
    with open(predictions_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            pred = str(row.get("prediction", ""))
            ref = row.get("reference")
            pred_norm = _normalize(pred)
            prediction_lengths.append(len(pred_norm))

            failed = False
            reasons = []

            if row.get("error"):
                failed = True
                reasons.append(str(row["error"]).split(":", 1)[0])
            if not pred_norm:
                failed = True
                reasons.append("empty_output")
            if _has_repetition(pred_norm):
                failed = True
                reasons.append("repetition")
            if row.get("input") and str(row.get("input")).strip() and pred_norm.startswith(_normalize(str(row["input"]))):
                failed = True
                reasons.append("prompt_echo")

            if ref:
                ref_norm = _normalize(str(ref))
                sim = _similarity(pred_norm, ref_norm)
                similarities.append(sim)
                if sim < 0.5:
                    failed = True
                    reasons.append(f"low_similarity_{sim:.2f}")
                if pred_norm == ref_norm:
                    reasons.append("exact_match")
            else:
                if len(pred_norm) < 5:
                    failed = True
                    reasons.append("too_short")

            if failed:
                row["label"] = _label_from_reasons(reasons)
                row["reasons"] = reasons
                failures.append(row)
                for reason in reasons:
                    reason_key = reason.split("_", 2)[0] if reason.startswith("low_similarity") else reason
                    reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1

    out_path = Path(out_dir) / "failures.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    failure_rate = len(failures) / total if total else 0.0
    metrics = {
        "num_predictions": total,
        "num_failures": len(failures),
        "failure_rate": failure_rate,
        "avg_similarity": sum(similarities) / len(similarities) if similarities else None,
        "avg_prediction_chars": sum(prediction_lengths) / len(prediction_lengths) if prediction_lengths else 0.0,
        "failure_reason_counts": reason_counts,
    }
    return str(out_path), metrics


def _label_from_reasons(reasons: list[str]) -> str:
    joined = " ".join(reasons)
    if "missing_image" in joined or "image_eval_failed" in joined:
        return "image_processing_failure"
    if "empty_output" in joined or "too_short" in joined:
        return "generation_empty_or_short"
    if "repetition" in joined or "prompt_echo" in joined:
        return "generation_repetition"
    if "low_similarity" in joined:
        return "semantic_mismatch"
    return "other_failure"
