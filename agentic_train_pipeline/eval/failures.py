"""Failure heuristics for predictions."""

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
    failures: List[Dict[str, Any]] = []
    with open(predictions_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pred = str(row.get("prediction", ""))
            ref = row.get("reference")
            pred_norm = _normalize(pred)

            failed = False
            reasons = []

            if not pred_norm:
                failed = True
                reasons.append("empty_output")
            if _has_repetition(pred_norm):
                failed = True
                reasons.append("repetition")

            if ref:
                ref_norm = _normalize(str(ref))
                sim = _similarity(pred_norm, ref_norm)
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
                row["reasons"] = reasons
                failures.append(row)

    out_path = Path(out_dir) / "failures.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(str(json.dumps(row)) + "
")
    return str(out_path)
