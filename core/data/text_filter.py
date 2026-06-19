"""Collection-time text filtering for low-value and duplicate web pages."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse


DEFAULT_MIN_CHARS = 300
DEFAULT_MIN_WORDS = 40
DEFAULT_MIN_UNIQUE_WORD_RATIO = 0.15
DEFAULT_SHINGLE_THRESHOLD = 0.90
DEFAULT_MAX_NEAR_DUPLICATE_ITEMS = 1000
DEFAULT_MAX_REPORTED_ROWS = 50


def filter_text_rows(
    rows: Iterable[dict[str, Any]],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_words: int = DEFAULT_MIN_WORDS,
    min_unique_word_ratio: float = DEFAULT_MIN_UNIQUE_WORD_RATIO,
    shingle_threshold: float = DEFAULT_SHINGLE_THRESHOLD,
    max_near_duplicate_items: int = DEFAULT_MAX_NEAR_DUPLICATE_ITEMS,
    max_reported_rows: int = DEFAULT_MAX_REPORTED_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop rows that are too short, repetitive, exact duplicates, or near-duplicates.

    Filtering keeps the first acceptable row for each URL/text/near-duplicate cluster
    and records concise removal reasons for later recovery decisions.
    """
    input_rows = list(rows)
    kept: list[dict[str, Any]] = []
    removed_examples: list[dict[str, Any]] = []
    removed_reason_counts: Counter[str] = Counter()
    seen_urls: dict[str, str] = {}
    seen_texts: dict[str, str] = {}
    kept_shingles: list[tuple[str, set[str]]] = []
    max_near = max(0, int(max_near_duplicate_items))
    max_reported = max(0, int(max_reported_rows))
    min_chars = max(0, int(min_chars))
    min_words = max(0, int(min_words))
    min_unique_word_ratio = max(0.0, float(min_unique_word_ratio))
    shingle_threshold = max(0.0, min(1.0, float(shingle_threshold)))

    for idx, row in enumerate(input_rows):
        text = str(row.get("text") or "").strip()
        normalized = _normalize(text)
        tokens = _tokens(normalized)
        row_id = _row_id(row, idx)
        source_url = str(row.get("source_url") or row.get("url") or "").strip()
        canonical_url = _canonical_url(source_url)
        unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
        reason = ""
        duplicate_of = ""
        score: float | None = None

        if not normalized:
            reason = "empty_text"
        elif len(text) < min_chars:
            reason = "too_short_chars"
        elif len(tokens) < min_words:
            reason = "too_few_words"
        elif unique_ratio < min_unique_word_ratio:
            reason = "low_unique_word_ratio"
        elif canonical_url and canonical_url in seen_urls:
            reason = "duplicate_url"
            duplicate_of = seen_urls[canonical_url]
        elif normalized in seen_texts:
            reason = "exact_duplicate_text"
            duplicate_of = seen_texts[normalized]
            score = 1.0
        else:
            shingles = _word_shingles(normalized)
            if len(kept_shingles) < max_near:
                for kept_id, kept_words in kept_shingles:
                    candidate_score = _jaccard(shingles, kept_words)
                    if candidate_score >= shingle_threshold:
                        reason = "near_duplicate_text"
                        duplicate_of = kept_id
                        score = candidate_score
                        break
            if not reason:
                kept.append(dict(row))
                if canonical_url:
                    seen_urls[canonical_url] = row_id
                seen_texts[normalized] = row_id
                if len(kept_shingles) < max_near:
                    kept_shingles.append((row_id, shingles))
                continue

        removed_reason_counts[reason] += 1
        if len(removed_examples) < max_reported:
            removed_examples.append(
                {
                    "id": row_id,
                    "reason": reason,
                    "duplicate_of": duplicate_of or None,
                    "score": round(score, 4) if score is not None else None,
                    "source_url": source_url or None,
                    "chars": len(text),
                    "words": len(tokens),
                    "unique_word_ratio": round(unique_ratio, 4),
                }
            )

    report = {
        "schema_version": "text_filter.v1",
        "enabled": True,
        "thresholds": {
            "min_chars": min_chars,
            "min_words": min_words,
            "min_unique_word_ratio": min_unique_word_ratio,
            "shingle_threshold": shingle_threshold,
            "max_near_duplicate_items": max_near,
        },
        "num_input": len(input_rows),
        "num_kept": len(kept),
        "num_removed": len(input_rows) - len(kept),
        "removal_rate": ((len(input_rows) - len(kept)) / len(input_rows)) if input_rows else 0.0,
        "removed_reason_counts": dict(removed_reason_counts),
        "removed_examples": removed_examples,
        "near_duplicate_comparison_limited": len(kept) > max_near,
    }
    return kept, report


def disabled_text_filter_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    input_rows = list(rows)
    return {
        "schema_version": "text_filter.v1",
        "enabled": False,
        "num_input": len(input_rows),
        "num_kept": len(input_rows),
        "num_removed": 0,
        "removal_rate": 0.0,
        "removed_reason_counts": {},
        "removed_examples": [],
        "near_duplicate_comparison_limited": False,
    }


def summarize_text_filter(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    return {
        "enabled": report.get("enabled"),
        "num_input": report.get("num_input"),
        "num_kept": report.get("num_kept"),
        "num_removed": report.get("num_removed"),
        "removal_rate": report.get("removal_rate"),
        "removed_reason_counts": report.get("removed_reason_counts") or {},
    }


def write_text_filter_report(report: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text, flags=re.UNICODE)


def _word_shingles(text: str, *, n: int = 5) -> set[str]:
    tokens = _tokens(text)
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _canonical_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().lstrip("www.")
    path = parsed.path.rstrip("/")
    return urlunparse(("", host, path, "", "", ""))


def _row_id(row: dict[str, Any], idx: int) -> str:
    for key in ("source_id", "group_key", "source_url", "url", "id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(idx)
