"""Embedding-based answerability checks for generated text SFT rows."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from core.data.source_page_types import detect_page_type_flags, hard_drop_page_type_flags
from core.data.text_quality import DEFAULT_EMBEDDING_MODEL, _cosine, _embed_texts
from core.redaction import sanitize_secret_text

EmbeddingFn = Callable[..., list[list[float]]]
NUMBER_RE = re.compile(r"\b\d[\d,./:-]*\b")


def filter_text_sft_examples_by_answerability(
    examples: list[dict[str, Any]],
    *,
    out_path: str | Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    min_answer_source_similarity: float = 0.35,
    min_question_source_similarity: float = 0.20,
    borderline_answer_source_similarity: float = 0.45,
    max_examples: int = 0,
    drop_low_value_pages: bool = True,
    drop_document_wrappers: bool = True,
    max_reported_rows: int = 100,
    embedding_fn: EmbeddingFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter text SFT examples that are not answerable from their source excerpt.

    The function is deterministic except for the supplied embedding model. If
    embedding inference fails, examples are left unchanged and the report records
    the error so the pipeline can continue in environments without the model.
    """

    candidate_indexes = [
        idx
        for idx, example in enumerate(examples)
        if max_examples <= 0 or idx < max_examples
    ]
    if not examples:
        report = _report([], num_examples=0, model=embedding_model, error=None)
        _maybe_write(out_path, report)
        return [], report

    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for idx in candidate_indexes:
        example = examples[idx]
        question = _user_text(example.get("messages"))
        answer = _assistant_text(example.get("messages"))
        evidence = _evidence_text(example)
        row = {
            "idx": idx,
            "source_url": example.get("source_url"),
            "group_key": example.get("group_key"),
            "source_id": example.get("source_id"),
            "question": question[:500],
            "answer": answer[:500],
            "evidence": evidence[:500],
            "page_type_flags": detect_page_type_flags(example.get("source_url"), evidence),
            "unsupported_numbers": sorted(_numbers(answer) - _numbers(evidence)),
            "question_source_similarity": None,
            "answer_source_similarity": None,
            "decision": "keep",
            "reasons": [],
        }
        rows.append(row)
        texts.extend([question, answer, evidence])

    try:
        embed = embedding_fn or _embed_texts
        embeddings = embed(texts, model_id=embedding_model)
    except Exception as exc:  # pragma: no cover - depends on optional model downloads/devices
        error = sanitize_secret_text(f"{type(exc).__name__}: {exc}")
        report = _report(rows, num_examples=len(examples), model=embedding_model, error=error)
        _maybe_write(out_path, report)
        return examples, report

    for pos, row in enumerate(rows):
        q_vec, a_vec, evidence_vec = embeddings[pos * 3 : pos * 3 + 3]
        question_source = _cosine(q_vec, evidence_vec)
        answer_source = _cosine(a_vec, evidence_vec)
        row["question_source_similarity"] = round(question_source, 4)
        row["answer_source_similarity"] = round(answer_source, 4)
        _decide_row(
            row,
            min_answer_source_similarity=min_answer_source_similarity,
            min_question_source_similarity=min_question_source_similarity,
            borderline_answer_source_similarity=borderline_answer_source_similarity,
            drop_low_value_pages=drop_low_value_pages,
            drop_document_wrappers=drop_document_wrappers,
        )

    dropped = {int(row["idx"]) for row in rows if row["decision"] == "drop"}
    kept_examples = [example for idx, example in enumerate(examples) if idx not in dropped]
    report = _report(rows, num_examples=len(examples), model=embedding_model, error=None, max_reported_rows=max_reported_rows)
    _maybe_write(out_path, report)
    return kept_examples, report


def _decide_row(
    row: dict[str, Any],
    *,
    min_answer_source_similarity: float,
    min_question_source_similarity: float,
    borderline_answer_source_similarity: float,
    drop_low_value_pages: bool,
    drop_document_wrappers: bool,
) -> None:
    evidence = str(row.get("evidence") or "").strip()
    reasons: list[str] = []
    if not evidence:
        reasons.append("answerability_evidence_missing")
    hard_flags = hard_drop_page_type_flags(
        [str(flag) for flag in row.get("page_type_flags") or []],
        drop_document_wrappers=drop_document_wrappers,
    )
    if drop_low_value_pages and hard_flags:
        reasons.extend(f"page_type:{flag}" for flag in hard_flags)

    answer_source = float(row.get("answer_source_similarity") or 0.0)
    question_source = float(row.get("question_source_similarity") or 0.0)
    if answer_source < min_answer_source_similarity:
        reasons.append("answer_source_similarity_below_minimum")
    elif answer_source < borderline_answer_source_similarity and row.get("unsupported_numbers"):
        reasons.append("answer_source_similarity_borderline_with_unsupported_numbers")
    if answer_source < borderline_answer_source_similarity and question_source < min_question_source_similarity:
        reasons.append("question_and_answer_source_similarity_low")

    row["reasons"] = reasons
    row["decision"] = "drop" if reasons else "keep"


def _report(
    rows: list[dict[str, Any]],
    *,
    num_examples: int,
    model: str,
    error: str | None,
    max_reported_rows: int = 100,
) -> dict[str, Any]:
    decisions = Counter(str(row.get("decision") or "unknown") for row in rows)
    reason_counts: Counter[str] = Counter()
    page_type_counts: Counter[str] = Counter()
    answer_scores: list[float] = []
    question_scores: list[float] = []
    for row in rows:
        reason_counts.update(str(reason) for reason in row.get("reasons") or [])
        page_type_counts.update(str(flag) for flag in row.get("page_type_flags") or [])
        if row.get("answer_source_similarity") is not None:
            answer_scores.append(float(row["answer_source_similarity"]))
        if row.get("question_source_similarity") is not None:
            question_scores.append(float(row["question_source_similarity"]))
    dropped = int(decisions.get("drop", 0))
    return {
        "schema_version": "sft_answerability.v1",
        "enabled": True,
        "embedding_model": model,
        "embedding_error": error,
        "num_examples": num_examples,
        "num_scored_examples": len(rows),
        "num_kept_examples": num_examples - dropped,
        "num_dropped_examples": dropped,
        "drop_rate": (dropped / num_examples) if num_examples else 0.0,
        "decision_counts": dict(decisions),
        "reason_counts": dict(reason_counts.most_common()),
        "page_type_flag_counts": dict(page_type_counts.most_common()),
        "avg_answer_source_similarity": _avg(answer_scores),
        "avg_question_source_similarity": _avg(question_scores),
        "rows": rows[: max(0, max_reported_rows)],
    }


def _maybe_write(path: str | Path | None, report: Mapping[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _user_text(messages: Any) -> str:
    return _messages_by_role(messages, "user")


def _assistant_text(messages: Any) -> str:
    return _messages_by_role(messages, "assistant")


def _messages_by_role(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == role:
            parts.append(_content_text(message.get("content")))
    return " ".join(part for part in parts if part).strip()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return " ".join(content.split())
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("text"):
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return " ".join(" ".join(parts).split())
    return str(content or "").strip()


def _evidence_text(example: Mapping[str, Any]) -> str:
    return " ".join(
        str(example.get(key) or "").strip()
        for key in ("source_excerpt", "source_text")
        if str(example.get(key) or "").strip()
    )[:2000]


def _numbers(text: str) -> set[str]:
    return {match.group(0) for match in NUMBER_RE.finditer(text or "")}


def _avg(values: Iterable[float]) -> float | None:
    value_list = [float(value) for value in values]
    if not value_list:
        return None
    return sum(value_list) / len(value_list)
