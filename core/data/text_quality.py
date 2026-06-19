"""Text quality diagnostics for collected text and SFT JSONL artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def records_from_serper_raw(raw_path: str | Path) -> list[dict[str, Any]]:
    path = Path(raw_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return records
    for query, pages in payload.items():
        if not isinstance(pages, list):
            continue
        for idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            text = str(page.get("full_text") or page.get("text") or "").strip()
            records.append(
                {
                    "id": f"{query}:{idx}",
                    "source": "collection",
                    "query": str(query),
                    "url": page.get("url"),
                    "text": text,
                }
            )
    return records


def records_from_sft_jsonl(sft_path: str | Path) -> list[dict[str, Any]]:
    path = Path(sft_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _messages_text(row.get("messages")) or _fallback_row_text(row)
            records.append(
                {
                    "id": str(idx),
                    "source": "sft",
                    "url": row.get("url"),
                    "text": text,
                }
            )
    return records


def write_text_quality_report(
    records: Iterable[dict[str, Any]],
    out_path: str | Path,
    *,
    source: str,
    enable_embeddings: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_threshold: float = 0.93,
    max_embedding_items: int = 256,
    shingle_threshold: float = 0.85,
    max_shingle_items: int = 1000,
    max_reported_pairs: int = 50,
) -> dict[str, Any]:
    report = analyze_text_quality(
        list(records),
        source=source,
        enable_embeddings=enable_embeddings,
        embedding_model=embedding_model,
        embedding_threshold=embedding_threshold,
        max_embedding_items=max_embedding_items,
        shingle_threshold=shingle_threshold,
        max_shingle_items=max_shingle_items,
        max_reported_pairs=max_reported_pairs,
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def analyze_text_quality(
    records: list[dict[str, Any]],
    *,
    source: str,
    enable_embeddings: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_threshold: float = 0.93,
    max_embedding_items: int = 256,
    shingle_threshold: float = 0.85,
    max_shingle_items: int = 1000,
    max_reported_pairs: int = 50,
) -> dict[str, Any]:
    normalized = [_normalize(record.get("text")) for record in records]
    non_empty = [text for text in normalized if text]
    exact_pairs = _exact_duplicate_pairs(records, normalized, max_reported_pairs=max_reported_pairs)
    shingle = _shingle_duplicate_report(
        records,
        normalized,
        threshold=shingle_threshold,
        max_items=max_shingle_items,
        max_reported_pairs=max_reported_pairs,
    )
    embedding = _embedding_duplicate_report(
        records,
        normalized,
        enabled=enable_embeddings,
        model_id=embedding_model,
        threshold=embedding_threshold,
        max_items=max_embedding_items,
        max_reported_pairs=max_reported_pairs,
    )
    return {
        "schema_version": "text_quality.v1",
        "source": source,
        "num_records": len(records),
        "num_non_empty": len(non_empty),
        "length_chars": _length_stats([len(text) for text in non_empty]),
        "exact_duplicate_rate": _duplicate_rate(non_empty),
        "exact_duplicate_count": _duplicate_count(non_empty),
        "exact_duplicate_pairs": exact_pairs,
        "url_duplicate_rate": _duplicate_rate([_canonical_url(record.get("url")) for record in records]),
        "url_duplicate_count": _duplicate_count([_canonical_url(record.get("url")) for record in records]),
        "domain_distribution": _domain_distribution(records),
        "script_distribution": _script_distribution(non_empty),
        "shingle_near_duplicate": shingle,
        "embedding_near_duplicate": embedding,
    }


def summarize_text_quality(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    shingle = report.get("shingle_near_duplicate") if isinstance(report.get("shingle_near_duplicate"), dict) else {}
    embedding = (
        report.get("embedding_near_duplicate")
        if isinstance(report.get("embedding_near_duplicate"), dict)
        else {}
    )
    return {
        "num_records": report.get("num_records"),
        "num_non_empty": report.get("num_non_empty"),
        "exact_duplicate_rate": report.get("exact_duplicate_rate"),
        "exact_duplicate_count": report.get("exact_duplicate_count"),
        "url_duplicate_rate": report.get("url_duplicate_rate"),
        "url_duplicate_count": report.get("url_duplicate_count"),
        "shingle_pair_count": shingle.get("pair_count"),
        "shingle_num_compared": shingle.get("num_compared"),
        "embedding_enabled": embedding.get("enabled"),
        "embedding_pair_count": embedding.get("pair_count"),
        "embedding_num_embedded": embedding.get("num_embedded"),
        "embedding_model": embedding.get("model"),
        "embedding_error": embedding.get("error"),
    }


def _messages_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or "unknown"
        parts.append(f"{role}: {_content_text(message.get('content'))}")
    return "\n".join(part for part in parts if part.strip())


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("text"):
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part.strip())
    return str(content or "")


def _fallback_row_text(row: dict[str, Any]) -> str:
    parts = [row.get(key) for key in ("prompt", "instruction", "response", "output", "text")]
    return "\n".join(str(part) for part in parts if part)


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _duplicate_rate(values: Iterable[str]) -> float:
    cleaned = [value for value in values if value]
    if not cleaned:
        return 0.0
    return _duplicate_count(cleaned) / len(cleaned)


def _duplicate_count(values: Iterable[str]) -> int:
    cleaned = [value for value in values if value]
    return len(cleaned) - len(set(cleaned))


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
    }


def _canonical_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().lstrip("www.")
    path = parsed.path.rstrip("/")
    return urlunparse(("", host, path, "", "", ""))


def _domain(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    host = (urlparse(raw).hostname or "").lower().lstrip("www.")
    return host


def _domain_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_domain(record.get("url")) for record in records)
    counts.pop("", None)
    return dict(counts.most_common(30))


def _script_distribution(texts: list[str]) -> dict[str, float]:
    counts = Counter(_dominant_script(text) for text in texts if text)
    total = sum(counts.values())
    if not total:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def _dominant_script(text: str) -> str:
    latin = 0
    cyrillic = 0
    other_letters = 0
    for char in text:
        code = ord(char)
        if "a" <= char <= "z":
            latin += 1
        elif 0x0400 <= code <= 0x052F:
            cyrillic += 1
        elif char.isalpha():
            other_letters += 1
    counts = {"latin": latin, "cyrillic": cyrillic, "other": other_letters}
    label, count = max(counts.items(), key=lambda item: item[1])
    return label if count else "unknown"


def _exact_duplicate_pairs(
    records: list[dict[str, Any]],
    normalized: list[str],
    *,
    max_reported_pairs: int,
) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    for idx, text in enumerate(normalized):
        if not text:
            continue
        if text in seen and len(pairs) < max_reported_pairs:
            pairs.append(_pair(records, seen[text], idx, score=1.0))
        else:
            seen[text] = idx
    return pairs


def _shingle_duplicate_report(
    records: list[dict[str, Any]],
    normalized: list[str],
    *,
    threshold: float,
    max_items: int,
    max_reported_pairs: int,
) -> dict[str, Any]:
    candidate_indexes = [idx for idx, text in enumerate(normalized) if text][:max(0, max_items)]
    shingles = [(idx, _word_shingles(normalized[idx])) for idx in candidate_indexes]
    pairs: list[dict[str, Any]] = []
    pair_count = 0
    for i in range(len(shingles)):
        left_idx, left_shingles = shingles[i]
        if not left_shingles:
            continue
        for j in range(i + 1, len(shingles)):
            right_idx, right_shingles = shingles[j]
            if not right_shingles:
                continue
            score = _jaccard(left_shingles, right_shingles)
            if score >= threshold:
                pair_count += 1
                if len(pairs) < max_reported_pairs:
                    pairs.append(_pair(records, left_idx, right_idx, score=score))
    return {
        "threshold": threshold,
        "max_items": max_items,
        "num_compared": len(shingles),
        "pair_count": pair_count,
        "pairs": pairs,
    }


def _word_shingles(text: str, *, n: int = 5) -> set[str]:
    tokens = re.findall(r"[\w']+", text, flags=re.UNICODE)
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _embedding_duplicate_report(
    records: list[dict[str, Any]],
    normalized: list[str],
    *,
    enabled: bool,
    model_id: str,
    threshold: float,
    max_items: int,
    max_reported_pairs: int,
) -> dict[str, Any]:
    report = {
        "enabled": enabled,
        "model": model_id,
        "threshold": threshold,
        "max_items": max_items,
        "num_embedded": 0,
        "pair_count": 0,
        "pairs": [],
    }
    if not enabled:
        return report
    candidates = [(idx, text) for idx, text in enumerate(normalized) if text][: max(0, max_items)]
    if len(candidates) < 2:
        return report
    try:
        embeddings = _embed_texts([text for _, text in candidates], model_id=model_id)
        pairs: list[dict[str, Any]] = []
        pair_count = 0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                score = _cosine(embeddings[i], embeddings[j])
                if score >= threshold:
                    pair_count += 1
                    if len(pairs) < max_reported_pairs:
                        pairs.append(_pair(records, candidates[i][0], candidates[j][0], score=score))
        report.update({"num_embedded": len(candidates), "pair_count": pair_count, "pairs": pairs})
    except Exception as exc:  # pragma: no cover - depends on optional model downloads/devices
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _embed_texts(texts: list[str], *, model_id: str) -> list[list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(device)
    model.eval()
    vectors: list[list[float]] = []
    batch_size = 16 if device == "cuda" else 4
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.extend(pooled.cpu().tolist())
    return vectors


def _last_token_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[range(batch_size), sequence_lengths]


def _cosine(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _pair(records: list[dict[str, Any]], left_idx: int, right_idx: int, *, score: float) -> dict[str, Any]:
    return {
        "left_id": records[left_idx].get("id", str(left_idx)),
        "right_id": records[right_idx].get("id", str(right_idx)),
        "score": round(float(score), 4) if math.isfinite(float(score)) else 0.0,
        "left_url": records[left_idx].get("url"),
        "right_url": records[right_idx].get("url"),
    }
