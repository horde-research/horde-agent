"""Coverage assessment between collection and SFT generation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

from core.agentic.action_space import ActionType
from core.agentic.models import PipelineState, QualityReport
from tools.generate_taxonomy.image_taxonomy import flatten_image_query_specs


def assess_coverage_and_refine_queries(state: PipelineState) -> dict[str, Any]:
    """Inspect collection output and return a bounded repair plan if needed."""
    cfg = state.config
    artifacts = state.artifacts
    metadata = artifacts.get("collection_metadata") or {}
    search_queries = _strings(artifacts.get("search_queries"))
    raw_counts = _query_page_counts(metadata.get("raw_result_path"))
    weak_text_queries = _weak_text_queries(raw_counts)
    num_samples = _int(artifacts.get("num_samples"), 0)

    blocking_issues: list[str] = []
    warnings: list[str] = []
    recommended_actions: list[str] = []
    suggested_adjustments: dict[str, Any] = {}

    min_text_samples = _int(cfg.get("coverage_min_text_samples"), 1)
    min_samples_per_query = _float(cfg.get("coverage_min_samples_per_query"), 0.0)
    sample_density = num_samples / len(search_queries) if search_queries else None

    if num_samples < min_text_samples:
        blocking_issues.append("coverage_text_samples_below_minimum")
        recommended_actions.append("refine_text_search_queries")
    if sample_density is not None and min_samples_per_query > 0 and sample_density < min_samples_per_query:
        blocking_issues.append("coverage_text_density_below_target")
        recommended_actions.append("add_targeted_text_queries")
    if weak_text_queries:
        warnings.append("coverage_some_queries_returned_no_pages")

    country = str(cfg.get("country") or "")
    added_queries = _build_text_repair_queries(
        weak_text_queries or search_queries[:3],
        country=country,
        limit=_int(cfg.get("coverage_max_added_queries"), 12),
        existing={query.lower() for query in search_queries},
    )
    repair_queries = added_queries if blocking_issues else []
    candidate_queries = added_queries if warnings and not blocking_issues else []
    if repair_queries:
        suggested_adjustments["coverage_added_queries"] = repair_queries

    image_review = _assess_image_coverage(state)
    blocking_issues.extend(image_review["blocking_issues"])
    warnings.extend(image_review["warnings"])
    recommended_actions.extend(image_review["recommended_actions"])
    if image_review["image_query_specs"]:
        suggested_adjustments["image_query_specs"] = image_review["image_query_specs"]
    if image_review["image_query_specs"] or image_review["blocking_issues"]:
        suggested_adjustments["image_search_results_per_query"] = "increase"

    gate_status = "repair" if blocking_issues else ("warn" if warnings else "pass")
    report = QualityReport(
        stage=ActionType.ASSESS_COVERAGE_AND_REFINE_QUERIES,
        passed=not blocking_issues,
        gate_status=gate_status,
        decision="repair" if blocking_issues else "continue",
        score=1.0 if not blocking_issues else 0.0,
        recoverable=True,
        issue_categories=_issue_categories(blocking_issues, warnings),
        blocking_issues=blocking_issues,
        warnings=warnings,
        metrics={
            "num_samples": num_samples,
            "num_search_queries": len(search_queries),
            "sample_density": sample_density,
            "num_weak_text_queries": len(weak_text_queries),
            **image_review["metrics"],
        },
        recommended_actions=sorted(set(recommended_actions)),
        suggested_adjustments=suggested_adjustments,
    )

    return {
        "report": report,
        "coverage_review": {
            "passed": report.passed,
            "gate_status": report.gate_status,
            "rationale": _rationale(report, repair_queries, candidate_queries, image_review),
            "weak_text_queries": weak_text_queries,
            "added_queries": repair_queries,
            "candidate_text_queries": candidate_queries,
            "weak_image_slots": image_review["weak_slots"],
            "image_query_specs": image_review["image_query_specs"],
            "metrics": report.metrics,
            "recommended_actions": report.recommended_actions,
        },
        "coverage_added_queries": repair_queries,
        "image_query_specs": image_review["image_query_specs"],
    }


def _assess_image_coverage(state: PipelineState) -> dict[str, Any]:
    cfg = state.config
    artifacts = state.artifacts
    metadata = artifacts.get("collection_metadata") or {}
    if not _as_bool(cfg.get("collect_images", False)):
        return _image_result()

    image_taxonomy = artifacts.get("image_taxonomy") or cfg.get("image_taxonomy") or {}
    slots = _taxonomy_slots(image_taxonomy)
    slot_ids = [str(slot["slot_id"]) for slot in slots]
    image_records = _read_image_records(artifacts.get("images_index") or metadata.get("images_index"))
    slot_counts = Counter(str(record.get("slot_id") or "") for record in image_records if isinstance(record, dict))
    covered_slots = {slot_id for slot_id in slot_ids if slot_counts.get(slot_id, 0) > 0}
    expected_slots = len(slot_ids) or _int(metadata.get("num_image_taxonomy_slots"), 0)
    covered_count = len(covered_slots) if slot_ids else 0
    slot_ratio = covered_count / expected_slots if expected_slots else None
    min_slot_ratio = _float(cfg.get("coverage_min_image_slot_ratio"), 0.0)
    min_images_per_slot = _int(cfg.get("coverage_min_images_per_slot"), 1)

    weak_slots = _weak_image_slots(slots, slot_counts, min_images_per_slot)

    blocking_issues: list[str] = []
    warnings: list[str] = []
    recommended_actions: list[str] = []
    if expected_slots and min_slot_ratio > 0 and (slot_ratio or 0.0) < min_slot_ratio:
        blocking_issues.append("coverage_image_slot_ratio_below_target")
        recommended_actions.append("collect_images_for_weak_taxonomy_slots")
    elif weak_slots:
        warnings.append("coverage_image_slots_underrepresented")

    image_specs = _image_specs_for_weak_slots(
        image_taxonomy,
        weak_slots,
        limit=_int(cfg.get("coverage_max_image_query_specs"), 12),
    )
    return _image_result(
        blocking_issues=blocking_issues,
        warnings=warnings,
        recommended_actions=recommended_actions,
        weak_slots=weak_slots,
        image_query_specs=image_specs,
        metrics={
            "num_images": _int(artifacts.get("num_images") or metadata.get("num_images"), len(image_records)),
            "num_image_records": len(image_records),
            "num_image_slots": expected_slots,
            "num_covered_image_slots": covered_count,
            "image_slot_coverage_ratio": slot_ratio,
            "num_weak_image_slots": len(weak_slots),
        },
    )


def _image_result(
    *,
    blocking_issues: list[str] | None = None,
    warnings: list[str] | None = None,
    recommended_actions: list[str] | None = None,
    weak_slots: list[dict[str, Any]] | None = None,
    image_query_specs: list[dict[str, str]] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "blocking_issues": blocking_issues or [],
        "warnings": warnings or [],
        "recommended_actions": recommended_actions or [],
        "weak_slots": weak_slots or [],
        "image_query_specs": image_query_specs or [],
        "metrics": metrics or {
            "num_images": None,
            "num_image_records": 0,
            "num_image_slots": 0,
            "num_covered_image_slots": 0,
            "image_slot_coverage_ratio": None,
            "num_weak_image_slots": 0,
        },
    }


def _taxonomy_slots(image_taxonomy: Any) -> list[dict[str, Any]]:
    if not isinstance(image_taxonomy, dict):
        return []
    slots = image_taxonomy.get("slots")
    if not isinstance(slots, list):
        return []
    return [slot for slot in slots if isinstance(slot, dict) and slot.get("slot_id")]


def _weak_image_slots(
    slots: list[dict[str, Any]],
    slot_counts: Counter[str],
    min_images_per_slot: int,
) -> list[dict[str, Any]]:
    weak_slots: list[dict[str, Any]] = []
    for slot in slots:
        slot_id = str(slot.get("slot_id") or "")
        count = int(slot_counts.get(slot_id, 0))
        if count < min_images_per_slot:
            weak_slots.append(
                {
                    "slot_id": slot_id,
                    "domain_label": slot.get("domain_label"),
                    "subdomain_label": slot.get("subdomain_label"),
                    "num_images": count,
                }
            )
    return weak_slots


def _query_page_counts(raw_result_path: Any) -> dict[str, int]:
    if not raw_result_path:
        return {}
    path = Path(str(raw_result_path))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    counts: dict[str, int] = {}
    for query, pages in payload.items():
        counts[str(query)] = len(pages) if isinstance(pages, list) else 0
    return counts


def _weak_text_queries(query_page_counts: Mapping[str, int]) -> list[str]:
    return [query for query, count in query_page_counts.items() if int(count or 0) <= 0]


def _build_text_repair_queries(
    seed_queries: list[str],
    *,
    country: str,
    limit: int,
    existing: set[str],
) -> list[str]:
    if limit <= 0:
        return []
    additions: list[str] = []
    added_keys: set[str] = set()
    suffixes = ["culture source", "history overview", "traditional practice", "local context"]
    for seed in seed_queries:
        base = " ".join(str(seed).split())
        if not base:
            continue
        for suffix in suffixes:
            query = f"{base} {suffix}".strip()
            if country and country.lower() not in query.lower():
                query = f"{country} {query}".strip()
            key = query.lower()
            if key in existing or key in added_keys:
                continue
            added_keys.add(key)
            additions.append(query)
            if len(additions) >= limit:
                return additions
    return additions


def _image_specs_for_weak_slots(
    image_taxonomy: Any,
    weak_slots: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, str]]:
    if limit <= 0 or not isinstance(image_taxonomy, dict) or not weak_slots:
        return []
    weak_ids = {str(slot.get("slot_id") or "") for slot in weak_slots}
    specs = [
        spec
        for spec in flatten_image_query_specs(image_taxonomy)
        if str(spec.get("slot_id") or "") in weak_ids
    ]
    return specs[:limit]


def _read_image_records(images_index: Any) -> list[dict[str, Any]]:
    if not images_index:
        return []
    path = Path(str(images_index))
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("images", []) if isinstance(payload, dict) else payload
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _issue_categories(blocking_issues: list[str], warnings: list[str]) -> list[str]:
    categories: set[str] = set()
    for issue in [*blocking_issues, *warnings]:
        if "image" in issue:
            categories.add("image_coverage")
        elif "query" in issue:
            categories.add("query_quality")
        elif "coverage" in issue or "sample" in issue:
            categories.add("missing_coverage")
        else:
            categories.add("other")
    return sorted(categories)


def _rationale(
    report: QualityReport,
    repair_queries: list[str],
    candidate_queries: list[str],
    image_review: dict[str, Any],
) -> str:
    if report.passed and not report.warnings:
        return "Collected data passed coverage checks; no query refinement is needed."
    parts: list[str] = []
    if report.blocking_issues:
        parts.append("Coverage needs repair: " + ", ".join(report.blocking_issues) + ".")
    if report.warnings:
        parts.append("Warnings: " + ", ".join(report.warnings) + ".")
    if repair_queries:
        parts.append(f"Prepared {len(repair_queries)} targeted text queries.")
    elif candidate_queries:
        parts.append(f"Found {len(candidate_queries)} candidate text queries for manual follow-up.")
    if image_review["image_query_specs"]:
        parts.append(f"Prepared {len(image_review['image_query_specs'])} targeted image query specs.")
    return " ".join(parts)
