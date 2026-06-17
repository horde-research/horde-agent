"""Quality gates for taxonomy generation and targeted repair."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict, List


DEFAULT_REQUIRED_DIMENSION_TERMS = {
    "food": ["food", "cuisine", "dish", "culinary"],
    "arts": ["art", "music", "dance", "literature", "craft"],
    "customs": ["custom", "ritual", "ceremony", "celebration", "tradition"],
    "language": ["language", "communication", "oral", "dialect"],
    "daily_life": ["daily", "family", "social", "community", "life"],
    "regional": ["region", "geography", "environment", "landscape", "rural", "urban"],
    "modern": ["modern", "contemporary", "trend", "youth", "technology"],
}


def infer_culture_profile(country_or_culture: str) -> Dict[str, Any]:
    normalized = country_or_culture.strip().lower()
    if "kazakhstan" in normalized or "kazakh" in normalized:
        return {
            "country_or_culture": country_or_culture,
            "native_languages": ["Kazakh", "Russian"],
            "native_scripts": ["Cyrillic"],
            "common_aliases": ["Kazakhstan", "Kazakh", "Қазақ", "Қазақстан"],
            "important_regions": ["steppe", "Almaty", "Astana", "Turkistan", "Altai"],
            "query_language_mix": {"english": 0.5, "native": 0.5},
        }
    return {
        "country_or_culture": country_or_culture,
        "native_languages": [],
        "native_scripts": [],
        "common_aliases": [country_or_culture],
        "important_regions": [],
        "query_language_mix": {"english": 1.0, "native": 0.0},
    }


def validate_categories(
    categories: List[Dict[str, str]],
    *,
    min_categories: int = 2,
    max_categories: int = 20,
    required_dimension_terms: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    required_dimension_terms = required_dimension_terms or DEFAULT_REQUIRED_DIMENSION_TERMS
    names = [_clean(item.get("name")) for item in categories]
    descriptions = [_clean(item.get("description")) for item in categories]
    duplicate_rate = _duplicate_rate(names)
    text = " ".join(names + descriptions).lower()
    missing_dimensions = [
        dimension
        for dimension, terms in required_dimension_terms.items()
        if not any(term in text for term in terms)
    ]

    blocking_issues: list[str] = []
    warnings: list[str] = []
    if len(categories) < min_categories:
        blocking_issues.append("category_count_below_minimum")
    if len(categories) > max_categories:
        blocking_issues.append("category_count_above_maximum")
    if any(not name for name in names):
        blocking_issues.append("empty_category_name")
    if any(not description for description in descriptions):
        blocking_issues.append("empty_category_description")
    if duplicate_rate > 0.2:
        blocking_issues.append("category_duplicate_rate_too_high")
    if missing_dimensions:
        warnings.append("missing_recommended_dimensions")

    issue_categories = _issue_categories(blocking_issues, warnings)
    gate_status = _gate_status(blocking_issues, warnings)
    score = _score(blocking_issues, warnings)
    return {
        "passed": not blocking_issues,
        "gate_status": gate_status,
        "decision": _decision_from_gate_status(gate_status),
        "issue_categories": issue_categories,
        "score": score,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "missing_dimensions": missing_dimensions,
        "metrics": {
            "num_categories": len(categories),
            "duplicate_rate": duplicate_rate,
            "missing_dimension_count": len(missing_dimensions),
        },
    }


def validate_subcategories(
    categories: List[Dict[str, str]],
    category_subcategories: Dict[str, List[Dict[str, str]]],
    *,
    min_subcategories_per_category: int = 1,
    max_subcategories_per_category: int = 12,
) -> Dict[str, Any]:
    failed_categories: list[str] = []
    per_category: dict[str, Any] = {}

    for category in categories:
        category_name = category["name"]
        subs = category_subcategories.get(category_name, [])
        names = [_clean(item.get("name")) for item in subs]
        descriptions = [_clean(item.get("description")) for item in subs]
        issues: list[str] = []
        if len(subs) < min_subcategories_per_category:
            issues.append("subcategory_count_below_minimum")
        if len(subs) > max_subcategories_per_category:
            issues.append("subcategory_count_above_maximum")
        if any(not name for name in names):
            issues.append("empty_subcategory_name")
        if any(not description for description in descriptions):
            issues.append("empty_subcategory_description")
        if _duplicate_rate(names) > 0.2:
            issues.append("subcategory_duplicate_rate_too_high")
        if issues:
            failed_categories.append(category_name)
        per_category[category_name] = {
            "passed": not issues,
            "blocking_issues": issues,
            "num_subcategories": len(subs),
        }

    blocking_issues = ["failed_subcategory_groups"] if failed_categories else []
    issue_categories = _issue_categories(blocking_issues, [])
    gate_status = _gate_status(blocking_issues, [])
    return {
        "passed": not failed_categories,
        "gate_status": gate_status,
        "decision": _decision_from_gate_status(gate_status),
        "issue_categories": issue_categories,
        "score": _score(blocking_issues, []),
        "blocking_issues": blocking_issues,
        "warnings": [],
        "failed_categories": failed_categories,
        "per_category": per_category,
        "metrics": {
            "num_categories": len(categories),
            "failed_category_count": len(failed_categories),
            "num_subcategories": sum(len(items) for items in category_subcategories.values()),
        },
    }


def validate_query_groups(
    categories: List[Dict[str, str]],
    category_subcategories: Dict[str, List[Dict[str, str]]],
    category_subcategory_queries: Dict[str, Dict[str, List[str]]],
    *,
    min_queries_per_subcategory: int = 1,
    max_queries_per_subcategory: int = 20,
    culture_aliases: List[str] | None = None,
) -> Dict[str, Any]:
    aliases = [alias.lower() for alias in (culture_aliases or []) if alias]
    failed_query_groups: list[dict[str, str]] = []
    per_group: dict[str, Any] = {}

    for category in categories:
        category_name = category["name"]
        for subcategory in category_subcategories.get(category_name, []):
            subcategory_name = subcategory["name"]
            queries = category_subcategory_queries.get(category_name, {}).get(subcategory_name, [])
            cleaned = [_clean(query) for query in queries if _clean(query)]
            issues: list[str] = []
            if len(cleaned) < min_queries_per_subcategory:
                issues.append("query_count_below_minimum")
            if len(cleaned) > max_queries_per_subcategory:
                issues.append("query_count_above_maximum")
            if _duplicate_rate([query.lower() for query in cleaned]) > 0.2:
                issues.append("query_duplicate_rate_too_high")
            if aliases and cleaned and not any(_has_alias(query, aliases) for query in cleaned):
                issues.append("culture_alias_missing")
            group_key = f"{category_name}||{subcategory_name}"
            if issues:
                failed_query_groups.append({"category": category_name, "subcategory": subcategory_name})
            per_group[group_key] = {
                "passed": not issues,
                "blocking_issues": issues,
                "num_queries": len(cleaned),
            }

    blocking_issues = ["failed_query_groups"] if failed_query_groups else []
    issue_categories = _issue_categories(blocking_issues, [])
    gate_status = _gate_status(blocking_issues, [])
    return {
        "passed": not failed_query_groups,
        "gate_status": gate_status,
        "decision": _decision_from_gate_status(gate_status),
        "issue_categories": issue_categories,
        "score": _score(blocking_issues, []),
        "blocking_issues": blocking_issues,
        "warnings": [],
        "failed_query_groups": failed_query_groups,
        "per_group": per_group,
        "metrics": {
            "failed_query_group_count": len(failed_query_groups),
            "num_queries": sum(
                len(queries)
                for subcategory_queries in category_subcategory_queries.values()
                for queries in subcategory_queries.values()
            ),
        },
    }


def build_taxonomy_quality(
    *,
    culture_profile: Dict[str, Any],
    category_report: Dict[str, Any],
    subcategory_report: Dict[str, Any],
    query_report: Dict[str, Any],
    repair_attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    child_statuses = [
        str(category_report.get("gate_status") or ("pass" if category_report.get("passed") else "repair")),
        str(subcategory_report.get("gate_status") or ("pass" if subcategory_report.get("passed") else "repair")),
        str(query_report.get("gate_status") or ("pass" if query_report.get("passed") else "repair")),
    ]
    gate_status = _aggregate_gate_status(child_statuses)
    passed = gate_status in {"pass", "warn"}
    issue_categories = sorted(
        set(category_report.get("issue_categories") or [])
        | set(subcategory_report.get("issue_categories") or [])
        | set(query_report.get("issue_categories") or [])
    )
    blocking_issues = sorted(
        set(category_report.get("blocking_issues") or [])
        | set(subcategory_report.get("blocking_issues") or [])
        | set(query_report.get("blocking_issues") or [])
    )
    warnings = sorted(
        set(category_report.get("warnings") or [])
        | set(subcategory_report.get("warnings") or [])
        | set(query_report.get("warnings") or [])
    )
    score = round(
        (
            float(category_report.get("score", 0.0))
            + float(subcategory_report.get("score", 0.0))
            + float(query_report.get("score", 0.0))
        )
        / 3,
        4,
    )
    return {
        "passed": passed,
        "gate_status": gate_status,
        "decision": _decision_from_gate_status(gate_status),
        "issue_categories": issue_categories,
        "score": score,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "culture_profile": culture_profile,
        "category_report": category_report,
        "subcategory_report": subcategory_report,
        "query_report": query_report,
        "repair_attempts": repair_attempts,
    }


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _duplicate_rate(values: Iterable[str]) -> float:
    cleaned = [value for value in values if value]
    if not cleaned:
        return 0.0
    return (len(cleaned) - len(set(cleaned))) / len(cleaned)


def _has_alias(query: str, aliases: List[str]) -> bool:
    lowered = query.lower()
    return any(alias in lowered for alias in aliases)


def _score(blocking_issues: list[str], warnings: list[str]) -> float:
    if blocking_issues:
        return 0.0
    return max(0.0, 1.0 - 0.05 * len(warnings))


def _gate_status(blocking_issues: list[str], warnings: list[str]) -> str:
    if blocking_issues:
        return "repair"
    if warnings:
        return "warn"
    return "pass"


def _aggregate_gate_status(child_statuses: list[str]) -> str:
    normalized = [status.strip().lower() for status in child_statuses]
    if any(status == "fail" for status in normalized):
        return "fail"
    if any(status == "repair" for status in normalized):
        return "repair"
    if any(status == "warn" for status in normalized):
        return "warn"
    return "pass"


def _decision_from_gate_status(gate_status: str) -> str:
    if gate_status in {"pass", "warn"}:
        return "continue"
    if gate_status == "repair":
        return "repair"
    return "stop"


def _issue_categories(blocking_issues: list[str], warnings: list[str]) -> list[str]:
    categories: set[str] = set()
    for issue in [*blocking_issues, *warnings]:
        if "alias" in issue:
            categories.add("culture_specificity")
        elif "duplicate" in issue:
            categories.add("duplicates")
        elif "empty" in issue:
            categories.add("schema")
        elif "count" in issue or "missing" in issue or "failed_" in issue:
            categories.add("missing_coverage")
        else:
            categories.add("other")
    return sorted(categories)
