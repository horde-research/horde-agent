"""Lightweight dataset validation."""

from typing import Any, List


def validate_text_columns(dataset: Any, text_columns: List[str]) -> List[str]:
    warnings: List[str] = []
    if not text_columns:
        warnings.append("No obvious text columns found.")
        return warnings
    sample = dataset.select(range(min(len(dataset), 50)))
    for col in text_columns:
        empty_count = 0
        for row in sample:
            value = row.get(col)
            if value is None or (isinstance(value, str) and not value.strip()):
                empty_count += 1
        if empty_count > 0:
            warnings.append(f"Column '{col}' has {empty_count} empty values in sample.")
    return warnings