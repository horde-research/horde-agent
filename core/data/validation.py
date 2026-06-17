"""Dataset validation helpers."""

from __future__ import annotations

from typing import Any, Iterable, List


def validate_text_columns(dataset: Any, text_columns: Iterable[str]) -> List[str]:
    columns = list(text_columns)
    warnings: list[str] = []
    if not columns:
        return ["No text columns detected."]

    dataset_columns = set(getattr(dataset, "column_names", []) or [])
    for column in columns:
        if column not in dataset_columns:
            warnings.append(f"Text column '{column}' is missing.")
            continue
        empty_count = 0
        for value in dataset[column]:
            if value is None or str(value).strip() == "":
                empty_count += 1
        if empty_count:
            warnings.append(f"Text column '{column}' has {empty_count} empty or missing values.")
    return warnings
