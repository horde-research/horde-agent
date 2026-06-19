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
        if column == "messages":
            warnings.extend(_validate_messages_column(dataset[column]))
            continue
        empty_count = 0
        for value in dataset[column]:
            if value is None or str(value).strip() == "":
                empty_count += 1
        if empty_count:
            warnings.append(f"Text column '{column}' has {empty_count} empty or missing values.")
    return warnings


def _validate_messages_column(values: Iterable[Any]) -> list[str]:
    empty_count = 0
    missing_user_count = 0
    missing_assistant_count = 0
    for value in values:
        if not isinstance(value, list) or not value:
            empty_count += 1
            continue
        roles = {
            str(message.get("role") or "").strip().lower()
            for message in value
            if isinstance(message, dict)
        }
        if "user" not in roles:
            missing_user_count += 1
        if "assistant" not in roles:
            missing_assistant_count += 1
    warnings: list[str] = []
    if empty_count:
        warnings.append(f"Text column 'messages' has {empty_count} empty or malformed chat rows.")
    if missing_user_count:
        warnings.append(f"Text column 'messages' has {missing_user_count} rows without a user message.")
    if missing_assistant_count:
        warnings.append(f"Text column 'messages' has {missing_assistant_count} rows without an assistant message.")
    return warnings
