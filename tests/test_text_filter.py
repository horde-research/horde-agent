from __future__ import annotations

from core.data.text_filter import filter_text_rows, summarize_text_filter


def test_text_filter_removes_low_value_and_duplicate_rows() -> None:
    base_text = " ".join(f"kazakh_term_{idx}" for idx in range(80))
    near_text = " ".join([*(f"kazakh_term_{idx}" for idx in range(72)), "new_detail", "extra_detail"])
    unique_text = " ".join(f"unique_term_{idx}" for idx in range(80))
    repeated_text = " ".join(["same"] * 80)

    rows = [
        {"source_id": "a", "source_url": "https://example.com/a", "text": base_text},
        {"source_id": "b", "source_url": "https://example.com/a?utm=1", "text": unique_text},
        {"source_id": "c", "source_url": "https://example.com/c", "text": base_text},
        {"source_id": "d", "source_url": "https://example.com/d", "text": near_text},
        {"source_id": "e", "source_url": "https://example.com/e", "text": repeated_text},
        {"source_id": "f", "source_url": "https://example.com/f", "text": "too short"},
    ]

    kept, report = filter_text_rows(
        rows,
        min_chars=20,
        min_words=5,
        min_unique_word_ratio=0.10,
        shingle_threshold=0.70,
    )

    assert [row["source_id"] for row in kept] == ["a"]
    assert report["num_input"] == 6
    assert report["num_kept"] == 1
    assert report["num_removed"] == 5
    assert report["removed_reason_counts"] == {
        "duplicate_url": 1,
        "exact_duplicate_text": 1,
        "near_duplicate_text": 1,
        "low_unique_word_ratio": 1,
        "too_short_chars": 1,
    }
    assert summarize_text_filter(report)["num_removed"] == 5
