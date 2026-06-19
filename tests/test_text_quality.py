from __future__ import annotations

import json
from pathlib import Path

from core.data.text_quality import analyze_text_quality, records_from_sft_jsonl


def test_text_quality_reports_exact_url_and_near_duplicates() -> None:
    records = [
        {
            "id": "a",
            "url": "https://example.com/page?utm=1",
            "text": "Same collected text",
        },
        {
            "id": "b",
            "url": "https://www.example.com/page",
            "text": "Same collected text",
        },
        {
            "id": "c",
            "url": "https://example.org/other",
            "text": "Same collected text with one extra phrase",
        },
    ]

    report = analyze_text_quality(
        records,
        source="test",
        enable_embeddings=False,
        shingle_threshold=0.5,
        max_reported_pairs=1,
    )

    assert report["exact_duplicate_count"] == 1
    assert report["exact_duplicate_rate"] == 1 / 3
    assert report["url_duplicate_count"] == 1
    assert report["shingle_near_duplicate"]["pair_count"] >= 1
    assert len(report["shingle_near_duplicate"]["pairs"]) == 1
    assert report["embedding_near_duplicate"]["enabled"] is False


def test_records_from_sft_jsonl_extracts_messages(tmp_path: Path) -> None:
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = records_from_sft_jsonl(sft_path)

    assert len(records) == 1
    assert "user: Question" in records[0]["text"]
    assert "assistant: Answer" in records[0]["text"]
