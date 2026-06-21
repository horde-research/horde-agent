from __future__ import annotations

import json
from pathlib import Path

from core.data.sft_answerability import filter_text_sft_examples_by_answerability


def test_sft_answerability_filters_unsupported_and_low_value_rows(tmp_path: Path) -> None:
    examples = [
        {
            "messages": [
                {"role": "user", "content": "What is the shanyrak?"},
                {"role": "assistant", "content": "The shanyrak is the circular crown at the top of a yurt."},
            ],
            "source_url": "https://good.example/yurt",
            "source_excerpt": "The shanyrak is the circular crown at the top of a yurt.",
        },
        {
            "messages": [
                {"role": "user", "content": "What is the shanyrak?"},
                {"role": "assistant", "content": "It is a search suggestion on a stock photography platform."},
            ],
            "source_url": "https://stock.adobe.com/search?k=shyrdak",
            "source_excerpt": "Shyrdak Images Browse stock photos vectors free trial Did you mean: shrank",
        },
        {
            "messages": [
                {"role": "user", "content": "What is the yurt interior layout?"},
                {"role": "assistant", "content": "The answer is about apple orchards and mountain weather."},
            ],
            "source_url": "https://good.example/yurt-layout",
            "source_excerpt": "A yurt interior has a place of honor, household storage, children's area, and hearth.",
        },
    ]

    def fake_embeddings(texts: list[str], model_id: str) -> list[list[float]]:  # noqa: ARG001
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "apple" in lowered or "weather" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors

    kept, report = filter_text_sft_examples_by_answerability(
        examples,
        out_path=tmp_path / "answerability.json",
        min_answer_source_similarity=0.35,
        min_question_source_similarity=0.20,
        embedding_fn=fake_embeddings,
    )

    assert len(kept) == 1
    assert kept[0]["source_url"] == "https://good.example/yurt"
    assert report["num_dropped_examples"] == 2
    assert report["reason_counts"]["page_type:known_stock_domain"] == 1
    assert report["reason_counts"]["answer_source_similarity_below_minimum"] == 1
    persisted = json.loads(Path(tmp_path / "answerability.json").read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "sft_answerability.v1"
