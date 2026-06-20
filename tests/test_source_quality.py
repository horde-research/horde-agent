from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_from_disk

from core.agentic.validators import validate_source_quality_output
from core.data.source_quality import assess_text_source_quality
from core.llm import LLMResponse


class FakeSourceQualityOracle:
    def generate_json_sync(self, request):  # noqa: ANN001
        return LLMResponse(
            request_id=request.request_id,
            success=True,
            data={
                "confidence": 0.9,
                "rationale": "Drop low-value search/navigation source cluster by domain rule.",
                "domain_rules": [
                    {"pattern": "bad.example", "decision": "drop", "reason": "navigation/search page"}
                ],
                "cluster_decisions": [],
                "query_refinements": ["traditional yurt shanyrak primary source"],
            },
        )


def _save_dataset(path: Path, rows: list[dict]) -> str:
    Dataset.from_list(rows).save_to_disk(str(path))
    return str(path)


def test_source_quality_oracle_policy_filters_domain_and_writes_dataset(tmp_path: Path) -> None:
    data_path = _save_dataset(
        tmp_path / "raw",
        [
            {
                "text": "Kazakh yurt construction uses the shanyrak crown, kerege lattice walls, "
                "uyk roof poles, and felt coverings for portable steppe shelter.",
                "source_url": "https://good.example/history/yurt",
                "source_query": "Kazakh yurt shanyrak",
                "group_key": "good-1",
            },
            {
                "text": "Menu Login Privacy Search Tags Categories Subscribe Copyright",
                "source_url": "https://bad.example/search?q=yurt",
                "source_query": "Kazakh yurt shanyrak",
                "group_key": "bad-1",
            },
        ],
    )

    result = assess_text_source_quality(
        data_path=data_path,
        output_dir=tmp_path / "source_quality",
        taxonomy={"categories": ["Kazakh yurt"], "category_subcategories": {"Kazakh yurt": ["shanyrak"]}},
        queries=["Kazakh yurt shanyrak"],
        config={
            "country": "Kazakhstan",
            "focus": "traditional culture",
            "source_quality_oracle_enable": True,
            "source_quality_min_quality_score": 0.10,
            "source_quality_accumulate_kept_sources": False,
        },
        llm_client=FakeSourceQualityOracle(),
    )

    filtered = load_from_disk(result["filtered_data_path"])
    assert len(filtered) == 1
    assert filtered[0]["source_url"] == "https://good.example/history/yurt"
    assert result["summary"]["oracle_used"] is True
    assert result["query_refinements"] == ["traditional yurt shanyrak primary source"]
    oracle_payload = json.loads(Path(result["oracle_payload_path"]).read_text(encoding="utf-8"))
    assert oracle_payload["target_entity"] == "Kazakhstan"
    assert oracle_payload["focus"] == "traditional culture"
    assert Path(result["policy_path"]).exists()
    assert Path(result["decisions_path"]).exists()


def test_source_quality_accumulates_kept_sources_across_attempts(tmp_path: Path) -> None:
    out_dir = tmp_path / "source_quality"
    first_path = _save_dataset(
        tmp_path / "first",
        [
            {
                "text": "Kazakh yurt felt construction uses a wooden frame and insulating felt cover.",
                "source_url": "https://one.example/yurt",
                "source_query": "Kazakh yurt felt",
                "group_key": "one",
            }
        ],
    )
    second_path = _save_dataset(
        tmp_path / "second",
        [
            {
                "text": "The shanyrak is the circular crown of a yurt and carries symbolic meaning.",
                "source_url": "https://two.example/shanyrak",
                "source_query": "Kazakh yurt shanyrak",
                "group_key": "two",
            }
        ],
    )

    first = assess_text_source_quality(
        data_path=first_path,
        output_dir=out_dir,
        taxonomy={"categories": ["Kazakh yurt"]},
        queries=["Kazakh yurt felt"],
        config={"source_quality_oracle_enable": False, "source_quality_min_quality_score": 0.05},
    )
    second = assess_text_source_quality(
        data_path=second_path,
        output_dir=out_dir,
        taxonomy={"categories": ["Kazakh yurt"]},
        queries=["Kazakh yurt shanyrak"],
        config={"source_quality_oracle_enable": False, "source_quality_min_quality_score": 0.05},
    )

    assert first["num_kept_rows"] == 1
    assert second["num_previous_accepted_rows"] == 1
    assert second["num_kept_rows"] == 2
    filtered = load_from_disk(second["filtered_data_path"])
    assert sorted(row["group_key"] for row in filtered) == ["one", "two"]


def test_source_quality_validator_blocks_insufficient_post_filter_corpus(tmp_path: Path) -> None:
    report = validate_source_quality_output(
        {
            "filtered_data_path": str(tmp_path / "missing"),
            "num_input_rows": 10,
            "num_kept_rows": 2,
            "num_removed_rows": 8,
            "summary": {
                "num_input_rows": 10,
                "num_kept_rows": 2,
                "num_kept_source_groups": 2,
                "num_kept_domains": 2,
                "top_domain_share": 0.5,
                "avg_kept_quality_score": 0.6,
                "removal_rate": 0.8,
            },
        },
        min_kept_rows=3,
        min_source_groups=3,
    )

    assert report.passed is False
    assert "source_quality_kept_rows_below_minimum" in report.blocking_issues
    assert report.decision == "repair"
