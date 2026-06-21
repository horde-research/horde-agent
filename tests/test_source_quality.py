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


def test_source_quality_embedding_alignment_drops_query_content_mismatch(tmp_path: Path) -> None:
    data_path = _save_dataset(
        tmp_path / "raw",
        [
            {
                "text": "Kazakh yurt construction uses the shanyrak crown and kerege lattice walls.",
                "source_url": "https://good.example/yurt",
                "source_query": "Kazakh yurt shanyrak",
                "group_key": "good",
            },
            {
                "text": "Additional education teachers develop communicative competence in schools.",
                "source_url": "https://mismatch.example/paper",
                "source_query": "Kazakh epic poetry dombra kobyz",
                "group_key": "bad",
            },
        ],
    )

    def fake_embeddings(texts: list[str], model_id: str) -> list[list[float]]:  # noqa: ARG001
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "additional education" in lowered or "teachers" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors

    result = assess_text_source_quality(
        data_path=data_path,
        output_dir=tmp_path / "source_quality",
        taxonomy={"categories": ["Kazakh yurt"]},
        queries=["Kazakh yurt shanyrak", "Kazakh epic poetry dombra kobyz"],
        config={
            "source_quality_oracle_enable": False,
            "source_quality_accumulate_kept_sources": False,
            "source_quality_min_quality_score": 0.0,
            "source_quality_enable_embeddings": True,
            "source_quality_embedding_fn": fake_embeddings,
            "source_quality_embedding_hard_min_similarity": 0.35,
        },
    )

    filtered = load_from_disk(result["filtered_data_path"])
    assert [row["group_key"] for row in filtered] == ["good"]
    decisions = [
        json.loads(line)
        for line in Path(result["decisions_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad = next(row for row in decisions if row["domain"] == "mismatch.example")
    assert bad["source_query_embedding_similarity"] == 0.0
    assert "source_query_embedding_alignment_below_hard_min" in bad["reasons"]


def test_source_quality_drops_stock_search_pages_without_oracle(tmp_path: Path) -> None:
    data_path = _save_dataset(
        tmp_path / "raw",
        [
            {
                "text": "Shyrdak Images Browse stock photos vectors free trial Did you mean: shrank, shirataki",
                "source_url": "https://stock.adobe.com/search?k=shyrdak",
                "source_query": "how to make a traditional Kazakh shyrdak rug",
                "group_key": "stock",
            },
            {
                "text": "A shyrdak is a traditional felt carpet made from patterned wool felt.",
                "source_url": "https://good.example/shyrdak",
                "source_query": "how to make a traditional Kazakh shyrdak rug",
                "group_key": "good",
            },
        ],
    )

    result = assess_text_source_quality(
        data_path=data_path,
        output_dir=tmp_path / "source_quality",
        taxonomy={"categories": ["Shyrdak"]},
        queries=["how to make a traditional Kazakh shyrdak rug"],
        config={
            "source_quality_oracle_enable": False,
            "source_quality_accumulate_kept_sources": False,
            "source_quality_min_quality_score": 0.0,
        },
    )

    filtered = load_from_disk(result["filtered_data_path"])
    assert [row["group_key"] for row in filtered] == ["good"]
    decisions = [
        json.loads(line)
        for line in Path(result["decisions_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stock = next(row for row in decisions if row["domain"] == "stock.adobe.com")
    assert "known_stock_domain" in stock["page_type_flags"]
    assert any(reason.startswith("page_type:") for reason in stock["reasons"])


def test_source_quality_revalidates_accumulated_sources(tmp_path: Path) -> None:
    out_dir = tmp_path / "source_quality"
    out_dir.mkdir()
    stale = {
        "text": "Shyrdak Images Browse stock photos vectors free trial Did you mean: shrank",
        "source_url": "https://stock.adobe.com/search?k=shyrdak",
        "source_query": "traditional shyrdak",
        "group_key": "stale-stock",
        "source_quality_keep": True,
        "source_quality_score": 0.9,
    }
    (out_dir / "accepted_sources.jsonl").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    data_path = _save_dataset(
        tmp_path / "raw",
        [
            {
                "text": "A shyrdak is a traditional felt carpet made from patterned wool felt.",
                "source_url": "https://good.example/shyrdak",
                "source_query": "traditional shyrdak",
                "group_key": "good",
            }
        ],
    )

    result = assess_text_source_quality(
        data_path=data_path,
        output_dir=out_dir,
        taxonomy={"categories": ["Shyrdak"]},
        queries=["traditional shyrdak"],
        config={
            "source_quality_oracle_enable": False,
            "source_quality_accumulate_kept_sources": True,
            "source_quality_min_quality_score": 0.0,
        },
    )

    filtered = load_from_disk(result["filtered_data_path"])
    assert [row["group_key"] for row in filtered] == ["good"]
    assert result["num_previous_accepted_rows_removed"] == 1
    assert result["summary"]["num_previous_accepted_rows_removed"] == 1


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
