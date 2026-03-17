"""Integration tests for the complete SFT pipeline.

Tests every stage independently and end-to-end across all three workflow modes
(full, workflow, minimal_agentic) with mocked LLM responses and mocked heavy
ops (training, evaluation).  No API keys or GPU required.

Run:
    pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if "HF_HOME" not in os.environ:
    _hf_cache = os.path.join(tempfile.gettempdir(), "hf_cache_test")
    os.makedirs(_hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = _hf_cache

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


# ─── Realistic "cut" fake LLM responses ──────────────────────────────────────
# Small but representative: 3 categories × 2 subcategories × 3 keywords = 18 kw

FAKE_CATEGORIES = {
    "categories": [
        {"name": "cuisine", "description": "Traditional dishes and cooking methods"},
        {"name": "music", "description": "Traditional and modern musical traditions"},
        {"name": "crafts", "description": "Traditional arts, textiles and handicrafts"},
    ]
}

FAKE_SUBCATEGORIES = {
    "subcategories": [
        {"name": "traditional_dishes", "description": "National dishes and recipes"},
        {"name": "cooking_methods", "description": "Traditional cooking techniques"},
    ]
}

FAKE_KEYWORDS = {
    "keywords": [
        "Kazakh beshbarmak recipe",
        "Kazakhstan traditional food",
        "қазақ тағамдары",
    ]
}

FAKE_TEXT_ANNOTATION = {
    "knowledge_qa": [
        {
            "question": "What is beshbarmak?",
            "answer": (
                "Beshbarmak is a traditional Kazakh dish made of boiled meat and "
                "flat noodles. It is considered the national dish of Kazakhstan and "
                "is served at important gatherings."
            ),
        },
        {
            "question": "How is kumis prepared?",
            "answer": (
                "Kumis is fermented mare's milk. It is prepared by fermenting "
                "fresh mare's milk in a leather bag called a saba, stirring "
                "regularly over several days."
            ),
        },
        {
            "question": "What role does hospitality play?",
            "answer": (
                "Hospitality is central to Kazakh culture. Guests are always "
                "offered the best food and a place at the table, reflecting "
                "deep-rooted customs of generosity."
            ),
        },
    ],
    "detailed_explanation": {
        "instruction": "Explain the significance of dastarkhan in Kazakh culture.",
        "response": (
            "The dastarkhan is a traditional tablecloth spread on the floor. "
            "It represents hospitality and abundance. When guests arrive, "
            "the dastarkhan is laid out with bread, sweets, and dried fruits "
            "before the main meal is served."
        ),
    },
    "analytical_reasoning": {
        "instruction": "Why has beshbarmak remained the national dish despite modernization?",
        "response": (
            "First, beshbarmak uses ingredients readily available on the steppe: "
            "horse or lamb meat and simple flour noodles. Second, it is deeply "
            "tied to nomadic traditions of communal eating. Combining these "
            "factors, it has survived as a cultural anchor."
        ),
    },
    "conversational_exchange": {
        "opening_question": "What are some typical Kazakh breakfast foods?",
        "opening_response": (
            "A typical Kazakh breakfast includes baursak (fried dough), kurt "
            "(dried cheese balls), and strong tea with milk. These foods reflect "
            "the nomadic heritage."
        ),
        "follow_up_question": "Are there regional variations in breakfast?",
        "follow_up_response": (
            "Yes, southern regions include more fruits and vegetables due to the "
            "warmer climate, while northern areas lean towards heavier dairy-based "
            "dishes."
        ),
    },
    "metadata": {
        "language": "en",
        "topics": ["cuisine", "beshbarmak", "kumis"],
        "domain": "cuisine",
    },
}

FAKE_MODALITY_DECISION = {
    "modality": "text",
    "confidence": 0.95,
    "rationale": "Dataset contains chat-format text messages.",
}

FAKE_COMPONENT_SELECTION = {
    "dataset_loader_key": "hf_text_default",
    "model_loader_key": "hf_causal_lm_default",
    "lora_preset_key": "lora_attn_small",
    "trainer_key": "static_sft_default",
    "hf_model_id": "test-model",
    "primary_metric": "eval_loss",
    "rationale": "Standard SFT setup for text data.",
}

FAKE_TRAINING_ADJUSTMENT = {
    "should_retry": False,
    "lr_multiplier": 1.0,
    "batch_size_delta": 0,
    "grad_accum_delta": 0,
    "max_steps_delta": 0,
    "switch_lora_preset_key": None,
    "stop_reason": "Converged.",
    "rationale": "Loss has stabilized, no further tuning needed.",
}

FAKE_ERROR_ANALYSIS = {
    "cluster_labels": ["formatting_errors"],
    "root_causes": ["Minor template mismatches"],
    "data_fixes": ["Standardize output templates"],
    "next_training_actions": ["Continue with current setup"],
}


# ─── Mock helpers ─────────────────────────────────────────────────────────────

def _make_fake_llm_response(data: Dict[str, Any], request_id: str = "test"):
    from core.llm.client import LLMResponse
    return LLMResponse(request_id=request_id, success=True, data=data)


def _fake_generate_json_sync(request):
    """Pick response based on schema names and keywords in the prompt."""
    msg = request.user_message.lower()
    sys_prompt = (request.system_prompt or "").lower()

    # Agentic decisions — match on the Pydantic schema name embedded in each
    # stage prompt, which uniquely identifies the request type.
    if "pipeline" in sys_prompt or "orchestrat" in sys_prompt:
        if "erroranalysisdecision" in msg or "cluster_labels" in msg:
            return _make_fake_llm_response(FAKE_ERROR_ANALYSIS, request.request_id)
        if "trainingadjustmentdecision" in msg or "should_retry" in msg:
            return _make_fake_llm_response(FAKE_TRAINING_ADJUSTMENT, request.request_id)
        if "componentselectiondecision" in msg:
            return _make_fake_llm_response(FAKE_COMPONENT_SELECTION, request.request_id)
        if "modalitydecision" in msg:
            return _make_fake_llm_response(FAKE_MODALITY_DECISION, request.request_id)

    # Taxonomy stages — check "keyword" before "subcategor" since keyword prompts
    # contain both words but subcategory prompts don't mention "keyword".
    if "keyword" in msg or "search keywords" in msg:
        return _make_fake_llm_response(FAKE_KEYWORDS, request.request_id)
    if "subcategor" in msg:
        return _make_fake_llm_response(FAKE_SUBCATEGORIES, request.request_id)
    if "categories" in msg:
        return _make_fake_llm_response(FAKE_CATEGORIES, request.request_id)

    # SFT annotation
    if any(k in msg for k in ("annotation", "knowledge", "distill", "analyze")):
        return _make_fake_llm_response(FAKE_TEXT_ANNOTATION, request.request_id)

    return _make_fake_llm_response(FAKE_CATEGORIES, request.request_id)


def _fake_generate_json_batch_sync(requests, *, batch_size=5, batch_delay_seconds=1.5):
    return [_fake_generate_json_sync(r) for r in requests]


def _make_fake_train_result(run_dir: str, iter_idx: int = 0) -> Dict[str, Any]:
    """Build a realistic fake TrainTool.execute() result."""
    adapter_path = os.path.join(run_dir, "adapters", f"iter_{iter_idx}")
    os.makedirs(adapter_path, exist_ok=True)

    train_log = os.path.join(run_dir, "train.log")
    Path(train_log).write_text("step 1 loss=3.0\nstep 5 loss=2.5\nstep 10 loss=2.2\n")

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    Path(metrics_path).write_text("")

    return {
        "adapter_path": adapter_path,
        "log_paths": {"train_log": train_log, "metrics": metrics_path},
        "metrics": {
            "steps": 10,
            "best_eval_loss": 2.3,
            "last_train_loss": 2.5,
            "last_eval_loss": 2.3,
        },
        "iteration_record": {
            "iter_idx": iter_idx,
            "config": {
                "lr": 2e-4,
                "batch_size": 4,
                "grad_accum": 4,
                "max_steps": 10,
                "warmup_ratio": 0.03,
                "weight_decay": 0.0,
                "max_seq_len": 512,
                "eval_steps": 50,
                "seed": 42,
            },
            "metrics": {
                "steps": 10,
                "best_eval_loss": 2.3,
                "last_train_loss": 2.5,
                "last_eval_loss": 2.3,
            },
            "adapter_path": adapter_path,
            "log_paths": {"train_log": train_log, "metrics": metrics_path},
        },
    }


def _make_fake_eval_result(run_dir: str) -> Dict[str, Any]:
    """Build a realistic fake EvalModelTool.execute() result."""
    predictions_path = os.path.join(run_dir, "predictions.jsonl")
    failures_path = os.path.join(run_dir, "failures.jsonl")

    with open(predictions_path, "w") as f:
        for i in range(5):
            f.write(json.dumps({
                "id": i,
                "input": f"Question {i}",
                "prediction": f"Predicted answer {i}",
                "reference": f"Reference answer {i}",
            }) + "\n")

    with open(failures_path, "w") as f:
        f.write(json.dumps({"id": 2, "reason": "mismatch"}) + "\n")

    return {
        "predictions_path": predictions_path,
        "failures_path": failures_path,
        "cluster_preview": {
            "clusters": [
                {"label": "formatting", "count": 1, "examples": ["mismatch on id 2"]},
            ]
        },
    }


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Patch LLMClient.from_env to return a mock client with fake responses."""
    mock_client = MagicMock()
    mock_client.generate_json_sync = _fake_generate_json_sync
    mock_client.generate_json_batch_sync = _fake_generate_json_batch_sync

    with patch("core.llm.client.LLMClient.from_env", return_value=mock_client):
        yield mock_client


@pytest.fixture
def run_dir():
    """Create a temporary run directory."""
    with tempfile.TemporaryDirectory(prefix="horde_test_") as d:
        yield d


# ─── Stage 1: Taxonomy generation ────────────────────────────────────────────

class TestTaxonomyGeneration:
    def test_generates_categories_subcategories_keywords(self, mock_llm):
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        tool = GenerateTaxonomyTool()
        result = tool.execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})

        assert "categories" in result
        assert len(result["categories"]) >= 2
        assert result["categories"][0]["name"] == "cuisine"

        assert "category_subcategories" in result
        for cat in result["categories"]:
            assert cat["name"] in result["category_subcategories"]
            subs = result["category_subcategories"][cat["name"]]
            assert len(subs) >= 1

        assert "category_subcategory_keywords" in result
        total_kws = sum(
            len(kws)
            for sub_dict in result["category_subcategory_keywords"].values()
            for kws in sub_dict.values()
        )
        assert total_kws > 0

    def test_keyword_count_matches_structure(self, mock_llm):
        """Verify the taxonomy tree is fully connected."""
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        result = GenerateTaxonomyTool().execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})

        for cat in result["categories"]:
            cat_name = cat["name"]
            assert cat_name in result["category_subcategories"]
            assert cat_name in result["category_subcategory_keywords"]

            for sub in result["category_subcategories"][cat_name]:
                sub_name = sub["name"]
                assert sub_name in result["category_subcategory_keywords"][cat_name], (
                    f"Subcategory '{sub_name}' missing from keywords dict"
                )
                assert len(result["category_subcategory_keywords"][cat_name][sub_name]) > 0

    def test_empty_country_raises(self, mock_llm):
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool

        with pytest.raises(ValueError, match="required"):
            GenerateTaxonomyTool().execute("", None)


# ─── Stage 2: SFT dataset generation from text ──────────────────────────────

class TestBuildSftDataset:
    def test_text_mode_from_directory(self, mock_llm, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        input_dir = os.path.join(project_root, "tests", "test_data", "texts")
        if not os.path.isdir(input_dir):
            pytest.skip("test_data/texts/ not found")

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_dir": input_dir,
            "output_annotations": os.path.join(run_dir, "annotations.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 2,
            "batch_delay": 0.0,
        })

        assert result["num_items"] >= 1
        assert result["num_examples"] >= 1
        assert os.path.exists(result["sft_path"])

        with open(result["sft_path"]) as f:
            first = json.loads(f.readline())
        assert "messages" in first
        assert first["messages"][0]["role"] == "user"
        assert first["messages"][1]["role"] == "assistant"

    def test_text_mode_from_jsonl(self, mock_llm, run_dir):
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl_path = os.path.join(run_dir, "input.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "Kazakhstan is a country in Central Asia with rich traditions."}) + "\n")
            f.write(json.dumps({"text": "Beshbarmak is a beloved dish of the Kazakh people."}) + "\n")
            f.write(json.dumps({"text": "The dombra is a two-stringed instrument played in Kazakhstan."}) + "\n")

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl_path,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "annotations.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 2,
            "batch_delay": 0.0,
        })

        assert result["num_items"] == 3
        assert result["num_examples"] >= 3
        assert result["num_failures"] == 0

        sft_examples = []
        with open(result["sft_path"]) as f:
            for line in f:
                sft_examples.append(json.loads(line))
        assert all("messages" in ex for ex in sft_examples)

    def test_sft_produces_valid_chat_format(self, mock_llm, run_dir):
        """Every SFT example must have user + assistant messages."""
        from tools.build_sft_dataset.tool import BuildSftDatasetTool

        jsonl_path = os.path.join(run_dir, "input.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "Traditional Kazakh horse games."}) + "\n")

        result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": jsonl_path,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "ann.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 1,
            "batch_delay": 0.0,
        })

        with open(result["sft_path"]) as f:
            for line in f:
                ex = json.loads(line)
                msgs = ex["messages"]
                assert len(msgs) >= 2
                assert len(msgs) % 2 == 0, "Messages should come in user/assistant pairs"
                for i, m in enumerate(msgs):
                    expected_role = "user" if i % 2 == 0 else "assistant"
                    assert m["role"] == expected_role
                    assert len(m["content"]) > 0


# ─── Stage 3: Core data utilities ────────────────────────────────────────────

class TestCoreData:
    def test_load_jsonl_dataset(self, run_dir):
        from core.data.hf_dataset import load_dataset_from_path

        jsonl_path = os.path.join(run_dir, "test.jsonl")
        rows = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "What?"}, {"role": "assistant", "content": "AI."}]},
        ]
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        ds, resolved_id = load_dataset_from_path(jsonl_path)
        assert len(ds) == 2
        assert "messages" in ds.column_names

    def test_format_text_for_sft_chat_format(self):
        from core.data.modality import format_text_for_sft

        example = {
            "messages": [
                {"role": "user", "content": "What is beshbarmak?"},
                {"role": "assistant", "content": "It is a traditional Kazakh dish."},
            ]
        }
        text = format_text_for_sft(example)
        assert "What is beshbarmak?" in text
        assert "traditional Kazakh dish" in text
        assert "<|user|>" in text
        assert "<|assistant|>" in text

    def test_format_text_for_sft_plain_text(self):
        from core.data.modality import format_text_for_sft

        example = {"text": "Kazakhstan is a large country in Central Asia."}
        text = format_text_for_sft(example)
        assert text == "Kazakhstan is a large country in Central Asia."

    def test_extract_text_input_output(self):
        from core.data.modality import extract_text_input_output

        example = {
            "messages": [
                {"role": "user", "content": "Tell me about kumis"},
                {"role": "assistant", "content": "Kumis is fermented mare's milk."},
            ]
        }
        prompt, ref = extract_text_input_output(example)
        assert "kumis" in prompt.lower()
        assert "fermented" in ref.lower()

    def test_infer_modality(self):
        from core.data.modality import infer_modality

        assert "text" in infer_modality(["messages", "id"], {})
        assert "text" in infer_modality(["text"], {})
        assert "image" in infer_modality(["image", "text"], {})

    def test_validate_text_columns(self, run_dir):
        from datasets import Dataset
        from core.data.validation import validate_text_columns

        ds = Dataset.from_list([{"text": "hello"}, {"text": ""}, {"text": "world"}])
        warnings = validate_text_columns(ds, ["text"])
        assert len(warnings) == 1
        assert "empty" in warnings[0].lower() or "missing" in warnings[0].lower()


# ─── Stage 4: Build dataset tool ─────────────────────────────────────────────

class TestBuildDatasetTool:
    def test_loads_jsonl_and_builds_summary(self, run_dir):
        from tools.build_dataset.tool import BuildDatasetTool

        jsonl_path = os.path.join(run_dir, "sft.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Answer {i}"},
                    ]
                }) + "\n")

        result = BuildDatasetTool().execute(jsonl_path, {"run_dir": run_dir})

        assert "dataset_ref" in result
        assert result["dataset_ref"]["data_path"] == jsonl_path
        assert result["dataset_summary"]["sample_count"] == 5
        assert "messages" in result["dataset_summary"]["columns"]
        assert os.path.exists(result["dataset_manifest_path"])

    def test_summary_detects_modality(self, run_dir):
        from tools.build_dataset.tool import BuildDatasetTool

        jsonl_path = os.path.join(run_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Q{i}"},
                        {"role": "assistant", "content": f"A{i}"},
                    ]
                }) + "\n")

        result = BuildDatasetTool().execute(jsonl_path, {"run_dir": run_dir})
        assert "text" in result["dataset_summary"]["modality_candidates"]


# ─── End-to-end: Full pipeline (country → taxonomy → collect → sft → … ) ────

class TestFullPipeline:
    """End-to-end test of the full pipeline with all heavy ops mocked."""

    def test_full_mode_taxonomy_to_report(self, mock_llm, run_dir):
        """full mode: country → taxonomy → collect → sft → build → train → eval → report."""
        from agent.orchestrator import Orchestrator
        from datasets import Dataset

        # Prepare fake collected data
        collect_dir = os.path.join(run_dir, "collect")
        os.makedirs(os.path.join(collect_dir, "dataset"), exist_ok=True)
        ds = Dataset.from_list([
            {"text": "Kazakhstan is known for its vast steppes and rich cultural heritage."},
            {"text": "Beshbarmak is a traditional Kazakh dish made from boiled meat and noodles."},
            {"text": "The dombra is a traditional two-stringed instrument played across Central Asia."},
            {"text": "Kazakh nomads developed a rich tradition of oral poetry and storytelling."},
            {"text": "Kumis, fermented mare's milk, has been a staple of the Kazakh diet for centuries."},
        ])
        ds.save_to_disk(os.path.join(collect_dir, "dataset"))

        fake_collect_result = {
            "data_path": os.path.join(collect_dir, "dataset"),
            "num_samples": 5,
            "metadata": {"provider": "test"},
        }
        fake_train_result = _make_fake_train_result(run_dir)
        fake_eval_result = _make_fake_eval_result(run_dir)

        with (
            patch("tools.collect_data.tool.CollectDataTool.execute", return_value=fake_collect_result),
            patch("tools.train.tool.TrainTool.execute", return_value=fake_train_result),
            patch("tools.eval_model.tool.EvalModelTool.execute", return_value=fake_eval_result),
        ):
            result = Orchestrator({
                "mode": "full",
                "country": "Kazakhstan",
                "run_dir": run_dir,
                "max_iters": 1,
                "max_steps": 10,
                "hf_model_id": "test-model",
                "batch_size": 2,
                "batch_delay": 0.0,
            }).run()

        assert result["mode"] == "full"

        # Report was generated
        assert "report_path" in result
        assert os.path.exists(result["report_path"])
        report_text = Path(result["report_path"]).read_text()
        assert "Dataset Summary" in report_text

        # Taxonomy produced correct structure
        assert result["taxonomy"]["num_categories"] >= 2
        assert result["taxonomy"]["num_keywords"] > 0

        # SFT generation worked
        assert result["sft_generation"]["num_examples"] >= 1

        # Adapter was "produced"
        assert "adapter_path" in result

    def test_full_mode_verifies_data_flow(self, mock_llm, run_dir):
        """Verify intermediate artifacts are created and wired correctly."""
        from agent.orchestrator import Orchestrator
        from datasets import Dataset

        collect_dir = os.path.join(run_dir, "collect")
        os.makedirs(os.path.join(collect_dir, "dataset"), exist_ok=True)
        Dataset.from_list([
            {"text": "Traditional Kazakh eagle hunting is practiced in the Altai mountains."},
            {"text": "Kazakh yurts are portable dwellings used by nomads on the steppe."},
        ]).save_to_disk(os.path.join(collect_dir, "dataset"))

        fake_collect_result = {
            "data_path": os.path.join(collect_dir, "dataset"),
            "num_samples": 2,
            "metadata": {"provider": "test"},
        }
        fake_train_result = _make_fake_train_result(run_dir)
        fake_eval_result = _make_fake_eval_result(run_dir)

        with (
            patch("tools.collect_data.tool.CollectDataTool.execute", return_value=fake_collect_result) as mock_collect,
            patch("tools.train.tool.TrainTool.execute", return_value=fake_train_result) as mock_train,
            patch("tools.eval_model.tool.EvalModelTool.execute", return_value=fake_eval_result) as mock_eval,
        ):
            result = Orchestrator({
                "mode": "full",
                "country": "Kazakhstan",
                "run_dir": run_dir,
                "max_iters": 1,
                "max_steps": 10,
                "hf_model_id": "test-model",
                "batch_size": 2,
                "batch_delay": 0.0,
            }).run()

        # CollectDataTool was called with keywords from taxonomy
        collect_call_args = mock_collect.call_args
        assert collect_call_args is not None
        collected_config = collect_call_args[0][0]
        assert "keywords" in collected_config
        assert len(collected_config["keywords"]) > 0

        # TrainTool received a dataset_ref
        train_call_args = mock_train.call_args[0]
        dataset_ref = train_call_args[0]
        assert "data_path" in dataset_ref

        # EvalModelTool received an adapter path
        eval_call_args = mock_eval.call_args[0]
        assert eval_call_args[0] == fake_train_result["adapter_path"]

        # SFT JSONL was created as an intermediate
        sft_dir = Path(run_dir) / "sft"
        assert sft_dir.exists()
        sft_jsonl = sft_dir / "sft.jsonl"
        assert sft_jsonl.exists()
        with open(sft_jsonl) as f:
            sft_lines = [json.loads(line) for line in f if line.strip()]
        assert len(sft_lines) >= 1
        assert "messages" in sft_lines[0]


# ─── End-to-end: Workflow mode (existing JSONL → train → eval → report) ──────

class TestWorkflowPipeline:
    def test_workflow_mode_from_jsonl(self, run_dir):
        """workflow mode: existing JSONL → build → train → eval → report."""
        from agent.orchestrator import Orchestrator

        jsonl_path = os.path.join(run_dir, "sft.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(8):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Question {i} about Kazakhstan culture"},
                        {"role": "assistant", "content": f"Answer {i}: detailed cultural information."},
                    ]
                }) + "\n")

        fake_train_result = _make_fake_train_result(run_dir)
        fake_eval_result = _make_fake_eval_result(run_dir)

        with (
            patch("tools.train.tool.TrainTool.execute", return_value=fake_train_result),
            patch("tools.eval_model.tool.EvalModelTool.execute", return_value=fake_eval_result),
        ):
            result = Orchestrator({
                "mode": "workflow",
                "data_path": jsonl_path,
                "run_dir": run_dir,
                "max_iters": 1,
                "max_steps": 10,
                "hf_model_id": "test-model",
            }).run()

        assert result["mode"] == "workflow"
        assert os.path.exists(result["report_path"])
        assert result["adapter_path"] == fake_train_result["adapter_path"]

        report_text = Path(result["report_path"]).read_text()
        assert "Dataset Summary" in report_text

    def test_workflow_mode_multiple_iterations(self, run_dir):
        """Verify workflow mode can run multiple training iterations."""
        from agent.orchestrator import Orchestrator

        jsonl_path = os.path.join(run_dir, "sft.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Q{i}"},
                        {"role": "assistant", "content": f"A{i}"},
                    ]
                }) + "\n")

        call_count = {"n": 0}

        def _train_side_effect(dataset_ref, config):
            idx = call_count["n"]
            call_count["n"] += 1
            return _make_fake_train_result(run_dir, iter_idx=idx)

        fake_eval_result = _make_fake_eval_result(run_dir)

        with (
            patch("tools.train.tool.TrainTool.execute", side_effect=_train_side_effect) as mock_train,
            patch("tools.eval_model.tool.EvalModelTool.execute", return_value=fake_eval_result),
        ):
            result = Orchestrator({
                "mode": "workflow",
                "data_path": jsonl_path,
                "run_dir": run_dir,
                "max_iters": 3,
                "max_steps": 5,
                "hf_model_id": "test-model",
            }).run()

        assert result["mode"] == "workflow"
        assert mock_train.call_count == 3


# ─── End-to-end: Minimal agentic mode ────────────────────────────────────────

class TestMinimalAgenticPipeline:
    """Test the LLM-driven minimal_agentic pipeline mode end-to-end."""

    def test_minimal_agentic_full_flow(self, mock_llm, run_dir):
        """minimal_agentic: data → agent decisions → train → eval → report."""
        from agent.orchestrator import Orchestrator

        jsonl_path = os.path.join(run_dir, "sft.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Question {i} about Kazakh culture"},
                        {"role": "assistant", "content": f"Detailed answer {i} about traditions."},
                    ]
                }) + "\n")

        fake_train_result = _make_fake_train_result(run_dir)
        fake_eval_result = _make_fake_eval_result(run_dir)

        # Train log needed by agentic mode for suggest_training_adjustments
        train_log_path = fake_train_result["log_paths"]["train_log"]
        Path(train_log_path).write_text("step 1 loss=3.0\nstep 10 loss=2.2\n")

        with (
            patch("tools.train.tool.TrainTool.execute", return_value=fake_train_result),
            patch("tools.eval_model.tool.EvalModelTool.execute", return_value=fake_eval_result),
        ):
            result = Orchestrator({
                "mode": "minimal_agentic",
                "data_path": jsonl_path,
                "run_dir": run_dir,
                "max_iters": 1,
                "max_steps": 10,
                "hf_model_id": "test-model",
            }).run()

        assert result["mode"] == "minimal_agentic"

        # Agent made component selection decisions
        assert "component_selection" in result
        cs = result["component_selection"]
        assert cs["dataset_loader_key"] == "hf_text_default"
        assert cs["trainer_key"] == "static_sft_default"
        assert cs["hf_model_id"] == "test-model"

        # Error analysis was performed
        assert "error_analysis" in result
        assert "cluster_labels" in result["error_analysis"]
        assert "root_causes" in result["error_analysis"]

        # Report was generated
        assert os.path.exists(result["report_path"])

        # Agent decisions log was created
        decisions_path = Path(run_dir) / "agent_decisions.jsonl"
        assert decisions_path.exists()
        decisions = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
        stages = [d["stage"] for d in decisions]
        assert "decide_modality" in stages
        assert "select_components" in stages
        assert "suggest_training_adjustments" in stages
        assert "analyze_errors" in stages

    def test_minimal_agentic_with_model_override(self, mock_llm, run_dir):
        """Verify hf_model_id_override takes precedence over agent decision."""
        from agent.orchestrator import Orchestrator

        jsonl_path = os.path.join(run_dir, "sft.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Q{i}"},
                        {"role": "assistant", "content": f"A{i}"},
                    ]
                }) + "\n")

        fake_train_result = _make_fake_train_result(run_dir)
        Path(fake_train_result["log_paths"]["train_log"]).write_text("step 1 loss=2.5\n")
        fake_eval_result = _make_fake_eval_result(run_dir)

        with (
            patch("tools.train.tool.TrainTool.execute", return_value=fake_train_result) as mock_train,
            patch("tools.eval_model.tool.EvalModelTool.execute", return_value=fake_eval_result),
        ):
            result = Orchestrator({
                "mode": "minimal_agentic",
                "data_path": jsonl_path,
                "run_dir": run_dir,
                "max_iters": 1,
                "hf_model_id_override": "my-custom-model",
            }).run()

        assert result["component_selection"]["hf_model_id"] == "my-custom-model"
        train_config_arg = mock_train.call_args[0][1]
        assert train_config_arg["hf_model_id"] == "my-custom-model"

    def test_minimal_agentic_respects_retry_decision(self, mock_llm, run_dir):
        """When agent says should_retry=True, training runs multiple iterations."""
        from agent.orchestrator import Orchestrator

        jsonl_path = os.path.join(run_dir, "sft.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Q{i}"},
                        {"role": "assistant", "content": f"A{i}"},
                    ]
                }) + "\n")

        retry_responses = [
            {
                "should_retry": True,
                "lr_multiplier": 0.5,
                "batch_size_delta": 0,
                "grad_accum_delta": 0,
                "max_steps_delta": 0,
                "switch_lora_preset_key": None,
                "stop_reason": None,
                "rationale": "Loss still decreasing, retry.",
            },
            FAKE_TRAINING_ADJUSTMENT,  # should_retry=False → stops
        ]
        retry_idx = {"n": 0}

        original_sync = mock_llm.generate_json_sync

        def _sync_with_retry(request):
            msg = request.user_message.lower()
            sys_prompt = (request.system_prompt or "").lower()
            if ("pipeline" in sys_prompt or "orchestrat" in sys_prompt) and (
                "trainingadjustmentdecision" in msg or "should_retry" in msg
            ):
                idx = min(retry_idx["n"], len(retry_responses) - 1)
                retry_idx["n"] += 1
                return _make_fake_llm_response(retry_responses[idx], request.request_id)
            return original_sync(request)

        mock_llm.generate_json_sync = _sync_with_retry

        call_count = {"n": 0}

        def _train_side_effect(dataset_ref, config):
            idx = call_count["n"]
            call_count["n"] += 1
            result = _make_fake_train_result(run_dir, iter_idx=idx)
            Path(result["log_paths"]["train_log"]).write_text(f"step 10 loss={2.5 - idx * 0.1}\n")
            return result

        fake_eval_result = _make_fake_eval_result(run_dir)

        with (
            patch("tools.train.tool.TrainTool.execute", side_effect=_train_side_effect) as mock_train,
            patch("tools.eval_model.tool.EvalModelTool.execute", return_value=fake_eval_result),
        ):
            result = Orchestrator({
                "mode": "minimal_agentic",
                "data_path": jsonl_path,
                "run_dir": run_dir,
                "max_iters": 5,
                "hf_model_id": "test-model",
            }).run()

        # The agent retried once then stopped, so 2 training calls total
        assert mock_train.call_count == 2


# ─── Cross-cutting: taxonomy → SFT → build → ready for train ─────────────────

class TestTaxonomySftBuildIntegration:
    """Test the data-preparation half of the pipeline without mocking
    taxonomy or SFT tools — only the LLM is mocked."""

    def test_taxonomy_to_sft_to_dataset(self, mock_llm, run_dir):
        """Taxonomy keywords → (skip collect) → SFT generation → BuildDataset."""
        from tools.generate_taxonomy.tool import GenerateTaxonomyTool
        from tools.build_sft_dataset.tool import BuildSftDatasetTool
        from tools.build_dataset.tool import BuildDatasetTool

        # Step 1: Generate taxonomy
        taxonomy = GenerateTaxonomyTool().execute("Kazakhstan", {"batch_size": 2, "batch_delay": 0.0})
        all_keywords = []
        for sub_dict in taxonomy["category_subcategory_keywords"].values():
            for kw_list in sub_dict.values():
                all_keywords.extend(kw_list)
        assert len(all_keywords) > 0

        # Step 2: Simulate collected texts from keywords
        collected_jsonl = os.path.join(run_dir, "collected.jsonl")
        with open(collected_jsonl, "w") as f:
            for kw in all_keywords[:6]:
                f.write(json.dumps({"text": f"Detailed article about {kw} in Kazakhstan."}) + "\n")

        # Step 3: Build SFT dataset
        sft_result = BuildSftDatasetTool().execute({
            "mode": "text",
            "input_jsonl": collected_jsonl,
            "text_field": "text",
            "output_annotations": os.path.join(run_dir, "annotations.jsonl"),
            "output_sft": os.path.join(run_dir, "sft.jsonl"),
            "batch_size": 3,
            "batch_delay": 0.0,
        })
        assert sft_result["num_examples"] >= 1

        # Step 4: Build training dataset
        dataset_result = BuildDatasetTool().execute(sft_result["sft_path"], {"run_dir": run_dir})
        assert dataset_result["dataset_summary"]["sample_count"] >= 1
        assert "messages" in dataset_result["dataset_summary"]["columns"]
        assert dataset_result["dataset_ref"]["data_path"] == sft_result["sft_path"]
