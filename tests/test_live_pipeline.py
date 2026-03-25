"""Live integration tests — real LLM calls + real (tiny) training.

These tests call the actual LLM API configured in .env and train a tiny
model on Mac CPU/MPS.  They are **skipped automatically** when LLM_API_KEY
is not set.

Each step feeds its real output into the next step:

    Taxonomy (LLM) → SFT annotation (LLM) → Dataset build → Train (tiny model) → Eval

All heavy operations use "cut" sizes:
    - 2 categories, ~2 subcategories each, ~3 keywords each
    - 3-5 text samples annotated
    - Training: 5 steps, batch_size=1, on a ~2 MB model (sshleifer/tiny-gpt2)

Run:
    pytest tests/test_live_pipeline.py -v -s

Expected runtime: ~1-3 minutes (mostly LLM API latency).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

if "HF_HOME" not in os.environ:
    _hf_cache = os.path.join(tempfile.gettempdir(), "hf_cache_live_test")
    os.makedirs(_hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = _hf_cache

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_has_api_key = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not _has_api_key, reason="LLM_API_KEY not set")

TINY_MODEL_ID = "sshleifer/tiny-gpt2"
MAX_CATEGORIES_TO_PROCESS = 2
MAX_TRAIN_STEPS = 5


# ─── Module-scoped fixtures (each runs once, chains into the next) ───────────

@pytest.fixture(scope="module")
def run_dir():
    with tempfile.TemporaryDirectory(prefix="horde_live_") as d:
        logger.info("Live test run dir: %s", d)
        yield d


@pytest.fixture(scope="module")
def taxonomy_result():
    """Step 1: Call real LLM to generate taxonomy."""
    from tools.generate_taxonomy.tool import GenerateTaxonomyTool

    logger.info("=== Step 1: Generating taxonomy (real LLM) ===")
    result = GenerateTaxonomyTool().execute("Kazakhstan", {
        "batch_size": 2,
        "batch_delay": 0.5,
    })

    cats = result["categories"][:MAX_CATEGORIES_TO_PROCESS]
    logger.info("Taxonomy: %d categories (cut to %d), using: %s",
                len(result["categories"]), len(cats),
                [c["name"] for c in cats])

    cut_subcats = {}
    cut_queries = {}
    for cat in cats:
        name = cat["name"]
        cut_subcats[name] = result["category_subcategories"].get(name, [])
        cut_queries[name] = result["category_subcategory_queries"].get(name, {})

    return {
        "categories": cats,
        "category_subcategories": cut_subcats,
        "category_subcategory_queries": cut_queries,
        "full_result": result,
    }


@pytest.fixture(scope="module")
def all_queries(taxonomy_result) -> List[str]:
    """Extract flat search query list from the cut taxonomy."""
    qs: List[str] = []
    for sub_dict in taxonomy_result["category_subcategory_queries"].values():
        for q_list in sub_dict.values():
            qs.extend(q_list)
    logger.info("Extracted %d search queries from taxonomy.", len(qs))
    return qs


@pytest.fixture(scope="module")
def sft_input_jsonl(run_dir, all_queries) -> str:
    """Synthesize short texts from search queries (simulates data collection)."""
    jsonl_path = os.path.join(run_dir, "collected_texts.jsonl")
    samples = [
        f"Kazakhstan is known for {q}. This is an important aspect of Kazakh culture "
        f"that has been preserved for centuries and continues to shape modern society."
        for q in all_queries[:5]
    ]
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for text in samples:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info("Wrote %d synthetic texts to %s", len(samples), jsonl_path)
    return jsonl_path


@pytest.fixture(scope="module")
def sft_result(run_dir, sft_input_jsonl):
    """Step 2: Call real LLM to annotate texts into SFT format."""
    from tools.build_sft_dataset.tool import BuildSftDatasetTool

    logger.info("=== Step 2: Building SFT dataset (real LLM) ===")
    result = BuildSftDatasetTool().execute({
        "mode": "text",
        "input_jsonl": sft_input_jsonl,
        "text_field": "text",
        "output_annotations": os.path.join(run_dir, "annotations.jsonl"),
        "output_sft": os.path.join(run_dir, "sft.jsonl"),
        "batch_size": 2,
        "batch_delay": 0.5,
    })
    logger.info("SFT: %d items → %d examples, %d failures",
                result["num_items"], result["num_examples"], result["num_failures"])
    return result


@pytest.fixture(scope="module")
def dataset_result(run_dir, sft_result):
    """Step 3: Load SFT JSONL as HF dataset."""
    from tools.build_dataset.tool import BuildDatasetTool

    logger.info("=== Step 3: Building HF dataset ===")
    result = BuildDatasetTool().execute(sft_result["sft_path"], {"run_dir": run_dir})
    logger.info("Dataset: %d samples, columns=%s, modality=%s",
                result["dataset_summary"]["sample_count"],
                result["dataset_summary"]["columns"],
                result["dataset_summary"]["modality_candidates"])
    return result


@pytest.fixture(scope="module")
def train_result(run_dir, dataset_result):
    """Step 4: Train a tiny model with LoRA for a few steps."""
    from tools.train.tool import TrainTool

    logger.info("=== Step 4: Training tiny model (%s, %d steps) ===",
                TINY_MODEL_ID, MAX_TRAIN_STEPS)
    result = TrainTool().execute(
        dataset_result["dataset_ref"],
        {
            "method": "sft",
            "run_dir": run_dir,
            "iter_idx": 0,
            "hf_model_id": TINY_MODEL_ID,
            "trainer_key": "static_sft_default",
            "lora_preset_key": "lora_attn_small",
            "model_loader_key": "hf_causal_lm_default",
            "train_config": {
                "max_steps": MAX_TRAIN_STEPS,
                "batch_size": 1,
                "grad_accum": 1,
                "eval_steps": MAX_TRAIN_STEPS + 1,
                "max_seq_len": 128,
            },
        },
    )
    logger.info("Training done: adapter=%s, metrics=%s",
                result["adapter_path"], result["metrics"])
    return result


@pytest.fixture(scope="module")
def eval_result(run_dir, train_result, dataset_result):
    """Step 5: Run inference on the trained model and evaluate."""
    from tools.eval_model.tool import EvalModelTool

    logger.info("=== Step 5: Evaluating model ===")
    result = EvalModelTool().execute(
        train_result["adapter_path"],
        dataset_result["dataset_ref"]["data_path"],
        {
            "run_dir": run_dir,
            "hf_model_id": TINY_MODEL_ID,
            "split": "train",
            "max_samples": 3,
            "max_new_tokens": 32,
        },
    )
    logger.info("Eval done: predictions=%s, failures=%s",
                result["predictions_path"], result["failures_path"])
    return result


# ─── Step 1 Tests: Taxonomy ──────────────────────────────────────────────────

class TestStep1_Taxonomy:
    def test_categories_returned(self, taxonomy_result):
        cats = taxonomy_result["categories"]
        assert len(cats) >= 1
        for cat in cats:
            assert "name" in cat
            assert "description" in cat
            assert len(cat["name"]) > 0

    def test_subcategories_per_category(self, taxonomy_result):
        for cat in taxonomy_result["categories"]:
            subs = taxonomy_result["category_subcategories"].get(cat["name"], [])
            assert len(subs) >= 1, f"No subcategories for category '{cat['name']}'"
            for sub in subs:
                assert "name" in sub

    def test_queries_per_subcategory(self, taxonomy_result):
        for cat in taxonomy_result["categories"]:
            q_dict = taxonomy_result["category_subcategory_queries"].get(cat["name"], {})
            subs = taxonomy_result["category_subcategories"].get(cat["name"], [])
            for sub in subs:
                qs = q_dict.get(sub["name"], [])
                assert len(qs) >= 1, (
                    f"No search queries for {cat['name']}/{sub['name']}"
                )

    def test_queries_are_strings(self, all_queries):
        assert len(all_queries) >= 2
        for q in all_queries:
            assert isinstance(q, str)
            assert len(q) > 0


# ─── Step 2 Tests: SFT Dataset ──────────────────────────────────────────────

class TestStep2_SftAnnotation:
    def test_items_were_processed(self, sft_result):
        assert sft_result["num_items"] >= 1

    def test_sft_examples_generated(self, sft_result):
        assert sft_result["num_examples"] >= 1

    def test_sft_file_exists(self, sft_result):
        assert os.path.exists(sft_result["sft_path"])

    def test_sft_examples_are_valid_chat_format(self, sft_result):
        with open(sft_result["sft_path"]) as f:
            for line in f:
                ex = json.loads(line)
                assert "messages" in ex
                msgs = ex["messages"]
                assert len(msgs) >= 2
                assert msgs[0]["role"] == "user"
                assert msgs[1]["role"] == "assistant"
                assert len(msgs[0]["content"]) > 10
                assert len(msgs[1]["content"]) > 10

    def test_annotations_file_exists(self, sft_result):
        assert os.path.exists(sft_result["annotations_path"])
        with open(sft_result["annotations_path"]) as f:
            annotations = [json.loads(line) for line in f if line.strip()]
        assert len(annotations) >= 1


# ─── Step 3 Tests: Dataset Building ─────────────────────────────────────────

class TestStep3_DatasetBuilding:
    def test_dataset_ref_created(self, dataset_result):
        ref = dataset_result["dataset_ref"]
        assert "data_path" in ref
        assert ref["kind"] == "hf"

    def test_summary_has_samples(self, dataset_result):
        summary = dataset_result["dataset_summary"]
        assert summary["sample_count"] >= 1

    def test_modality_detected_as_text(self, dataset_result):
        summary = dataset_result["dataset_summary"]
        assert "text" in summary["modality_candidates"]

    def test_manifest_file_created(self, dataset_result):
        assert os.path.exists(dataset_result["dataset_manifest_path"])


# ─── Step 4 Tests: Training ─────────────────────────────────────────────────

class TestStep4_Training:
    def test_adapter_produced(self, train_result):
        assert os.path.isdir(train_result["adapter_path"])

    def test_adapter_contains_files(self, train_result):
        adapter_dir = Path(train_result["adapter_path"])
        files = list(adapter_dir.iterdir())
        assert len(files) >= 1, "Adapter directory is empty"

    def test_metrics_returned(self, train_result):
        m = train_result["metrics"]
        assert "steps" in m
        assert m["steps"] >= 1

    def test_train_loss_is_finite(self, train_result):
        m = train_result["metrics"]
        if m.get("last_train_loss") is not None:
            assert 0.0 < m["last_train_loss"] < 100.0

    def test_log_files_created(self, train_result):
        for key in ("train_log", "metrics"):
            path = train_result["log_paths"][key]
            assert os.path.exists(path), f"Missing log file: {key}"

    def test_iteration_record_valid(self, train_result):
        rec = train_result["iteration_record"]
        assert rec["iter_idx"] == 0
        assert rec["adapter_path"] == train_result["adapter_path"]
        assert rec["config"]["max_steps"] == MAX_TRAIN_STEPS


# ─── Step 5 Tests: Evaluation ───────────────────────────────────────────────

class TestStep5_Evaluation:
    def test_predictions_file_created(self, eval_result):
        assert os.path.exists(eval_result["predictions_path"])

    def test_predictions_contain_expected_fields(self, eval_result):
        with open(eval_result["predictions_path"]) as f:
            for line in f:
                row = json.loads(line)
                assert "id" in row
                assert "input" in row
                assert "prediction" in row
                assert "reference" in row

    def test_predictions_have_nonempty_output(self, eval_result):
        with open(eval_result["predictions_path"]) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) >= 1
        for row in rows:
            assert len(row["prediction"]) > 0, f"Empty prediction for id={row['id']}"

    def test_failures_file_created(self, eval_result):
        assert os.path.exists(eval_result["failures_path"])

    def test_cluster_preview_returned(self, eval_result):
        assert "cluster_preview" in eval_result
        assert "clusters" in eval_result["cluster_preview"]
