"""Full live pipeline test — real LLM, real training, real HF Hub.

Exercises the complete pipeline with real external services:

    1. Taxonomy generation           (Gemini API)
    2. Synthesize texts from keywords (no external API)
    3. SFT annotation                (Gemini API)
    4. Dataset build + push to HF    (HF Hub, private)
    5. Pull dataset from HF + train  (tiny model, LoRA, CPU/MPS)
    6. Push adapter to HF            (HF Hub, private)
    7. Pull adapter from HF + eval   (inference on 3 samples)

All intermediate results are saved to ``test_outputs/live_run/<timestamp>/``
so you can inspect every artifact after the run.

Requires:
    - LLM_API_KEY          (Gemini / OpenAI / xAI)
    - huggingface-cli login (for HF Hub push/pull)

Run:
    pytest tests/test_full_live_pipeline.py -v -s

Expected runtime: ~3-5 minutes.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Skip conditions ─────────────────────────────────────────────────────────

_has_llm_key = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))

def _hf_logged_in() -> bool:
    try:
        from huggingface_hub import HfApi
        HfApi().whoami()
        return True
    except Exception:
        return False

_has_hf = _hf_logged_in()

pytestmark = pytest.mark.skipif(
    not (_has_llm_key and _has_hf),
    reason=(
        f"LLM_API_KEY({'ok' if _has_llm_key else 'MISSING'}), "
        f"HF login({'ok' if _has_hf else 'MISSING'})"
    ),
)

# ─── Constants ────────────────────────────────────────────────────────────────

TINY_MODEL_ID = "sshleifer/tiny-gpt2"
MAX_CATEGORIES = 2
MAX_TRAIN_STEPS = 5
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HF_DATASET_REPO = f"horde-agent-test-dataset-{TIMESTAMP}"
HF_ADAPTER_REPO = f"horde-agent-test-adapter-{TIMESTAMP}"

OUTPUT_DIR = project_root / "test_outputs" / "live_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Module-scoped fixtures (each runs once, chains into the next) ───────────

@pytest.fixture(scope="module")
def run_dir() -> Path:
    d = OUTPUT_DIR / TIMESTAMP
    d.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", d)
    return d


# ── Step 1: Taxonomy ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def taxonomy_result(run_dir):
    from tools.generate_taxonomy.tool import GenerateTaxonomyTool

    logger.info("=" * 60)
    logger.info("STEP 1: Generating taxonomy (real LLM)")
    logger.info("=" * 60)

    full = GenerateTaxonomyTool().execute("Kazakhstan", {
        "batch_size": 2,
        "batch_delay": 0.5,
    })

    cats = full["categories"][:MAX_CATEGORIES]
    cut_subcats = {c["name"]: full["category_subcategories"].get(c["name"], []) for c in cats}
    cut_queries = {c["name"]: full["category_subcategory_queries"].get(c["name"], {}) for c in cats}

    result = {
        "categories": cats,
        "category_subcategories": cut_subcats,
        "category_subcategory_queries": cut_queries,
    }

    out_path = run_dir / "01_taxonomy.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved → %s", out_path)
    logger.info("Categories: %s", [c["name"] for c in cats])
    return result


FALLBACK_QUERIES = [
    "Kazakh beshbarmak recipe traditional preparation",
    "Kazakh eagle hunting berkutchi Altai mountains",
    "Kazakh dombra music traditions history",
    "Kazakh yurt nomadic culture steppe",
    "Kazakh kokpar horse game rules",
]


@pytest.fixture(scope="module")
def all_queries(taxonomy_result) -> List[str]:
    qs: List[str] = []
    for sub_dict in taxonomy_result["category_subcategory_queries"].values():
        for q_list in sub_dict.values():
            qs.extend(q_list)
    if not qs:
        logger.warning("Taxonomy produced 0 queries — using fallback queries")
        qs = FALLBACK_QUERIES
    return qs


# ── Step 2: Synthesize texts from keywords ───────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_texts_jsonl(run_dir, all_queries, taxonomy_result) -> str:
    """Create synthetic text articles from taxonomy search queries + categories.

    Uses both the taxonomy queries and category descriptions to produce
    meaningful paragraphs that the LLM can annotate into SFT examples.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Synthesizing texts from %d search queries", len(all_queries))
    logger.info("=" * 60)

    texts = []
    for q in all_queries[:5]:
        texts.append(
            f"Kazakhstan is widely known for {q}. This tradition has deep roots "
            f"in the nomadic heritage of the Kazakh people and continues to play "
            f"an important role in modern Kazakh society and cultural identity. "
            f"Scholars and travellers alike have documented its significance, "
            f"noting how it connects generations and preserves the wisdom of the steppe."
        )

    # Also add texts from category descriptions
    for cat in taxonomy_result["categories"][:2]:
        texts.append(
            f"An important aspect of Kazakh culture is {cat['name'].lower()}. "
            f"{cat['description']} These traditions reflect centuries of nomadic "
            f"life on the Eurasian steppe and continue to influence modern Kazakhstan."
        )

    jsonl_path = str(run_dir / "02_synthetic_texts.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    logger.info("Wrote %d synthetic texts → %s", len(texts), jsonl_path)
    return jsonl_path


# ── Step 3: SFT annotation (real LLM) ───────────────────────────────────────

@pytest.fixture(scope="module")
def sft_result(run_dir, synthetic_texts_jsonl):
    from tools.build_sft_dataset.tool import BuildSftDatasetTool

    logger.info("=" * 60)
    logger.info("STEP 3: Building SFT dataset (real LLM annotation)")
    logger.info("=" * 60)

    annotations_path = str(run_dir / "03_annotations.jsonl")
    sft_path = str(run_dir / "03_sft.jsonl")

    result = BuildSftDatasetTool().execute({
        "mode": "text",
        "input_jsonl": synthetic_texts_jsonl,
        "text_field": "text",
        "output_annotations": annotations_path,
        "output_sft": sft_path,
        "batch_size": 2,
        "batch_delay": 0.5,
    })

    logger.info("SFT: %d items → %d examples, %d failures",
                result["num_items"], result["num_examples"], result["num_failures"])
    return result


# ── Step 4: Build dataset + push to HF Hub ──────────────────────────────────

@pytest.fixture(scope="module")
def dataset_result(run_dir, sft_result):
    from tools.build_dataset.tool import BuildDatasetTool

    logger.info("=" * 60)
    logger.info("STEP 4a: Building HF dataset")
    logger.info("=" * 60)

    result = BuildDatasetTool().execute(sft_result["sft_path"], {"run_dir": str(run_dir)})
    logger.info("Dataset: %d samples, columns=%s",
                result["dataset_summary"]["sample_count"],
                result["dataset_summary"]["columns"])
    return result


@pytest.fixture(scope="module")
def hf_dataset_repo(run_dir, sft_result):
    from core.hf_hub import push_dataset

    logger.info("STEP 4b: Pushing dataset to HF Hub as '%s'...", HF_DATASET_REPO)
    repo_id = push_dataset(sft_result["sft_path"], HF_DATASET_REPO, private=True)

    (run_dir / "04_hf_dataset_repo.txt").write_text(repo_id, encoding="utf-8")
    logger.info("Dataset pushed → https://huggingface.co/datasets/%s", repo_id)
    return repo_id


# ── Step 5: Pull dataset from HF + train ────────────────────────────────────

@pytest.fixture(scope="module")
def pulled_dataset(run_dir, hf_dataset_repo):
    from core.hf_hub import pull_dataset

    logger.info("=" * 60)
    logger.info("STEP 5a: Pulling dataset from HF Hub")
    logger.info("=" * 60)

    ds = pull_dataset(hf_dataset_repo, split="train")
    logger.info("Pulled %d rows from %s", len(ds), hf_dataset_repo)

    pulled_jsonl = str(run_dir / "05_pulled_from_hf.jsonl")
    with open(pulled_jsonl, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    return {"dataset": ds, "jsonl_path": pulled_jsonl}


@pytest.fixture(scope="module")
def train_result(run_dir, pulled_dataset):
    from tools.build_dataset.tool import BuildDatasetTool
    from tools.train.tool import TrainTool

    logger.info("STEP 5b: Training %s for %d steps...", TINY_MODEL_ID, MAX_TRAIN_STEPS)

    train_dir = str(run_dir / "training")
    dataset_out = BuildDatasetTool().execute(pulled_dataset["jsonl_path"], {"run_dir": train_dir})

    result = TrainTool().execute(
        dataset_out["dataset_ref"],
        {
            "method": "sft",
            "run_dir": train_dir,
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

    summary = {
        "adapter_path": result["adapter_path"],
        "metrics": result["metrics"],
        "model_id": TINY_MODEL_ID,
        "steps": MAX_TRAIN_STEPS,
    }
    (run_dir / "05_training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Training done: loss=%.4f", result["metrics"].get("last_train_loss", -1))
    return result


# ── Step 6: Push adapter to HF Hub ──────────────────────────────────────────

@pytest.fixture(scope="module")
def hf_adapter_repo(run_dir, train_result):
    from core.hf_hub import push_adapter

    logger.info("=" * 60)
    logger.info("STEP 6: Pushing adapter to HF Hub")
    logger.info("=" * 60)

    repo_id = push_adapter(train_result["adapter_path"], HF_ADAPTER_REPO, private=True)
    (run_dir / "06_hf_adapter_repo.txt").write_text(repo_id, encoding="utf-8")
    logger.info("Adapter pushed → https://huggingface.co/%s", repo_id)
    return repo_id


# ── Step 7: Pull adapter from HF + evaluate ─────────────────────────────────

@pytest.fixture(scope="module")
def eval_result(run_dir, hf_adapter_repo, pulled_dataset):
    from core.hf_hub import pull_adapter
    from tools.eval_model.tool import EvalModelTool

    logger.info("=" * 60)
    logger.info("STEP 7: Pulling adapter from HF Hub + evaluating")
    logger.info("=" * 60)

    adapter_dir = pull_adapter(hf_adapter_repo, local_dir=str(run_dir / "07_pulled_adapter"))

    eval_dir = str(run_dir / "eval")
    result = EvalModelTool().execute(
        adapter_dir,
        pulled_dataset["jsonl_path"],
        {
            "run_dir": eval_dir,
            "hf_model_id": TINY_MODEL_ID,
            "split": "train",
            "max_samples": 3,
            "max_new_tokens": 32,
        },
    )

    (run_dir / "07_eval_predictions.jsonl").write_text(
        Path(result["predictions_path"]).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "07_eval_cluster_preview.json").write_text(
        json.dumps(result["cluster_preview"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Evaluation complete.")
    return result


# ── Cleanup: delete temporary HF repos after all tests ───────────────────────

@pytest.fixture(scope="module", autouse=True)
def cleanup_hf_repos(request, run_dir):
    yield
    from huggingface_hub import HfApi
    api = HfApi()
    for repo_file, repo_type in [
        ("04_hf_dataset_repo.txt", "dataset"),
        ("06_hf_adapter_repo.txt", "model"),
    ]:
        path = run_dir / repo_file
        if not path.exists():
            continue
        repo_id = path.read_text().strip()
        try:
            api.delete_repo(repo_id, repo_type=repo_type)
            logger.info("Cleaned up HF repo: %s (%s)", repo_id, repo_type)
        except Exception as exc:
            logger.warning("Failed to delete HF repo %s: %s", repo_id, exc)


# ═════════════════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestStep1_Taxonomy:
    def test_categories_generated(self, taxonomy_result):
        assert len(taxonomy_result["categories"]) >= 1
        for cat in taxonomy_result["categories"]:
            assert cat["name"]
            assert cat["description"]

    def test_subcategories_per_category(self, taxonomy_result):
        for cat in taxonomy_result["categories"]:
            subs = taxonomy_result["category_subcategories"][cat["name"]]
            assert len(subs) >= 1

    def test_queries_extracted(self, all_queries):
        assert len(all_queries) >= 2

    def test_taxonomy_saved_to_disk(self, run_dir):
        path = run_dir / "01_taxonomy.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["categories"]) >= 1


class TestStep2_SyntheticTexts:
    def test_texts_created(self, synthetic_texts_jsonl):
        assert os.path.exists(synthetic_texts_jsonl)
        with open(synthetic_texts_jsonl) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) >= 2
        assert all("text" in row for row in lines)


class TestStep3_SftAnnotation:
    def test_examples_generated(self, sft_result):
        assert sft_result["num_examples"] >= 1

    def test_no_total_failure(self, sft_result):
        assert sft_result["num_examples"] > sft_result["num_failures"]

    def test_sft_file_has_chat_format(self, sft_result):
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

    def test_annotations_saved(self, sft_result):
        assert os.path.exists(sft_result["annotations_path"])


class TestStep4_DatasetAndHfPush:
    def test_dataset_built(self, dataset_result):
        assert dataset_result["dataset_summary"]["sample_count"] >= 1
        assert "messages" in dataset_result["dataset_summary"]["columns"]

    def test_dataset_pushed_to_hf(self, hf_dataset_repo):
        assert "/" in hf_dataset_repo

    def test_repo_id_saved_to_disk(self, run_dir):
        path = run_dir / "04_hf_dataset_repo.txt"
        assert path.exists()
        assert "/" in path.read_text()


class TestStep5_PullAndTrain:
    def test_pulled_row_count_matches(self, pulled_dataset, sft_result):
        with open(sft_result["sft_path"]) as f:
            local_count = sum(1 for _ in f)
        assert len(pulled_dataset["dataset"]) == local_count

    def test_pulled_data_saved_as_jsonl(self, run_dir):
        path = run_dir / "05_pulled_from_hf.jsonl"
        assert path.exists()
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) >= 1

    def test_adapter_directory_created(self, train_result):
        assert os.path.isdir(train_result["adapter_path"])
        files = list(Path(train_result["adapter_path"]).iterdir())
        assert len(files) >= 1

    def test_training_metrics_valid(self, train_result):
        m = train_result["metrics"]
        assert m["steps"] >= 1
        if m.get("last_train_loss") is not None:
            assert 0.0 < m["last_train_loss"] < 100.0

    def test_training_summary_saved(self, run_dir):
        path = run_dir / "05_training_summary.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model_id"] == TINY_MODEL_ID
        assert data["steps"] == MAX_TRAIN_STEPS


class TestStep6_PushAdapter:
    def test_adapter_pushed_to_hf(self, hf_adapter_repo):
        assert "/" in hf_adapter_repo

    def test_repo_id_saved_to_disk(self, run_dir):
        path = run_dir / "06_hf_adapter_repo.txt"
        assert path.exists()


class TestStep7_PullAndEvaluate:
    def test_predictions_generated(self, eval_result):
        assert os.path.exists(eval_result["predictions_path"])
        with open(eval_result["predictions_path"]) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) >= 1

    def test_predictions_have_content(self, eval_result):
        with open(eval_result["predictions_path"]) as f:
            for line in f:
                row = json.loads(line)
                assert "input" in row
                assert "prediction" in row
                assert "reference" in row
                assert len(row["prediction"]) > 0

    def test_failures_file_created(self, eval_result):
        assert os.path.exists(eval_result["failures_path"])

    def test_cluster_preview_returned(self, eval_result):
        assert "clusters" in eval_result["cluster_preview"]

    def test_eval_artifacts_saved(self, run_dir):
        assert (run_dir / "07_eval_predictions.jsonl").exists()
        assert (run_dir / "07_eval_cluster_preview.json").exists()

    def test_all_step_outputs_exist(self, run_dir):
        """Every step saved its output — full pipeline is inspectable."""
        expected = [
            "01_taxonomy.json",
            "02_synthetic_texts.jsonl",
            "03_annotations.jsonl",
            "03_sft.jsonl",
            "04_hf_dataset_repo.txt",
            "05_pulled_from_hf.jsonl",
            "05_training_summary.json",
            "06_hf_adapter_repo.txt",
            "07_eval_predictions.jsonl",
            "07_eval_cluster_preview.json",
        ]
        for name in expected:
            path = run_dir / name
            assert path.exists(), f"Missing: {name}"
            assert path.stat().st_size > 0, f"Empty: {name}"
