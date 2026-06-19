# Horde Agent

Horde Agent is a constrained LangGraph pipeline for creating domain-specific SFT data, collecting source material, building a train/validation dataset, training a LoRA adapter, evaluating it, and generating a report.

The main v1 path is `full_agentic`. It is not an open-ended autonomous agent. It is a bounded controller over known tools and a known graph, with quality gates, recovery decisions, resume support, and LangSmith tracing.

## Current Flow

`generate_taxonomy -> collect_data -> assess_coverage_and_refine_queries -> build_sft_dataset -> build_dataset -> train_model -> evaluate_model -> generate_report`

The stages do the following:

- `generate_taxonomy`: builds culture/domain categories, subcategories, text search queries, and optional language-agnostic image taxonomy slots.
- `collect_data`: uses Serper text search and, when enabled, Serper Google Images to collect raw text and image data.
- `assess_coverage_and_refine_queries`: checks whether collected text/images cover enough queries and image taxonomy slots; if not, it routes back to collection with refined text queries or targeted image query specs.
- `build_sft_dataset`: converts collected text or images into SFT examples using the configured LLM; when `HF_DATASET_REPO` is configured, the SFT JSONL is pushed to Hugging Face Hub.
- `build_dataset`: builds a Hugging Face dataset with `train` and `validation` splits.
- `train_model`: runs text LoRA SFT or image-text LoRA SFT, unless debug stubbing is enabled; when `HF_ADAPTER_REPO` is configured, the produced LoRA adapter is pushed to Hugging Face Hub.
- `evaluate_model`: checks train-health logs, runs deterministic validation evaluation, and can optionally run a categorical LLM-as-judge quality gate.
- `generate_report`: writes the final run report from collected artifacts and metrics.

Agentic behavior currently happens in bounded places:

- Taxonomy mini-loop: failed category, subcategory, or query quality gates trigger targeted LLM repair calls before collection.
- Coverage mini-loop: weak collection coverage can trigger new text queries or image query specs and rerun collection.
- Recovery planner: failed quality reports choose a legal upstream stage and bounded config deltas instead of blindly continuing.
- Evaluation feedback: eval failures can request more collection coverage, stricter SFT prompts, or training stabilization.
- Resume controller: existing completed stages require confirmation or explicit CLI flags before reuse.

Everything else is deterministic tool execution.

## Project Layout

- `agent/main.py`: CLI entrypoint, logging setup, and CLI-to-config overrides.
- `agent/orchestrator.py`: creates tool instances and starts the runner.
- `agent/workflow.py`: routes `full`, `workflow`, `minimal_agentic`, and `full_agentic` modes.
- `config/pipeline_config.py`: single source of truth for env and CLI configuration.
- `core/agentic/`: LangGraph runtime, action space, state model, quality validators, recovery planner, coverage review, resume, and LangSmith observer.
- `core/data/`: dataset manifest, modality, and validation helpers.
- `core/ml/`: Hugging Face loading and LoRA helpers.
- `tools/generate_taxonomy/`: category/subcategory/query generation, quality gates, and image taxonomy.
- `tools/collect_data/`: Serper text collection, Serper image collection, and legacy HTML image parsing.
- `tools/build_sft_dataset/`: text and image SFT example generation.
- `tools/build_dataset/`: Hugging Face dataset creation and train/validation split.
- `tools/train/`: text and image LoRA SFT trainers.
- `tools/eval_model/`: evaluation, train-health checks, failure analysis, and optional LLM judge.
- `tools/reporting/`: final report generation.
- `tests/`: unit/integration tests, plus live tests that require external services.

## Setup

Use the existing `agents` pyenv environment if it is already installed:

```bash
PYENV_VERSION=agents pyenv exec python -m agent.main --help
```

For a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
```

GPU runs should use `requirements.txt` in an environment with the correct CUDA/PyTorch stack.

Create `.env` from `env.example`:

```bash
cp env.example .env
```

Required for `full_agentic`:

- `LANGSMITH_API_KEY`: required because agent observability is hard-wired through LangSmith.
- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`: used for taxonomy and SFT annotation.
- `SERPER_API_KEY`: used for text and image collection.
- `COUNTRY`, `RUN_DIR`, `SFT_TARGET_LANGUAGE`, `HF_MODEL_ID`: core run settings.

Optional but common:

- `HF_TOKEN`: needed for private/gated Hugging Face models and optional pushes.
- `HF_DATASET_REPO`, `HF_ADAPTER_REPO`: push the generated SFT dataset and LoRA adapter to Hugging Face Hub from `full_agentic` and `workflow` runs. Values may be either repo names such as `horde-agent-kazakhstan-lora` or full repo ids such as `my-org/horde-agent-kazakhstan-lora`.
- `LANGSMITH_PROJECT`: defaults to `horde-agent`.
- `TEXT_QUALITY_ENABLE_EMBEDDINGS=true`: enable embedding near-duplicate diagnostics. The default model is `Qwen/Qwen3-Embedding-0.6B`; leave this disabled for quick smoke runs if you do not want an extra model download.

## CPU Debug Run

This exercises taxonomy, collection, SFT annotation, dataset build, report generation, state persistence, recovery logic, and LangSmith traces. It stubs GPU-heavy train/eval.

```bash
PYENV_VERSION=agents pyenv exec python -m agent.main \
  --mode full_agentic \
  --country Kazakhstan \
  --out_dir output/debug_cpu \
  --fresh-run \
  --max_queries 20 \
  --max_queries_per_category 5 \
  --max_steps 1 \
  --max_samples 16 \
  --dataset-val-ratio 0.2 \
  --debug-stub-train \
  --debug-stub-eval \
  --resume-confirm-completed
```

This still calls the configured LLM and Serper APIs. For no-network validation, run the test suite instead.

## GPU Smoke Runs

Text SFT:

```bash
PYENV_VERSION=agents pyenv exec python -m agent.main \
  --mode full_agentic \
  --country Kazakhstan \
  --training-modality text \
  --out_dir output/gpu_smoke_text \
  --fresh-run \
  --max_queries 20 \
  --max_queries_per_category 5 \
  --max_steps 20 \
  --max_samples 64 \
  --dataset-val-ratio 0.2 \
  --eval-split validation \
  --resume-confirm-completed
```

Image SFT needs image collection and an image-text base model:

```bash
PYENV_VERSION=agents pyenv exec python -m agent.main \
  --mode full_agentic \
  --country Kazakhstan \
  --training-modality image \
  --out_dir output/gpu_smoke_image \
  --fresh-run \
  --max_queries 10 \
  --max_steps 10 \
  --max_samples 16 \
  --resume-confirm-completed
```

Set these in `.env` for image runs:

```bash
COLLECT_IMAGES=true
ENABLE_IMAGE_TAXONOMY=true
IMAGE_COLLECTION_MODE=serper
IMAGE_TAXONOMY_MAX_SLOTS=3
HF_MODEL_ID=<image-text-model-id>
```

Optional image near-duplicate filtering can run after image download and before image SFT annotation:

```bash
IMAGE_DEDUP_ENABLE=true
IMAGE_DEDUP_THRESHOLD=0.90
IMAGE_DEDUP_MODEL_PATH=models/sscd_disc_mixup.torchscript.pt
```

When enabled, the SSCD TorchScript model is downloaded from Facebook Research if missing, then used for batched inference. The pipeline keeps `collect/images_raw.json`, writes the deduped active manifest to `collect/images.json`, and stores cluster/pair details in `collect/image_dedup_report.json`.

Keep `EVAL_ENABLE_LLM_JUDGE=false` for the first GPU smoke run. Enable it only after the deterministic train/eval path works.

Judge-enabled eval can be run from an existing trained adapter:

```bash
EVAL_ENABLE_LLM_JUDGE=true EVAL_MAX_SAMPLES=8 PYENV_VERSION=agents pyenv exec python -m agent.main \
  --mode full_agentic \
  --country Kazakhstan \
  --out_dir output/gpu_smoke_text \
  --restart-from-stage evaluate_model \
  --resume-confirm-completed \
  --eval-enable-llm-judge
```

The judge reuses the configured `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_API_KEY` path. It asks for minimal categorical JSON:

```json
{
  "verdict": "pass|minor_issue|major_failure",
  "grounding": "supported|unsupported|insufficient_source",
  "categories": ["wrong_fact|missing_key_point|hallucination|irrelevant|format|language|unsafe|other"]
}
```

When judge is enabled, deterministic string-similarity failures are kept as diagnostics, but the judge gate is the primary quality signal. If source metadata is available, the judge checks whether predictions are supported by retained source excerpts.

## Held-Out Source Evaluation

For text runs, `full_agentic` splits collected source groups before SFT annotation:

- train source groups are converted into SFT examples for LoRA training.
- held-out source groups are converted into eval-only QA examples and are never used for training.

The split is deterministic by `SEED` and can be tuned from `.env`:

```bash
SOURCE_EVAL_ENABLE=true
SOURCE_EVAL_RATIO=0.10
SOURCE_EVAL_MAX_ITEMS=8
EVAL_COMPARE_BASE_MODEL=true
```

When held-out eval examples exist, `evaluate_model` uses them instead of the internal dataset validation split. It first predicts with the base model, then predicts with the trained LoRA adapter on the same examples, and writes `eval/attempt_N/lift_summary.json` with deltas such as quality score, failure rate, major failure rate, and unsupported grounding rate.

## Hugging Face Uploads

Set these in `.env` to publish artifacts from `full_agentic`:

```bash
HF_TOKEN=hf_...
HF_DATASET_REPO=horde-agent-kazakhstan-sft
HF_ADAPTER_REPO=horde-agent-kazakhstan-lora
```

`HF_USERNAME` is optional. If it is omitted, the Hub username is resolved from `HF_TOKEN`. If a repo value already includes an owner, for example `my-org/horde-agent-kazakhstan-lora`, that owner is used.

Adapter upload happens only after real `train_model` succeeds. `debug_stub_train` skips adapter upload to avoid publishing dummy adapters. Dataset upload happens after `build_sft_dataset` succeeds.

The upload path writes Hub `README.md` cards:

- Dataset cards describe modality, source/filtering quality, duplicate diagnostics, and train/validation split metadata once `build_dataset` succeeds.
- Adapter cards describe the base model, training settings/metrics, and are updated after `evaluate_model` with base-vs-adapter lift, failure rate, train-health, judge quality, and unsupported-grounding metrics.

Upload repo ids, card update status, or upload errors are recorded in agent artifacts and the final report.

## Text Filtering

Collection applies conservative text filtering before SFT annotation. It drops empty/short pages, highly repetitive text, duplicate URLs, exact duplicate text, and cheap shingle near-duplicates. Defaults can be tuned from `.env`:

```bash
TEXT_FILTER_ENABLE=true
TEXT_FILTER_MIN_CHARS=300
TEXT_FILTER_MIN_WORDS=40
TEXT_FILTER_MIN_UNIQUE_WORD_RATIO=0.15
TEXT_FILTER_SHINGLE_THRESHOLD=0.90
```

The filter writes `collect/text_filter_report.json`. If every real page is filtered out, the recovery policy expands collection and relaxes length thresholds instead of silently training on the placeholder row.

## Text Quality Diagnostics

`full_agentic` writes compact text-quality summaries after collection and SFT generation:

- `collect/text_quality.json`: diagnostics over scraped page text from `serper_raw.json`.
- `sft/text_quality.json`: diagnostics over the generated chat SFT JSONL.

The reports include exact normalized duplicate rate, canonical URL duplicate rate, shingle/Jaccard near-duplicate counts, length stats, script mix, domain distribution, and a capped list of example duplicate pairs. Embedding near-duplicate detection is opt-in:

```bash
TEXT_QUALITY_ENABLE_EMBEDDINGS=true
TEXT_QUALITY_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
TEXT_QUALITY_EMBEDDING_THRESHOLD=0.93
TEXT_QUALITY_MAX_EMBEDDING_ITEMS=256
```

The embedding pass records errors inside `text_quality.json` instead of failing the pipeline, so a missing model download or device issue does not hide the primary train/eval result.

## Resume and Restart

`full_agentic` writes state into `RUN_DIR`. If a run directory already contains `agent_state.json`, the controller resumes from it.

Useful flags:

- `--resume-confirm-completed`: reuse completed stages without pausing.
- `--restart-from-stage collect_data`: clear that stage and downstream stages, then resume.
- `--fresh-run`: clear the whole agentic stage state in the run directory.

If you change data collection or taxonomy settings, prefer a new `RUN_DIR` or restart from the affected upstream stage.

Recovery is bounded by retry count and by failure signatures. If a failed stage would repeat the same recovery reason, config delta, blocking issues, and key metrics, the controller stops with `recovery_stalled_same_failure_signature` instead of spending another retry on an unchanged loop. Explicit restart/rebuild paths clear old recovery signatures.

## Outputs and Traces

Local artifacts are written under `RUN_DIR`:

- `agent_state.json`: latest serialized pipeline state with credential-like values redacted, including recorded recovery fingerprints.
- `artifact_manifest.json`: active local artifact paths, owning stage, existence, size, attempt, and iteration metadata.
- `agent_trace.jsonl`: stage trajectory.
- `decision_history.jsonl`, `quality_history.jsonl`, `result_history.jsonl`, `config_history.jsonl`: inspectable controller history.
- `collect/`: raw collection output, text filter report, metadata, collection text-quality diagnostics, and optional image dedup reports.
- `sft/`: train-source annotations/SFT JSONL, held-out source eval JSONL when available, and SFT text-quality diagnostics.
- `dataset/`: Hugging Face train/validation dataset.
- `eval/attempt_N/`: adapter validation outputs, base-model validation outputs under `base/`, deterministic diagnostics, clustered failures, judge artifacts, and lift summary for each evaluation attempt.
- Final report path is logged at the end when report generation succeeds.

LangSmith traces are available in the project configured by `LANGSMITH_PROJECT` or `horde-agent`. Each run has a root span and one child span per stage.

## Tests

Run the non-live suite:

```bash
PYENV_VERSION=agents pyenv exec python -m pytest \
  tests/test_agentic_tool_adapters.py \
  tests/test_agentic_workflow_integration.py \
  tests/test_pipeline_integration.py \
  tests/test_minimal_agentic_flow.py \
  tests/test_agentic_langgraph_runtime.py \
  tests/test_agentic_v1_core.py \
  tests/test_cli_logging.py \
  tests/test_agentic_observability.py \
  tests/test_image_manifest_loader.py \
  tests/test_taxonomy_mini_loop.py \
  tests/test_image_taxonomy.py \
  tests/test_collect_data_image_search.py \
  tests/test_eval_quality.py \
  tests/test_image_training.py \
  tests/test_text_quality.py \
  tests/test_text_filter.py \
  tests/test_image_dedup.py
```

Live tests such as `tests/test_live_pipeline.py` and `tests/test_full_live_pipeline.py` require real API keys and network access.

## Known Caveats

- `full_agentic` is constrained-agentic, not a general planner. It chooses among legal graph actions and bounded recovery deltas.
- Image training is newer than text training. Validate the chosen image-text model, chat template, processor behavior, and LoRA target modules on a tiny GPU run first.
- LLM-as-judge is optional. When enabled, it is the primary semantic quality gate, while train-health checks and deterministic validation metrics still run as diagnostics.
- CPU debug mode does not prove GPU training stability. It only validates orchestration, artifacts, state, and reporting.
