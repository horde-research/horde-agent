# Horde Agent

Horde Agent is a constrained LangGraph pipeline for creating domain-specific SFT data, collecting source material, building a train/validation dataset, training a LoRA adapter, evaluating it, and generating a report.

The main v1 path is `full_agentic`. It is not an open-ended autonomous agent. It is a bounded controller over known tools and a known graph, with quality gates, recovery decisions, resume support, and LangSmith tracing.

## Current Flow

`generate_taxonomy -> collect_data -> assess_coverage_and_refine_queries -> build_sft_dataset -> build_dataset -> train_model -> evaluate_model -> generate_report`

The stages do the following:

- `generate_taxonomy`: builds culture/domain categories, subcategories, text search queries, and optional language-agnostic image taxonomy slots.
- `collect_data`: uses Serper text search and, when enabled, Serper Google Images to collect raw text and image data.
- `assess_coverage_and_refine_queries`: checks whether collected text/images cover enough queries and image taxonomy slots; if not, it routes back to collection with refined text queries or targeted image query specs.
- `build_sft_dataset`: converts collected text or images into SFT examples using the configured LLM.
- `build_dataset`: builds a Hugging Face dataset with `train` and `validation` splits.
- `train_model`: runs text LoRA SFT or image-text LoRA SFT, unless debug stubbing is enabled.
- `evaluate_model`: checks train-health logs, runs deterministic validation evaluation, and can optionally run LLM-as-judge.
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
- `HF_DATASET_REPO`, `HF_ADAPTER_REPO`: push outputs to Hugging Face Hub.
- `LANGSMITH_PROJECT`: defaults to `horde-agent`.

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

Keep `EVAL_ENABLE_LLM_JUDGE=false` for the first GPU smoke run. Enable it only after the deterministic train/eval path works.

## Resume and Restart

`full_agentic` writes state into `RUN_DIR`. If a run directory already contains `agent_state.json`, the controller resumes from it.

Useful flags:

- `--resume-confirm-completed`: reuse completed stages without pausing.
- `--restart-from-stage collect_data`: clear that stage and downstream stages, then resume.
- `--fresh-run`: clear the whole agentic stage state in the run directory.

If you change data collection or taxonomy settings, prefer a new `RUN_DIR` or restart from the affected upstream stage.

## Outputs and Traces

Local artifacts are written under `RUN_DIR`:

- `agent_state.json`: latest serialized pipeline state.
- `agent_trace.jsonl`: stage trajectory.
- `decision_history.jsonl`, `quality_history.jsonl`, `result_history.jsonl`, `config_history.jsonl`: inspectable controller history.
- `collect/`: raw collection output and metadata.
- `sft/`: annotations and SFT JSONL.
- `dataset/`: Hugging Face train/validation dataset.
- `eval/`: evaluation metrics and failure analysis.
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
  tests/test_image_training.py
```

Live tests such as `tests/test_live_pipeline.py` and `tests/test_full_live_pipeline.py` require real API keys and network access.

## Known Caveats

- `full_agentic` is constrained-agentic, not a general planner. It chooses among legal graph actions and bounded recovery deltas.
- Image training is newer than text training. Validate the chosen image-text model, chat template, processor behavior, and LoRA target modules on a tiny GPU run first.
- LLM-as-judge is optional and should not be the only eval signal. Train-health checks and deterministic validation metrics still run without it.
- CPU debug mode does not prove GPU training stability. It only validates orchestration, artifacts, state, and reporting.
