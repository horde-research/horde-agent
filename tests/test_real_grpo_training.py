"""Opt-in real SFT -> GRPO smoke test.

Run mocked-judge training manually:
    RUN_REAL_GRPO_TRAINING_TEST=1 PYTHONPATH=. pytest tests/test_real_grpo_training.py -q

Run real-LLM judge training manually:
    RUN_REAL_GRPO_LLM_TEST=1 PYTHONPATH=. pytest tests/test_real_grpo_training.py::test_real_tiny_model_trains_sft_then_grpo_with_real_llm_judge -q -s

The test loads a real Hugging Face causal LM, trains LoRA SFT on the SFT split,
then trains GRPO on the RL split. The LLM judge is mocked so the test exercises
the RL trainer/optimizer path without spending API credits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_GRPO_TRAINING_TEST") != "1"
    and os.getenv("RUN_REAL_GRPO_LLM_TEST") != "1",
    reason="Set RUN_REAL_GRPO_TRAINING_TEST=1 or RUN_REAL_GRPO_LLM_TEST=1 to run real training smoke tests.",
)


class FakeJudgeClient:
    def generate_json_batch_sync(self, requests, *, batch_size=5, batch_delay_seconds=1.5):
        from core.llm.client import LLMResponse

        return [
            LLMResponse(
                request_id=request.request_id,
                success=True,
                data={"score": 0.8, "rationale": "Deterministic smoke-test reward."},
            )
            for request in requests
        ]


def _write_culture_sft_jsonl(path: Path) -> None:
    examples = [
        (
            "What is beshbarmak?",
            "Beshbarmak is a traditional Kazakh dish of boiled meat served over flat noodles, often shared at gatherings.",
        ),
        (
            "What is kumis?",
            "Kumis is a fermented drink made from mare milk and associated with Central Asian nomadic food traditions.",
        ),
        (
            "What is a dombra?",
            "A dombra is a long-necked two-string instrument important in Kazakh music and oral storytelling.",
        ),
        (
            "What is a yurt?",
            "A yurt is a portable round dwelling used by nomadic peoples, built with a wooden frame and felt coverings.",
        ),
        (
            "Why is hospitality important in Kazakh culture?",
            "Hospitality matters because welcoming guests with food, tea, and respect expresses generosity and social honor.",
        ),
        (
            "What is baursak?",
            "Baursak is fried dough served with tea and festive meals in Kazakh and other Central Asian traditions.",
        ),
        (
            "What is Nauryz?",
            "Nauryz is a spring renewal holiday celebrated with food, visits, and customs marking a new year in the region.",
        ),
        (
            "What is shubat?",
            "Shubat is fermented camel milk, traditionally consumed in parts of Kazakhstan and Central Asia.",
        ),
        (
            "How should an answer about culture be written?",
            "It should be respectful, specific, and avoid stereotypes or unsupported generalizations.",
        ),
        (
            "How can local terms be handled?",
            "Local terms can be used with brief explanations so readers understand their meaning and cultural context.",
        ),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for prompt, answer in examples:
            handle.write(json.dumps({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }, ensure_ascii=False) + "\n")


def _tiny_grpo_config(tmp_path, *, run_name: str) -> dict:
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "culture_sft.jsonl"
    _write_culture_sft_jsonl(data_path)

    return {
        "mode": "workflow",
        "data_path": str(data_path),
        "run_dir": str(run_dir),
        "country": "Kazakhstan",
        "sft_target_language": "English",
        "hf_model_id": "sshleifer/tiny-gpt2",
        "max_iters": 1,
        "max_steps": 1,
        "max_samples": 10,
        "sft_train_fraction": 0.8,
        "train_batch_size": 1,
        "train_grad_accum": 1,
        "train_max_seq_len": 64,
        "train_eval_steps": 1,
        "enable_grpo": True,
        "grpo_steps": 1,
        "grpo_batch_size": 1,
        "grpo_grad_accum": 1,
        "grpo_num_generations": 2,
        "grpo_max_prompt_length": 64,
        "grpo_max_completion_length": 16,
        "grpo_judge_batch_size": 2,
        "grpo_judge_batch_delay": 0.0,
        "eval_max_samples": 2,
        "eval_max_new_tokens": 8,
    }


def _assert_sft_then_grpo_result(result: dict, run_dir: Path, *, expected_judge_text: str | None = None) -> None:
    assert result["adapter_path"]
    assert Path(result["adapter_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["predictions_path"]).exists()

    iterations = json.loads((run_dir / "iterations.json").read_text(encoding="utf-8"))
    assert [record["method"] for record in iterations] == ["sft", "grpo"]
    assert Path(iterations[0]["adapter_path"]).exists()
    assert Path(iterations[1]["adapter_path"]).exists()

    judge_log = Path(iterations[1]["log_paths"]["judge_rewards"])
    assert judge_log.exists()
    judge_text = judge_log.read_text(encoding="utf-8")
    assert '"score"' in judge_text
    if expected_judge_text is not None:
        assert expected_judge_text in judge_text


def _require_real_training_deps() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    pytest.importorskip("peft")
    pytest.importorskip("trl")


def test_real_tiny_model_trains_sft_then_grpo_on_split(tmp_path):
    if os.getenv("RUN_REAL_GRPO_TRAINING_TEST") != "1":
        pytest.skip("Set RUN_REAL_GRPO_TRAINING_TEST=1 to run the mocked-judge training smoke test.")
    _require_real_training_deps()

    from agent.orchestrator import Orchestrator

    config = _tiny_grpo_config(tmp_path, run_name="grpo_smoke_mock_judge")
    run_dir = Path(config["run_dir"])

    with patch("core.llm.client.LLMClient.from_env", return_value=FakeJudgeClient()):
        result = Orchestrator(config).run()

    _assert_sft_then_grpo_result(result, run_dir, expected_judge_text="Deterministic smoke-test reward")


def test_real_tiny_model_trains_sft_then_grpo_with_real_llm_judge(tmp_path):
    if os.getenv("RUN_REAL_GRPO_LLM_TEST") != "1":
        pytest.skip("Set RUN_REAL_GRPO_LLM_TEST=1 to run the real-LLM judge training smoke test.")
    _require_real_training_deps()

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv()
    if not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("LLM_API_KEY is required for the real-LLM judge smoke test.")

    from agent.orchestrator import Orchestrator

    config = _tiny_grpo_config(tmp_path, run_name="grpo_smoke_real_llm_judge")
    run_dir = Path(config["run_dir"])

    result = Orchestrator(config).run()

    _assert_sft_then_grpo_result(result, run_dir)
