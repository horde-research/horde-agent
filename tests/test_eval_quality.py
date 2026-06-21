from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image

from core.agentic.validators import validate_eval_output
from core.llm.client import LLMResponse
from tools.eval_model.eval.error_analysis import cluster_failures
from tools.eval_model.eval.failures import collect_failures_with_metrics
from tools.eval_model.eval.inference import run_inference
from tools.eval_model.eval.llm_judge import run_llm_judge
from tools.eval_model.eval.train_health import evaluate_training_health
from tools.eval_model.tool import EvalModelTool


class FakeTextTokenizer:
    model_max_length = 64

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if tokens == [101, 102]:
            return "Paris."
        if tokens == [11, 12, 101, 102]:
            return "What is the capital of France? Paris."
        return " ".join(str(token) for token in tokens)


class ChatTemplateTextTokenizer(FakeTextTokenizer):
    def __init__(self) -> None:
        self.rendered_texts: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: ANN001
        rendered = "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(self, text, **kwargs):
        self.rendered_texts.append(text)
        return super().__call__(text, **kwargs)


class HugeMaxTextTokenizer(FakeTextTokenizer):
    model_max_length = 10**30

    def __init__(self) -> None:
        self.max_lengths: list[int] = []

    def __call__(self, text, **kwargs):
        self.max_lengths.append(kwargs["max_length"])
        return super().__call__(text, **kwargs)


class FakeTextModel:
    device = torch.device("cpu")

    def eval(self):
        return self

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[101, 102]], dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)


class FakeImageTokenizer:
    model_max_length = 64


class FakeImageProcessor:
    tokenizer = FakeImageTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "USER: <image>\nDescribe the image.\nASSISTANT:"

    def __call__(self, **kwargs):
        return {
            "input_ids": torch.tensor([[21, 22]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if tokens == [201, 202]:
            return "A red square."
        if tokens == [21, 22, 201, 202]:
            return "USER: <image>\nDescribe the image.\nASSISTANT: A red square."
        return " ".join(str(token) for token in tokens)


class FakeImageModel:
    def eval(self):
        return self

    def parameters(self):
        return iter([])

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[201, 202]], dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)


class FakeJudgeClient:
    def generate_json_batch_sync(self, requests, *, batch_size=5, batch_delay_seconds=1.5):
        responses = []
        for index, request in enumerate(requests):
            responses.append(
                LLMResponse(
                    request_id=request.request_id,
                    success=True,
                    data={
                        "verdict": "pass" if index == 0 else "major_failure",
                        "grounding": "supported" if index == 0 else "unsupported",
                        "categories": [] if index == 0 else ["irrelevant"],
                    },
                )
            )
        return responses


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_training_health_flags_exploding_loss(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {"step": 1, "loss": 1.0, "grad_norm": 1.0},
            {"step": 2, "loss": 2.0, "grad_norm": 2.0},
            {"step": 3, "loss": 4.0, "grad_norm": 3.0},
        ],
    )

    report = evaluate_training_health({"metrics": str(metrics_path)}, expected_steps=3)

    assert report["gate_status"] == "repair"
    assert "training_loss_exploding" in report["blocking_issues"]
    assert report["metrics"]["loss_trend"] == "exploding"


def test_collect_failures_with_metrics_and_single_cluster(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {"id": 0, "input": "Say hello", "prediction": "", "reference": "Hello"},
            {"id": 1, "input": "Capital?", "prediction": "Paris", "reference": "Paris"},
        ],
    )

    failures_path, metrics = collect_failures_with_metrics(str(predictions_path), str(tmp_path))
    preview = cluster_failures(failures_path, str(tmp_path))

    assert metrics["num_predictions"] == 2
    assert metrics["num_failures"] == 1
    assert metrics["failure_rate"] == 0.5
    assert preview["clusters"][0]["label"] == "generation_empty_or_short"


def test_llm_judge_aggregates_major_failures(monkeypatch, tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {"id": 0, "input": "Q1", "prediction": "A1", "reference": "A1"},
            {
                "id": 1,
                "input": "Q2",
                "prediction": "wrong",
                "reference": "A2",
                "source_url": "https://example.com/source",
                "source_excerpt": "The answer is A2.",
                "group_key": "source-1",
            },
        ],
    )
    monkeypatch.setattr("tools.eval_model.eval.llm_judge.LLMClient.from_env", lambda **kwargs: FakeJudgeClient())

    summary = run_llm_judge(
        str(predictions_path),
        str(tmp_path),
        modality="text",
        target_language="English",
        provider="fake",
        model="fake",
        api_key="fake",
        batch_size=2,
        batch_delay=0.0,
    )

    assert summary["enabled"] is True
    assert summary["gate_status"] == "repair"
    assert summary["quality_score"] == 0.5
    assert summary["pass_count"] == 1
    assert summary["major_failure_count"] == 1
    assert summary["failure_category_counts"] == {"irrelevant": 1}
    assert summary["grounding_counts"] == {"supported": 1, "unsupported": 1}
    assert summary["unsupported_grounding_count"] == 1
    assert Path(summary["judge_results_path"]).exists()
    judge_text = Path(summary["judge_results_path"]).read_text(encoding="utf-8")
    assert "reason" not in judge_text
    assert "source-1" in judge_text


def test_llm_judge_includes_focus_in_prompt_and_summary(monkeypatch, tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": 0,
                "input": "What is the shanyrak?",
                "prediction": "A yurt crown.",
                "reference": "The shanyrak is the circular crown of a Kazakh yurt.",
                "source_excerpt": "The shanyrak is the circular crown of a Kazakh yurt.",
            },
        ],
    )
    captured_messages: list[str] = []

    class CapturingJudgeClient:
        def generate_json_batch_sync(self, requests, *, batch_size=5, batch_delay_seconds=1.5):  # noqa: ANN001
            captured_messages.extend(request.user_message for request in requests)
            return [
                LLMResponse(
                    request_id=requests[0].request_id,
                    success=True,
                    data={"verdict": "pass", "grounding": "supported", "categories": []},
                )
            ]

    monkeypatch.setattr("tools.eval_model.eval.llm_judge.LLMClient.from_env", lambda **kwargs: CapturingJudgeClient())

    summary = run_llm_judge(
        str(predictions_path),
        str(tmp_path),
        modality="text",
        target_language="English",
        focus="traditional culture",
        provider="fake",
        model="fake",
        api_key="fake",
        batch_size=1,
        batch_delay=0.0,
    )

    assert summary["focus"] == "traditional culture"
    assert captured_messages
    assert "Scope/focus: traditional culture" in captured_messages[0]


def test_eval_validator_uses_judge_gate_over_heuristic_failure_rate(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    predictions_path.write_text("{}", encoding="utf-8")
    failures_path.write_text("{}", encoding="utf-8")

    report = validate_eval_output(
        {
            "predictions_path": str(predictions_path),
            "failures_path": str(failures_path),
            "cluster_preview": {"clusters": [{"label": "semantic_mismatch", "count": 10}]},
            "metrics": {
                "failure_rate": 1.0,
                "training_health": {"gate_status": "pass"},
                "judge": {
                    "enabled": True,
                    "gate_status": "pass",
                    "major_failure_rate": 0.0,
                    "quality_score": 1.0,
                    "failure_category_counts": {},
                },
            },
        }
    )

    assert report.passed
    assert "eval_failure_rate_too_high" not in report.blocking_issues
    assert "eval_failure_clusters_present" not in report.warnings


def test_eval_validator_flags_unsupported_grounding(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    predictions_path.write_text("{}", encoding="utf-8")
    failures_path.write_text("{}", encoding="utf-8")

    report = validate_eval_output(
        {
            "predictions_path": str(predictions_path),
            "failures_path": str(failures_path),
            "cluster_preview": {"clusters": []},
            "metrics": {
                "failure_rate": 0.0,
                "training_health": {"gate_status": "pass"},
                "judge": {
                    "enabled": True,
                    "gate_status": "repair",
                    "major_failure_rate": 0.0,
                    "unsupported_grounding_rate": 0.5,
                    "failure_category_counts": {},
                },
            },
        }
    )

    assert not report.passed
    assert "eval_grounding_failure" in report.blocking_issues
    assert "eval_knowledge_missing" in report.blocking_issues
    assert report.metrics["judge_unsupported_grounding_rate"] == 0.5


def test_eval_validator_blocks_semantic_mismatch_rate_without_judge(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    predictions_path.write_text("{}", encoding="utf-8")
    failures_path.write_text("{}", encoding="utf-8")

    report = validate_eval_output(
        {
            "predictions_path": str(predictions_path),
            "failures_path": str(failures_path),
            "cluster_preview": {"clusters": [{"label": "semantic_mismatch", "count": 6}]},
            "metrics": {
                "failure_rate": 0.30,
                "num_predictions": 20,
                "training_health": {"gate_status": "pass"},
                "judge": {"enabled": False},
            },
        }
    )

    assert not report.passed
    assert "eval_semantic_mismatch_rate_too_high" in report.blocking_issues
    assert report.metrics["semantic_mismatch_rate"] == 0.3


def test_eval_validator_blocks_missing_training_metrics(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    predictions_path.write_text("{}", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")

    report = validate_eval_output(
        {
            "predictions_path": str(predictions_path),
            "failures_path": str(failures_path),
            "cluster_preview": {"clusters": []},
            "metrics": {
                "failure_rate": 0.0,
                "training_health": {
                    "gate_status": "warn",
                    "warnings": ["training_metrics_missing", "training_loss_missing"],
                },
                "judge": {"enabled": False},
            },
        }
    )

    assert not report.passed
    assert "eval_training_metrics_missing" in report.blocking_issues


def test_eval_validator_blocks_insufficient_judge_grounding(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    predictions_path.write_text("{}", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")

    report = validate_eval_output(
        {
            "predictions_path": str(predictions_path),
            "failures_path": str(failures_path),
            "cluster_preview": {"clusters": []},
            "metrics": {
                "failure_rate": 0.0,
                "training_health": {"gate_status": "pass"},
                "judge": {
                    "enabled": True,
                    "gate_status": "pass",
                    "major_failure_rate": 0.0,
                    "unsupported_grounding_rate": 0.0,
                    "grounding_counts": {"supported": 1, "insufficient_source": 3},
                    "num_judged": 4,
                    "failure_category_counts": {},
                },
            },
        }
    )

    assert not report.passed
    assert "eval_grounding_insufficient_source" in report.blocking_issues
    assert report.metrics["judge_insufficient_grounding_rate"] == 0.75


def test_eval_model_text_path_writes_metrics_and_disabled_judge(monkeypatch, tmp_path: Path) -> None:
    test_dataset = tmp_path / "text_sft.jsonl"
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_jsonl(
        test_dataset,
        [
            {
                "prompt": "What is the capital of France?",
                "response": "Paris.",
                "group_key": "source-france",
                "source_url": "https://example.com/france",
                "source_excerpt": "Paris is the capital city of France.",
            }
        ],
    )
    metrics_path = tmp_path / "train_metrics.jsonl"
    _write_jsonl(metrics_path, [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}])
    monkeypatch.setattr("tools.eval_model.tool.load_hf_causal_lm", lambda model_id: (FakeTextModel(), FakeTextTokenizer()))
    monkeypatch.setattr("tools.eval_model.tool.load_lora_adapters", lambda model, adapter_dir: model)

    result = EvalModelTool().execute(
        str(adapter_dir),
        str(test_dataset),
        {
            "run_dir": str(tmp_path),
            "hf_model_id": "fake-text-model",
            "training_modality": "text",
            "train_log_paths": {"metrics": str(metrics_path)},
            "max_steps": 2,
            "eval_enable_llm_judge": False,
            "max_samples": 1,
        },
    )

    assert Path(result["predictions_path"]).exists()
    prediction = json.loads(Path(result["predictions_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert prediction["source_url"] == "https://example.com/france"
    assert prediction["group_key"] == "source-france"
    assert prediction["source_excerpt"] == "Paris is the capital city of France."
    assert Path(result["eval_metrics_path"]).exists()
    assert result["metrics"]["training_modality"] == "text"
    assert result["training_health"]["gate_status"] == "pass"
    assert result["judge_summary"]["enabled"] is False
    assert Path(result["base_predictions_path"]).exists()
    assert Path(result["base_failures_path"]).exists()
    assert result["lift_summary"]["enabled"] is True
    assert result["lift_summary"]["failure_rate_delta"] == 0.0


def test_text_eval_uses_safe_input_length_for_tokenizer_sentinel(tmp_path: Path) -> None:
    tokenizer = HugeMaxTextTokenizer()
    dataset = Dataset.from_list([{"prompt": "Q", "response": "A"}])

    run_inference(
        model=FakeTextModel(),
        tokenizer=tokenizer,
        dataset=dataset,
        out_dir=str(tmp_path),
        max_samples=1,
        max_new_tokens=2,
    )

    assert tokenizer.max_lengths == [2048]


def test_text_eval_uses_chat_template_generation_prompt(tmp_path: Path) -> None:
    tokenizer = ChatTemplateTextTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris."},
                ]
            }
        ]
    )

    run_inference(
        model=FakeTextModel(),
        tokenizer=tokenizer,
        dataset=dataset,
        out_dir=str(tmp_path),
        max_samples=1,
        max_new_tokens=2,
    )

    assert tokenizer.rendered_texts
    assert tokenizer.rendered_texts[0] == "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"


def test_eval_model_image_path_uses_vlm_loader(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "red.jpg"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    test_dataset = tmp_path / "image_sft.jsonl"
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_jsonl(
        test_dataset,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image."},
                            {"type": "image", "image": str(image_path)},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]},
                ]
            }
        ],
    )
    monkeypatch.setattr(
        "tools.eval_model.tool.load_hf_image_text_model",
        lambda model_id: (FakeImageModel(), FakeImageProcessor()),
    )
    monkeypatch.setattr("tools.eval_model.tool.load_lora_adapters", lambda model, adapter_dir: model)

    result = EvalModelTool().execute(
        str(adapter_dir),
        str(test_dataset),
        {
            "run_dir": str(tmp_path),
            "hf_model_id": "fake-vlm",
            "training_modality": "image",
            "eval_enable_llm_judge": False,
            "max_samples": 1,
        },
    )

    prediction = json.loads(Path(result["predictions_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert prediction["image_path"] == str(image_path)
    assert prediction["prediction"] == "A red square."
    assert result["metrics"]["training_modality"] == "image"
    assert result["judge_summary"]["enabled"] is False
    assert Path(result["base_predictions_path"]).exists()
    assert result["lift_summary"]["enabled"] is True
