from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch


MODALITY_DECISION = {
    "modality": "text",
    "confidence": 0.96,
    "rationale": "The dataset contains chat-format text records.",
}

COMPONENT_SELECTION = {
    "dataset_loader_key": "hf_text_default",
    "model_loader_key": "hf_causal_lm_default",
    "lora_preset_key": "lora_attn_small",
    "trainer_key": "static_sft_default",
    "hf_model_id": "mock-agent-model",
    "primary_metric": "eval_loss",
    "rationale": "Use the default text SFT stack for this smoke test.",
}

TRAINING_ADJUSTMENT = {
    "should_retry": False,
    "lr_multiplier": 1.0,
    "batch_size_delta": 0,
    "grad_accum_delta": 0,
    "max_steps_delta": 0,
    "switch_lora_preset_key": None,
    "stop_reason": "Single mocked iteration is enough.",
    "rationale": "Mocked metrics are healthy.",
}

ERROR_ANALYSIS = {
    "cluster_labels": ["formatting"],
    "root_causes": ["Mocked mismatch cluster."],
    "data_fixes": ["Keep the current synthetic data."],
    "next_training_actions": ["No retry needed."],
}


def _fake_llm_response(data: Dict[str, Any], request_id: str):
    from core.llm.client import LLMResponse

    return LLMResponse(request_id=request_id, success=True, data=data)


def _fake_generate_json_sync(request):
    message = request.user_message.lower()
    if "modalitydecision" in message:
        return _fake_llm_response(MODALITY_DECISION, request.request_id)
    if "componentselectiondecision" in message:
        return _fake_llm_response(COMPONENT_SELECTION, request.request_id)
    if "trainingadjustmentdecision" in message:
        return _fake_llm_response(TRAINING_ADJUSTMENT, request.request_id)
    if "erroranalysisdecision" in message:
        return _fake_llm_response(ERROR_ANALYSIS, request.request_id)
    raise AssertionError(f"Unexpected LLM prompt: {request.user_message[:200]}")


def _write_sft_jsonl(path: Path) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i} about Kazakh culture"},
                {"role": "assistant", "content": f"Answer {i} about traditions."},
            ]
        }
        for i in range(6)
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fake_train_result(run_dir: Path) -> Dict[str, Any]:
    adapter_path = run_dir / "adapters" / "iter_0"
    adapter_path.mkdir(parents=True)
    train_log = run_dir / "train.log"
    metrics_path = run_dir / "metrics.jsonl"
    train_log.write_text("step 1 loss=2.4\nstep 5 loss=2.1\n", encoding="utf-8")
    metrics_path.write_text("", encoding="utf-8")
    metrics = {
        "steps": 5,
        "best_eval_loss": 2.0,
        "last_train_loss": 2.1,
        "last_eval_loss": 2.0,
    }
    config = {
        "lr": 0.0002,
        "batch_size": 4,
        "grad_accum": 4,
        "max_steps": 5,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_seq_len": 512,
        "eval_steps": 50,
        "seed": 42,
    }
    return {
        "adapter_path": str(adapter_path),
        "log_paths": {"train_log": str(train_log), "metrics": str(metrics_path)},
        "metrics": metrics,
        "iteration_record": {
            "iter_idx": 0,
            "config": config,
            "metrics": metrics,
            "adapter_path": str(adapter_path),
            "log_paths": {"train_log": str(train_log), "metrics": str(metrics_path)},
        },
    }


def _fake_eval_result(run_dir: Path) -> Dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    failures_path = run_dir / "failures.jsonl"
    predictions_path.write_text(
        json.dumps({"input": "Q", "prediction": "A", "reference": "A"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failures_path.write_text(
        json.dumps({"id": 0, "reason": "mocked"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "predictions_path": str(predictions_path),
        "failures_path": str(failures_path),
        "cluster_preview": {"clusters": [{"label": "formatting", "count": 1}]},
    }


def test_minimal_agentic_flow_smoke_checks_each_step(tmp_path: Path) -> None:
    from agent.orchestrator import Orchestrator
    from tools.build_dataset.tool import BuildDatasetTool

    data_path = tmp_path / "sft.jsonl"
    _write_sft_jsonl(data_path)

    mock_llm = MagicMock()
    mock_llm.generate_json_sync.side_effect = _fake_generate_json_sync
    train_result = _fake_train_result(tmp_path)
    eval_result = _fake_eval_result(tmp_path)

    original_build_execute = BuildDatasetTool.execute

    def _build_execute(self, data_path_arg, config_arg):
        return original_build_execute(self, data_path_arg, config_arg)

    with (
        patch("core.llm.client.LLMClient.from_env", return_value=mock_llm),
        patch.object(BuildDatasetTool, "execute", autospec=True, side_effect=_build_execute) as mock_build_dataset,
        patch("tools.train.tool.TrainTool.execute", return_value=train_result) as mock_train,
        patch("tools.eval_model.tool.EvalModelTool.execute", return_value=eval_result) as mock_eval,
    ):
        result = Orchestrator(
            {
                "mode": "minimal_agentic",
                "data_path": str(data_path),
                "run_dir": str(tmp_path),
                "max_iters": 1,
                "max_steps": 5,
                "hf_model_id": "test-model",
                "sft_target_language": "English",
            }
        ).run()

    assert result["mode"] == "minimal_agentic"
    assert result["component_selection"]["trainer_key"] == "static_sft_default"
    assert result["error_analysis"]["cluster_labels"] == ["formatting"]
    assert Path(result["report_path"]).exists()

    assert mock_build_dataset.call_count == 1
    build_args = mock_build_dataset.call_args.args
    assert build_args[1] == str(data_path)
    assert build_args[2]["run_dir"] == str(tmp_path)

    assert mock_train.call_count == 1
    train_dataset_ref, train_config = mock_train.call_args.args
    assert train_dataset_ref["data_path"] == str(tmp_path / "dataset")
    assert train_dataset_ref["source_data_path"] == str(data_path)
    assert train_dataset_ref["eval_split"] == "validation"
    assert train_config["trainer_key"] == "static_sft_default"
    assert train_config["model_loader_key"] == "hf_causal_lm_default"
    assert train_config["lora_preset_key"] == "lora_attn_small"
    assert train_config["hf_model_id"] == "test-model"

    assert mock_eval.call_count == 1
    eval_args = mock_eval.call_args.args
    assert eval_args[0] == train_result["adapter_path"]
    assert eval_args[1] == str(tmp_path / "dataset")
    assert eval_args[2]["split"] == "validation"

    decisions_path = tmp_path / "agent_decisions.jsonl"
    decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
    assert [decision["stage"] for decision in decisions] == [
        "decide_modality",
        "select_components",
        "suggest_training_adjustments",
        "analyze_errors",
    ]

    report_text = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "Dataset Summary" in report_text
    assert "Agent Decisions" in report_text
    assert "Training Iterations" in report_text
