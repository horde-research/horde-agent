from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from core.registry.builtins import build_registry
from core.types.pipeline_types import MetricsSummary
from tools.train.tool import TrainTool
from tools.train.trainers.vision_language_sft_trainer import VisionLanguageSFTCollator


class FakeTokenizer:
    pad_token_id = 0


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.templates = []
        self.images = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.templates.append(messages)
        return "USER: <image> Describe\nASSISTANT: Answer"

    def __call__(self, **kwargs):
        self.images = kwargs["images"]
        return {
            "input_ids": torch.tensor([[1, 2, 0], [1, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            "pixel_values": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
        }


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)


def test_vision_language_collator_loads_existing_image_sft_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)
    processor = FakeProcessor()
    collator = VisionLanguageSFTCollator(processor, max_seq_len=16)

    batch = collator(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
                            {"type": "image", "image": str(image_path)},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]},
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is visible?"},
                            {"type": "image", "image": str(image_path)},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "Red."}]},
                ]
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 3)
    assert batch["pixel_values"].shape == (2, 3, 8, 8)
    assert batch["labels"].tolist() == [[1, 2, -100], [1, 3, 4]]
    assert len(processor.images) == 2


def test_registry_exposes_image_training_components() -> None:
    registry = build_registry()

    snapshot = registry.snapshot()

    assert "hf_image_text_default" in snapshot.model_loader_keys
    assert "vision_language_sft" in snapshot.trainer_keys


def test_train_tool_auto_selects_image_loader_and_trainer(monkeypatch, tmp_path: Path) -> None:
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")
    calls: dict[str, str] = {}

    class FakeDataset:
        def __len__(self):
            return 1

    class FakeRegistry:
        def get_model_loader(self, key: str):
            calls["model_loader_key"] = key
            return lambda model_id: ("model", "processor")

        def get_trainer(self, key: str):
            calls["trainer_key"] = key
            return object

        def get_lora_preset(self, key: str):
            calls["lora_preset_key"] = key
            return {"r": 1, "alpha": 1, "dropout": 0.0, "target_modules": ["q_proj"]}

    monkeypatch.setattr("tools.train.tool.build_registry", lambda: FakeRegistry())
    monkeypatch.setattr("tools.train.tool.load_dataset_from_path", lambda data_path, split: (FakeDataset(), data_path))
    monkeypatch.setattr("tools.train.tool.attach_lora", lambda model, preset: model)
    monkeypatch.setattr(
        "tools.train.tool.run_sft_iteration",
        lambda **kwargs: (str(tmp_path / "adapter"), {"train_log": str(tmp_path / "train.log"), "metrics": str(tmp_path / "metrics.jsonl")}),
    )
    monkeypatch.setattr(
        "tools.train.tool.parse_metrics",
        lambda path: MetricsSummary(steps=1, best_eval_loss=None, last_train_loss=0.1, last_eval_loss=None),
    )

    result = TrainTool().execute(
        {"kind": "hf", "data_path": str(sft_path), "split": "train"},
        {
            "method": "sft",
            "run_dir": str(tmp_path),
            "hf_model_id": "fake-vlm",
            "training_modality": "image",
            "train_config": {"max_steps": 1},
        },
    )

    assert calls["model_loader_key"] == "hf_image_text_default"
    assert calls["trainer_key"] == "vision_language_sft"
    assert result["adapter_path"] == str(tmp_path / "adapter")
