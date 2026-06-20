from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image

from core.registry.builtins import build_registry
from core.types.pipeline_types import MetricsSummary, TrainConfig
from tools.train.tool import TrainTool
from tools.train.trainers.vision_language_sft_trainer import VisionLanguageSFTCollator, VisionLanguageSFTTrainer


class FakeTokenizer:
    pad_token_id = 0


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.templates = []
        self.images = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.templates.append(messages)
        assistant_text = ""
        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, list):
                    assistant_text = " ".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
                else:
                    assistant_text = str(content or "")
        return f"USER: <image> Describe\nASSISTANT: {assistant_text}".rstrip()

    def __call__(self, **kwargs):
        self.images = kwargs["images"]
        rows = []
        masks = []
        for text in kwargs["text"]:
            if text.endswith("ASSISTANT:"):
                rows.append([1, 2, 0])
                masks.append([1, 1, 0])
            elif "Red" in text:
                rows.append([1, 3, 4])
                masks.append([1, 1, 1])
            else:
                rows.append([1, 2, 5])
                masks.append([1, 1, 1])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "pixel_values": torch.zeros((len(rows), 3, 8, 8), dtype=torch.float32),
        }


class FilteringProcessor(FakeProcessor):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.templates.append(messages)
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text") == "short prompt":
                        return "short prompt"
        return "very long prompt"

    def __call__(self, **kwargs):
        text = kwargs["text"][0]
        length = 4 if "short" in text else 12
        return {"input_ids": torch.zeros((1, length), dtype=torch.long)}


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
                    {"role": "assistant", "content": "A red square."},
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
                    {"role": "assistant", "content": "Red."},
                ]
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 3)
    assert batch["pixel_values"].shape == (2, 3, 8, 8)
    assert batch["labels"].tolist() == [[-100, -100, 5], [-100, -100, 4]]
    assert len(processor.images) == 2


def test_vision_language_collator_collapses_text_only_message_content(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)
    processor = FakeProcessor()
    collator = VisionLanguageSFTCollator(processor, max_seq_len=16)

    collator(
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
            }
        ]
    )

    assert processor.templates[0][1]["content"] == "A red square."
    assert processor.templates[0][0]["content"][0]["type"] == "image"


def test_vision_language_trainer_filters_overlong_examples(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "short prompt"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "short answer"}]},
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": "very long prompt"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "long answer"}]},
                ]
            },
        ]
    )
    trainer = VisionLanguageSFTTrainer(
        model=None,
        tokenizer=FilteringProcessor(),
        train_dataset=dataset,
        eval_dataset=None,
        out_dir=str(tmp_path / "out"),
        config=TrainConfig(max_seq_len=6, max_steps=1),
    )

    filtered = trainer._filter_overlong_examples(dataset, split_name="train")

    assert len(filtered) == 1


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
