from __future__ import annotations

from core.types.pipeline_types import TrainConfig
from tools.train.trainers.static_sft_trainer import StaticSFTTrainer


class CharOffsetTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.last_text = ""

    def __call__(
        self,
        text,
        *,
        max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=False,
    ):
        self.last_text = text
        chars = list(text)[:max_length]
        input_ids = [idx + 1 for idx in range(len(chars))]
        attention_mask = [1] * len(input_ids)
        offsets = [(idx, idx + 1) for idx in range(len(chars))]
        while len(input_ids) < max_length:
            input_ids.append(self.pad_token_id)
            attention_mask.append(0)
            offsets.append((0, 0))
        payload = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return payload


class ChatTemplateTokenizer(CharOffsetTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: ANN001
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def test_static_sft_tokenize_masks_user_prompt_tokens(tmp_path) -> None:
    tokenizer = CharOffsetTokenizer()
    trainer = StaticSFTTrainer(
        model=None,
        tokenizer=tokenizer,
        train_dataset=[],
        eval_dataset=None,
        out_dir=str(tmp_path),
        config=TrainConfig(max_seq_len=128),
    )

    encoded = trainer._tokenize(
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
    )

    question_pos = tokenizer.last_text.index("Question")
    answer_pos = tokenizer.last_text.index("Answer")
    assert encoded["labels"][question_pos] == -100
    assert encoded["labels"][answer_pos] == encoded["input_ids"][answer_pos]
    assert encoded["labels"][answer_pos + len("Answer") - 1] == encoded["input_ids"][answer_pos + len("Answer") - 1]
    assert encoded["labels"][-1] == -100


def test_static_sft_tokenize_uses_chat_template_when_available(tmp_path) -> None:
    tokenizer = ChatTemplateTokenizer()
    trainer = StaticSFTTrainer(
        model=None,
        tokenizer=tokenizer,
        train_dataset=[],
        eval_dataset=None,
        out_dir=str(tmp_path),
        config=TrainConfig(max_seq_len=128),
    )

    encoded = trainer._tokenize(
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }
    )

    assert "<|im_start|>user" in tokenizer.last_text
    assert "<|im_start|>assistant" in tokenizer.last_text
    question_pos = tokenizer.last_text.index("Question")
    answer_pos = tokenizer.last_text.index("Answer")
    assert encoded["labels"][question_pos] == -100
    assert encoded["labels"][answer_pos] == encoded["input_ids"][answer_pos]
