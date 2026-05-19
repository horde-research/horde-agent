"""Reward functions for GRPO training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from pydantic import BaseModel, Field

from core.llm.client import LLMClient, LLMRequest, format_instructions, parse_response


class JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


JUDGE_SYSTEM_PROMPT = """You are an expert reward judge for RL fine-tuning a language model.

You will evaluate answers across many cultures, countries, languages, and domains
such as history, food, arts, religion, geography, law, education, and everyday
customs. Be culturally careful: accept valid local terms, transliterations, and
alternate spellings when they are compatible with the reference. Do not reward
stereotypes, unsupported generalizations, invented facts, or confident claims
that contradict the reference.

Primary grading basis:
1. Factual alignment with the reference answer and prompt.
2. Completeness for the user request.
3. Cultural specificity and respectful wording.
4. Clarity, coherence, and usefulness.
5. No repetition, empty output, irrelevant text, or hallucinated details.

Use general knowledge only to detect obvious contradictions or unsafe claims; do
not require facts that are absent from the reference. Return only the requested
JSON score object."""


def build_judge_user_message(*, prompt: str, reference: str, completion: str) -> str:
    reference_block = reference or (
        "No reference answer was provided. Judge using the prompt, internal consistency, "
        "cultural care, specificity, and absence of hallucination."
    )
    return (
        "Score this candidate answer from 0.0 to 1.0 for RL training.\n\n"
        f"USER PROMPT:\n{prompt}\n\n"
        f"REFERENCE ANSWER OR SOURCE-DERIVED TARGET:\n{reference_block}\n\n"
        f"CANDIDATE ANSWER:\n{completion}\n\n"
        "Scoring guide:\n"
        "- 1.0: accurate, complete, culturally specific, respectful, and directly answers the prompt.\n"
        "- 0.7: mostly correct but missing useful detail or minor nuance.\n"
        "- 0.5: partially correct but incomplete, generic, or weakly grounded.\n"
        "- 0.2: mostly wrong, vague, repetitive, or contains unsupported cultural claims.\n"
        "- 0.0: empty, irrelevant, unsafe, or contradicts the reference.\n"
        "Explain the main reason briefly in `rationale`."
        f"{format_instructions(JudgeScore)}"
    )


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: List[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(str(part.get("text", "")) for part in content if isinstance(part, dict))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(completion)


def _coerce_list(values: Any, length: int, default: str = "") -> List[str]:
    if values is None:
        return [default] * length
    if isinstance(values, str):
        return [values] * length
    result = list(values)
    if len(result) == length:
        return [str(v) for v in result]
    if not result:
        return [default] * length
    if length % len(result) == 0:
        repeated: List[str] = []
        for value in result:
            repeated.extend([str(value)] * (length // len(result)))
        return repeated
    return [str(result[min(i, len(result) - 1)]) for i in range(length)]


class LLMJudgeReward:
    """LLM-as-judge reward callable compatible with TRL GRPOTrainer."""

    __name__ = "llm_judge_reward"

    def __init__(
        self,
        *,
        log_path: str,
        llm_client: Optional[LLMClient] = None,
        batch_size: int = 5,
        batch_delay_seconds: float = 1.5,
    ) -> None:
        self.llm_client = llm_client or LLMClient.from_env()
        self.batch_size = batch_size
        self.batch_delay_seconds = batch_delay_seconds
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompts: Iterable[Any], completions: Iterable[Any], reference=None, **kwargs) -> List[float]:
        completions_list = [_completion_to_text(completion) for completion in completions]
        prompts_list = _coerce_list(prompts, len(completions_list))
        references = _coerce_list(reference or kwargs.get("references"), len(completions_list))

        requests = [
            LLMRequest(
                request_id=f"grpo-judge-{idx}",
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_message=build_judge_user_message(prompt=prompt, reference=ref, completion=completion),
                generation_config={"temperature": 0.0},
            )
            for idx, (prompt, ref, completion) in enumerate(zip(prompts_list, references, completions_list))
        ]

        responses = self.llm_client.generate_json_batch_sync(
            requests,
            batch_size=self.batch_size,
            batch_delay_seconds=self.batch_delay_seconds,
        )

        rewards: List[float] = []
        with self.log_path.open("a", encoding="utf-8") as handle:
            for prompt, ref, completion, response in zip(prompts_list, references, completions_list, responses):
                score = 0.0
                rationale = ""
                error = response.error
                if response.success and response.data is not None:
                    try:
                        parsed = parse_response(response.data, JudgeScore)
                        score = max(0.0, min(1.0, float(parsed.score)))
                        rationale = parsed.rationale
                    except Exception as exc:
                        error = str(exc)
                rewards.append(score)
                handle.write(json.dumps({
                    "prompt": prompt,
                    "reference": ref,
                    "completion": completion,
                    "score": score,
                    "rationale": rationale,
                    "error": error,
                }, ensure_ascii=False) + "\n")
        return rewards
