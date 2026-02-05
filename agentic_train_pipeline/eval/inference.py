"""Inference helpers for small evaluation samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from agentic_train_pipeline.parser.modality import extract_text_input_output


def run_inference(
    model,
    tokenizer,
    dataset,
    out_dir: str,
    max_samples: int = 64,
    max_new_tokens: int = 128,
) -> str:
    out_path = Path(out_dir) / "predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample = dataset.select(range(min(len(dataset), max_samples)))
    results: List[Dict[str, Any]] = []

    model.eval()
    for idx, example in enumerate(sample):
        prompt, reference = extract_text_input_output(example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append(
            {
                "id": idx,
                "input": prompt,
                "prediction": pred_text,
                "reference": reference,
            }
        )

    with out_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(str(json.dumps(row)) + "\n")

    return str(out_path)
