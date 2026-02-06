"""Markdown reporting for pipeline results.

Copied from `agentic_train_pipeline/reporting/report.py` and adjusted for new package layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types.pipeline_types import IterationRecord


def _parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        start = line.find("{")
        end = line.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(line[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_json_line(line)
            if parsed is not None:
                records.append(parsed)
    return records


def write_report(
    out_dir: str,
    dataset_summary: Dict[str, Any],
    component_selection: Dict[str, Any],
    iterations: List[IterationRecord],
    failures_path: str,
    cluster_preview: Dict[str, Any],
    error_analysis: Dict[str, Any],
) -> str:
    out_path = Path(out_dir) / "report.md"
    decisions_path = Path(out_dir) / "agent_decisions.jsonl"
    decisions = _read_jsonl(str(decisions_path))

    lines: List[str] = []
    lines.append("# Agentic LoRA SFT Report\n")
    lines.append("## Dataset Summary\n")
    lines.append("```\n" + json.dumps(dataset_summary, indent=2) + "\n```\n")

    lines.append("## Agent Decisions\n")
    lines.append("```\n" + json.dumps(decisions, indent=2) + "\n```\n")

    lines.append("## Selected Components\n")
    lines.append("```\n" + json.dumps(component_selection, indent=2) + "\n```\n")

    lines.append("## Training Iterations\n")
    for record in iterations:
        lines.append(f"### Iteration {record.iter_idx}\n")
        lines.append("```\n" + json.dumps(record.model_dump(), indent=2) + "\n```\n")

    lines.append("## Failure Clusters\n")
    lines.append("```\n" + json.dumps(cluster_preview, indent=2) + "\n```\n")

    lines.append("## Error Analysis\n")
    lines.append("```\n" + json.dumps(error_analysis, indent=2) + "\n```\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)

