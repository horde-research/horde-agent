"""Dataset manifest writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_manifest(run_dir: str, summary: Dict[str, Any]) -> str:
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)
