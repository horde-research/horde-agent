"""Dataset manifest writer."""

import json
from pathlib import Path
from typing import Dict, Any


def write_manifest(out_dir: str, dataset_summary: Dict[str, Any]) -> str:
    out_path = Path(out_dir) / "dataset_manifest.json"
    out_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    return str(out_path)