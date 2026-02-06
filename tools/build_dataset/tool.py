"""
Dataset building tool.

Prepares training-ready datasets from raw data.
"""

from tools.base_tool import BaseTool

from core.data.hf_dataset import load_dataset_from_path
from core.data.manifest import write_manifest
from core.data.modality import build_example_preview, infer_modality
from core.data.validation import validate_text_columns
from core.types.pipeline_types import DatasetSummary
from pathlib import Path

class BuildDatasetTool(BaseTool):
    """
    Builds formatted datasets for training.
    
    Operations:
    - Train/validation/test split
    - Tokenization
    - Format conversion (SFT/GRPO/DPO specific)
    - Padding and truncation
    - Dataset saving in HF format
    
    Output format depends on training method:
    - SFT: (input, output) pairs
    - GRPO: (prompt, chosen, rejected) tuples
    - DPO: (prompt, chosen, rejected) tuples
    """
    
    def execute(self, data_path, config):
        """
        Build training dataset.
        
        Args:
            data_path (str): Path to evaluated data
            config (dict):
                - split_ratio: float
                - max_length: int
                - training_method: str (sft|grpo|dpo)
                
        Returns:
            dict: {
                'dataset_ref': dict,
                'dataset_summary': dict,
                'dataset_manifest_path': str
            }
        """
        split = config.get("split", "train")
        run_dir = config.get("run_dir") or config.get("out_dir")
        if not run_dir:
            raise ValueError("BuildDatasetTool requires config['run_dir'] (or 'out_dir').")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        dataset, resolved_id = load_dataset_from_path(data_path, split=split)
        columns = list(getattr(dataset, "column_names", []))
        features = getattr(dataset, "features", {}) or {}
        modality_candidates = infer_modality(columns, features)
        example = build_example_preview(dataset[0]) if len(dataset) > 0 else {}
        text_columns = [c for c in columns if c in {"text", "prompt", "response", "instruction", "output"}]
        warnings = validate_text_columns(dataset, text_columns)

        summary = DatasetSummary(
            data_path=data_path,
            resolved_data_id=resolved_id,
            columns=columns,
            sample_count=len(dataset),
            example=example,
            modality_candidates=modality_candidates,
            validation_warnings=warnings,
        )
        manifest_path = write_manifest(run_dir, summary.model_dump())

        return {
            "dataset_ref": {"kind": "hf", "data_path": data_path, "split": split, "resolved_id": resolved_id},
            "dataset_summary": summary.model_dump(),
            "dataset_manifest_path": manifest_path,
        }
