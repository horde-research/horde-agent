"""
Unified reporting tool.

Handles initialization, continuous logging, and final report generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.base_tool import BaseTool

from core.types.pipeline_types import IterationRecord
from tools.reporting.report import write_report

class ReportingTool(BaseTool):
    """
    Multi-phase reporting tool.
    
    Phase 1 - Initialize:
        Setup wandb/tensorboard/mlflow before training
        
    Phase 2 - Log:
        Continuous logging during training (called from trainer)
        
    Phase 3 - Finalize:
        Generate comprehensive HTML/PDF report after evaluation
    
    Supports multiple reporters simultaneously.
    """
    
    def __init__(self, config):
        """
        Initialize reporters based on config.
        
        Args:
            config (dict):
                - reporters: list[str] (wandb, tensorboard, mlflow, file)
        """
        super().__init__(config)
        self.reporters = []
        self.run_id = None
        self.report_data = {}
    
    def initialize(self, experiment_config):
        """
        Phase 1: Initialize before training.
        
        Args:
            experiment_config (dict): Full workflow config
            
        Returns:
            str: run_id
        """
        self.run_id = experiment_config.get("run_id")
        self.report_data["experiment_config"] = experiment_config
        return self.run_id or "run"
    
    def log(self, metrics, step=None, context='train'):
        """
        Phase 2: Log metrics during training.
        
        Args:
            metrics (dict): Metrics to log
            step (int): Training step/epoch
            context (str): 'train' or 'val' or 'eval'
        """
        history = self.report_data.setdefault("history", [])
        history.append({"context": context, "step": step, "metrics": metrics})
    
    def log_artifact(self, artifact_path, artifact_type='file'):
        """Log artifacts (models, plots, etc.)."""
        artifacts = self.report_data.setdefault("artifacts", [])
        artifacts.append({"type": artifact_type, "path": artifact_path})
    
    def finalize(self, eval_results):
        """
        Phase 3: Generate final report.
        
        Args:
            eval_results (dict): Results from eval_model tool
            
        Returns:
            str: Path to final report
        """
        out_dir = self.config.get("run_dir") or self.config.get("out_dir")
        if not out_dir:
            raise ValueError("ReportingTool requires config['run_dir'] (or 'out_dir').")

        dataset_summary = eval_results.get("dataset_summary") or {}
        component_selection = eval_results.get("component_selection") or {}
        failures_path = eval_results.get("failures_path") or ""
        cluster_preview = eval_results.get("cluster_preview") or {}
        error_analysis = eval_results.get("error_analysis") or {}

        iterations_raw = eval_results.get("iterations") or []
        iterations: List[IterationRecord] = []
        for item in iterations_raw:
            if isinstance(item, IterationRecord):
                iterations.append(item)
            else:
                iterations.append(IterationRecord.model_validate(item))

        return write_report(
            out_dir=out_dir,
            dataset_summary=dataset_summary,
            component_selection=component_selection,
            iterations=iterations,
            failures_path=failures_path,
            cluster_preview=cluster_preview,
            error_analysis=error_analysis,
        )
