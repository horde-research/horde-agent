"""
Unified reporting tool.

Handles initialization, continuous logging, and final report generation.
"""

from tools.base_tool import BaseTool

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
        pass
    
    def log(self, metrics, step=None, context='train'):
        """
        Phase 2: Log metrics during training.
        
        Args:
            metrics (dict): Metrics to log
            step (int): Training step/epoch
            context (str): 'train' or 'val' or 'eval'
        """
        pass
    
    def log_artifact(self, artifact_path, artifact_type='file'):
        """Log artifacts (models, plots, etc.)."""
        pass
    
    def finalize(self, eval_results):
        """
        Phase 3: Generate final report.
        
        Args:
            eval_results (dict): Results from eval_model tool
            
        Returns:
            str: Path to final report
        """
        pass
