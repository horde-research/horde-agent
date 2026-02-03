"""
Training tool.

Orchestrates model training using specified method (SFT/GRPO/DPO).
"""

from tools.base_tool import BaseTool

class TrainTool(BaseTool):
    """
    Main training orchestrator.
    
    Delegates to specific trainer based on method:
    - SFT: Supervised Fine-Tuning
    - GRPO: Group Relative Policy Optimization
    - DPO: Direct Preference Optimization
    
    Handles:
    - Trainer initialization
    - Training loop execution
    - Checkpointing
    - Reporting callbacks
    """
    
    def __init__(self, reporting_callback=None):
        """
        Initialize training tool.
        
        Args:
            reporting_callback: Function to call with training metrics
        """
        super().__init__()
        self.reporting_callback = reporting_callback
    
    def execute(self, dataset_paths, config):
        """
        Train model using specified method.
        
        Args:
            dataset_paths (dict): Paths to train/val/test datasets
            config (dict):
                - method: str (sft|grpo|dpo)
                - model: str (base model)
                - epochs: int
                - batch_size: int
                - learning_rate: float
                - lora_r: int (optional)
                - lora_alpha: int (optional)
                
        Returns:
            dict: {
                'model_path': str,
                'final_metrics': dict,
                'training_history': list
            }
        """
        pass
