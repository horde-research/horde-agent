"""
Dataset building tool.

Prepares training-ready datasets from raw data.
"""

from tools.base_tool import BaseTool

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
                'train_path': str,
                'val_path': str,
                'test_path': str,
                'num_train': int,
                'num_val': int,
                'num_test': int
            }
        """
        pass
