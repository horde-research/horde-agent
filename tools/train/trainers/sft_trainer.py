"""
SFT (Supervised Fine-Tuning) trainer.

Standard supervised learning on (input, output) pairs.
"""

class SFTTrainer:
    """
    Supervised Fine-Tuning trainer.
    
    Uses standard cross-entropy loss on next-token prediction.
    Supports LoRA/QLoRA for efficient training.
    """
    
    def __init__(self, config, reporting_callback=None):
        """Initialize SFT trainer."""
        pass
    
    def train(self, train_dataset, val_dataset):
        """
        Run SFT training.
        
        Returns:
            dict: Training results
        """
        pass
