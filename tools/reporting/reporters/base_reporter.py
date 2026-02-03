"""
Base reporter interface.

All reporters (wandb, tensorboard, etc.) implement this interface.
"""

class BaseReporter:
    """
    Base class for all reporters.
    
    Subclasses implement specific reporting backends.
    """
    
    def init(self, **kwargs):
        """Initialize reporter (e.g., wandb.init())."""
        raise NotImplementedError
    
    def log(self, metrics, step=None):
        """Log metrics."""
        raise NotImplementedError
    
    def log_artifact(self, artifact_path, artifact_type='file'):
        """Log artifact."""
        pass
    
    def finish(self):
        """Cleanup and close reporter."""
        pass
