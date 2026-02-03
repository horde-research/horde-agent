"""Simple file-based reporter."""

from tools.reporting.reporters.base_reporter import BaseReporter

class FileReporter(BaseReporter):
    """
    Logs metrics to local JSON files.
    
    Useful for simple experiments or when online tracking is unavailable.
    """
    
    def init(self, log_dir='data/logs', **kwargs):
        """Initialize file logging."""
        pass
    
    def log(self, metrics, step=None):
        """Append metrics to JSON file."""
        pass
    
    def finish(self):
        """Close log file."""
        pass
