"""
Configuration management.

Loads and merges configuration from multiple sources.
"""

class ConfigManager:
    """
    Manages configuration loading and merging.
    
    Priority order:
    1. Environment variables
    2. User-provided config
    3. Default config
    """
    
    def __init__(self, config_path='config/default.yaml'):
        """Load configuration."""
        pass
    
    def get(self, key, default=None):
        """Get configuration value."""
        pass
    
    def merge(self, user_config):
        """Merge user config with defaults."""
        pass
