"""
Logging configuration.

Centralized logging setup for all components.
"""

import logging

def setup_logging(level=logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_logger(name):
    """Get logger instance for module."""
    return logging.getLogger(name)
