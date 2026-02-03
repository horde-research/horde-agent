"""
Base class for all tools.

All tools inherit from this base class.
"""

class BaseTool:
    """
    Base class for all agent tools.
    
    Each tool implements:
    - execute(): Main tool logic
    - validate_input(): Input validation
    - validate_output(): Output validation
    """
    
    def __init__(self, config=None):
        """Initialize tool with configuration."""
        self.config = config or {}
    
    def execute(self, *args, **kwargs):
        """
        Execute the tool's main function.
        
        Must be implemented by subclasses.
        
        Returns:
            Tool-specific output
        """
        raise NotImplementedError
    
    def validate_input(self, *args, **kwargs):
        """Validate input parameters."""
        return True
    
    def validate_output(self, output):
        """Validate output before returning."""
        return output
