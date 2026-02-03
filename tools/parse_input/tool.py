"""
Input parsing tool.

Converts natural language or YAML into structured workflow config.
"""

from tools.base_tool import BaseTool

class ParseInputTool(BaseTool):
    """
    Parses user input into validated workflow configuration.
    
    Handles three input types:
    1. Natural language -> uses LLM to generate YAML
    2. YAML string -> validates and fills defaults
    3. Dict -> validates and fills defaults
    
    Output: Validated workflow config dict with all required fields
    """
    
    def __init__(self, llm_client=None, schema_path='config/schemas/workflow_config.schema.yaml'):
        """
        Initialize parser.
        
        Args:
            llm_client: LLM client for natural language parsing (optional)
            schema_path: Path to workflow config schema
        """
        super().__init__()
    
    def execute(self, user_input):
        """
        Parse and validate user input.
        
        Args:
            user_input: Natural language str, YAML str, or dict
            
        Returns:
            dict: Validated workflow configuration
        """
        pass
