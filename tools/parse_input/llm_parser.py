"""
LLM-based natural language parser.

Converts natural language descriptions to structured YAML config.
"""

class LLMParser:
    """
    Uses LLM to convert natural language to workflow config.
    
    Example:
        Input: "Train SFT model on Kazakh language for 3 epochs"
        Output: {
            'project_name': 'kazakh_sft',
            'language': 'kk',
            'data': {'source': '...'},
            'train': {'method': 'sft', 'epochs': 3}
        }
    """
    
    def __init__(self, llm_client):
        """Initialize with LLM client."""
        pass
    
    def parse(self, natural_language_input):
        """
        Convert natural language to config dict.
        
        Args:
            natural_language_input (str): User's natural language description
            
        Returns:
            dict: Parsed configuration
        """
        pass
