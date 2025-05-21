"""
Parser module for processing the output of the first LLM into proper JSON format.
Uses a second LLM to ensure the response follows the required format.
"""

import json
import re
from llm_client import LLMClient

class LLMOutputParser:
    """Class for parsing LLM output and ensuring it follows the required format."""
    
    def __init__(self, provider="hunyuan", model=None):
        """Initialize the LLM parser.
        
        Args:
            provider: The LLM provider to use for parsing
            model: The specific model to use with the provider
        """
        self.llm_client = LLMClient(provider=provider, model=model)
        
    def parse_and_format(self, llm_response, original_prompt=None):
        """Parse the output from the first LLM and convert it to the required JSON format.
        
        Args:
            llm_response: The raw response from the first LLM
            original_prompt: Optional, the original prompt for context
            
        Returns:
            A properly formatted JSON response
        """
        # First check if the response already contains valid JSON
        json_data = self._extract_valid_json(llm_response)
        if json_data and self._validate_json_format(json_data):
            print("First LLM response already contains valid JSON, no need for second LLM")
            return json.dumps(json_data)
            
        # If we can't extract valid JSON, use the second LLM to fix it
        parser_prompt = self._create_parser_prompt(llm_response, original_prompt)
        
        # Get response from the second LLM
        print("Using second LLM to parse and format response")
        formatted_response = self.llm_client.generate_response(parser_prompt)
        
        # Extract JSON from the second LLM's response
        json_data = self._extract_valid_json(formatted_response)
        
        # If we still don't have valid JSON, create a fallback response
        if not json_data or not self._validate_json_format(json_data):
            print("Warning: Second LLM failed to generate valid JSON, using fallback")
            return json.dumps({
                "moves": [],
                "reasoning": "ERROR: Could not generate valid moves from LLM response"
            })
            
        return json.dumps(json_data)
    
    def _extract_valid_json(self, text):
        """Extract valid JSON data from text.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON data or None if no valid JSON found
        """
        try:
            # First try to parse the entire text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from a code block
            try:
                json_block_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', text, re.DOTALL)
                if json_block_match:
                    json_str = json_block_match.group(1)
                    return json.loads(json_str)
            except:
                pass
                
            # Try to extract JSON object with regex
            try:
                json_match = re.search(r'\{[\s\S]*?"moves"\s*:\s*\[[\s\S]*?\][\s\S]*?\}', text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Clean up the string (replace single quotes with double quotes, etc.)
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                    # Remove trailing commas
                    json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)
                    return json.loads(json_str)
            except:
                pass
                
        return None
    
    def _validate_json_format(self, json_data):
        """Validate that JSON data follows the required format.
        
        Args:
            json_data: Parsed JSON data
            
        Returns:
            Boolean indicating if the JSON follows the required format
        """
        # Check that json_data is a dict with the required keys
        if not isinstance(json_data, dict):
            return False
            
        # Check for required keys
        if "moves" not in json_data or "reasoning" not in json_data:
            return False
            
        # Check that moves is a list
        if not isinstance(json_data["moves"], list):
            return False
            
        # Check that each move is a valid direction
        valid_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        for move in json_data["moves"]:
            if not isinstance(move, str) or move.upper() not in valid_directions:
                return False
                
        # Check that reasoning is a string
        if not isinstance(json_data["reasoning"], str):
            return False
            
        return True
    
    def _create_parser_prompt(self, llm_response, original_prompt=None):
        """Create a prompt for the second LLM to parse the output of the first LLM.
        
        Args:
            llm_response: The raw response from the first LLM
            original_prompt: Optional, the original prompt for context
            
        Returns:
            Prompt for the second LLM
        """
        context = ""
        if original_prompt:
            context = f"""
I previously sent the following prompt to an LLM:

{original_prompt}

"""

        prompt = f"""{context}I received the following response from the LLM:

```
{llm_response}
```

The response should be a valid JSON object with the following format:
{{
  "moves": ["MOVE1", "MOVE2", ...],
  "reasoning": "..." 
}}

Where:
- "moves" is a list of valid directions ("UP", "DOWN", "LEFT", "RIGHT")
- "reasoning" is a brief explanation of the path-planning rationale

Please extract the intended moves and reasoning from the response and format them as a valid JSON object following the exact format above. If you cannot determine the moves, return an empty moves list and set reasoning to "NO_PATH_FOUND".

Return ONLY the JSON object with no additional text or explanation.
"""
        return prompt 