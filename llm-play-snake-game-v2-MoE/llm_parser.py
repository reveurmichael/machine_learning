"""
Parser module for processing the output of the first LLM into proper JSON format.
Uses a second LLM to ensure the response follows the required format.
"""

import json
from llm_client import LLMClient
from config import PARSER_PROMPT_TEMPLATE
from json_utils import extract_valid_json, validate_json_format

class LLMOutputParser:
    """Class for parsing LLM output and ensuring it follows the required format."""
    
    def __init__(self, provider="hunyuan", model=None):
        """Initialize the LLM parser.
        
        Args:
            provider: The LLM provider to use for parsing
            model: The specific model to use with the provider
        """
        self.llm_client = LLMClient(provider=provider, model=model)
        
    def parse_and_format(self, llm_response):
        """Parse the output from the first LLM and convert it to the required JSON format.
        
        Args:
            llm_response: The raw response from the first LLM
            
        Returns:
            A tuple containing (formatted_response, parser_prompt) where:
              - formatted_response: The properly formatted JSON response
              - parser_prompt: The prompt that was sent to the parser LLM
        """
        # First check if the response already contains valid JSON
        json_data = extract_valid_json(llm_response)
        if json_data and validate_json_format(json_data):
            print("First LLM response already contains valid JSON, no need for second LLM")
            # We didn't use a parser prompt, so return None for that
            return json.dumps(json_data), None
            
        # If we can't extract valid JSON, use the second LLM to fix it
        parser_prompt = self._create_parser_prompt(llm_response)
        
        # Get response from the second LLM
        print("Using second LLM to parse and format response")
        formatted_response = self.llm_client.generate_response(parser_prompt)
        
        # Extract JSON from the second LLM's response
        json_data = extract_valid_json(formatted_response)
        
        # If we still don't have valid JSON, create a fallback response
        if not json_data or not validate_json_format(json_data):
            print("Warning: Second LLM failed to generate valid JSON, using fallback")
            return json.dumps({
                "moves": [],
                "reasoning": "ERROR: Could not generate valid moves from LLM response"
            }), parser_prompt
            
        return json.dumps(json_data), parser_prompt

    def _create_parser_prompt(self, llm_response):
        """Create a prompt for the second LLM to parse the output of the first LLM.
        
        Args:
            llm_response: The raw response from the first LLM
            
        Returns:
            Prompt for the second LLM
        """
        # Use the template from config.py with the llm_response
        return PARSER_PROMPT_TEMPLATE.format(response=llm_response) 