"""
Parser module for processing the output of the primary LLM into proper JSON format.
Uses a secondary LLM to ensure the response follows the required format.
"""

import json
import time
import traceback
from datetime import datetime
from colorama import Fore
from llm_client import LLMClient
from config import PARSER_PROMPT_TEMPLATE
from utils.json_utils import extract_valid_json, validate_json_format
from utils.log_utils import save_to_file, format_parsed_llm_response

class LLMOutputParser:
    """Class for parsing primary LLM output and ensuring it follows the required format."""
    
    def __init__(self, provider="hunyuan", model=None):
        """Initialize the LLM parser.
        
        Args:
            provider: The LLM provider to use for secondary LLM
            model: The specific model to use with the provider
        """
        self.llm_client = LLMClient(provider=provider, model=model)
        
    def parse_and_format(self, llm_response, head_pos=None, apple_pos=None, body_cells=None):
        """Parse the output from the primary LLM and convert it to the required JSON format.
        
        Args:
            llm_response: The raw response from the primary LLM
            head_pos: Optional head position string in format "(x, y)"
            apple_pos: Optional apple position string in format "(x, y)"
            body_cells: Optional body cells string in format "[(x1, y1), (x2, y2), ...]"
            
        Returns:
            A tuple containing (formatted_response, parser_prompt) where:
              - formatted_response: The properly formatted JSON response
              - parser_prompt: The prompt that was sent to the secondary LLM
        """
        # Check if the primary LLM response contains valid JSON (for logging purposes only)
        first_json_valid = False
        json_data = extract_valid_json(llm_response)
        if json_data and validate_json_format(json_data):
            first_json_valid = True
            print("Primary LLM response contains valid JSON, but will still use secondary LLM for consistency")
        
        # Create the prompt for the secondary LLM
        parser_prompt = self._create_parser_prompt(llm_response, head_pos, apple_pos, body_cells)
        
        # Get response from the secondary LLM
        print("Using secondary LLM to parse and format response")
        formatted_response = self.llm_client.generate_response(parser_prompt)
        
        # Extract JSON from the secondary LLM's response
        json_data = extract_valid_json(formatted_response)
        
        # If we don't have valid JSON from the secondary LLM, create a fallback response
        if not json_data or not validate_json_format(json_data):
            print("Warning: Secondary LLM failed to generate valid JSON, using fallback")
            fallback_data = {
                "moves": [],
                "reasoning": "ERROR: Could not generate valid moves from LLM response"
            }
            return json.dumps(fallback_data), parser_prompt
            
        return json.dumps(json_data), parser_prompt

    def _create_parser_prompt(self, llm_response, head_pos=None, apple_pos=None, body_cells=None):
        """Create a prompt for the secondary LLM to parse the output of the primary LLM.
        
        Args:
            llm_response: The raw response from the primary LLM
            head_pos: Optional head position string in format "(x, y)"
            apple_pos: Optional apple position string in format "(x, y)"
            body_cells: Optional body cells string in format "[(x1, y1), (x2, y2), ...]"
            
        Returns:
            Prompt for the secondary LLM
        """
        # Use string replacement for the prompt template
        parser_prompt = PARSER_PROMPT_TEMPLATE.replace("TEXT_TO_BE_REPLACED_FIRST_LLM_RESPONSE", llm_response)
        
        # Replace head and apple position placeholders if provided
        if head_pos:
            parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", head_pos)
        if apple_pos:
            parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", apple_pos)
        if body_cells:
            parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_BODY_CELLS", body_cells)
            
        return parser_prompt 