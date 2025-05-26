"""
LLM parser module for the Snake game.
This module is deprecated - use LLMOutputParser from llm_client.py instead.
"""

import json
from typing import List, Tuple, Dict, Any


def parse_llm_response(response_text: str, head_pos: Tuple[int, int], apple_pos: Tuple[int, int], body_cells: List[Tuple[int, int]]) -> Tuple[str, str]:
    """Parse and format an LLM response.
    
    This function is deprecated. Use LLMOutputParser.parse_and_format() instead.
    
    Args:
        response_text: The raw LLM response to parse
        head_pos: Current head position
        apple_pos: Current apple position
        body_cells: Current body cell positions
        
    Returns:
        Tuple of (parsed_response, parser_prompt)
    """
    # Create the parser prompt
    parser_prompt = _create_parser_prompt(response_text, head_pos, apple_pos, body_cells)
    
    # Get the parsed response
    parsed_response = _parse_response(response_text)
    
    return parsed_response, parser_prompt


def _create_parser_prompt(response_text: str, head_pos: Tuple[int, int], apple_pos: Tuple[int, int], body_cells: List[Tuple[int, int]]) -> str:
    """Create a prompt for the parser LLM.
    
    This function is deprecated. Use LLMOutputParser._create_parser_prompt() instead.
    
    Args:
        response_text: The raw LLM response to parse
        head_pos: Current head position
        apple_pos: Current apple position
        body_cells: Current body cell positions
        
    Returns:
        The parser prompt
    """
    return f"""Please parse and format the following LLM response for a Snake game. The response should be a valid JSON object with a "moves" array containing the sequence of moves to make.

Current game state:
- Head position: {head_pos}
- Apple position: {apple_pos}
- Body cells: {body_cells}

Raw LLM response:
{response_text}

Please extract the moves and format them as a JSON object like this:
{{
    "moves": ["UP", "RIGHT", "DOWN", "LEFT"]
}}

Only include valid moves (UP, DOWN, LEFT, RIGHT). If no valid moves are found, return an empty moves array.
"""


def _parse_response(response_text: str) -> str:
    """Parse the LLM response into a JSON object.
    
    This function is deprecated. Use LLMOutputParser.parse_and_format() instead.
    
    Args:
        response_text: The raw LLM response to parse
        
    Returns:
        The parsed response as a JSON string
    """
    try:
        # Try to parse the response as JSON
        data = json.loads(response_text)
        
        # If it's already a valid JSON object with moves, return it
        if isinstance(data, dict) and "moves" in data:
            return json.dumps(data)
            
        # Otherwise, try to extract moves from the text
        moves = []
        for line in response_text.split("\n"):
            line = line.strip().upper()
            if line in ["UP", "DOWN", "LEFT", "RIGHT"]:
                moves.append(line)
                
        return json.dumps({"moves": moves})
        
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract moves from the text
        moves = []
        for line in response_text.split("\n"):
            line = line.strip().upper()
            if line in ["UP", "DOWN", "LEFT", "RIGHT"]:
                moves.append(line)
                
        return json.dumps({"moves": moves}) 