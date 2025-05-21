"""
Utility module for JSON processing in the Snake game.
Consolidates JSON extraction and validation functions used by multiple components.
"""

import json
import re

def extract_valid_json(text):
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
        # Try extracting from code block
        json_data = extract_json_from_code_block(text)
        if json_data:
            return json_data
            
        # Try extracting from regular text
        json_data = extract_json_from_text(text)
        if json_data:
            return json_data
                
    return None

def validate_json_format(json_data):
    """Validate that JSON data follows the required format for snake moves.
    
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

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    try:
        # Look for JSON code block which is common in LLM responses - Use a more robust pattern
        json_block_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response, re.DOTALL)
        if not json_block_match:
            return None
            
        json_str = json_block_match.group(1)
        # Clean the JSON string
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        # Remove any trailing commas before closing brackets or braces
        json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)
        
        data = json.loads(json_str)
        return data
    except Exception as e:
        print(f"Failed to parse JSON code block: {e}")
        # Try direct JSON parsing as a fallback
        try:
            # Try to parse the entire response as JSON directly
            data = json.loads(response)
            if isinstance(data, dict) and "moves" in data:
                return data
        except:
            pass
        return None

def extract_json_from_text(response):
    """Extract JSON data directly from text without code block.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    try:
        # First try direct JSON parsing (for clean JSON responses)
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "moves" in data:
                return data
        except:
            pass
            
        # Try extracting JSON object outside of code blocks with a more robust pattern
        json_match = re.search(r'\{[\s\S]*?"moves"\s*:\s*\[[\s\S]*?\][\s\S]*?\}', response, re.DOTALL)
        if not json_match:
            return None
            
        json_str = json_match.group(0)
        # Clean the JSON string
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        # Remove any trailing commas before closing brackets or braces
        json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)
        
        # Now try to parse the cleaned JSON
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}, trying simplified parsing")
        
        # Try to extract just the moves array
        try:
            moves_array_match = re.search(r'"moves"\s*:\s*\[([\s\S]*?)\]', json_str, re.DOTALL)
            if not moves_array_match:
                return None
                
            moves_array = moves_array_match.group(1)
            # Extract quoted strings
            move_matches = re.findall(r'"([^"]+)"', moves_array)
            valid_moves = [move.upper() for move in move_matches 
                          if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            if valid_moves:
                return {"moves": valid_moves}
        except Exception:
            pass
            
        return None
    except Exception as e:
        print(f"Error in JSON parsing: {e}")
        return None

def extract_moves_from_arrays(response):
    """Extract moves from array notation in text.
    
    Args:
        response: LLM response text
        
    Returns:
        List of extracted moves, or empty list if none found
    """
    moves = []
    
    # First try a more comprehensive approach to find arrays of direction strings
    array_match = re.search(r'\[\s*("(?:UP|DOWN|LEFT|RIGHT)"(?:\s*,\s*"(?:UP|DOWN|LEFT|RIGHT)")*)\s*\]', 
                          response, re.IGNORECASE | re.DOTALL)
    
    if array_match:
        # Extract all quoted direction strings from the found array
        directions = re.findall(r'"([^"]+)"', array_match.group(1))
        if directions:
            moves = [move.upper() for move in directions 
                   if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            return moves
    
    # Fallback: Look for arrays of directions in quotes (original method)
    move_arrays = re.findall(r'\[\s*"([^"]+)"\s*(?:,\s*"([^"]+)"\s*)*\]', response)
    if move_arrays:
        for move_group in move_arrays:
            for move in move_group:
                if move and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    moves.append(move.upper())
                    
    return moves 