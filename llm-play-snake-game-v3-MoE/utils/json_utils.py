"""
JSON processing system for the Snake game.
Comprehensive utilities for JSON parsing, validation, and extraction from LLM responses,
with special handling for common formatting variations and error conditions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from config.game_constants import VALID_MOVES

def preprocess_json_string(json_str: str) -> str:
    """Preprocess a JSON string to fix common formatting issues.
    
    Args:
        json_str: JSON string to preprocess
        
    Returns:
        Preprocessed JSON string
    """
    # Remove comments
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Fix unquoted keys
    json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)
    
    # Fix trailing commas in arrays
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    # Fix trailing commas in objects
    json_str = re.sub(r',\s*\}', '}', json_str)
    
    # Fix single quotes to double quotes (carefully, to avoid breaking JSON)
    in_string = False
    in_escape = False
    result = []
    
    for char in json_str:
        if char == '\\' and not in_escape:
            in_escape = True
            result.append(char)
        elif in_escape:
            in_escape = False
            result.append(char)
        elif char == '"' and not in_escape:
            in_string = not in_string
            result.append(char)
        elif char == "'" and not in_string and not in_escape:
            result.append('"')
        else:
            result.append(char)
    
    return ''.join(result)

def validate_json_format(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate that the JSON data has the expected format.
    
    Args:
        data: JSON data to validate
        
    Returns:
        (is_valid, error_message) tuple. If is_valid is False, error_message
        contains a description of the error.
    """
    if not isinstance(data, dict):
        return False, "JSON data is not a dictionary"
        
    if "moves" not in data:
        return False, "JSON data does not contain a 'moves' key"
        
    if not isinstance(data["moves"], list):
        return False, "The 'moves' field is not a list"
        
    # Validate all moves are valid directions
    valid_moves = VALID_MOVES
    
    for i, move in enumerate(data["moves"]):
        if not isinstance(move, str):
            return False, f"Move is not a string: {move}"
        
        # Convert move to uppercase for case-insensitive validation
        move_upper = move.upper()
        if move_upper not in valid_moves:
            print(f"JSON validation error: Invalid move: '{move}' (upper: '{move_upper}'), valid moves are {valid_moves}")
            return False, f"Invalid move: {move}"
        
        # Ensure all moves are in uppercase format
        data["moves"][i] = move_upper
    
    return True, None

def extract_json_from_code_block(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON data from a code block in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    # Match JSON code blocks with comprehensive language identifier patterns
    # Covers: ```json, ```javascript, ```js, and plain ```
    code_block_matches = re.findall(r'```(?:json|javascript|js)?\s*([\s\S]*?)```', response, re.DOTALL)
    
    for i, code_block in enumerate(code_block_matches):
        try:
            # Preprocess the code block
            processed_block = preprocess_json_string(code_block)
            
            # Try to parse the JSON
            data = json.loads(processed_block)
            
            # Validate it has the expected format
            if isinstance(data, dict) and "moves" in data:
                print(f"✅ Valid JSON found in code block {i+1}")
                return data
            else:
                print(f"❌ Code block {i+1} does not contain a 'moves' key")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in code block {i+1}: {e}")
            continue
        except Exception as e:
            print(f"❌ Error during code block extraction for block {i+1}: {e}")
    
    # If we get here, we couldn't extract valid JSON from any code block
    if len(code_block_matches) > 0:
        pass

    # If the normal extraction fails, try a more aggressive pattern matching approach
    # This handles cases where the LLM includes formatting characters around JSON
    try:
        # Look for JSON-like patterns in the text
        json_pattern = r'\{\s*"moves"\s*:\s*\[(.*?)\]'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            moves_content = match.group(1)
            # Extract the move strings from the array
            move_matches = re.findall(r'"([^"]+)"', moves_content)
            
            # Validate and convert to uppercase
            valid_moves = []
            for move in move_matches:
                move_upper = move.upper()
                if move_upper in VALID_MOVES:
                    valid_moves.append(move_upper)
            
            if valid_moves:
                print(f"✅ Successfully extracted moves using pattern matching: {valid_moves}")
                return {"moves": valid_moves}
    except Exception as e:
        print(f"❌ Error during pattern extraction: {e}")
    
    return None

def extract_valid_json(
    text: str,
    attempt_id: int = 0,
) -> Optional[Dict[str, Any]]:
    """Extract valid JSON data from text.
    
    Attempts multiple extraction strategies:
    1. Direct JSON parsing
    2. Code block extraction
    3. Text-based extraction
    4. Move array extraction
    
    Args:
        text: Text that may contain JSON
        game_state: Optional GameData instance to record parsing stats
        attempt_id: Attempt number (0 for first attempt, incremented for retries)
        
    Returns:
        Parsed JSON data or None if no valid JSON found
    """
    try:
        # First try to parse the entire text as JSON
        data = json.loads(text)
        if isinstance(data, dict) and "moves" in data:
            # Validate the moves format
            is_valid, error_msg = validate_json_format(data)
            if not is_valid:
                # Only print error message on first attempt
                if attempt_id == 0:
                    print(f"JSON format validation error: {error_msg}")
                return None
                
            # Print detailed info about the moves
            print(f"✅ Successfully extracted moves: {data['moves']}")
            
            return data
        else:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON missing 'moves' key or not a dict. Keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
    except json.JSONDecodeError as e:
        # Only print error message on first attempt
        if attempt_id == 0:
            print(f"JSON decode error: {e}")
    
    # Try extracting from code block
    json_data = extract_json_from_code_block(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON format validation error (code block): {error_msg}")
            return None
            
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from code block: {json_data['moves']}")
        
        return json_data

    # Try extracting from regular text
    json_data = extract_json_from_text(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON format validation error (text): {error_msg}")
            return None
            
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from text: {json_data['moves']}")
        
        return json_data
            
    # As a last resort, try to extract move arrays directly
    json_data = extract_moves_from_arrays(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            # Only print error message on first attempt
            if attempt_id == 0:
                print(f"JSON format validation error (arrays): {error_msg}")
            return None
            
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from arrays: {json_data['moves']}")
        return json_data
            
    # No valid JSON found
    return None

def extract_json_from_text(response: str) -> Optional[Dict[str, List[str]]]:
    """Extract JSON data from text response.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    # Special case pattern matching for moves array
    moves_pattern = r'"moves"\s*:\s*\[(.*?)\]'
    match = re.search(moves_pattern, response, re.DOTALL)
    
    if match:
        moves_content = match.group(1)
        
        # Extract quoted strings
        moves = re.findall(r'["\']([^"\']+)["\']', moves_content)
        
        if moves:
            # Convert all moves to uppercase
            standardized_moves = [move.upper() for move in moves]
            
            # Filter to only valid moves
            valid_moves = [move for move in standardized_moves if move in VALID_MOVES]
            
            if valid_moves:
                print(f"Extracted moves from text: {valid_moves}")
                return {"moves": valid_moves}
    
    # Try extracting moves pattern
    moves = extract_moves_pattern(response)
    if moves:
        # Convert all moves to uppercase
        standardized_moves = [move.upper() for move in moves]
        
        # Filter to only valid moves
        valid_moves = [move for move in standardized_moves if move in VALID_MOVES]
        
        if valid_moves:
            print(f"Extracted moves using pattern matching: {valid_moves}")
            return {"moves": valid_moves}
    
    return None

def extract_moves_pattern(json_str: str) -> Optional[Dict[str, List[str]]]:
    """Extract moves from a JSON string using pattern matching.
    
    Args:
        json_str: JSON string to extract moves from
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    try:
        # Extract moves array
        moves_array_match = re.search(r'["\']?moves["\']?\s*:\s*\[([\s\S]*?)\]', json_str, re.DOTALL)
        if not moves_array_match:
            return None
            
        moves_array = moves_array_match.group(1)
        # Extract valid move strings
        move_matches = re.findall(r'["\']([^"\']+)["\']', moves_array)
        
        # Convert all moves to uppercase for case-insensitive validation
        valid_moves = []
        for move in move_matches:
            move_upper = move.upper()
            if move_upper in VALID_MOVES:
                valid_moves.append(move_upper)
    
        if valid_moves:
            return {"moves": valid_moves}
        return None
    except Exception as e:
        print(f"Move extraction error: {e}")
        return None

def extract_moves_from_arrays(response: str) -> Optional[Dict[str, List[str]]]:
    """Extract moves from arrays in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with moves key or None if extraction failed
    """
    # Look for arrays in the response
    array_matches = re.findall(r'\[(.*?)\]', response, re.DOTALL)
    
    for array_str in array_matches:
        # Extract quoted strings
        quoted_items = re.findall(r'["\']([^"\']+)["\']', array_str)
        
        # Convert all items to uppercase for case-insensitive validation
        valid_moves = []
        for item in quoted_items:
            item_upper = item.upper()
            if item_upper in VALID_MOVES:
                valid_moves.append(item_upper)  # Store in uppercase format
        
        if valid_moves and len(valid_moves) > 0:
            return {"moves": valid_moves}
    
    return None
