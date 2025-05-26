"""
Utility module for JSON processing in the Snake game.
Consolidates JSON extraction and validation functions used by multiple components.
"""

import os
import json
import re

# Global counter for JSON extraction errors
json_error_stats = {
    "total_extraction_attempts": 0,
    "successful_extractions": 0,
    "failed_extractions": 0,
    "json_decode_errors": 0,
    "format_validation_errors": 0,
    "code_block_extraction_errors": 0,
    "text_extraction_errors": 0,
    "fallback_extraction_success": 0
}

def get_json_error_stats():
    """Get the current JSON error statistics.
    
    Returns:
        Dictionary with error statistics
    """
    return json_error_stats

def reset_json_error_stats():
    """Reset the JSON error statistics."""
    global json_error_stats
    json_error_stats = {
        "total_extraction_attempts": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "json_decode_errors": 0,
        "format_validation_errors": 0,
        "code_block_extraction_errors": 0,
        "text_extraction_errors": 0,
        "fallback_extraction_success": 0
    }

def preprocess_json_string(json_str):
    """Preprocess a JSON string to handle common issues.
    
    Args:
        json_str: JSON string to preprocess
        
    Returns:
        Preprocessed JSON string
    """
    # Handle single quotes as double quotes
    # Only do this conversion for keys and values, not within string content
    in_string = False
    processed = ""
    for i, char in enumerate(json_str):
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            in_string = not in_string
            processed += char
        elif char == "'" and not in_string and (i == 0 or json_str[i-1] != '\\'):
            processed += '"'  # Replace ' with " for JSON keys
        else:
            processed += char
    
    # Handle unquoted keys
    processed = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', processed)
    
    # Handle trailing commas in arrays and objects
    processed = re.sub(r',\s*([}\]])', r'\1', processed)
    
    # Handle multiline strings
    multiline_strings = {}
    counter = 0
    
    # First, extract all string values and replace them with placeholders
    for match in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', processed):
        value = match.group(1)
        if '\n' in value:
            key = f"__MULTILINE_STRING_{counter}__"
            multiline_strings[key] = value
            counter += 1
    
    # Replace placeholders with properly escaped strings
    for placeholder, value in multiline_strings.items():
        escaped_value = value.replace('\n', '\\n').replace('\r', '\\r')
        processed = processed.replace(placeholder, escaped_value)
    
    return processed

def validate_json_format(data):
    """Validate the format of parsed JSON data.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Boolean indicating if the format is valid
    """
    global json_error_stats
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        json_error_stats["format_validation_errors"] += 1
        return False
    
    # Check if it has a 'moves' key
    if 'moves' not in data:
        json_error_stats["format_validation_errors"] += 1
        return False
    
    # Check if 'moves' is a list
    if not isinstance(data['moves'], list):
        json_error_stats["format_validation_errors"] += 1
        return False
    
    # Check if all moves are valid directions
    valid_directions = {'UP', 'DOWN', 'LEFT', 'RIGHT'}
    for move in data['moves']:
        if not isinstance(move, str) or move.upper() not in valid_directions:
            json_error_stats["format_validation_errors"] += 1
            return False
    
    return True

def extract_moves_fallback(json_str):
    """Extract moves from a JSON string using pattern matching when normal parsing fails.
    
    Args:
        json_str: JSON-like string that couldn't be parsed normally.
        
    Returns:
        Dictionary with "moves" key or None if extraction failed.
    """
    try:
        moves_array_match = re.search(r'["\']?moves["\']?\s*:\s*\[([\s\S]*?)\]', json_str, re.DOTALL)
        if not moves_array_match:
            return None
            
        moves_array = moves_array_match.group(1)
        move_matches = re.findall(r'["\']([^"\']+)["\']', moves_array)
        valid_moves = [move.upper() for move in move_matches 
                      if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
    
        if valid_moves:
            json_error_stats["fallback_extraction_success"] += 1
            return {"moves": valid_moves}
        return None
    except Exception as e:
        print(f"Error in extract_moves_fallback: {e}")
        return None

def extract_moves_from_arrays(response_text):
    """Extract moves from array-like structures in the LLM response text.
    
    Args:
        response_text: LLM response text.
        
    Returns:
        Dictionary with "moves" key or None if extraction failed.
    """
    try:
        array_matches = re.findall(r'\[(.*?)\]', response_text, re.DOTALL)
        for array_str in array_matches:
            quoted_items = re.findall(r'["\']([^"\']+)["\']', array_str)
            valid_moves = [item.upper() for item in quoted_items 
                          if item.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
            if valid_moves:
                json_error_stats["fallback_extraction_success"] += 1
                return {"moves": valid_moves}
        return None
    except Exception as e:
        print(f"Error in extract_moves_from_arrays: {e}")
        return None

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in text.
    
    Args:
        response: Text that may contain a code block with JSON
        
    Returns:
        Parsed JSON data or None if extraction failed
    """
    try:
        # Match code blocks with optional language specifier
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response, re.MULTILINE)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            try:
                # Try direct parsing first
                data = json.loads(json_str)
                if validate_json_format(data):
                    return data
            except json.JSONDecodeError:
                # Try preprocessing
                try:
                    preprocessed = preprocess_json_string(json_str)
                    data = json.loads(preprocessed)
                    if validate_json_format(data):
                        return data
                except json.JSONDecodeError:
                    # Failed after preprocessing
                    json_error_stats["code_block_extraction_errors"] += 1
        return None
    except Exception as e:
        json_error_stats["code_block_extraction_errors"] += 1
        print(f"Error extracting JSON from code block: {e}")
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
        except json.JSONDecodeError:
            # Expected error for malformed JSON, just continue to next method
            pass
        except Exception as e:
            # Unexpected error during direct parsing
            print(f"Unexpected error during direct JSON parsing: {e}")
            
        # Try to match both double-quoted and single-quoted JSON patterns
        json_match = re.search(r'\{[\s\S]*?["\']moves["\']?\s*:\s*\[[\s\S]*?\][\s\S]*?\}', response, re.DOTALL)
        
        # If that fails, try a more permissive pattern that might match unquoted keys
        if not json_match:
            json_match = re.search(r'\{\s*moves\s*:[\s\S]*?\}', response, re.DOTALL)
            
        if not json_match:
            # No JSON-like structure found
            return None
            
        json_str = json_match.group(0)
        # Use our comprehensive preprocessing function
        json_str = preprocess_json_string(json_str)
        
        # Now try to parse the cleaned JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            json_error_stats["text_extraction_errors"] += 1
            print(f"Failed to parse JSON after preprocessing: {e}, trying fallback extraction")
            
            # Try to extract just the moves array using fallback method
            return extract_moves_fallback(json_str)
    except Exception as e:
        json_error_stats["text_extraction_errors"] += 1
        print(f"Unexpected error in JSON extraction: {e}")
        return None

def extract_valid_json(text):
    """Extract valid JSON data from text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON data or None if no valid JSON found
    """
    json_error_stats["total_extraction_attempts"] += 1
    
    try:
        # First try to parse the entire text as JSON
        data = json.loads(text)
        if validate_json_format(data):
            json_error_stats["successful_extractions"] += 1
            return data
    except json.JSONDecodeError:
        json_error_stats["json_decode_errors"] += 1
        # Try with our preprocessing for single quotes and unquoted keys
        try:
            preprocessed_text = preprocess_json_string(text)
            data = json.loads(preprocessed_text)
            if validate_json_format(data):
                json_error_stats["successful_extractions"] += 1
                return data
        except json.JSONDecodeError:
            json_error_stats["json_decode_errors"] += 1
        
        # Try extracting from code block
        json_data = extract_json_from_code_block(text)
        if json_data:
            json_error_stats["successful_extractions"] += 1
            return json_data
            
        # Try extracting from regular text
        json_data = extract_json_from_text(text)
        if json_data:
            json_error_stats["successful_extractions"] += 1
            return json_data
                
    json_error_stats["failed_extractions"] += 1
    return None 