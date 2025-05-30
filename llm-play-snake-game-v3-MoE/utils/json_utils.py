"""
JSON processing system for the Snake game.
Comprehensive utilities for JSON parsing, validation, and extraction from LLM responses,
with special handling for common formatting variations and error conditions.
"""

import json
import re
import os
import numpy as np
from datetime import datetime

# Global tracking of JSON parsing statistics
json_error_stats = {
    "total_extraction_attempts": 0,
    "successful_extractions": 0,
    "failed_extractions": 0,
    "json_decode_errors": 0,
    "text_extraction_errors": 0,
    "pattern_extraction_success": 0,
    "format_validation_errors": 0
}

class NumPyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        """Handle NumPy types for JSON serialization.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_json_error_stats():
    """Get the current JSON error statistics.
    
    Returns:
        Dictionary with error statistics
    """
    return json_error_stats

def reset_json_error_stats():
    """Reset all JSON error statistics to zero."""
    for key in json_error_stats:
        json_error_stats[key] = 0

def save_experiment_info_json(args, directory):
    """Save experiment configuration information to a JSON file.
    
    Args:
        args: Command line arguments
        directory: Directory to save the file in
        
    Returns:
        Experiment info dictionary
    """
    # Convert args to dict
    args_dict = vars(args)
    
    # Create experiment info
    experiment_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": args_dict,
        "game_statistics": {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        },
        "time_statistics": {
            "total_llm_communication_time": 0,
            "total_game_movement_time": 0,
            "total_waiting_time": 0
        },
        "token_usage_stats": {
            "primary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            },
            "secondary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        },
        "step_stats": {
            "empty_steps": 0,
            "error_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0
        },
        "json_parsing_stats": json_error_stats.copy()
    }
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save to file
    file_path = os.path.join(directory, "summary.json")
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(experiment_info, f, indent=2, cls=NumPyJSONEncoder)
    
    return experiment_info

def save_session_stats(log_dir, **kwargs):
    """Save session statistics to the summary JSON file.
    
    Args:
        log_dir: Directory containing the summary.json file
        **kwargs: Statistics fields to save
    """
    # Read existing summary file
    summary_path = os.path.join(log_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        return
    
    try:
        with open(summary_path, "r", encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"Error reading summary.json: {e}")
        return
    
    # Ensure all required sections exist
    if "game_statistics" not in summary:
        summary["game_statistics"] = {
            "total_games": 0,
            "total_score": 0,
            "total_steps": 0,
            "scores": []
        }
    
    if "time_statistics" not in summary:
        summary["time_statistics"] = {
            "total_llm_communication_time": 0,
            "total_game_movement_time": 0,
            "total_waiting_time": 0
        }
    
    if "token_usage_stats" not in summary:
        summary["token_usage_stats"] = {
            "primary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            },
            "secondary_llm": {
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0
            }
        }
    
    if "step_stats" not in summary:
        summary["step_stats"] = {
            "empty_steps": 0,
            "error_steps": 0,
            "valid_steps": 0,
            "invalid_reversals": 0
        }
    
    # Apply new statistics values to the appropriate sections
    for key, value in kwargs.items():
        if key == "json_error_stats":
            summary["json_parsing_stats"] = value
        elif key == "game_count":
            summary["game_statistics"]["total_games"] = value
        elif key == "total_score":
            summary["game_statistics"]["total_score"] = value
        elif key == "total_steps":
            summary["game_statistics"]["total_steps"] = value
        elif key == "game_scores":
            summary["game_statistics"]["scores"] = value
        elif key == "empty_steps":
            summary["step_stats"]["empty_steps"] = value
        elif key == "error_steps":
            summary["step_stats"]["error_steps"] = value
        elif key == "valid_steps":
            summary["step_stats"]["valid_steps"] = value
        elif key == "invalid_reversals":
            summary["step_stats"]["invalid_reversals"] = value
        elif key == "time_stats":
            # Handle time statistics if provided
            if value and isinstance(value, dict):
                if "llm_communication_time" in value:
                    summary["time_statistics"]["total_llm_communication_time"] = value["llm_communication_time"]
                if "game_movement_time" in value:
                    summary["time_statistics"]["total_game_movement_time"] = value["game_movement_time"]
                if "waiting_time" in value:
                    summary["time_statistics"]["total_waiting_time"] = value["waiting_time"]
        elif key == "token_stats":
            # Handle token statistics if provided
            if value and isinstance(value, dict):
                if "primary" in value and isinstance(value["primary"], dict):
                    primary = value["primary"]
                    if "total_tokens" in primary:
                        summary["token_usage_stats"]["primary_llm"]["total_tokens"] = primary["total_tokens"]
                    if "total_prompt_tokens" in primary:
                        summary["token_usage_stats"]["primary_llm"]["total_prompt_tokens"] = primary["total_prompt_tokens"]
                    if "total_completion_tokens" in primary:
                        summary["token_usage_stats"]["primary_llm"]["total_completion_tokens"] = primary["total_completion_tokens"]
                
                if "secondary" in value and isinstance(value["secondary"], dict):
                    secondary = value["secondary"]
                    if "total_tokens" in secondary:
                        summary["token_usage_stats"]["secondary_llm"]["total_tokens"] = secondary["total_tokens"]
                    if "total_prompt_tokens" in secondary:
                        summary["token_usage_stats"]["secondary_llm"]["total_prompt_tokens"] = secondary["total_prompt_tokens"]
                    if "total_completion_tokens" in secondary:
                        summary["token_usage_stats"]["secondary_llm"]["total_completion_tokens"] = secondary["total_completion_tokens"]
        elif key == "step_stats":
            # Handle step statistics if provided as a complete dictionary
            if value and isinstance(value, dict):
                if "empty_steps" in value:
                    summary["step_stats"]["empty_steps"] = value["empty_steps"]
                if "error_steps" in value:
                    summary["step_stats"]["error_steps"] = value["error_steps"]
                if "valid_steps" in value:
                    summary["step_stats"]["valid_steps"] = value["valid_steps"]
                if "invalid_reversals" in value:
                    summary["step_stats"]["invalid_reversals"] = value["invalid_reversals"]
        elif key == "parser_usage_count":
            if "metadata" not in summary:
                summary["metadata"] = {}
            summary["metadata"]["parser_usage_count"] = value
        elif key == "max_empty_moves":
            if "metadata" not in summary:
                summary["metadata"] = {}
            summary["metadata"]["max_empty_moves"] = value
        elif key == "max_consecutive_errors_allowed":
            if "metadata" not in summary:
                summary["metadata"] = {}
            summary["metadata"]["max_consecutive_errors_allowed"] = value
        else:
            # For any other fields, add them at the top level
            summary[key] = value
    
    # Save the summary file
    try:
        with open(summary_path, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, cls=NumPyJSONEncoder)
    except Exception as e:
        print(f"Error writing summary.json: {e}")

def merge_nested_dicts(target, source):
    """Recursively merge two dictionaries, including nested dictionaries.
    
    Args:
        target: Target dictionary to merge into
        source: Source dictionary with values to merge
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge_nested_dicts(target[key], value)
        else:
            target[key] = value

def preprocess_json_string(json_str):
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

def validate_json_format(data):
    """Validate that data contains the expected format for the game.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        Tuple of (is_valid, message) where:
            is_valid: Boolean indicating if the data is valid
            message: Error message if not valid, None otherwise
    """
    if not isinstance(data, dict):
        print(f"JSON validation error: Data is not a dictionary, it's {type(data)}")
        return False, "Data is not a dictionary"
    
    if "moves" not in data:
        print(f"JSON validation error: Missing 'moves' key in {data.keys()}")
        return False, "Missing 'moves' key"
    
    if not isinstance(data["moves"], list):
        print(f"JSON validation error: Moves is not a list, it's {type(data['moves'])}")
        return False, "Moves is not a list"
    
    # Validate moves
    valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
    for i, move in enumerate(data["moves"]):
        if not isinstance(move, str):
            print(f"JSON validation error: Move {i} is not a string, it's {type(move)}")
            return False, f"Move {i} is not a string"
        
        move_upper = move.upper()
        if move_upper not in valid_moves:
            print(f"JSON validation error: Invalid move: '{move}' (upper: '{move_upper}'), valid moves are {valid_moves}")
            return False, f"Invalid move: {move}"
    
    return True, None

def extract_json_from_code_block(response):
    """Extract JSON data from a code block in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    # Match JSON code blocks with comprehensive language identifier patterns
    # Covers: ```json, ```javascript, ```js, and plain ```
    code_block_matches = re.findall(r'```(?:json|javascript|js)?\s*([\s\S]*?)```', response, re.DOTALL)
    
    print(f"Found {len(code_block_matches)} code blocks in response")
    
    for i, code_block in enumerate(code_block_matches):
        try:
            # Preprocess the code block
            processed_block = preprocess_json_string(code_block)
            
            # Print first part of processed block for debugging
            preview = processed_block[:100] + "..." if len(processed_block) > 100 else processed_block
            print(f"Processing code block {i+1}: {preview}")
            
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
                if move_upper in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    valid_moves.append(move_upper)
            
            if valid_moves:
                print(f"✅ Successfully extracted moves using pattern matching: {valid_moves}")
                return {"moves": valid_moves}
    except Exception as e:
        print(f"❌ Error during pattern extraction: {e}")
    
    return None

def extract_valid_json(text, game_state=None):
    """Extract valid JSON data from text.
    
    Attempts multiple extraction strategies:
    1. Direct JSON parsing
    2. Code block extraction
    3. Text-based extraction
    4. Move array extraction
    
    Args:
        text: Text that may contain JSON
        game_state: Optional GameData instance to record parsing stats
        
    Returns:
        Parsed JSON data or None if no valid JSON found
    """
    json_error_stats["total_extraction_attempts"] += 1
    
    # Initialize tracking in game_state if provided
    if game_state is not None:
        game_state.record_json_extraction_attempt(success=False)
    
    try:
        # First try to parse the entire text as JSON
        data = json.loads(text)
        if isinstance(data, dict) and "moves" in data:
            # Validate the moves format
            is_valid, error_msg = validate_json_format(data)
            if not is_valid:
                print(f"JSON format validation error: {error_msg}")
                json_error_stats["format_validation_errors"] += 1
                if game_state is not None:
                    game_state.record_json_extraction_attempt(success=False, error_type="format")
                return None
                
            json_error_stats["successful_extractions"] += 1
            
            # Print detailed info about the moves
            print(f"✅ Successfully extracted moves: {data['moves']}")
            
            # Record success in game_state if provided
            if game_state is not None:
                game_state.record_json_extraction_attempt(success=True)
                
            return data
        else:
            print(f"JSON missing 'moves' key or not a dict. Keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        json_error_stats["json_decode_errors"] += 1
        
        # Record error type in game_state if provided
        if game_state is not None:
            game_state.record_json_extraction_attempt(success=False, error_type="decode")
    
    # Try extracting from code block
    json_data = extract_json_from_code_block(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            print(f"JSON format validation error (code block): {error_msg}")
            json_error_stats["format_validation_errors"] += 1
            if game_state is not None:
                game_state.record_json_extraction_attempt(success=False, error_type="format")
            return None
            
        json_error_stats["successful_extractions"] += 1
        
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from code block: {json_data['moves']}")
        
        # Record success in game_state if provided
        if game_state is not None:
            game_state.record_json_extraction_attempt(success=True)
            
        return json_data
        
    # Try extracting from regular text
    json_data = extract_json_from_text(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            print(f"JSON format validation error (text): {error_msg}")
            json_error_stats["format_validation_errors"] += 1
            if game_state is not None:
                game_state.record_json_extraction_attempt(success=False, error_type="format")
            return None
            
        json_error_stats["successful_extractions"] += 1
        
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from text: {json_data['moves']}")
        
        # Record success in game_state if provided
        if game_state is not None:
            game_state.record_json_extraction_attempt(success=True)
            
        return json_data
            
    # As a last resort, try to extract move arrays directly
    json_data = extract_moves_from_arrays(text)
    if json_data:
        # Validate the moves format
        is_valid, error_msg = validate_json_format(json_data)
        if not is_valid:
            print(f"JSON format validation error (arrays): {error_msg}")
            json_error_stats["format_validation_errors"] += 1
            if game_state is not None:
                game_state.record_json_extraction_attempt(success=False, error_type="format")
            return None
            
        json_error_stats["successful_extractions"] += 1
        
        # Print detailed info about the moves
        print(f"✅ Successfully extracted moves from arrays: {json_data['moves']}")
        
        # Record success in game_state if provided
        if game_state is not None:
            game_state.record_json_extraction_attempt(success=True)
            
        return json_data
            
    json_error_stats["failed_extractions"] += 1
    print("❌ Failed to extract any valid moves from the response")
    
    # No valid JSON found
    return None

def extract_json_from_text(response):
    """Extract JSON data from text response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted JSON data as a dictionary, or None if not found/invalid
    """
    try:
        # Direct JSON parsing for clean responses
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "moves" in data:
                return data
        except json.JSONDecodeError:
            # Continue to alternative extraction methods
            pass
        except Exception as e:
            print(f"Error during direct JSON parsing: {e}")
            
        # Match JSON patterns with quoted or unquoted keys
        json_match = re.search(r'\{[\s\S]*?["\']moves["\']?\s*:\s*\[[\s\S]*?\][\s\S]*?\}', response, re.DOTALL)
        
        # More permissive pattern for unquoted keys
        if not json_match:
            json_match = re.search(r'\{\s*moves\s*:[\s\S]*?\}', response, re.DOTALL)
            
        if not json_match:
            # No JSON structure found
            json_error_stats["text_extraction_errors"] += 1
            return None
            
        json_str = json_match.group(0)
        # Preprocess to standardize format
        json_str = preprocess_json_string(json_str)
        
        # Parse the preprocessed JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            json_error_stats["text_extraction_errors"] += 1
            print(f"JSON parsing error: {e}")
            
            # Extract moves using pattern matching
            return extract_moves_pattern(json_str)
            
    except Exception as e:
        print(f"JSON extraction error: {e}")
        return None

def extract_moves_pattern(json_str):
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
        valid_moves = [move.upper() for move in move_matches 
                      if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
    
        if valid_moves:
            json_error_stats["pattern_extraction_success"] += 1
            return {"moves": valid_moves}
        return None
    except Exception as e:
        print(f"Move extraction error: {e}")
        return None

def extract_moves_from_arrays(response):
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
        
        # Check if these are valid moves
        valid_moves = [item.upper() for item in quoted_items 
                      if item.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
        
        if valid_moves and len(valid_moves) > 0:
            return {"moves": valid_moves}
    
    return None 